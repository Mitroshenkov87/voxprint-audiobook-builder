"""``voxprint bench``: narrate one fixed text with the batched path and with CUDA Graphs, print speed and peak VRAM.

For the comparison on a real PC before the fast decode path (:mod:`core.fast_decode`) may become the default.  Every mode
loads its own engine, synthesizes one warm-up phrase (not timed; with CUDA Graphs that includes the capture), then the
fixed text phrase by phrase - in batches planned by the VRAM policy for ``batched``, one by one for ``graphs``.

* ``x_realtime`` = seconds of audio per second of synthesis (higher is faster); ``rtf`` = the inverse (as in
  docs/HOW-IT-WORKS.md, lower is faster);
* ``peak_vram_gb`` = the peak PyTorch held on the card during the timed part (``max_memory_reserved``);
  ``card_peak_used_gb`` = that plus what other programs and the driver used when the mode started.

Qt-free; the engine factory is injected so tests run without a GPU.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

MODES = ("batched", "graphs")

#: Fixed Russian text (``core/data/bench_ru.txt``): a warm-up phrase, then 12 phrases of 20-95 characters (dialogue, long
#: and short sentences, letter yo).
TEXT_FILE = Path(__file__).with_name("data") / "bench_ru.txt"


def load_text(path: Path = TEXT_FILE) -> Tuple[str, List[str]]:
    """``(warm-up phrase, timed phrases)`` from the bench text file (lines starting with ``#`` are comments)."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    return lines[0], lines[1:]


WARMUP, _PHRASES = load_text()
TEXT_RU: Sequence[str] = tuple(_PHRASES)


@dataclass
class ModeResult:
    """Timing and video memory for one benchmark mode."""

    mode: str
    ok: bool
    reason: str = ""
    phrases: int = 0
    batch: int = 0
    audio_s: float = 0.0
    synth_s: float = 0.0
    load_s: float = 0.0
    warmup_s: float = 0.0
    x_realtime: float = 0.0
    rtf: float = 0.0
    peak_vram_gb: Optional[float] = None
    card_peak_used_gb: Optional[float] = None
    card_total_gb: Optional[float] = None
    wavs: List[str] = field(default_factory=list)

    def line(self) -> str:
        """One terminal line: the speed, or why this mode was skipped."""
        if not self.ok:
            return f"{self.mode:8s} skipped: {self.reason}"
        vram = "" if self.peak_vram_gb is None else (
            f", peak VRAM {self.peak_vram_gb:.2f} GB (card {self.card_peak_used_gb:.2f} of {self.card_total_gb:.1f} GB)")
        return (f"{self.mode:8s} {self.x_realtime:5.2f}x realtime (RTF {self.rtf:.3f}): {self.audio_s:.1f} s of audio in "
                f"{self.synth_s:.1f} s, batch {self.batch}, load {self.load_s:.1f} s, warm-up {self.warmup_s:.1f} s{vram}")


def _cuda(engine: Any) -> Optional[Any]:
    try:
        import torch

        dev = str(getattr(engine, "device", "cpu"))
        if torch.cuda.is_available() and dev.startswith("cuda"):
            return torch, torch.device(dev)
    except Exception:  # noqa: BLE001
        pass
    return None


def _batches(texts: Sequence[str], size: int) -> List[List[str]]:
    size = max(1, int(size))
    return [list(texts[i:i + size]) for i in range(0, len(texts), size)]


def run_mode(mode: str, make_engine: Callable[[str], Any], texts: Sequence[str] = TEXT_RU,
             out_dir: Optional[Path] = None, log: Callable[[str], None] = lambda _m: None) -> ModeResult:
    """Load an engine for ``mode``, warm it up, synthesize ``texts`` and measure."""
    res = ModeResult(mode=mode, ok=False)
    t0 = time.perf_counter()
    try:
        engine = make_engine(mode)
    except Exception as exc:  # noqa: BLE001
        res.reason = f"engine could not be loaded: {exc}"
        return res
    res.load_s = time.perf_counter() - t0
    try:
        actual = getattr(engine, "decode_mode", mode)
        if mode == "graphs" and actual != "graphs":
            res.reason = "CUDA Graphs are not available here (see the log: faster-qwen3-tts missing, no GPU or capture failed)"
            return res
        cu = _cuda(engine)
        t1 = time.perf_counter()
        engine.synthesize(WARMUP)
        res.warmup_s = time.perf_counter() - t1
        base_used = None
        if cu is not None:
            torch, dev = cu
            torch.cuda.synchronize(dev)
            free, total = torch.cuda.mem_get_info(dev)
            res.card_total_gb = total / 1024 ** 3
            base_used = (total - free) / 1024 ** 3 - torch.cuda.memory_reserved(dev) / 1024 ** 3
            torch.cuda.reset_peak_memory_stats(dev)
        size = 1 if mode == "graphs" else int(engine.max_batch()) if callable(getattr(engine, "max_batch", None)) else 1
        res.batch = size
        log(f"{mode}: {len(texts)} phrases, batch {size}")
        audio: List[np.ndarray] = []
        t2 = time.perf_counter()
        for group in _batches(texts, size):
            if len(group) > 1 and callable(getattr(engine, "synthesize_batch", None)):
                audio.extend(engine.synthesize_batch(group))
            else:
                audio.extend(engine.synthesize(t) for t in group)
        if cu is not None:
            cu[0].cuda.synchronize(cu[1])
        res.synth_s = time.perf_counter() - t2
        sr = int(getattr(engine, "sample_rate", 24000))
        res.audio_s = float(sum(len(a) for a in audio)) / sr
        res.phrases = len(audio)
        if res.synth_s > 0 and res.audio_s > 0:
            res.x_realtime = res.audio_s / res.synth_s
            res.rtf = res.synth_s / res.audio_s
        if cu is not None:
            torch, dev = cu
            res.peak_vram_gb = torch.cuda.max_memory_reserved(dev) / 1024 ** 3
            res.card_peak_used_gb = (base_used or 0.0) + res.peak_vram_gb
        if out_dir is not None:
            import soundfile as sf

            out_dir.mkdir(parents=True, exist_ok=True)
            gap = np.zeros(int(sr * 0.4), dtype=np.float32)
            joined = np.concatenate([np.concatenate([a, gap]) for a in audio]) if audio else gap
            f = out_dir / f"bench-{mode}.wav"
            sf.write(str(f), joined, sr)
            res.wavs.append(str(f))
        res.ok = True
        return res
    except Exception as exc:  # noqa: BLE001
        res.reason = f"{type(exc).__name__}: {exc}"
        return res
    finally:
        try:
            engine.close()
        except Exception:  # noqa: BLE001
            pass


def run(modes: Sequence[str], make_engine: Callable[[str], Any], texts: Sequence[str] = TEXT_RU,
        out_dir: Optional[Path] = None, log: Callable[[str], None] = lambda _m: None) -> List[ModeResult]:
    """Run each name in ``modes`` and return one result per mode."""
    return [run_mode(m, make_engine, texts, out_dir, log) for m in modes]


def summary(results: Sequence[ModeResult]) -> Dict[str, Any]:
    """A dict of the mode results, plus how CUDA Graphs compared with batching when both ran."""
    out: Dict[str, Any] = {"modes": [asdict(r) for r in results]}
    ok = {r.mode: r for r in results if r.ok}
    if "batched" in ok and "graphs" in ok and ok["batched"].x_realtime > 0:
        out["graphs_vs_batched"] = round(ok["graphs"].x_realtime / ok["batched"].x_realtime, 3)
    return out
