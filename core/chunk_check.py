"""Per-chunk speech-recognition check during narration, with regeneration of chunks that do not say their text.

After a batch is synthesized, each new chunk is re-read by Qwen3-ASR and its character error rate (CER) against the chunk's
text is computed.  Above the threshold (default 15 %) the chunk is generated again, up to ``retries`` times, with the sampling
the community uses for fine-tuned Qwen3-TTS voices (temperature 0.8, top_p 0.85, top_k 30; research notes, item 5) and a
seed derived from the chunk's cache key and the attempt number, so a rerun of the same job makes the same choices.  The attempt
with the lowest CER is kept (the first one wins a tie).  Typical failures caught: skipped or repeated words, babbling, a chunk
cut short.

The check never changes the cache key (``engine tag + text``): a cached chunk is the accepted result and is never checked again,
so resumed jobs and the CPU overlap of chapter assembly / encodes (:mod:`core.narration`) work exactly as before.

Memory: the recogniser is loaded lazily at the first check; it goes to the GPU only if enough video memory is left next to the
TTS model (otherwise it runs on the CPU), and it is unloaded when synthesis ends (:meth:`ChunkChecker.close`).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from core import yo
from core.voice_check import cer

log = logging.getLogger("voxprint.narration")

DEFAULT_MAX_CER = 0.15
DEFAULT_RETRIES = 2
MAX_RETRIES = 5
#: Sampling for the regenerated attempts (Instavar: "Generation parameters" for fine-tuned voices); the first attempt keeps the
#: engine's normal settings, so a job with every chunk passing sounds exactly as without the check.
RETRY_SAMPLING: Dict[str, float] = {"temperature": 0.8, "top_p": 0.85, "top_k": 30}
#: Video memory that must stay free for the TTS batches when the recogniser is put on the GPU (GB, on top of its weights).
VRAM_RESERVE_GB = 2.0


@dataclass
class ChunkCheckOptions:
    """Threshold and number of regeneration attempts."""
    max_cer: float = DEFAULT_MAX_CER
    retries: int = DEFAULT_RETRIES
    sampling: Dict[str, float] = field(default_factory=lambda: dict(RETRY_SAMPLING))
    #: Read ordinals in the recognised text the way the chunk text was prepared (``NarrationOptions.ordinals``).
    ordinals: bool = True
    #: Restore Russian yo in the recognised text the way the chunk was spoken (``NarrationOptions.yo``).
    yo: bool = True


def chunk_seed(key: str, attempt: int) -> int:
    """Deterministic 31-bit seed of a chunk (its cache key) and attempt number."""
    return int(hashlib.sha256(f"{key}|{attempt}".encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def free_vram_gb() -> float:
    """Free CUDA memory in GB (0 without CUDA)."""
    try:
        import torch

        return torch.cuda.mem_get_info()[0] / 1024 ** 3 if torch.cuda.is_available() else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def asr_device(weights_gb: float, free_gb: Callable[[], float] = free_vram_gb) -> str:
    """``cuda`` if the recogniser (+ reserve for the TTS batches) fits into the free video memory, else ``cpu``."""
    return "cuda" if free_gb() >= weights_gb * 1.2 + VRAM_RESERVE_GB else "cpu"


class ChunkChecker:
    """Scores chunks with a recogniser and regenerates the bad ones (see the module docstring)."""

    def __init__(self, asr_factory: Callable[[], object], language: Optional[str] = None,
                 options: Optional[ChunkCheckOptions] = None, normalize: Optional[Callable[[str], str]] = None) -> None:
        self.asr_factory, self.language, self.normalize = asr_factory, language, normalize
        self.options = options or ChunkCheckOptions()
        self.options.retries = max(0, min(MAX_RETRIES, int(self.options.retries)))
        self._asr = None
        self.stats = {"checked": 0, "regenerated": 0, "fixed": 0, "still_bad": 0, "attempts": 0, "failed": 0}

    def _recogniser(self):
        if self._asr is None:
            self._asr = self.asr_factory()
            self._asr.load()
        return self._asr

    def score(self, audio: np.ndarray, sr: int, text: str) -> Optional[float]:
        """CER of ``audio`` against ``text`` (None when recognition fails: the chunk is then accepted as it is)."""
        try:
            hyp = self._recogniser().transcribe(np.asarray(audio, dtype=np.float32), sr, self.language).text
            if self.normalize is not None:
                try:
                    hyp = self.normalize(hyp)
                except Exception:  # noqa: BLE001
                    pass
            return cer(text, hyp)
        except Exception as exc:  # noqa: BLE001 - the check is a bonus; never stop the book for it
            self.stats["failed"] += 1
            log.warning("chunk check: recognition failed: %s", exc)
            return None

    def _again(self, engine, text: str, seed: int) -> Optional[np.ndarray]:
        try:
            fn = getattr(engine, "synthesize_sampled", None)
            audio = fn(text, seed=seed, **self.options.sampling) if callable(fn) else engine.synthesize(text)
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            return audio if audio.size else None
        except Exception as exc:  # noqa: BLE001 - keep what we have
            log.warning("chunk check: regeneration failed: %s", exc)
            return None

    def check(self, engine, text: str, key: str, index: int, audio: np.ndarray, sr: int) -> np.ndarray:
        """The audio to keep for one freshly synthesized chunk."""
        self.stats["checked"] += 1
        best = self.score(audio, sr, text)
        if best is None or best <= self.options.max_cer:
            return audio
        self.stats["regenerated"] += 1
        first = best
        for attempt in range(1, self.options.retries + 1):
            again = self._again(engine, text, chunk_seed(key, attempt))
            self.stats["attempts"] += 1
            if again is None:
                continue
            c = self.score(again, getattr(engine, "sample_rate", sr), text)
            if c is not None and c < best:
                best, audio = c, again
            if best <= self.options.max_cer:
                break
        self.stats["fixed" if best <= self.options.max_cer else "still_bad"] += 1
        log.info("chunk %d: CER %.2f -> %.2f after regeneration", index + 1, first, best)
        return audio

    def close(self) -> None:
        """Free the recogniser."""
        if self._asr is not None:
            try:
                self._asr.unload()
            except Exception:  # noqa: BLE001
                pass
            self._asr = None


def make_default_checker(language: str, options: Optional[ChunkCheckOptions] = None,
                         ready: Optional[Callable[[str], Optional[Path]]] = None) -> Optional[ChunkChecker]:
    """The checker narration uses, or None when no recogniser is installed (never downloads).

    Prefers the fast Qwen3-ASR-0.6B when it is installed (speed matters here more than the last percent), else the 1.7B model.
    The device is decided when the recogniser is first needed, i.e. after the TTS model is on the GPU."""
    from core.asr import make_default_asr
    from core.narration import chain_steps, default_normalizer
    from core import ordinals
    from infra import asr_choice, preload

    found = asr_choice.ready(ready, choice=asr_choice.USE_SMALL)
    if found is None:
        log.warning("chunk check skipped: no speech recognition model is installed")
        return None
    path = Path(found[1])

    def factory():
        gb = preload.weights_bytes(path) / 1024 ** 3
        dev = asr_device(gb)
        log.info("chunk check: %s on %s", found[0], dev)
        return make_default_asr(str(path), dev)

    name = (language or "").strip()
    # the recogniser writes "глава 2" where the chunk text says "глава вторая": prepare its output the same way
    opts = options or ChunkCheckOptions()
    ordinal = ordinals.ordinal_step(language) if opts.ordinals else None
    letter = yo.as_step(language) if opts.yo else None
    # same order as narration.text_steps: ordinals, yo, then the number reader
    return ChunkChecker(factory, name if name and name.lower() != "auto" else None, options,
                        chain_steps(ordinal, letter, default_normalizer(language)))
