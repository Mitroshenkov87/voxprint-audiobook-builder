"""Automatic pick of the best checkpoint and adapter strength after a training run (Qt-free).

The training loss says little about how a voice sounds (a falling loss over-fits as often as it improves), so after the run
the last :data:`PICK_EPOCHS` epoch checkpoints are each tried at a few adapter strengths (:mod:`core.adapter_strength`) on up
to :data:`MAX_PHRASES` sentences the training never saw (the held-out clips' texts, ``checkpoints/holdout.json``; the preview
sentence when the dataset was too small for a holdout).  Every sample is scored by

* the character error rate of a Qwen3-ASR re-reading (does it say the text?),
* the speaker similarity to the voice (Qwen3-TTS's own encoder, against the averaged speaker embedding when there is one),
* the predicted MOS (DNSMOS OVRL) when that model is downloaded,

and a sample that runs to the length cap (no end of speech) is penalised.  The winner's adapter replaces the final one, its
strength goes to ``voice.json``, ``checkpoints/pick.json`` keeps every score, and the other epoch folders are deleted (disk
use stays at one checkpoint).  Synthesis, recognition and MOS are injectable (tests use fakes).

NEEDS VALIDATION ON A GPU: weights, phrase count and candidate grid are research-based guesses (~20-40 s per candidate on an
RTX 4090 was the estimate; 3 epochs x 3 strengths x 4 phrases here).
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core import adapter_strength, voice_check
from core.events import CancelToken, ProgressCallback, Stage, noop_progress
from core.i18n import tr

log = logging.getLogger("voxprint.checkpoint_pick")

PICK_EPOCHS = 3
MAX_PHRASES = 4
MIN_PHRASE_CHARS = 20            # very short texts say little about a voice and often trip the end-of-speech check
WEIGHTS = {"cer": 0.5, "sim": 0.3, "mos": 0.2}
NO_STOP_PENALTY = 0.5            # subtracted x the share of samples that never stopped
PICK_FILE = "pick.json"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_config.json")


def epoch_folders(adapter_dir: Path) -> List[Tuple[int, Path]]:
    """``[(epoch, folder)]`` of the saved checkpoints, oldest first."""
    out = []
    for d in (Path(adapter_dir) / "checkpoints").glob("epoch_*"):
        try:
            n = int(d.name.split("_", 1)[1])
        except ValueError:
            continue
        if (d / "adapter_model.safetensors").is_file():
            out.append((n, d))
    return sorted(out)


def pick_phrases(adapter_dir: Path, language: str, n: int = MAX_PHRASES) -> List[str]:
    """Up to ``n`` held-out texts spread from short to long; the preview sentence when there is no holdout."""
    from core.lora_trainer import HOLDOUT_FILE

    texts: List[str] = []
    try:
        data = json.loads((Path(adapter_dir) / "checkpoints" / HOLDOUT_FILE).read_text(encoding="utf-8"))
        texts = [str(r.get("text", "")).strip() for r in data.get("rows", [])]
    except (OSError, ValueError, AttributeError):
        pass
    texts = [t for t in texts if t]
    long_enough = [t for t in texts if len(t) >= MIN_PHRASE_CHARS]
    texts = sorted(long_enough or texts, key=len)
    if not texts:
        from workers.preview_runner import SAMPLE_TEXT

        return [SAMPLE_TEXT.get((language or "english").lower(), SAMPLE_TEXT["english"])]
    if len(texts) <= n:
        return texts
    return [texts[round(i * (len(texts) - 1) / (n - 1))] for i in range(n)] if n > 1 else texts[-1:]


def score(cer: Optional[float], sim: Optional[float], mos: Optional[float], no_stop: float) -> float:
    """Higher is better.  Available metrics are mixed with :data:`WEIGHTS` (re-normalised over the ones present)."""
    parts = {"cer": None if cer is None else 1.0 - min(1.0, max(0.0, cer)),
             "sim": None if sim is None else min(1.0, max(0.0, sim)),
             "mos": None if mos is None else min(1.0, max(0.0, (mos - 1.0) / 4.0))}
    have = {k: v for k, v in parts.items() if v is not None}
    base = sum(WEIGHTS[k] * v for k, v in have.items()) / sum(WEIGHTS[k] for k in have) if have else 0.0
    return base - NO_STOP_PENALTY * no_stop


def _mean(vals: Sequence[Optional[float]]) -> Optional[float]:
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def run_pick(adapter_dir: Path, language: str, *, engine_factory: Callable[[Path, str], Any], asr=None, mos=None,
             scales: Sequence[float] = adapter_strength.PREVIEW_SCALES, last_n: int = PICK_EPOCHS,
             progress: ProgressCallback = noop_progress, cancel: Optional[CancelToken] = None) -> Optional[Dict[str, Any]]:
    """Score every (epoch, strength) candidate; returns ``{"best_epoch", "best_scale", "phrases", "candidates"}`` or ``None``
    when there is nothing to choose (no checkpoints, or an engine that cannot switch adapters / strengths).

    ``engine_factory(adapter_dir, language)`` must return an *unmerged* engine (``switch_adapter``, ``set_adapter_scale``,
    ``synthesize`` and optionally ``synthesize_batch`` / ``speaker_embedding``)."""
    from core import audio_utils as au
    from core.tts_engine import FRAMES_PER_SECOND, max_tokens_for
    from workers.preview_runner import mos_of, reference_embedding, sample_similarity

    cancel = cancel or CancelToken()
    adapter_dir = Path(adapter_dir)
    folders = epoch_folders(adapter_dir)[-max(1, last_n):]
    if not folders:
        return None
    phrases = pick_phrases(adapter_dir, language)
    caps = [max_tokens_for(t) / FRAMES_PER_SECOND - 0.6 for t in phrases]
    ref, ref_sr = au.load_audio(adapter_dir / "ref_sample.wav", 24000)
    grid = [(e, f, s) for e, f in folders for s in scales]
    samples: Dict[Tuple[int, float], List[Tuple[np.ndarray, int, Optional[float]]]] = {}
    eng = engine_factory(adapter_dir, language)
    try:
        if not (callable(getattr(eng, "switch_adapter", None)) and callable(getattr(eng, "set_adapter_scale", None))):
            log.info("checkpoint pick skipped: the engine cannot switch adapters")
            return None
        ref_emb = reference_embedding(eng, adapter_dir, ref, ref_sr)
        current = None
        for i, (epoch, folder, scale) in enumerate(grid):
            cancel.check()
            progress(Stage.SAVE, i / len(grid), tr("progress.pick", i=i + 1, n=len(grid)))
            if folder != current:
                eng.switch_adapter(folder)
                current = folder
            eng.set_adapter_scale(scale)
            batch = getattr(eng, "synthesize_batch", None)
            audios = batch(phrases) if callable(batch) and len(phrases) > 1 else [eng.synthesize(t) for t in phrases]
            sr = int(getattr(eng, "sample_rate", 24000))
            samples[(epoch, scale)] = [(np.asarray(a, dtype=np.float32).reshape(-1), sr,
                                        sample_similarity(eng, np.asarray(a, dtype=np.float32).reshape(-1), sr, ref_emb))
                                       for a in audios]
    finally:
        try:
            eng.close()
        except Exception:  # noqa: BLE001
            pass
    # recognition after the TTS model is gone (both on one GPU would need more VRAM)
    if asr is not None and callable(getattr(asr, "load", None)):
        asr.load()
    try:
        cands = []
        for (epoch, scale), items in samples.items():
            cancel.check()
            cers, sims, moss, stops = [], [], [], []
            for (audio, sr, sim), text, cap in zip(items, phrases, caps):
                hyp = asr.transcribe(audio, sr, (language or "").capitalize() or None).text if asr is not None else None
                cers.append(voice_check.cer(text, hyp) if hyp is not None else None)
                sims.append(sim)
                moss.append(mos_of(mos, audio, sr))
                stops.append(1.0 if len(audio) / sr >= cap else 0.0)
            c = {"epoch": epoch, "scale": scale, "cer": _mean(cers), "sim": _mean(sims), "mos": _mean(moss),
                 "no_stop": float(np.mean(stops)) if stops else 0.0}
            c["score"] = round(score(c["cer"], c["sim"], c["mos"], c["no_stop"]), 4)
            cands.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
    finally:
        if asr is not None and callable(getattr(asr, "unload", None)):
            asr.unload()
    # Ties: the strength nearer the default, then the later epoch (more training on the same evidence)
    best = max(cands, key=lambda c: (c["score"], -abs(c["scale"] - adapter_strength.DEFAULT_SCALE), c["epoch"]))
    progress(Stage.SAVE, 1.0, tr("progress.pick_done", epoch=best["epoch"], scale=f"{best['scale']:.2f}"))
    return {"best_epoch": best["epoch"], "best_scale": best["scale"], "phrases": phrases, "candidates": cands}


def apply_pick(adapter_dir: Path, result: Dict[str, Any], delete_losers: bool = True) -> None:
    """Make the winner the voice's adapter, record the pick, and delete the other epoch checkpoints."""
    adapter_dir = Path(adapter_dir)
    log.info("checkpoint pick: winner epoch %s strength %s; candidates %s", result.get("best_epoch"), result.get("best_scale"),
             json.dumps(result.get("candidates", []), ensure_ascii=False, default=str))
    folders = dict(epoch_folders(adapter_dir))
    win = folders.get(int(result["best_epoch"]))
    if win is None:
        raise FileNotFoundError(f"checkpoint of epoch {result['best_epoch']} is missing")
    for name in ADAPTER_WEIGHTS:
        shutil.copy2(win / name, adapter_dir / name)
    meta_f = adapter_dir / "training_meta.json"
    try:   # the adapter now is the state after best_epoch passes (the keys stay those Alexandria reads)
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        meta["epochs"] = int(result["best_epoch"])
        meta_f.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError):
        log.warning("training_meta.json not updated after the checkpoint pick")
    (adapter_dir / "checkpoints" / PICK_FILE).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if delete_losers:
        for epoch, folder in folders.items():
            if folder != win:
                shutil.rmtree(folder, ignore_errors=True)
