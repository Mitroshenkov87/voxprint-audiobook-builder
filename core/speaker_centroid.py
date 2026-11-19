"""Averaged speaker embedding ("centroid") over many clean clips, instead of the embedding of the single reference clip.

Qwen3-TTS conditions every utterance on a speaker embedding (x-vector) of its speaker encoder.  Until now Voxprint took it
from one reference clip (``ref.wav``) - for training, for the universal model's speaker row and for the voice-clone prompt
at narration.  Embeddings of single clips of one speaker agree only at ~0.7 cosine (mood, loudness, words), the average of
30-64 clean clips at 0.85+ (Baseten's Qwen3-TTS voice-cloning notes), so the voice gets a steadier identity.

* clips: 3-15 s, no clipping, not too quiet, the cleanest first (estimated SNR), at most :data:`MAX_CLIPS`;
  fewer than :data:`MIN_CLIPS` usable clips -> no centroid (the old single-clip path);
* stored next to the adapter as :data:`FILENAME` (float32 vector, safetensors) and used by training, the universal model and
  narration; voices without the file (all older voices) keep the old path;
* the ICL part of the voice-clone prompt (reference audio codes + its text) is unchanged - only the x-vector is replaced;
* ``VOXPRINT_SPEAKER_CENTROID=0`` switches it off everywhere (training then uses ``ref.wav``; narration ignores the file).

NEEDS VALIDATION ON A GPU: A/B a voice with and without the centroid (similarity, steadiness across chapters).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("voxprint.speaker_centroid")

FILENAME = "speaker_centroid.safetensors"
MIN_CLIPS = 4
MAX_CLIPS = 64
MIN_SECONDS, MAX_SECONDS = 3.0, 15.0
MIN_RMS_DBFS = -40.0


def enabled() -> bool:
    """False when switched off with ``VOXPRINT_SPEAKER_CENTROID=0``."""
    return os.environ.get("VOXPRINT_SPEAKER_CENTROID", "1").strip().lower() not in ("0", "false", "no", "off")


def select_clips(paths: Sequence[str], sr: int = 24000, max_clips: int = MAX_CLIPS) -> List[Tuple[str, np.ndarray]]:
    """The cleanest usable clips as ``(path, samples at sr)``, best first (estimated SNR; see the module docstring)."""
    from core import audio_utils as au
    from core.quality import audio_stats

    scored = []
    for p in paths:
        try:
            x, _ = au.load_audio(p, sr)
        except Exception as exc:  # noqa: BLE001 - an unreadable clip is simply not used here
            log.warning("centroid: cannot read %s: %s", p, exc)
            continue
        secs = len(x) / sr
        if not MIN_SECONDS <= secs <= MAX_SECONDS:
            continue
        st = audio_stats(x, sr)
        if st["clipped_samples"] > 0 or st["rms_dbfs"] < MIN_RMS_DBFS:
            continue
        scored.append((st["estimated_snr_db"], str(p), x))
    scored.sort(key=lambda t: -t[0])
    return [(p, x) for _, p, x in scored[:max_clips]]


def average(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """Centroid of speaker embeddings.

    The raw vectors are not averaged directly: their lengths differ from clip to clip and the mean of vectors pointing in
    slightly different directions is shorter than any of them, so the talker would see an embedding of an unusual
    magnitude.  Each vector is normalised to unit length, the directions are averaged and re-normalised, and the result
    gets the median length of the individual embeddings back."""
    e = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in embeddings])
    norms = np.linalg.norm(e, axis=1)
    keep = norms > 0
    e, norms = e[keep], norms[keep]
    if not len(e):
        raise ValueError("no non-zero embeddings")
    c = (e / norms[:, None]).mean(axis=0)
    c = c / (np.linalg.norm(c) or 1.0)
    return (c * float(np.median(norms))).astype(np.float32)


def compute(embed: Callable[[np.ndarray, int], np.ndarray], paths: Sequence[str], sr: int = 24000,
            max_clips: int = MAX_CLIPS) -> Optional[Tuple[np.ndarray, List[str]]]:
    """``(centroid, used clip paths)`` from ``embed(samples, sr)`` over the selected clips, or ``None`` with fewer than
    :data:`MIN_CLIPS` usable clips or when switched off."""
    if not enabled():
        return None
    clips = select_clips(paths, sr, max_clips)
    if len(clips) < MIN_CLIPS:
        log.info("centroid: only %d usable clips - using the single reference clip", len(clips))
        return None
    embs = [np.asarray(embed(x, sr), dtype=np.float32).reshape(-1) for _, x in clips]
    return average(embs), [p for p, _ in clips]


def save(folder: Path, vec: np.ndarray, clips: Sequence[str] = ()) -> Path:
    """Write :data:`FILENAME` into ``folder`` (with the clip count and names as metadata)."""
    from safetensors.numpy import save_file

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / FILENAME
    save_file({"speaker_embedding": np.asarray(vec, dtype=np.float32).reshape(-1)}, str(path),
              metadata={"format": "np", "clips": str(len(clips)),
                        "files": json.dumps([Path(c).name for c in clips][:MAX_CLIPS])})
    return path


def load(folder: Path) -> Optional[np.ndarray]:
    """The voice's centroid, or ``None`` (no file, switched off, or unreadable - then the old single-clip path is used)."""
    path = Path(folder) / FILENAME
    if not enabled() or not path.is_file():
        return None
    try:
        from safetensors.numpy import load_file

        return np.asarray(load_file(str(path))["speaker_embedding"], dtype=np.float32).reshape(-1)
    except Exception as exc:  # noqa: BLE001
        log.warning("centroid unreadable (%s): using the reference clip", exc)
        return None
