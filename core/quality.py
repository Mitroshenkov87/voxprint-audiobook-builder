"""Automatic segment quality filter (no user interaction).

Metrics and thresholds follow ``voice_clone_lab`` (``audio_stats``): clipping, level (RMS in dBFS) and an SNR estimate
taken as the difference between the 90th and 10th percentile of frame loudness (30 ms frames, 10 ms hop).
A segment is dropped when:

* clipping: more than ``max(10, 0.05 % of its length)`` samples have ``|x| >= 0.999``;
* RMS < -42 dBFS (too quiet);
* estimated SNR < 8 dB (too noisy or too uneven).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

CLIP_LEVEL = 0.999
CLIP_MIN_SAMPLES = 10
CLIP_FRACTION = 0.0005
MIN_RMS_DBFS = -42.0
MIN_SNR_DB = 8.0
#: If the filter would drop more than this share of segments, the SNR criterion is considered unreliable and disabled.
MAX_DROP_FRACTION = 0.5


def audio_stats(samples: np.ndarray, sr: int) -> Dict[str, float]:
    """Compute peak, RMS (dBFS), clipped-sample count and the percentile-based SNR estimate of a clip.

    Multi-channel input is averaged to mono; an empty clip yields zeros (and a very low RMS).
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = len(x)
    peak = float(np.max(np.abs(x))) if n else 0.0
    rms = float(np.sqrt(np.mean(np.square(x)))) if n else 0.0
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
    clipped = int(np.sum(np.abs(x) >= CLIP_LEVEL))
    frame, hop = max(1, int(sr * 0.03)), max(1, int(sr * 0.01))
    if n >= frame:
        starts = np.arange(0, n - frame + 1, hop)
        frames = np.lib.stride_tricks.sliding_window_view(x, frame)[starts]
        fr = np.sqrt(np.mean(np.square(frames), axis=1))
    elif n:
        fr = np.array([rms])
    else:
        fr = np.empty(0)
    if fr.size:
        db = 20.0 * np.log10(np.maximum(fr, 1e-12))
        snr = float(np.percentile(db, 90) - np.percentile(db, 10))
    else:
        snr = 0.0
    return {"peak": peak, "rms_dbfs": rms_dbfs, "clipped_samples": clipped, "estimated_snr_db": snr,
            "samples": n}


def reject_reason(stats: Dict[str, float], use_snr: bool = True) -> Optional[str]:
    """Reason code for rejecting a segment, or None if the segment is fine."""
    limit = max(CLIP_MIN_SAMPLES, int(CLIP_FRACTION * stats["samples"]))
    if stats["clipped_samples"] > limit:
        return "clipping"
    if stats["rms_dbfs"] < MIN_RMS_DBFS:
        return "too_quiet"
    if use_snr and stats["estimated_snr_db"] < MIN_SNR_DB:
        return "low_snr"
    return None


@dataclass
class QualityResult:
    """Per-segment verdicts of :func:`assess_segments` (parallel lists, one entry per input piece)."""
    keep: List[bool]
    reasons: List[Optional[str]]
    stats: List[Dict[str, float]]
    snr_disabled: bool = False

    @property
    def n_dropped(self) -> int:
        """Number of segments that did not pass the filter."""
        return sum(1 for k in self.keep if not k)

    def summary(self) -> Dict[str, int]:
        """Count of dropped segments per reason code (``clipping``, ``too_quiet``, ``low_snr``)."""
        out: Dict[str, int] = {}
        for r in self.reasons:
            if r:
                out[r] = out.get(r, 0) + 1
        return out


def assess_segments(pieces: Sequence[np.ndarray], sr: int) -> QualityResult:
    """Assess every piece.  If too many are rejected (> 50 %), the SNR criterion is switched off."""
    stats = [audio_stats(p, sr) for p in pieces]
    reasons = [reject_reason(s) for s in stats]
    snr_disabled = False
    if pieces and sum(1 for r in reasons if r) > MAX_DROP_FRACTION * len(pieces):
        reasons2 = [reject_reason(s, use_snr=False) for s in stats]
        if sum(1 for r in reasons2 if r) <= MAX_DROP_FRACTION * len(pieces):
            reasons, snr_disabled = reasons2, True
    return QualityResult([r is None for r in reasons], reasons, stats, snr_disabled)
