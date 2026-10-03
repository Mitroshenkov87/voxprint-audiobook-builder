"""Objective checks of a synthesized sample against the speaker's reference clip (no Qt, no torch).

Used by the automatic post-training check and by the quick previews.  Metrics: whether the speech stopped by itself
(babbling to the token limit was the failure of the strong-learning-rate runs), recognition error rate (WER) of the sample
against the text it should say, and the median pitch (F0) compared with the reference, in semitones - a male voice that
comes out 4+ semitones higher sounds female.  The verdict is only a hint; the ear has the last word.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

GOOD, WARN, BAD = "good", "warn", "bad"
SEMITONES_WARN, SEMITONES_BAD = 3.0, 6.0
WER_WARN, WER_BAD = 0.25, 0.5
MIN_RMS_DBFS = -45.0


@dataclass
class Check:
    seconds: float
    expected_seconds: float
    wer: Optional[float]
    f0: float
    ref_f0: float
    semitones: Optional[float]
    rms_dbfs: float
    verdict: str = GOOD
    issues: List[str] = field(default_factory=list)      # codes: no_stop, quiet, wer, pitch, no_pitch

    def as_dict(self) -> dict:
        return {"seconds": round(self.seconds, 1), "wer": None if self.wer is None else round(self.wer, 2),
                "f0_hz": round(self.f0, 1), "ref_f0_hz": round(self.ref_f0, 1),
                "semitones": None if self.semitones is None else round(self.semitones, 1),
                "verdict": self.verdict, "issues": list(self.issues)}


def f0_median(x: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 400.0) -> float:
    """Median fundamental frequency (Hz) over voiced frames by normalised autocorrelation; 0.0 when nothing is voiced."""
    x = np.asarray(x, dtype=np.float32)
    n = int(sr * 0.04)
    hop = int(sr * 0.02)
    lo, hi = int(sr / fmax), int(sr / fmin)
    if len(x) < n + hi:
        return 0.0
    gate = 0.1 * float(np.sqrt(np.mean(x ** 2))) + 1e-9
    f0s = []
    for s in range(0, len(x) - n - hi, hop):
        fr = x[s:s + n + hi]
        a = fr[:n] - fr[:n].mean()
        e = float(np.sqrt(np.mean(a ** 2)))
        if e < gate * 2:
            continue
        rs = []
        for L in range(lo, hi):
            b = fr[L:L + n] - fr[L:L + n].mean()
            rs.append(float(np.sum(a * b)) / (float(np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))) + 1e-12))
        best = max(rs)
        if best > 0.6:
            i = next(k for k, r in enumerate(rs) if r >= 0.92 * best)     # the first strong peak, not a multiple of the period (octave errors)
            while i + 1 < len(rs) and rs[i + 1] > rs[i]:
                i += 1
            f0s.append(sr / (lo + i))
    return float(np.median(f0s)) if len(f0s) >= 5 else 0.0


def semitones(f: float, ref: float) -> Optional[float]:
    """Pitch distance sample vs reference in semitones (positive = higher); None if either is unknown."""
    return 12.0 * math.log2(f / ref) if f > 0 and ref > 0 else None


def _words(s: str) -> List[str]:
    return re.sub(r"[^\w\s]", " ", s.lower().replace("\u0451", "\u0435")).split()


def wer(ref: str, hyp: str) -> float:
    """Word error rate (edit distance over words / reference length)."""
    r, h = _words(ref), _words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def check_sample(audio: np.ndarray, sr: int, text: str, ref_audio: np.ndarray, ref_sr: int, *, asr=None,
                 language: Optional[str] = None, max_seconds: float = 0.0, asr_text: Optional[str] = None) -> Check:
    """Check one synthesized ``audio`` that should say ``text``.

    ``max_seconds`` is the generation cap used for it (``tts_engine.max_tokens_for(text) / 12.5``): a sample that reaches the
    cap never stopped.  ``asr`` (a :class:`core.asr.BaseASR`) or a ready ``asr_text`` provides the recognised text for the WER.
    """
    secs = len(audio) / sr
    rms = 20 * math.log10(max(float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0, 1e-9))
    hyp = asr_text if asr_text is not None else (asr.transcribe(audio, sr, language).text if asr is not None else None)
    w = wer(text, hyp) if hyp is not None else None
    f0, ref_f0 = f0_median(audio, sr), f0_median(ref_audio, ref_sr)
    st = semitones(f0, ref_f0)
    c = Check(secs, max_seconds, w, f0, ref_f0, st, rms)
    bad = warn = False
    if max_seconds and secs >= max_seconds - 0.6:
        c.issues.append("no_stop"); bad = True
    if rms < MIN_RMS_DBFS:
        c.issues.append("quiet"); bad = True
    if w is not None and w > WER_WARN:
        c.issues.append("wer"); bad |= w > WER_BAD; warn = True
    if st is None:
        c.issues.append("no_pitch")
    elif abs(st) > SEMITONES_WARN:
        c.issues.append("pitch"); bad |= abs(st) > SEMITONES_BAD; warn = True
    c.verdict = BAD if bad else (WARN if warn else GOOD)
    return c
