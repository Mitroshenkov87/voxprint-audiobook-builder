"""Speech-recognition cross-check of dataset clips ("audio + text" mode): drop clips whose recognised text does not match
the transcript.

The forced aligner always places the given words somewhere, so a clip with a slip of the tongue, a skipped or repeated word or
an alignment error still gets the *script's* text.  Trained on such pairs, the model learns wrong pronunciations and babbling
(research notes, item 6; QwenLM/Qwen3-TTS issue #39: "fewer but clean clips").  Each clip is re-read by Qwen3-ASR and its
character error rate (CER) against the clip's text is computed; clips above the threshold (default 12 %, 10-15 % allowed) are
dropped.  The dataset never shrinks below :data:`MIN_KEEP_CLIPS` because of this check: then only the worst clips go, and a
warning says how many doubtful ones stayed.  A clip whose recognition fails is kept (no evidence against it).

Auto-transcribed datasets (no text) are not checked: their text *is* the recogniser's output (see :mod:`core.asr_dataset`,
which has its own plausibility gate).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from core.voice_check import cer

log = logging.getLogger("voxprint.dataset")

DEFAULT_MAX_CER = 0.12
MIN_MAX_CER, MAX_MAX_CER = 0.10, 0.15
#: The check never takes the dataset below this many clips (the trainer warns below ~20 clips anyway).
MIN_KEEP_CLIPS = 20


def clamp_threshold(value: Optional[float]) -> float:
    """A usable threshold in [0.10, 0.15]; 0 / None / garbage -> 0.0 (the check is off)."""
    try:
        v = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if v <= 0 or v != v:
        return 0.0
    return min(MAX_MAX_CER, max(MIN_MAX_CER, v))


@dataclass
class ClipCheckReport:
    """Outcome of :func:`check_clips` (also written to the dataset's ``report.json``)."""
    threshold: float
    checked: int = 0
    dropped: int = 0
    kept_over: int = 0                  # clips above the threshold kept because of MIN_KEEP_CLIPS
    failed: int = 0                     # recognition errors (clip kept)
    cers: List[Optional[float]] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)        # what the recogniser heard, per clip

    def as_dict(self) -> dict:
        return {"threshold": self.threshold, "checked": self.checked, "dropped": self.dropped, "kept_over": self.kept_over,
                "failed": self.failed, "min_keep": MIN_KEEP_CLIPS}


def check_clips(pieces: Sequence[np.ndarray], sr: int, texts: Sequence[str], asr, language: Optional[str],
                max_cer: float = DEFAULT_MAX_CER, min_keep: int = MIN_KEEP_CLIPS,
                normalize: Optional[Callable[[str], str]] = None,
                progress: Callable[[float], None] = lambda f: None, cancel_check: Callable[[], None] = lambda: None):
    """``(keep, report)``: ``keep[i]`` says whether clip i stays.

    ``asr`` is a :class:`core.asr.BaseASR` (loaded and unloaded here); ``normalize`` turns the recognised text into the spoken
    form the dataset text uses (numbers written out ...), so "2026" vs "две тысячи двадцать шесть" is not counted as an error.
    """
    n = len(pieces)
    rep = ClipCheckReport(threshold=max_cer, cers=[None] * n, texts=[""] * n)
    keep = [True] * n
    if n == 0:
        return keep, rep
    asr.load()
    try:
        for i, (x, ref) in enumerate(zip(pieces, texts)):
            cancel_check()
            try:
                hyp = asr.transcribe(x, sr, language).text
                rep.texts[i] = hyp
                if normalize is not None:
                    try:
                        hyp = normalize(hyp)
                    except Exception:  # noqa: BLE001 - compare the raw text then
                        pass
                rep.cers[i] = cer(ref, hyp)
                rep.checked += 1
            except Exception as exc:  # noqa: BLE001 - one failed clip is no reason to drop it or to stop
                rep.failed += 1
                log.warning("clip check: recognition of clip %d failed: %s", i + 1, exc)
            progress((i + 1) / n)
    finally:
        asr.unload()
    def _worst(item: tuple[int, float]) -> float:
        return -item[1]

    scored = [(i, c) for i, c in enumerate(rep.cers) if c is not None and c > max_cer]
    bad = [i for i, _cer in sorted(scored, key=_worst)]
    allowed = max(0, n - max(0, min_keep))                 # how many clips may go without falling below the minimum
    for i in bad[:allowed]:                                 # the worst first
        keep[i] = False
    rep.dropped = min(len(bad), allowed)
    rep.kept_over = len(bad) - rep.dropped
    log.info("clip check (CER > %.0f %%): %d of %d clips dropped, %d doubtful kept (minimum %d), %d not recognised",
             max_cer * 100, rep.dropped, n, rep.kept_over, min_keep, rep.failed)
    return keep, rep
