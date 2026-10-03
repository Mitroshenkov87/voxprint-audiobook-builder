"""Small shared data structures of the dataset pipeline (no dependencies on torch or Qt).

``WordTiming`` is what the forced aligner returns; ``Segment`` is one training clip (an audio interval plus its text)
that ends up as ``segment_NNN.wav`` + a row in ``metadata.jsonl``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class WordTiming:
    """One word with timestamps (seconds from the start of the *whole* recording).

    ``word`` is the token as the aligner returned it (no punctuation).  ``char_start`` / ``char_end`` give the range of
    the word in the normalized text; they are filled in by ``core.text_utils.attach_spans``.  ``sentence_end`` marks the
    last word of a sentence and is used by the slicer to prefer cuts at sentence boundaries.
    """

    word: str
    start: float
    end: float
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    sentence_end: bool = False

    @property
    def duration(self) -> float:
        """Length of the word in seconds."""
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        """Compact JSON-friendly form (times rounded to milliseconds)."""
        return {"word": self.word, "start": round(self.start, 3), "end": round(self.end, 3)}


@dataclass
class Segment:
    """A dataset segment: an audio interval and its clean text.

    ``index`` is 1-based and defines the file name.  ``extra`` carries optional per-segment data, e.g. ``text_raw`` (the
    text before number/abbreviation expansion) which is written to ``report.json``.
    """

    index: int  # 1-based
    start: float
    end: float
    text: str
    n_words: int = 0
    ends_sentence: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Length of the segment in seconds."""
        return self.end - self.start

    @property
    def filename(self) -> str:
        """File name of the cut clip inside the dataset folder, e.g. ``segment_007.wav``."""
        return f"segment_{self.index:03d}.wav"
