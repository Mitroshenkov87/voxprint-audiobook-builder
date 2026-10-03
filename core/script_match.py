"""Matching recognised clips to the lines of a recording script (tolerant of stumbles and re-read sentences).

A person reading a script in one continuous take will sometimes stumble, stay silent for a second and read the sentence
again.  The recording is cut at pauses and every piece is recognised on its own (``core.asr_dataset``); this module then
decides which script line(s) each piece really contains:

* :func:`parse_script_lines` extracts the lines that are read aloud (block headers ``=== ...``, ``[stage directions]`` and
  lines that consist only of a ``(hint)`` are never read, speaker labels are bracketed so they are skipped too; the
  parsing stops at the consent block);
* :func:`match_clips` compares each recognised piece with every window of 1..``MAX_WINDOW`` consecutive script lines.  A piece
  is accepted only if it matches a window almost completely in both directions (so a half-sentence stumble is rejected);
  when several pieces match the same line the best one wins (the later one on a tie, a re-read is usually the better one),
  the others are ignored.  The accepted piece gets the clean script text as its transcript.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Sequence

MAX_WINDOW = 4          #: a piece may contain at most this many consecutive script lines
MIN_RATIO = 0.82        #: minimal letter similarity between a recognised piece and the script window


#: A "=== ... ===" header containing one of these (any case) starts the consent block: ru / en / de.
CONSENT_WORDS = ("СОГЛАСИЕ", "CONSENT", "EINWILLIGUNG")


def _letters(text: str) -> str:
    return "".join(ch for ch in text.lower().replace("\u0451", "\u0435") if ch.isalpha())


def parse_script_lines(script: str) -> List[str]:
    """The lines of ``script`` that are read aloud, in order (see the module docstring for what is skipped)."""
    lines: List[str] = []
    for raw in script.replace("\r\n", "\n").split("\n"):
        s = re.sub(r"\[[^\]]*\]", " ", raw)            # [stage directions / speaker labels / emotion tags]
        s = re.sub(r"\s+", " ", s).strip()
        if raw.strip().startswith("===") and any(w in raw.upper() for w in CONSENT_WORDS):   # the spoken consent block is not training text
            break
        if not s or s.startswith("===") or s.startswith("---"):
            continue
        if raw.strip().startswith("(") and raw.strip().endswith(")"):   # a whole-line (hint), as in script v2
            continue
        if not _letters(s):
            continue
        lines.append(s)
    return lines


@dataclass
class ClipMatch:
    """Result for one piece: ``line_start``/``n_lines`` (None if rejected), ``ratio`` and the ``status``."""
    status: str                      # "matched" | "repeat" | "no_match"
    ratio: float = 0.0
    line_start: Optional[int] = None
    n_lines: int = 0
    text: str = ""


def _best_window(letters: str, keys: Sequence[str], lines: Sequence[str]) -> Optional[ClipMatch]:
    n = len(letters)
    best: Optional[ClipMatch] = None
    for i in range(len(keys)):
        acc = ""
        for k in range(1, MAX_WINDOW + 1):
            if i + k > len(keys):
                break
            acc += keys[i + k - 1]
            if len(acc) > n * 1.4 + 3:
                break
            if len(acc) < n * 0.7 - 3:
                continue
            sm = SequenceMatcher(None, letters, acc, autojunk=False)
            if sm.real_quick_ratio() < MIN_RATIO or sm.quick_ratio() < MIN_RATIO:
                continue
            r = sm.ratio()
            if best is None or r > best.ratio:
                best = ClipMatch("matched", r, i, k, " ".join(lines[i:i + k]))
    return best


def match_clips(clip_texts: Sequence[str], script_lines: Sequence[str], min_ratio: float = MIN_RATIO) -> List[ClipMatch]:
    """One :class:`ClipMatch` per recognised piece (same order).  Pieces that are stumbles or re-reads get a non-"matched" status."""
    keys = [_letters(s) for s in script_lines]
    cands: List[Optional[ClipMatch]] = []
    for t in clip_texts:
        lt = _letters(t)
        m = _best_window(lt, keys, script_lines) if lt else None
        cands.append(m if m is not None and m.ratio >= min_ratio else None)
    out = [ClipMatch("no_match") if c is None else c for c in cands]
    claimed = set()
    order = sorted((i for i, c in enumerate(cands) if c is not None), key=lambda i: (-round(cands[i].ratio, 3), -i))
    for i in order:
        c = cands[i]
        span = set(range(c.line_start, c.line_start + c.n_lines))
        if span & claimed:
            out[i] = ClipMatch("repeat", c.ratio, c.line_start, c.n_lines)
        else:
            claimed |= span
    return out
