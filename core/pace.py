"""Reading speed that follows the text: a tempo factor per chunk, applied after synthesis by time-stretching (pitch kept).

Qwen3-TTS has no speed parameter, so the narration stretches each synthesized piece (:func:`core.audio_utils.time_stretch`,
WSOLA) before it is joined; the chunk cache keeps the unstretched audio, so changing the speed never re-synthesizes.

* ``speed`` - the global factor from Settings / ``--speed`` (1.0 = as the voice speaks, 0.8 slower ... 1.2 faster).
* ``style`` - a per-book preset: ``auto`` (detected), ``scripture`` (solemn: slower overall), ``fiction``, ``dialogue``
  (dialogue-heavy: close to the voice's own speed).
* :func:`segment_tempo` - long, comma-rich or descriptive sentences and scripture-like lines are read a little slower;
  dialogue lines and short lines at the normal speed.  The factor is kept within :data:`MIN_TEMPO` ... :data:`MAX_TEMPO`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

AUTO, SCRIPTURE, FICTION, DIALOGUE = "auto", "scripture", "fiction", "dialogue"
STYLES = (AUTO, SCRIPTURE, FICTION, DIALOGUE)
MIN_SPEED, MAX_SPEED = 0.7, 1.3
MIN_TEMPO, MAX_TEMPO = 0.6, 1.4
#: Base factor of each style (before the per-sentence adjustments).
STYLE_BASE = {SCRIPTURE: 0.92, FICTION: 1.0, DIALOGUE: 1.0}

_WORD = re.compile(r"\w+", re.UNICODE)
_DIALOGUE = re.compile(r"^\s*[—–\-«\"“„']")
_NUMBERED = re.compile(r"^\s*\d{1,3}(?::\d{1,3})?[.):]?\s+\S", re.MULTILINE)
_SOLEMN = re.compile(r"\b(?:бог|господь|господи|сотворил|воистину|аминь|god|lord|amen|behold|thou|thee|gott|herr)\b",
                     re.IGNORECASE)


def clamp_speed(value: object) -> float:
    """A reading speed inside 0.7-1.3, or 1.0 when ``value`` is not a number."""
    try:
        v = float(value)                      # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1.0
    return round(max(MIN_SPEED, min(MAX_SPEED, v)), 2)


def clamp_style(value: object) -> str:
    """A known style name (``auto``, ``scripture``, ``fiction``, ``dialogue``), or ``auto``."""
    v = str(value or "").strip().lower()
    return v if v in STYLES else AUTO


@dataclass
class Pace:
    """Reading speed of one narration job."""
    speed: float = 1.0
    style: str = AUTO

    def __post_init__(self) -> None:
        """Clamp the speed and the style after the dataclass is built."""
        self.speed, self.style = clamp_speed(self.speed), clamp_style(self.style)


def book_style(book, style: str = AUTO) -> str:
    """The style for ``book``: ``style`` unless it is ``auto``; then scripture when many lines are numbered verses or the
    vocabulary is solemn, dialogue when many paragraphs are spoken lines, else fiction."""
    style = clamp_style(style)
    if style != AUTO:
        return style
    text = "\n\n".join(c.text for c in getattr(book, "chapters", []))[:200_000]
    paras = [p for p in re.split(r"\n\s*\n|\n(?=\s*\d{1,3}[ .:)])", text) if p.strip()]
    if not paras:
        return FICTION
    words = max(1, len(_WORD.findall(text)))
    if len(_NUMBERED.findall(text)) >= 0.3 * len(paras) or len(_SOLEMN.findall(text)) / words > 0.01:
        return SCRIPTURE
    if sum(bool(_DIALOGUE.match(p)) for p in paras) >= 0.35 * len(paras):
        return DIALOGUE
    return FICTION


def segment_tempo(text: str, style: str = FICTION, speed: float = 1.0, kind: str = "") -> float:
    """Tempo factor of one chunk (< 1 = slower).  Simple, explainable heuristics; see the module docstring."""
    words = _WORD.findall(text)
    n = len(words)
    factor = STYLE_BASE.get(style, 1.0)
    if kind == "title":
        factor *= 0.9                                  # titles a little slower and clearer
    elif _DIALOGUE.match(text) and style != SCRIPTURE:
        factor = max(factor, 1.0)                      # dialogue: the voice's own speed
    elif n >= 6:
        if n > 25:
            factor *= 0.95                             # long sentence
        punct = len(re.findall(r"[,;:—–]", text)) / n
        if punct > 0.15:
            factor *= 0.97                             # dense with clauses
        if sum(len(w) for w in words) / n > 6.5:
            factor *= 0.97                             # long, descriptive words
    return round(max(MIN_TEMPO, min(MAX_TEMPO, factor * clamp_speed(speed))), 3)


def _file() -> Path:
    from infra import paths
    return paths.state_dir() / "narration_pace.json"


def load() -> Pace:
    """Speed and style from Settings (defaults when never set)."""
    try:
        d = json.loads(_file().read_text(encoding="utf-8"))
        return Pace(d.get("speed", 1.0), d.get("style", AUTO)) if isinstance(d, dict) else Pace()
    except (OSError, ValueError):
        return Pace()


def save(pace: Pace) -> None:
    """Remember speed and style (a failure to write is not an error)."""
    try:
        p = _file()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"speed": pace.speed, "style": pace.style}), encoding="utf-8")
    except OSError:
        pass
