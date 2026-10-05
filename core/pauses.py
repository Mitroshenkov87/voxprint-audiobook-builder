"""Pause strength of a narration: explicit silence between the spoken pieces, independent of the model's prosody.

A TTS model decides by itself how long a comma or a full stop sounds, and it often decides "almost nothing".  The narration
therefore cuts the text at the places where a listener expects a pause (comma, sentence end, ellipsis, dash, paragraph, scene
break, chapter end) and joins the pieces with *measured silence*; :class:`PauseProfile` says how long each kind is.

* ``BASE_MS`` - the length of every kind at the default strength.  The defaults are a little longer than the earlier
  fixed values (sentence 300 -> 420 ms, paragraph 750 -> 1050 ms, chapter end 1200 -> 1900 ms ...) so that pauses are
  clearly audible out of the box.
* ``LEVELS`` - the five positions of the "Pauses" slider (shorter ... normal ... longer); ``DEFAULT_LEVEL`` is "normal".
* ``multipliers`` - a separate factor for every kind (default 1.0) on top of the slider, for fine tuning in code or tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

COMMA, SENTENCE, ELLIPSIS, DASH, PARAGRAPH, SCENE, CHAPTER, TITLE = (
    "comma", "sentence", "ellipsis", "dash", "paragraph", "scene", "chapter", "title")
KINDS = (COMMA, SENTENCE, ELLIPSIS, DASH, PARAGRAPH, SCENE, CHAPTER, TITLE)

#: Silence in milliseconds at the default strength ("normal").
BASE_MS: Dict[str, int] = {
    COMMA: 240, SENTENCE: 430, ELLIPSIS: 680, DASH: 320, PARAGRAPH: 1050, SCENE: 1900, CHAPTER: 1900, TITLE: 1100,
}
#: How much the slider moves each kind (1 = fully, 0 = not at all): small pauses change less than big ones.
SENSITIVITY: Dict[str, float] = {
    COMMA: 0.8, SENTENCE: 1.0, ELLIPSIS: 1.0, DASH: 0.8, PARAGRAPH: 1.0, SCENE: 1.0, CHAPTER: 1.0, TITLE: 1.0,
}
#: The slider positions: factor applied to ``BASE_MS``.  Index 2 is "normal".
LEVELS = (0.5, 0.75, 1.0, 1.4, 1.9)
DEFAULT_LEVEL = 2
MAX_PAUSE_MS = 6000


def clamp_level(level: object) -> int:
    """A valid slider position for anything (strings from a file, out-of-range numbers)."""
    try:
        n = int(level)                         # type: ignore[call-overload]
    except (TypeError, ValueError):
        return DEFAULT_LEVEL
    return max(0, min(len(LEVELS) - 1, n))


@dataclass
class PauseProfile:
    """Silence lengths for one narration job."""
    level: int = DEFAULT_LEVEL
    multipliers: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.level = clamp_level(self.level)

    def ms(self, kind: str) -> int:
        """Silence after a piece that ends with ``kind`` (a name from :data:`KINDS`)."""
        base = BASE_MS.get(kind, BASE_MS[SENTENCE])
        factor = 1.0 + (LEVELS[self.level] - 1.0) * SENSITIVITY.get(kind, 1.0)
        value = base * factor * float(self.multipliers.get(kind, 1.0))
        return int(max(0, min(MAX_PAUSE_MS, round(value))))

    def table(self) -> Dict[str, int]:
        """Every kind with its length (for tests, logs and the manual)."""
        return {k: self.ms(k) for k in KINDS}


def _file() -> Path:
    from infra import paths
    return paths.state_dir() / "pause_level.txt"


def load_level() -> int:
    """The slider position the user chose last time (the default if there is none)."""
    try:
        return clamp_level(_file().read_text(encoding="utf-8").strip())
    except OSError:
        return DEFAULT_LEVEL


def save_level(level: int) -> None:
    """Remember the slider position (a failure to write is not an error)."""
    try:
        p = _file()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(clamp_level(level)), encoding="utf-8")
    except OSError:
        pass


# Explicit pauses cut the text at every comma / dash / ellipsis; on short pieces the voice model tends to "bubble" and
# swallow words, so the feature is opt-in (default OFF) and remembered in a separate small file next to the level.
def _enabled_file():
    return _file().with_name(_file().stem + "_enabled.txt")


def load_enabled() -> bool:
    """Whether the user turned explicit pauses on (False when never chosen)."""
    try:
        return _enabled_file().read_text(encoding="utf-8").strip() == "1"
    except OSError:
        return False


def save_enabled(on: bool) -> None:
    """Remember the explicit-pauses switch (a failure to write is not an error)."""
    try:
        p = _enabled_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("1" if on else "0", encoding="utf-8")
    except OSError:
        pass
