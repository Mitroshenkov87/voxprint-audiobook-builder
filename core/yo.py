"""Restore the Russian letter yo before synthesis, and only where a dictionary is sure.

The narration model is stock ``Qwen/Qwen3-TTS-12Hz-1.7B-Base``. It was not trained on stress
marks, so a combining acute (U+0301) or a ``+`` before the stressed vowel is not a pronunciation
hint for it: those characters are extra tokens. This module therefore does not insert them.
Writing yo (U+0451) is a different letter the tokenizer already knows. Words the safe dictionary
does not list, including ``текст`` and ambiguous pairs such as ``все`` / ``всё``, are left as
written. A plain ``е`` the model itself decides to read as yo is outside what the text can force.

The word list is ``core/data/yo_safe.txt``, the MIT ``dictionary/safe.txt`` of
https://github.com/e2yo/eyo-kernel (see ``yo_safe.LICENSE``). The lookup follows that project's
rules: parentheses expand endings, a leading underscore is lowercase-only, a lowercase entry also
covers the capitalised form, and a dotted abbreviation such as ``мед. училище`` is not rewritten.
The TypeScript library is not vendored.
"""
from __future__ import annotations

import re
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, Optional

# Cyrillic ranges, built from code points so this module has no Cyrillic literals.
_YE_UP, _YO_UP = chr(0x0415), chr(0x0401)
_YE_LO, _YO_LO = chr(0x0435), chr(0x0451)
_UPPER = "".join(chr(c) for c in range(0x0410, 0x0430)) + _YO_UP
_LOWER = "".join(chr(c) for c in range(0x0430, 0x0450)) + _YO_LO
_LETTERS = _UPPER + _LOWER
_UPPER_SET = frozenset(_UPPER)
# Punctuation the eyo word pattern treats as "not a continuation of an abbreviation".
_PUNCT = "".join(chr(c) for c in (
    0x5B, 0x7B, 0x7D, 0x28, 0x29, 0x7C, 0x3C, 0x3E, 0x3D, 0x5F, 0x22, 0x27,
    0xAB, 0xBB, 0x201E, 0x201C, 0x23, 0x24, 0x5E, 0x25, 0x26, 0x2A, 0x2B,
    0x3A, 0x3B, 0x2C, 0x3F, 0x21, 0x2011, 0x5C, 0x2D, 0x5D,
))

_HAS_EYO = re.compile("[" + _YE_UP + _YO_UP + _YE_LO + _YO_LO + "]")
_SPLIT = re.compile("[(|)]")
_LOCK = threading.Lock()
_CACHE: Optional[Dict[str, str]] = None


def _charset(chars: str) -> str:
    """A regex character class. ``-``, ``]``, ``\\`` and ``^`` are escaped."""
    body = []
    for ch in chars:
        body.append("\\" + ch if ch in r"[]\^-" else ch)
    return "[" + "".join(body) + "]"


def _word_pattern() -> "re.Pattern[str]":
    """Words of two or more letters whose tail is lowercase, skipping dotted abbreviations.

    The first letter may be any case. An all-caps token does not match. ``мед. училище`` does
    not match ``мед``: a dot, spaces, then a lowercase letter (or two capitals, or punctuation)
    means the token is an abbreviation. A dot glued to punctuation is the same.
    """
    first = _charset(_LETTERS)
    lower = _charset(_LOWER)
    upper = _charset(_UPPER)
    punct = _charset(_PUNCT)
    return re.compile(
        first + lower + "+"
        + "(?!" + lower
        + r"|\.[ \u00a0\t]+(?:" + lower + "|" + upper + "{2}|" + punct + ")"
        + r"|\." + punct + ")"
    )


_WORD = _word_pattern()


def dictionary_path() -> Path:
    """``yo_safe.txt`` beside this module, or under ``core/data`` in a frozen build."""
    name = Path("data") / "yo_safe.txt"
    here = Path(__file__).resolve().parent / name
    if here.is_file():
        return here
    base = getattr(sys, "_MEIPASS", None)
    if base:
        packed = Path(base) / "core" / name
        if packed.is_file():
            return packed
    return here


def fold(word: str) -> str:
    """``word`` with yo folded to ``е``, the dictionary key."""
    return word.replace(_YO_UP, _YE_UP).replace(_YO_LO, _YE_LO)


def _capitalise(text: str) -> str:
    """First character uppercased; the rest is kept (not :meth:`str.capitalize`)."""
    return text[0].upper() + text[1:] if text else text


def _add_inner(table: Dict[str, str], word: str) -> None:
    lower_only = word.startswith("_")
    if lower_only:
        word = word[1:]
    if not word:
        return
    key = fold(word)
    table[key] = word
    if not lower_only and word[0] not in _UPPER_SET:
        table[_capitalise(key)] = _capitalise(word)


def _add_line(table: Dict[str, str], raw: str) -> None:
    word = raw.split("#", 1)[0].strip()
    if not word:
        return
    if "(" in word:
        parts = _SPLIT.split(word)
        for part in parts[1:-1]:
            _add_inner(table, parts[0] + part)
        return
    _add_inner(table, word)


def load(text: str) -> Dict[str, str]:
    """Build ``{ye-form: yo-form}`` from a dictionary file (one entry per line)."""
    table: Dict[str, str] = {}
    for line in text.splitlines():
        _add_line(table, line)
    return table


def dictionary() -> Dict[str, str]:
    """The shipped safe dictionary, loaded once."""
    global _CACHE
    if _CACHE is None:
        with _LOCK:
            if _CACHE is None:
                path = dictionary_path()
                _CACHE = load(path.read_text(encoding="utf-8"))
    return _CACHE


def restore(text: str, table: Optional[Dict[str, str]] = None) -> str:
    """``text`` with yo written back where ``table`` (default: the shipped dictionary) is sure."""
    if not text or _HAS_EYO.search(text) is None:
        return text or ""
    words = table if table is not None else dictionary()

    def repl(match: "re.Match[str]") -> str:
        word = match.group(0)
        return words.get(fold(word), word)

    return _WORD.sub(repl, text)


def restore_counted(text: str, table: Optional[Dict[str, str]] = None) -> tuple:
    """``(restored text, how many words changed)``."""
    if not text or _HAS_EYO.search(text) is None:
        return text or "", 0
    words = table if table is not None else dictionary()
    changed = 0

    def repl(match: "re.Match[str]") -> str:
        nonlocal changed
        word = match.group(0)
        out = words.get(fold(word), word)
        if out != word:
            changed += 1
        return out

    return _WORD.sub(repl, text), changed


def applies(language: Optional[str]) -> bool:
    """True for a Russian narration language (``ru`` or ``Russian``)."""
    return (language or "").strip().lower() in ("ru", "russian")


def as_step(language: Optional[str]) -> Optional[Callable[[str], str]]:
    """A ``text -> text`` step for ``language``, or ``None`` when yo restoration does not apply."""
    if not applies(language):
        return None
    return restore
