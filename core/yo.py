"""Restore the Russian letter yo where a dictionary is sure.

Standalone. Standard library only. Copy this file and ``core/data/yo_runtime.tsv.gz`` (the lookup
table). The source lists and the dataset card live next to that table; see ``core/data/YO_DATASET.md``.

    from core.yo import restore
    restore("text with a sure yo word")

The stock Qwen3-TTS base model does not read stress marks, so this module never inserts U+0301 or
``+``. It writes yo only when the runtime table has a different spelling. Ambiguous words stay as
written. Licence: Apache-2.0 for this file; the word list is MIT (e2yo/eyo-kernel).

Credit: Voxprint AI Audiobook Builder https://github.com/Mitroshenkov87/voxprint-audiobook-builder
"""
from __future__ import annotations

import gzip
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
_VOWELS = frozenset(chr(c) for c in (
    0x0430, 0x0435, 0x0451, 0x0438, 0x043E, 0x0443, 0x044B, 0x044D, 0x044E, 0x044F,
    0x0410, 0x0415, 0x0401, 0x0418, 0x041E, 0x0423, 0x042B, 0x042D, 0x042E, 0x042F,
))
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


def _data_file(name: str) -> Path:
    """A file in ``core/data``, or the same path inside a frozen build."""
    here = Path(__file__).resolve().parent / "data" / name
    if here.is_file():
        return here
    base = getattr(sys, "_MEIPASS", None)
    if base:
        packed = Path(base) / "core" / "data" / name
        if packed.is_file():
            return packed
    return here


def dictionary_path() -> Path:
    """The eyo-kernel safe source list (``yo_safe.txt``). The app loads :func:`runtime_path` instead."""
    return _data_file("yo_safe.txt")


def runtime_path() -> Path:
    """Gzipped ``ye-form<TAB>yo-form`` table built by ``tools/build_yo_dataset.py``."""
    return _data_file("yo_runtime.tsv.gz")


def fold(word: str) -> str:
    """``word`` with yo folded to ye, the dictionary key."""
    return word.replace(_YO_UP, _YE_UP).replace(_YO_LO, _YE_LO)


def stress_index(form: str) -> Optional[int]:
    """0-based index of yo among the vowels of ``form``, or ``None`` when there is no yo.

    In Russian, yo is always the stressed vowel. This is recorded in the published dataset.
    Narration does not write the mark into the text.
    """
    seen = 0
    found: Optional[int] = None
    for ch in form:
        if ch not in _VOWELS:
            continue
        if ch in (_YO_LO, _YO_UP):
            found = seen
        seen += 1
    return found


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


def load_runtime(text: str) -> Dict[str, str]:
    """``{ye-form: yo-form}`` from the compact runtime table (one pair per line)."""
    table: Dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.split("\t", 1)
        table[key] = value
    return table


def dictionary() -> Dict[str, str]:
    """The shipped runtime table, loaded once. Falls back to expanding ``yo_safe.txt`` if the table is absent."""
    global _CACHE
    if _CACHE is None:
        with _LOCK:
            if _CACHE is None:
                path = runtime_path()
                if path.is_file():
                    _CACHE = load_runtime(gzip.decompress(path.read_bytes()).decode("utf-8"))
                else:
                    _CACHE = load(dictionary_path().read_text(encoding="utf-8"))
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
