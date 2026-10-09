"""Restore the Russian letter yo where a dictionary is sure.

Standalone. Standard library only. Copy this file and ``core/data/yo_runtime.tsv.gz`` (the lookup
table). The source lists and the dataset card live next to that table; see ``core/data/YO_DATASET.md``.

    from core.yo import restore
    restore("text with a sure yo word")

The stock Qwen3-TTS base model does not read stress marks, so this module never inserts U+0301 or
``+``. It writes yo only when the runtime table has a different spelling. Ambiguous words stay as
written, except the most frequent one: "vse" (all, plural) versus "vsyo" (everything, still), which is
decided from its neighbours by the small rule table ``core/data/yo_context.json`` (copy it too; without
it that word is left alone). Licence: Apache-2.0 for this file and the rule table; the word list is MIT
(e2yo/eyo-kernel).

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


def restore(text: str, table: Optional[Dict[str, str]] = None, context: bool = True) -> str:
    """``text`` with yo written back where ``table`` (default: the shipped dictionary) is sure.

    ``context`` (default on) also decides "vse" / "vsyo" from the neighbouring words; see :func:`restore_vse`.
    """
    return restore_counted(text, table, context)[0]


def restore_counted(text: str, table: Optional[Dict[str, str]] = None, context: bool = True) -> tuple:
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

    out = _WORD.sub(repl, text)
    if context:
        out, more = restore_vse_counted(out)
        changed += more
    return out, changed


# ----------------------------------------------------------------------------------------------- "vse" by context
_CTX_LOCK = threading.Lock()
_CTX: Optional[dict] = None
_NBSP_SPACE = " \t\u00a0"


def context_path() -> Path:
    """The rule table for "vse" / "vsyo" (``core/data/yo_context.json``)."""
    return _data_file("yo_context.json")


def _context_rules() -> dict:
    """The rule table, loaded once (an empty dict when the file is missing: the word is then left alone)."""
    global _CTX
    if _CTX is None:
        with _CTX_LOCK:
            if _CTX is None:
                import json

                path = context_path()
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    raw = {}
                rules: dict = {}
                for key, value in raw.items():
                    if isinstance(value, list):
                        items = [fold(str(v)).lower() for v in value]
                        if key.endswith("endings"):
                            rules[key] = tuple(sorted(set(items), key=len, reverse=True))
                        else:
                            rules[key] = frozenset(items)
                    else:
                        rules[key] = value
                _CTX = rules
    return _CTX


def _skip_spaces(text: str, pos: int) -> int:
    while pos < len(text) and text[pos] in _NBSP_SPACE:
        pos += 1
    return pos


def _word_at(text: str, pos: int) -> str:
    """The run of Cyrillic letters (and inner hyphens) starting at ``pos`` (``""`` when there is none)."""
    end = pos
    while end < len(text) and (text[end] in _LETTERS or (text[end] == "-" and end + 1 < len(text)
                                                          and text[end + 1] in _LETTERS and end > pos)):
        end += 1
    return text[pos:end]


def _word_before(text: str, pos: int) -> str:
    """The word that ends right before ``pos`` with only spaces between (``""`` after punctuation)."""
    end = pos
    while end > 0 and text[end - 1] in _NBSP_SPACE:
        end -= 1
    start = end
    while start > 0 and text[start - 1] in _LETTERS:
        start -= 1
    return text[start:end]


def _ends(word: str, endings) -> bool:
    return any(word.endswith(e) and len(word) > len(e) for e in endings)


def _plural_like(word: str, rules: dict) -> bool:
    return len(word) >= 3 and _ends(word, rules.get("plural_endings", ()))


def _vse_is_yo(text: str, start: int, end: int, rules: dict) -> bool:
    """True when the "vse" at ``text[start:end]`` should be read "vsyo"."""
    prev = fold(_word_before(text, start)).lower()
    prev_plural = prev in rules.get("prev_plural", ()) or _ends(prev, rules.get("prev_plural_endings", ()))
    if end < len(text) and text[end] == "-":
        tail = fold(_word_at(text, end + 1)).lower()
        return tail in rules.get("hyphen_tails", ())
    pos = _skip_spaces(text, end)
    if pos >= len(text):
        return not prev_plural
    ch = text[pos]
    if ch in rules.get("end_punct", "") or ch in rules.get("closing", ""):
        return not prev_plural
    if ch == ",":
        nxt = fold(_word_at(text, _skip_spaces(text, pos + 1))).lower()
        return nxt in rules.get("after_comma_yo", ())
    w1 = fold(_word_at(text, pos)).lower()
    if not w1:
        return False
    after = _skip_spaces(text, pos + len(w1))
    w2 = fold(_word_at(text, after)).lower() if after > pos + len(w1) else ""
    if w1 in rules.get("next_keep", ()):
        return False
    if w1 in rules.get("next_yo", ()):
        return True
    adverb = (len(w1) >= 3 and w1.endswith(fold(chr(0x043E)))
              and not _ends(w1, rules.get("not_adverb_endings", ())))
    if w1 in rules.get("next_check", ()) or adverb:
        return not _plural_like(w2, rules)
    if _ends(w1, rules.get("verb_sg_endings", ())) or _ends(w1, rules.get("adj_sg_endings", ())):
        return True
    if prev in rules.get("subject_sg", ()) and _ends(w1, rules.get("subject_sg_verb_endings", ())):
        return True
    return False


def restore_vse_counted(text: str) -> tuple:
    """``(text, changes)`` with "vse" written as "vsyo" where the neighbouring words show the singular.

    Kept as "vse": before a plural word or pronoun ("vse lyudi", "vse oni"), after a plural subject or verb, and
    whenever no rule fires. Written as "vsyo": before a singular verb or a neuter adjective, at the end of a clause,
    before ", chto", in fixed phrases ("vsyo ravno", "vsyo-taki", "vsyo eshchyo") and before a comparative. The
    word lists are data (``core/data/yo_context.json``); an all-caps "VSE" is left alone like every other word.
    """
    rules = _context_rules()
    word = rules.get("word")
    if not text or not word:
        return text or "", 0
    yo_word = str(rules.get("yo_word") or "")
    pattern = re.compile("(?<!" + _charset(_LETTERS) + ")(" + _charset(word[0] + word[0].upper()) + re.escape(word[1:])
                         + ")(?!" + _charset(_LETTERS) + ")")
    out = []
    last = 0
    changed = 0
    for m in pattern.finditer(text):
        if _vse_is_yo(text, m.start(), m.end(), rules):
            found = m.group(1)
            out.append(text[last:m.start()])
            out.append(found[0] + yo_word[1:])
            last = m.end()
            changed += 1
    if not changed:
        return text, 0
    out.append(text[last:])
    return "".join(out), changed


def restore_vse(text: str) -> str:
    """:func:`restore_vse_counted` without the count."""
    return restore_vse_counted(text)[0]


def applies(language: Optional[str]) -> bool:
    """True for a Russian narration language (``ru`` or ``Russian``)."""
    return (language or "").strip().lower() in ("ru", "russian")


def as_step(language: Optional[str]) -> Optional[Callable[[str], str]]:
    """A ``text -> text`` step for ``language``, or ``None`` when yo restoration does not apply."""
    if not applies(language):
        return None
    return restore
