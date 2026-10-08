"""Ordinal numbers in narration text: "день 1" -> "день первый", "в 3-м томе" -> "в третьем томе", "21st" -> "twenty first".

A text-normalization step that runs on every chunk *before* the language normalizer (which reads the remaining digits as
cardinals) and before synthesis.  What triggers an ordinal is data (:mod:`core.ordinal_rules`), per language:

* **Russian** - a listed noun followed by a number or Roman numeral ("глава 2", "Глава IV", "в главе 3", "псалом 22"): the
  ordinal takes the noun's gender and case ("глава вторая", "в главе третьей", "псалом двадцать второй"); written endings
  ("1-й", "2-я", "3-го", "в 5-м томе", "90-х") are read with the gender/case of the ending, refined by the next word.
* **English** - written ordinals ("1st", "22nd", "103rd"); a heading Roman numeral after a listed noun is read as the
  cardinal it stands for ("Chapter IV" -> "Chapter four"), the usual English reading of "Chapter 4".
* **German** - a number with a dot before a listed noun ("3. Kapitel" -> "drittes Kapitel", "am 3. Oktober" -> "am dritten
  Oktober"), the ending chosen by the article before it.

Numbers that are clearly something else are left alone: ranges ("главы 1-3"), references ("глава 1:5", "стих 3.16"),
lists ("главы 1 и 2"), decimals, dates after "день" ("день 1 сентября").  Disable with Settings -> Narration or
``voxprint narrate --no-ordinals``.  Docs: docs/ORDINALS.md.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

from core import num_words as nw
from core.ordinal_rules import RULES

#: Russian month names in the genitive: "день 1 сентября" is a date, not "день первый".
RU_MONTHS_GEN = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
                 "ноября", "декабря")
MAX_NUMBER = 9999

_ROMAN_RE = re.compile(r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_value(token: str) -> Optional[int]:
    """Value of an upper-case Roman numeral ("XIV" -> 14), ``None`` if ``token`` is not a valid one."""
    if not token or not _ROMAN_RE.fullmatch(token):
        return None
    total = 0
    for i, ch in enumerate(token):
        v = _ROMAN_VALUES[ch]
        total += -v if i + 1 < len(token) and _ROMAN_VALUES[token[i + 1]] > v else v
    return total or None


def _number(token: str) -> Optional[int]:
    if token.isdigit():
        n = int(token)
    else:
        n = roman_value(token) or 0
    return n if 0 < n <= MAX_NUMBER else None


def _alternation(words) -> str:
    return "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True))


#: After the number: a range, a reference ("1:5", "3.16", "1,5"), a list ("1 и 2", "1, 2"), a written ending or a letter.
_NOT_ALONE = r"(?![\w])(?!\s*[-–—]\s*\d)(?![.,:]\d)(?!\s*(?:,|и|and|und)\s*\d)(?!-\w)"


# --------------------------------------------------------------------------- Russian
@lru_cache(maxsize=1)
def _ru_tables():
    rules = RULES["ru"]
    forms: Dict[str, Tuple[str, str]] = {}
    for gender, table in rules["after"].values():          # type: ignore[union-attr]
        for form, case in table.items():
            forms[form] = (gender, case)
    after = re.compile(r"(?<![\w-])(?i:(" + _alternation(forms) + r"))(\s+)(\d{1,4}|[IVXLCDM]+)" + _NOT_ALONE
                       + r"(?!\s+(?i:" + _alternation(RU_MONTHS_GEN) + r")\b)")
    suffix = rules["suffix"]
    suffix_re = re.compile(r"(?<![\w.,])(\d{1,4})\s?-\s?(" + _alternation(suffix) + r")(?![\w])", re.IGNORECASE)
    return forms, after, suffix, suffix_re, rules["suffix_default"], set(rules["prepositional"])


_WORD_AFTER = re.compile(r"\s+([\w]+)")
_WORD_BEFORE = re.compile(r"([\w]+)\s*$")


def _ru(text: str) -> str:
    forms, after, suffix, suffix_re, defaults, preps = _ru_tables()

    def by_ending(m: "re.Match[str]") -> str:
        n = int(m.group(1))
        if not 0 < n <= MAX_NUMBER:
            return m.group(0)
        ending = m.group(2).lower()
        nxt = _WORD_AFTER.match(text, m.end())
        noun = forms.get(nxt.group(1).lower()) if nxt else None
        prev = _WORD_BEFORE.search(text, 0, m.start())
        prev_word = prev.group(1).lower() if prev else ""
        spec = suffix.get(ending)
        if spec is None:
            if noun is not None:
                spec = noun
            elif ending == "м":
                spec = ("m", "prep") if prev_word in preps else defaults["м"]
            else:
                spec = defaults.get(ending, ("m", "nom"))
        gender, case = spec                                  # type: ignore[misc]
        if ending in ("е", "ые") and noun is None and n % 10 == 0 and prev_word in ("в", "во"):
            gender, case = "p", "acc"                        # "в 90-е" - the nineties
        if ending == "е" and nxt and nxt.group(1).lower() in ("годы", "гг"):
            gender, case = "p", "nom"
        return nw.ru_ordinal(n, case, gender)

    def by_noun(m: "re.Match[str]") -> str:
        n = _number(m.group(3))
        spec = forms.get(m.group(1).lower())
        if n is None or spec is None:
            return m.group(0)
        gender, case = spec
        return f"{m.group(1)}{m.group(2)}{nw.ru_ordinal(n, case, gender)}"

    text = suffix_re.sub(by_ending, text)
    return after.sub(by_noun, text)


# --------------------------------------------------------------------------- English
@lru_cache(maxsize=1)
def _en_tables():
    rules = RULES["en"]
    nouns = list(rules["after"])                             # type: ignore[arg-type]
    suffix = re.compile(r"(?<![\w.,])(\d{1,6})(?i:(" + _alternation(rules["suffix"]) + r"))(?![\w])")
    # a Roman numeral only as a heading ("Chapter IV", "Part I." / "Part I:"): "part I explain" keeps its pronoun
    roman = re.compile(r"(?<![\w-])(?i:(" + _alternation(nouns) + r"))(\s+)([IVXLCDM]+)(?=\s*(?:$|[.:\n—–]))", re.MULTILINE)
    return suffix, roman


def _en(text: str) -> str:
    suffix, roman = _en_tables()
    text = suffix.sub(lambda m: nw.en_ordinal(int(m.group(1))) if int(m.group(1)) > 0 else m.group(0), text)

    def by_noun(m: "re.Match[str]") -> str:
        n = roman_value(m.group(3))
        return f"{m.group(1)}{m.group(2)}{nw.en_cardinal(n)}" if n else m.group(0)

    return roman.sub(by_noun, text)


# --------------------------------------------------------------------------- German
_DE_UNITS = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn", "elf", "zwölf",
             "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn"]
_DE_TENS = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"]
_DE_ORD_SMALL = {1: "erst", 3: "dritt", 7: "siebt", 8: "acht"}
#: Ending of the ordinal after these words (weak "-e" / "-en", mixed after "ein"); no article = strong, by gender.
_DE_WEAK_E = ("der", "die", "das", "ins", "ans", "eine")
_DE_WEAK_EN = ("dem", "den", "des", "am", "im", "vom", "zum", "zur", "beim", "einem", "einen", "eines", "einer")
_DE_STRONG = {"m": "er", "f": "e", "n": "es"}


def _de_below100(n: int) -> str:
    if n < 20:
        return _DE_UNITS[n]
    t, u = divmod(n, 10)
    return (("ein" if u == 1 else _DE_UNITS[u]) + "und" if u else "") + _DE_TENS[t]


def de_cardinal(n: int) -> str:
    """German cardinal as one word ("einundzwanzig", "zweihundertdrei", "tausendneunhundert"), 0..999999."""
    if n < 100:
        return _de_below100(n)
    if n < 1000:
        h, r = divmod(n, 100)
        return ("ein" if h == 1 else _DE_UNITS[h]) + "hundert" + (_de_below100(r) if r else "")
    th, r = divmod(n, 1000)
    return ("ein" if th == 1 else de_cardinal(th)) + "tausend" + (de_cardinal(r) if r else "")


def de_ordinal_stem(n: int) -> str:
    """Stem before the ending: 3 "dritt", 21 "einundzwanzigst", 103 "einhundertdritt"."""
    r = n % 100
    if 1 <= r <= 19:
        head = de_cardinal(n - r) if n - r else ""
        return head + _DE_ORD_SMALL.get(r, _DE_UNITS[r] + "t")
    return de_cardinal(n) + "st"


@lru_cache(maxsize=1)
def _de_tables():
    nouns: Dict[str, str] = RULES["de"]["before"]           # type: ignore[assignment]
    pattern = re.compile(r"(?<![\w.,])(\d{1,4})\.(\s+)(?i:(" + _alternation(nouns) + r"))\b")
    return nouns, pattern


def _de(text: str) -> str:
    nouns, pattern = _de_tables()

    def sub(m: "re.Match[str]") -> str:
        n = int(m.group(1))
        if not 0 < n <= MAX_NUMBER:
            return m.group(0)
        prev = _WORD_BEFORE.search(text, 0, m.start())
        article = prev.group(1).lower() if prev else ""
        gender = nouns[m.group(3).lower()]
        if article in _DE_WEAK_EN:
            ending = "en"
        elif article in _DE_WEAK_E:
            ending = "e"
        elif article == "ein":
            ending = "es" if gender == "n" else "er"
        else:
            ending = _DE_STRONG[gender]
        return f"{de_ordinal_stem(n)}{ending}{m.group(2)}{m.group(3)}"

    return pattern.sub(sub, text)


# --------------------------------------------------------------------------- facade
_ENGINES: Dict[str, Callable[[str], str]] = {"ru": _ru, "en": _en, "de": _de}
SUPPORTED: Tuple[str, ...] = tuple(sorted(set(_ENGINES) & set(RULES)))


def language_key(language: Optional[str]) -> str:
    """``ru`` / ``en`` / ``de`` for a language name or code ("Russian", "ru-RU", "deutsch"); ``""`` if not supported."""
    from core.languages import language_code

    code = language_code(language or "").split("-")[0].lower()
    return code if code in SUPPORTED else ""


def apply_ordinals(text: str, language: Optional[str]) -> str:
    """``text`` with the ordinals of ``language`` spelled out (unchanged for an unsupported language)."""
    key = language_key(language)
    if not key or not text or not any(c.isdigit() or c in "IVXLCDM" for c in text):
        return text
    return _ENGINES[key](text)


def ordinal_step(language: Optional[str]) -> Optional[Callable[[str], str]]:
    """``text -> text`` for the narration pipeline, or ``None`` when ``language`` has no rules."""
    key = language_key(language)
    if not key:
        return None
    return lambda t: apply_ordinals(t, key)


def supported_languages() -> List[str]:
    """Codes with ordinal rules (for the docs and ``voxprint narrate --help``)."""
    return list(SUPPORTED)


# --------------------------------------------------------------------------- the setting (Settings -> Narration)
def _file():
    from infra import paths
    return paths.state_dir() / "narration_ordinals.json"


def load_enabled() -> bool:
    """Whether narration reads ordinals by context (default on; a missing or unreadable file means on)."""
    import json

    try:
        d = json.loads(_file().read_text(encoding="utf-8"))
        return bool(d.get("enabled", True)) if isinstance(d, dict) else True
    except (OSError, ValueError):
        return True


def save_enabled(enabled: bool) -> None:
    """Remember the setting (a failure to write is not an error)."""
    import json

    try:
        p = _file()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"enabled": bool(enabled)}), encoding="utf-8")
    except OSError:
        pass
