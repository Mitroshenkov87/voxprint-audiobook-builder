"""Dates and numbers spoken in words ("третьего октября две тысячи двадцать шестого года", "third of October twenty twenty-six",
"dritten Oktober zweitausendsechsundzwanzig") -> ISO date.  Recognisers often write numbers as words, so the digit-only date reader of
:mod:`core.consent` misses the date of a spoken consent statement; this module is the fallback (Russian, English, German; no dependencies).
"""
from __future__ import annotations

import re
from datetime import date as _date
from typing import Dict, List, Optional, Tuple

VAL, HUNDRED, THOUSAND = "val", "hundred", "thousand"

MONTHS: Dict[str, int] = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7, "августа": 8, "сентября": 9,
    "октября": 10, "ноября": 11, "декабря": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
    "januar": 1, "februar": 2, "marz": 3, "mai": 5, "juni": 6, "juli": 7, "oktober": 10, "dezember": 12,
}

# --- Russian: stems (the longest matching stem wins; endings of 0..6 letters are free, so cardinals and ordinals share a stem).
_RU = {
    "один": 1, "одн": 1, "перв": 1, "дв": 2, "втор": 2, "три": 3, "трет": 3, "четыр": 4, "четв": 4, "пят": 5, "шест": 6, "сем": 7, "седьм": 7,
    "восем": 8, "восьм": 8, "девят": 9, "десят": 10, "одиннадцат": 11, "двенадцат": 12, "тринадцат": 13, "четырнадцат": 14,
    "пятнадцат": 15, "шестнадцат": 16, "семнадцат": 17, "восемнадцат": 18, "девятнадцат": 19, "двадцат": 20, "тридцат": 30,
    "сорок": 40, "пятьдесят": 50, "пятидесят": 50, "шестьдесят": 60, "шестидесят": 60, "семьдесят": 70, "семидесят": 70,
    "восемьдесят": 80, "восьмидесят": 80, "девяност": 90, "сто": 100, "двест": 200, "триста": 300, "четыреста": 400,
    "пятьсот": 500, "пятисот": 500, "шестьсот": 600, "шестисот": 600, "семьсот": 700, "семисот": 700, "восемьсот": 800,
    "восьмисот": 800, "девятьсот": 900, "девятисот": 900,
}
_RU_THOUSAND = ("тысяч",)

# --- English (cardinals and ordinals; "hundred"/"thousand" multiply).
_EN = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_EN.update({"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90})
_EN_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
           "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17,
           "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30}

# --- German (compounds are split into pieces; accents folded: fünf -> funf, zwölf -> zwolf, ß -> ss).
_DE = {"ein": 1, "eins": 1, "zwei": 2, "drei": 3, "vier": 4, "funf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10, "elf": 11,
       "zwolf": 12, "dreizehn": 13, "vierzehn": 14, "funfzehn": 15, "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19,
       "zwanzig": 20, "dreissig": 30, "vierzig": 40, "funfzig": 50, "sechzig": 60, "siebzig": 70, "achtzig": 80, "neunzig": 90,
       "ers": 1, "drit": 3, "sieb": 7, "ach": 8}
_DE_PIECES = sorted(list(_DE) + ["hundert", "tausend", "und"], key=len, reverse=True)
_DE_SUFFIX = ("sten", "ster", "stes", "ste", "ten", "ter", "tes", "te")

Token = Tuple[str, int]


def _fold(s: str) -> str:
    return (s.lower().replace("\u0451", "\u0435").replace("\u00e4", "a").replace("\u00f6", "o").replace("\u00fc", "u")
            .replace("\u00df", "ss"))


def _ru_token(w: str) -> Optional[Token]:
    if any(w.startswith(s) and len(w) - len(s) <= 7 for s in _RU_THOUSAND):
        return (THOUSAND, 1000)
    best = None
    for stem, v in _RU.items():
        if w.startswith(stem) and len(w) - len(stem) <= 6 and (best is None or len(stem) > len(best[0])):
            best = (stem, v)
    return (VAL, best[1]) if best else None


def _en_tokens(w: str) -> Optional[List[Token]]:
    if w in _EN:
        return [(VAL, _EN[w])]
    if w in _EN_ORD:
        return [(VAL, _EN_ORD[w])]
    if w == "hundred":
        return [(HUNDRED, 100)]
    if w == "thousand":
        return [(THOUSAND, 1000)]
    if w in ("and", "of", "the", "th", "st", "nd", "rd"):
        return []
    return None


def _de_pieces(w: str) -> Optional[List[Token]]:
    out: List[Token] = []
    while w:
        for p in _DE_PIECES:
            if w.startswith(p):
                out.append((VAL, 0) if p == "und" else (HUNDRED, 100) if p == "hundert" else (THOUSAND, 1000) if p == "tausend" else (VAL, _DE[p]))
                w = w[len(p):]
                break
        else:
            return None
    return out


def _de_tokens(w: str) -> Optional[List[Token]]:
    for cand in [w] + [w[:-len(s)] for s in _DE_SUFFIX if w.endswith(s) and len(w) > len(s) + 1]:
        got = _de_pieces(cand)
        if got:
            return got
    return None


def _total(tokens: List[Token]) -> int:
    total = cur = 0
    for kind, v in tokens:
        if kind == VAL:
            cur += v
        elif kind == HUNDRED:
            cur = (cur or 1) * 100
        else:
            total += (cur or 1) * 1000
            cur = 0
    return total + cur


def _en_year(tokens: List[Token]) -> int:
    """"two thousand twenty six" or "twenty twenty-six" / "nineteen ninety nine"."""
    if any(k in (HUNDRED, THOUSAND) for k, _ in tokens):
        return _total(tokens)
    groups: List[int] = []
    for _, v in tokens:
        if not groups or (v >= 10 and groups[-1] >= 10) or (v == 0 and False):
            groups.append(v)
        else:
            groups[-1] += v
    return groups[0] * 100 + groups[1] if len(groups) == 2 else (groups[0] if groups else 0)


def _tokens_for(lang: str, w: str) -> Optional[List[Token]]:
    if w.isdigit():
        return [(VAL, int(w))]
    if lang == "ru":
        t = _ru_token(w)
        return [t] if t else None
    if lang == "en":
        return _en_tokens(w)
    return _de_tokens(w)


def _is_filler(lang: str, w: str) -> bool:
    return w in {"ru": ("года", "год", "г", "и"), "en": ("and", "of", "the", "th", "st", "nd", "rd", "in", "year"),
                 "de": ("und", "am", "den", "dem", "jahr", "jahres", "im")}[lang]


def _collect(lang: str, words: List[str], rng) -> List[Token]:
    out: List[Token] = []
    for j in rng:
        w = words[j]
        toks = _tokens_for(lang, w)
        if toks is None:
            if _is_filler(lang, w):
                continue
            break
        out.extend(toks)
    return out


def find_date(text: str) -> str:
    """ISO date from a statement with the date written in words, or "" (digit dates are handled by the caller)."""
    words = [_fold(w) for w in re.findall(r"[^\W\d_]+|\d+", text)]
    for i, w in enumerate(words):
        mo = MONTHS.get(w)
        if not mo:
            continue
        for lang in ("ru", "en", "de"):
            if lang == "ru" and not re.match(r"[\u0430-\u044f]", w):
                continue
            if lang != "ru" and re.match(r"[\u0430-\u044f]", w):
                continue
            before = _collect(lang, words, range(i - 1, max(-1, i - 4), -1))[::-1]
            after_rng = range(i + 1, min(len(words), i + 10))
            after = _collect(lang, words, after_rng)
            day = _total(before) if before else 0
            if not day and lang == "en" and after and after[0][1] in range(1, 32) and words[i + 1] in _EN_ORD:
                day, after = after[0][1], after[1:]          # "October third, twenty twenty six"
            if not day and lang == "en" and words[i + 1:i + 2] and words[i + 1].isdigit() and len(words[i + 1]) <= 2:
                day, after = int(words[i + 1]), after[1:]    # "October 3 twenty twenty six"
            year = (_en_year(after) if lang == "en" else _total(after)) if after else 0
            if 1 <= day <= 31 and 1900 <= year <= 2100:
                try:
                    return _date(year, mo, day).isoformat()
                except ValueError:
                    pass
    return ""
