"""Numbers as words for narration: cardinals, ordinals (with case / gender), years and decimals in Russian and English.

Why own code: ``num2words`` is LGPL-2.1 (an extra obligation for a bundled, frozen application) and does not decline
Russian ordinals by the suffix found in the text ("5-му" -> "пятому").  This module is small, dependency-free and covers
what a book needs.  Russian is the detailed one (gender of "один/одна", plural forms after numerals, six cases for
ordinals); English has cardinals, ordinals and the usual year reading.  Other languages are not supported (callers
leave the digits unchanged).
"""
from __future__ import annotations

from typing import List, Tuple

# Russian word tables are data.
_UNITS = {"m": ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"],
          "f": ["ноль", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"],
          "n": ["ноль", "одно", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]}
_TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
          "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
_HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]
_SCALES = [("тысяча", "тысячи", "тысяч", "f"), ("миллион", "миллиона", "миллионов", "m"),
           ("миллиард", "миллиарда", "миллиардов", "m"), ("триллион", "триллиона", "триллионов", "m")]

#: Case / gender codes accepted by :func:`ru_ordinal`.
CASES = ("nom", "gen", "dat", "acc", "ins", "prep")


def ru_plural(n: int, forms: Tuple[str, str, str]) -> str:
    """Pick the Russian plural form for ``n`` (1 / 2-4 / 5+ rule with the 11-14 exception)."""
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return forms[0]
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return forms[1]
    return forms[2]


def _ru_triplet(n: int, gender: str) -> List[str]:
    """Words for 0..999 in the given gender (``m``/``f``/``n``)."""
    words: List[str] = []
    h, rest = divmod(n, 100)
    if h:
        words.append(_HUNDREDS[h])
    if rest >= 20:
        t, u = divmod(rest, 10)
        words.append(_TENS[t])
        if u:
            words.append(_UNITS[gender][u])
    elif rest >= 10:
        words.append(_TEENS[rest - 10])
    elif rest:
        words.append(_UNITS[gender][rest])
    return words


def ru_cardinal(n: int, gender: str = "m") -> str:
    """Russian cardinal in the nominative case; ``gender`` applies to the last group ("одна", "две").

    1000 is "тысяча" (not "одна тысяча"), as people say it.  From 10**15 up the digits are read one by one.
    """
    if n < 0:
        return "минус " + ru_cardinal(-n, gender)
    if n == 0:
        return "ноль"
    if n >= 10 ** 15:
        return " ".join(_UNITS["m"][int(c)] for c in str(n))
    words: List[str] = []
    for i in range(4, -1, -1):
        v = (n // 1000 ** i) % 1000
        if not v:
            continue
        if i == 0:
            words.extend(_ru_triplet(v, gender))
        else:
            one, few, many, g = _SCALES[i - 1]
            words.extend(_ru_triplet(v, g))
            words.append(ru_plural(v, (one, few, many)))
    if words[:2] == ["один", "тысяча"] or words[:2] == ["одна", "тысяча"]:
        words = words[1:]
    return " ".join(words)


# --------------------------------------------------------------------------- Russian ordinals

#: Nominative masculine forms of the last component of an ordinal.
_ORD = {1: "первый", 2: "второй", 3: "третий", 4: "четвёртый", 5: "пятый", 6: "шестой", 7: "седьмой", 8: "восьмой",
        9: "девятый", 10: "десятый", 11: "одиннадцатый", 12: "двенадцатый", 13: "тринадцатый", 14: "четырнадцатый",
        15: "пятнадцатый", 16: "шестнадцатый", 17: "семнадцатый", 18: "восемнадцатый", 19: "девятнадцатый",
        20: "двадцатый", 30: "тридцатый", 40: "сороковой", 50: "пятидесятый", 60: "шестидесятый",
        70: "семидесятый", 80: "восьмидесятый", 90: "девяностый", 100: "сотый", 200: "двухсотый",
        300: "трёхсотый", 400: "четырёхсотый", 500: "пятисотый", 600: "шестисотый", 700: "семисотый",
        800: "восьмисотый", 900: "девятисотый"}
#: Round thousands that are written as one word ("двухтысячный").
_ORD_THOUSANDS = {1000: "тысячный", 2000: "двухтысячный", 3000: "трёхтысячный", 4000: "четырёхтысячный",
                  5000: "пятитысячный", 10000: "десятитысячный"}

_HARD = {  # (gender/number, case) -> ending after the stem
    ("m", "nom"): "ый", ("m", "gen"): "ого", ("m", "dat"): "ому", ("m", "acc"): "ый", ("m", "ins"): "ым", ("m", "prep"): "ом",
    ("f", "nom"): "ая", ("f", "gen"): "ой", ("f", "dat"): "ой", ("f", "acc"): "ую", ("f", "ins"): "ой", ("f", "prep"): "ой",
    ("n", "nom"): "ое", ("n", "gen"): "ого", ("n", "dat"): "ому", ("n", "acc"): "ое", ("n", "ins"): "ым", ("n", "prep"): "ом",
    ("p", "nom"): "ые", ("p", "gen"): "ых", ("p", "dat"): "ым", ("p", "acc"): "ые", ("p", "ins"): "ыми", ("p", "prep"): "ых",
}
_SOFT = {  # "третий"
    ("m", "nom"): "ий", ("m", "gen"): "ьего", ("m", "dat"): "ьему", ("m", "acc"): "ий", ("m", "ins"): "ьим", ("m", "prep"): "ьем",
    ("f", "nom"): "ья", ("f", "gen"): "ьей", ("f", "dat"): "ьей", ("f", "acc"): "ью", ("f", "ins"): "ьей", ("f", "prep"): "ьей",
    ("n", "nom"): "ье", ("n", "gen"): "ьего", ("n", "dat"): "ьему", ("n", "acc"): "ье", ("n", "ins"): "ьим", ("n", "prep"): "ьем",
    ("p", "nom"): "ьи", ("p", "gen"): "ьих", ("p", "dat"): "ьим", ("p", "acc"): "ьи", ("p", "ins"): "ьими", ("p", "prep"): "ьих",
}


def _decline_ordinal_word(word: str, case: str, gender: str) -> str:
    """Decline one nominative-masculine ordinal word ("пятый", "второй", "третий") to ``case`` / ``gender`` (m f n p)."""
    table = _HARD
    if word.endswith("ий"):
        table, stem = _SOFT, word[:-2]
    elif word.endswith("ой") or word.endswith("ый"):
        stem = word[:-2]
        if word.endswith("ой") and gender == "m" and case in ("nom", "acc"):
            return word                        # stressed ending: "второй", "шестой", "сороковой" (not "вторый")
    else:
        return word
    return stem + table[(gender, case)]


def ru_ordinal(n: int, case: str = "nom", gender: str = "m", year: bool = False) -> str:
    """Russian ordinal numeral: ``ru_ordinal(1999, "gen")`` -> "тысяча девятьсот девяносто девятого".

    ``gender`` is ``m`` / ``f`` / ``n`` / ``p`` (plural); the case is one of :data:`CASES` (accusative equals the
    nominative for inanimate nouns except feminine singular).  Only the last word is an ordinal, the rest is cardinal.
    ``year`` is kept for call-site clarity: "тысяча" never gets "одна".
    """
    if n <= 0:
        return "нулевой" if n == 0 and (case, gender) == ("nom", "m") else ru_cardinal(n)
    if n in _ORD_THOUSANDS:
        return _decline_ordinal_word(_ORD_THOUSANDS[n], case, gender)
    if n % 1000 == 0:
        return ru_cardinal(n)                                      # rare; the cardinal reads acceptably
    if n % 10 and n % 100 > 20:
        last = n % 10
    elif n % 100:
        last = n % 100
    else:
        last = n % 1000
    head = n - last
    words = ru_cardinal(head) + " " if head else ""
    return words + _decline_ordinal_word(_ORD[last], case, gender)


# --------------------------------------------------------------------------- fractions and plural helpers (ru)

_FRACTION_NAMES = {1: ("десятая", "десятых"), 2: ("сотая", "сотых"), 3: ("тысячная", "тысячных"),
                   4: ("десятитысячная", "десятитысячных"), 5: ("стотысячная", "стотысячных"),
                   6: ("миллионная", "миллионных")}


def ru_decimal(int_part: str, frac: str) -> str:
    """"2,5" -> "две целых пять десятых"; fractions longer than 6 digits are read digit by digit after "запятая"."""
    n = int(int_part or "0")
    if not frac:
        return ru_cardinal(n)
    if len(frac) > 6:
        return ru_cardinal(n) + " запятая " + " ".join(_UNITS["m"][int(c)] for c in frac)
    whole = ru_cardinal(n, "f") + (" целая" if n % 10 == 1 and n % 100 != 11 else " целых")
    k = int(frac)
    one, many = _FRACTION_NAMES[len(frac)]
    return f"{whole} {ru_cardinal(k, 'f')} {one if k % 10 == 1 and k % 100 != 11 else many}"


# --------------------------------------------------------------------------- English

_EN_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
             "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_EN_SCALES = ["", "thousand", "million", "billion", "trillion"]
_EN_ORD_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                     "nine": "ninth", "twelve": "twelfth"}


def _en_triplet(n: int) -> List[str]:
    """Words for 0..999."""
    words: List[str] = []
    h, rest = divmod(n, 100)
    if h:
        words += [_EN_UNITS[h], "hundred"]
    if rest >= 20:
        t, u = divmod(rest, 10)
        words.append(_EN_TENS[t] + (" " + _EN_UNITS[u] if u else ""))
    elif rest:
        words.append(_EN_UNITS[rest])
    return words


def en_cardinal(n: int) -> str:
    """English cardinal ("one thousand two hundred thirty four")."""
    if n < 0:
        return "minus " + en_cardinal(-n)
    if n == 0:
        return "zero"
    if n >= 10 ** 15:
        return " ".join(_EN_UNITS[int(c)] for c in str(n))
    words: List[str] = []
    for i in range(4, -1, -1):
        v = (n // 1000 ** i) % 1000
        if v:
            words.extend(_en_triplet(v))
            if i:
                words.append(_EN_SCALES[i])
    return " ".join(words)


def en_ordinal(n: int) -> str:
    """English ordinal ("twenty first")."""
    words = en_cardinal(n).split(" ")
    last = words[-1]
    if last in _EN_ORD_IRREGULAR:
        words[-1] = _EN_ORD_IRREGULAR[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def en_year(n: int) -> str:
    """Year reading: 1999 "nineteen ninety nine", 1905 "nineteen oh five", 2000 "two thousand", 2010 "twenty ten"."""
    if n < 1100 or 2000 <= n <= 2009 or n > 9999:
        return en_cardinal(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        return en_cardinal(hi) + " hundred"
    if lo < 10:
        return f"{en_cardinal(hi)} oh {_EN_UNITS[lo]}"
    return f"{en_cardinal(hi)} {en_cardinal(lo)}"


def en_decimal(int_part: str, frac: str) -> str:
    """"2.5" -> "two point five" (digits after the point are read one by one)."""
    out = en_cardinal(int(int_part or "0"))
    return out + (" point " + " ".join(_EN_UNITS[int(c)] for c in frac) if frac else "")


# --------------------------------------------------------------------------- facade

SUPPORTED = ("ru", "en")


def lang_code(language: str) -> str:
    """Normalize a language name or code ("Russian", "ru", "english") to ``ru`` / ``en`` / ``""`` (unsupported)."""
    s = (language or "").strip().lower()
    if s in ("ru", "russian", "rus", "русский"):
        return "ru"
    if s in ("en", "english", "eng"):
        return "en"
    if s in ("de", "german", "deutsch"):
        return "de"
    return ""


def cardinal(n: int, lang: str, gender: str = "m") -> str:
    """Cardinal in ``lang`` (``ru`` or ``en``)."""
    return ru_cardinal(n, gender) if lang == "ru" else en_cardinal(n)
