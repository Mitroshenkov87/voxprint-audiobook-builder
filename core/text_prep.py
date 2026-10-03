"""Rule-based preparation of a book for narration (offline, standard library only, Russian and English).

What a TTS engine stumbles over is mostly *layout* and *notation*, not grammar: hard-wrapped lines, soft hyphens,
footnote marks, page numbers, URLs, digits, dates, abbreviations, Roman numerals in headings.  This module turns them into
plain spoken text.  It is the first stage of the automatic pipeline (the user never reviews the text):

1. rule-based preparation (here), 2. optional neural clean-up (:mod:`core.text_cleanup`), 3. synthesis.

Steps (keys of :data:`STEP_KEYS`, each can be switched off):

``layout``     invisible characters, ligatures, soft hyphens, hyphenation at line ends, hard-wrapped lines, spacing;
``quotes``     typographic quotes / apostrophes, dashes (``--``, `` - ``, dialogue dashes), ellipses;
``noise``      footnote marks (``[1]``, ``¹``, ``*``), page numbers, running headers repeated on many pages;
``links``      URLs and e-mail addresses become a short phrase ("ссылка", "link");
``numbers``    integers, decimals, ordinals ("5-му"), years, dates, percents, currency, units -> words;
``abbrev``     "т. д.", "г.", "им.", "Mr.", "Dr.", "etc." ... -> full words;
``headings``   "Глава XII" -> "Глава двенадцатая", ALL-CAPS headings -> normal case, bare Roman numerals -> words.

Very long sentences need no step of their own: the chunker (:mod:`core.chunker`) always cuts them at clause boundaries
with the aligner's clause splitter (:func:`core.text_utils.split_clauses`).

Languages: ``ru`` and ``en`` get every step; for other languages only the language-neutral steps run (layout, quotes,
noise, links) and digits are left for the engine.  Everything is deterministic, so a resumed job produces identical text
(the narration cache is keyed by the prepared text).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

from core import num_words as nw
from core.book_parsers import Book, Chapter

STEP_LAYOUT, STEP_QUOTES, STEP_NOISE, STEP_LINKS, STEP_NUMBERS, STEP_ABBREV, STEP_HEADINGS = (
    "layout", "quotes", "noise", "links", "numbers", "abbrev", "headings")
STEP_KEYS: Tuple[str, ...] = (STEP_LAYOUT, STEP_NOISE, STEP_QUOTES, STEP_LINKS, STEP_HEADINGS, STEP_NUMBERS, STEP_ABBREV)
#: Steps that need language knowledge (digits / abbreviations / headings).
LANGUAGE_STEPS = frozenset({STEP_NUMBERS, STEP_ABBREV, STEP_HEADINGS})
#: Execution order (numbers before abbreviations: "1999 г." must still be seen as a year).
_ORDER: Tuple[str, ...] = (STEP_LAYOUT, STEP_NOISE, STEP_QUOTES, STEP_LINKS, STEP_HEADINGS, STEP_NUMBERS, STEP_ABBREV)


@dataclass(frozen=True)
class PrepOptions:
    """Which rule-based steps run (all by default)."""
    steps: FrozenSet[str] = field(default_factory=lambda: frozenset(STEP_KEYS))

    @classmethod
    def none(cls) -> "PrepOptions":
        """No rule-based preparation."""
        return cls(frozenset())


@dataclass
class PrepReport:
    """What the preparation did: language, applied steps and the number of replacements per step."""
    language: str = ""
    steps: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)       # steps that need a language this book does not have


class _Ctx:
    """Language and counters shared by the step functions of one run."""

    def __init__(self, lang: str, counts: Optional[Counter] = None) -> None:
        self.lang = lang
        self.counts: Counter = counts if counts is not None else Counter()

    def sub(self, key: str, rx: "re.Pattern[str]", repl, text: str) -> str:
        """``rx.sub`` that counts the replacements under ``key``."""
        text, n = rx.subn(repl, text)
        if n:
            self.counts[key] += n
        return text


# =========================================================================== layout

_INVISIBLE = re.compile("[\u00ad\u200b\u200c\u200d\u2060\ufeff\u200e\u200f\u202a-\u202e]")
_SPACES = re.compile("[\u00a0\u2000-\u200a\u202f\u205f\u3000\t\u000b\u000c]")
_CONTROL = re.compile("[\u0000-\u0008\u000e-\u001f\u007f-\u009f]")
_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"}
_TERMINAL = tuple(".!?…:;»”\"')]")
_DASH_START = re.compile(r"^\s*[—–]\s|^\s*-\s+\S")
#: Parts that keep their hyphen when a word is broken at the end of a line ("кое-\nчто", "twenty-\none").
_KEEP_LEFT = {"кое", "кой", "по", "во", "что", "кто", "как", "где", "куда", "когда", "чей", "какой", "twenty", "thirty",
              "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "well", "self", "ex", "non", "co", "anti"}
_KEEP_RIGHT = {"то", "либо", "нибудь", "таки", "ка", "де", "ли", "же", "бы", "one", "two", "three", "four", "five", "six",
               "seven", "eight", "nine", "like", "based", "known", "free", "made"}
_HYPHEN_BREAK = re.compile(r"(\w+)[-\u2010\u2011]\n[ \t]*(\w+)")


def _join_hyphen(m: "re.Match[str]") -> str:
    """Join "перенос-\\nчение" -> "перенесение"; keep the hyphen for particles and compounds."""
    left, right = m.group(1), m.group(2)
    if not right[:1].islower():
        return f"{left}-\n{right}"                                  # capital: a name or a new sentence, leave it
    if left.lower() in _KEEP_LEFT or right.lower() in _KEEP_RIGHT:
        return f"{left}-{right}"
    return left + right


def _is_poem(lines: List[str]) -> bool:
    """A block of short lines is verse: keep its line breaks."""
    if len(lines) < 3:
        return False
    lens = [len(x) for x in lines]
    return max(lens) < 60 and sum(lens) / len(lens) < 40


def _unwrap_block(block: str) -> str:
    """Join hard-wrapped lines of one block (a block is text between blank lines)."""
    lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
    if len(lines) <= 1:
        return lines[0] if lines else ""
    if _is_poem(lines):
        return "\n".join(lines)
    paras: List[str] = [lines[0]]
    for ln in lines[1:]:
        prev = paras[-1]
        starts_dialogue = bool(_DASH_START.match(ln))
        continues = (not prev.endswith(_TERMINAL) or ln[:1].islower()) and not starts_dialogue
        if continues:
            paras[-1] = prev + " " + ln
        else:
            paras.append(ln)                                        # line break after a full stop = new paragraph
    return "\n\n".join(paras)


def step_layout(text: str, ctx: _Ctx) -> str:
    """Invisible characters, ligatures, hyphenation, hard-wrapped lines and spacing."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)
    text = ctx.sub(STEP_LAYOUT, _INVISIBLE, "", text)
    text = _CONTROL.sub("", text)
    text = _SPACES.sub(" ", text)
    text = re.sub(r"[ ]*\n[ ]*", "\n", text)
    text = ctx.sub(STEP_LAYOUT, _HYPHEN_BREAK, _join_hyphen, text)
    blocks = [b for b in re.split(r"\n{2,}", text)]
    out: List[str] = []
    for b in blocks:
        u = _unwrap_block(b)
        if u != b.strip() and "\n" in b:
            ctx.counts[STEP_LAYOUT] += b.count("\n")
        if u:
            out.append(u)
    text = "\n\n".join(out)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" +([,.;:!?…»”)])", r"\1", text)                  # no space before closing punctuation
    text = re.sub(r"([(«“]) +", r"\1", text)
    return text.strip()


# =========================================================================== quotes and dashes

_DQUOTES = re.compile("[\u201c\u201d\u201e\u201f\u00ab\u00bb\u2039\u203a\u301d\u301e\u301f\uff02]")
_SQUOTES = re.compile("[\u2018\u2019\u201a\u201b\u02bc\u2032\uff07`\u00b4]")
_DASH_VARIANTS = re.compile(r"\s*(?:--+|\u2015|\u2212|\u2012|\u2013|\u2014)\s*")


def step_quotes(text: str, ctx: _Ctx) -> str:
    """Typographic quotes -> plain, all dash variants -> an em dash with spaces, ellipses, repeated marks."""
    text = ctx.sub(STEP_QUOTES, _DQUOTES, '"', text)
    text = ctx.sub(STEP_QUOTES, _SQUOTES, "'", text)
    # dialogue dash at the start of a line: "- Привет", "– Привет", "--Привет"
    text = ctx.sub(STEP_QUOTES, re.compile(r"(?m)^[ ]*(?:--+|[\u2013\u2014\u2015-])[ ]*(?=\S)"), "\u2014 ", text)
    # between two digits an en dash is a range: 1914-1918 stays readable as "1914 — 1918"
    text = ctx.sub(STEP_QUOTES, re.compile(r"(?<=\d)\s*[\u2013\u2014]\s*(?=\d)"), " \u2014 ", text)
    text = ctx.sub(STEP_QUOTES, re.compile(r"(?<=\S) +[-\u2013\u2014\u2015\u2212]+ +(?=\S)|(?<=[^\s\d])--+(?=[^\s])|(?<=\S)\s*\u2015\s*(?=\S)"), " \u2014 ", text)
    text = ctx.sub(STEP_QUOTES, re.compile(r"(?m)(?<=\S)\u2013(?=\s)|(?<=\s)\u2013(?=\S)"), "\u2014", text)
    text = ctx.sub(STEP_QUOTES, re.compile(r"\.{3,}|\. \. \.|\u2026{2,}"), "\u2026", text)
    text = ctx.sub(STEP_QUOTES, re.compile(r"([!?])\1{1,}"), r"\1", text)
    text = re.sub(r" {2,}", " ", text)
    return text


# =========================================================================== footnotes, page numbers

_FOOTNOTES = re.compile(r"\[\d{1,3}\]|\[\*+\]|\{\d{1,3}\}|[\u00b9\u00b2\u00b3\u2070-\u2079]+|(?<=[\w.,;:!?»”\"')])\*{1,3}(?=\s|$)")
_PAGE_NUMBER = re.compile(r"^\s*(?:[-\u2013\u2014]\s*)?(?:(?:стр\.?|с\.|page|p\.|pp\.)\s*)?\d{1,4}\s*(?:[-\u2013\u2014])?\s*$", re.I)


def step_noise(text: str, ctx: _Ctx) -> str:
    """Remove footnote marks, page-number paragraphs and headers repeated on many pages."""
    text = ctx.sub(STEP_NOISE, _FOOTNOTES, "", text)
    paras = text.split("\n\n")
    counts = Counter(p.strip() for p in paras if p.strip() and "\n" not in p.strip())
    kept: List[str] = []
    for p in paras:
        s = p.strip()
        if not s:
            continue
        if _PAGE_NUMBER.match(s):
            ctx.counts[STEP_NOISE] += 1
            continue
        if counts[s] >= 5 and len(s) <= 60 and not s.endswith(_TERMINAL) and any(c.isalpha() for c in s):
            ctx.counts[STEP_NOISE] += 1                           # a running header repeated page after page
            continue
        kept.append(p)
    return "\n\n".join(kept)


# =========================================================================== links

_URL = re.compile(r"(?i)\b(?:https?://|ftp://|www\.)[^\s<>\"«»]+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_LINK_WORDS = {"ru": ("ссылка", "адрес электронной почты"), "en": ("link", "email address"), "de": ("Link", "E-Mail-Adresse")}


def step_links(text: str, ctx: _Ctx) -> str:
    """URLs and e-mail addresses -> a short phrase (spelling them out is unlistenable)."""
    link, mail = _LINK_WORDS.get(ctx.lang, _LINK_WORDS["en"])

    def url(m: "re.Match[str]") -> str:
        """Keep sentence punctuation that was glued to the URL."""
        raw = m.group(0)
        tail = ""
        while raw and raw[-1] in ".,;:!?)\u00bb\"'":
            tail = raw[-1] + tail
            raw = raw[:-1]
        return link + tail
    text = ctx.sub(STEP_LINKS, _EMAIL, mail, text)
    return ctx.sub(STEP_LINKS, _URL, url, text)


# =========================================================================== numbers

_RU_MONTHS = {"января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7, "августа": 8,
              "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12}
_RU_MONTH_BY_NUM = {v: k for k, v in _RU_MONTHS.items()}
_EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
_PREPS_PREP = {"в", "во", "на", "при", "о", "об", "обо"}
_ROMAN_RX = re.compile(r"^(?=[MDCLXVI])M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(s: str) -> int:
    """Value of a valid Roman numeral (0 if ``s`` is not one)."""
    s = s.upper()
    if not s or not _ROMAN_RX.match(s):
        return 0
    total = 0
    for i, ch in enumerate(s):
        v = _ROMAN_VALUES[ch]
        total += -v if i + 1 < len(s) and _ROMAN_VALUES[s[i + 1]] > v else v
    return total


def _prev_word(text: str, pos: int) -> str:
    """Lower-case word that ends right before ``pos`` (spaces skipped)."""
    m = re.search(r"([^\W\d_]+)\s*$", text[max(0, pos - 24):pos])
    return m.group(1).lower() if m else ""


def _next_word(text: str, pos: int) -> str:
    """Lower-case word that starts right after ``pos`` (spaces skipped)."""
    m = re.match(r"\s*([^\W\d_]+)", text[pos:pos + 40])
    return m.group(1).lower() if m else ""


def _ru_gender_for(n: int, next_word: str) -> str:
    """Gender of "один/одна/одно", "два/две" guessed from the noun that follows the numeral."""
    last = n % 10
    if n % 100 in (11, 12, 13, 14):
        return "m"
    if last == 1:
        if next_word.endswith(("а", "я")):
            return "f"
        if next_word.endswith(("о", "е")):
            return "n"
    elif last == 2 and next_word.endswith(("ы", "и")):
        return "f"
    return "m"


def _int(s: str) -> int:
    """Integer value of a digit string that may contain thin spaces / separators."""
    return int(re.sub(r"\D", "", s) or "0")


def _split_decimal(s: str) -> Tuple[str, str]:
    """"2,50" -> ("2", "50")."""
    ip, _, fp = re.sub(r"[ \u00a0]", "", s).replace(",", ".").partition(".")
    return ip, fp


# currency tables: code -> (language -> (major forms, major gender, minor forms, minor gender))
_CURRENCY = {
    "RUB": {"ru": (("рубль", "рубля", "рублей"), "m", ("копейка", "копейки", "копеек"), "f"),
            "en": (("ruble", "rubles", "rubles"), "m", ("kopeck", "kopecks", "kopecks"), "m")},
    "USD": {"ru": (("доллар", "доллара", "долларов"), "m", ("цент", "цента", "центов"), "m"),
            "en": (("dollar", "dollars", "dollars"), "m", ("cent", "cents", "cents"), "m")},
    "EUR": {"ru": (("евро", "евро", "евро"), "m", ("цент", "цента", "центов"), "m"),
            "en": (("euro", "euros", "euros"), "m", ("cent", "cents", "cents"), "m")},
    "GBP": {"ru": (("фунт", "фунта", "фунтов"), "m", ("пенс", "пенса", "пенсов"), "m"),
            "en": (("pound", "pounds", "pounds"), "m", ("penny", "pence", "pence"), "m")},
}
_CUR_SYMBOLS = {"₽": "RUB", "руб": "RUB", "руб.": "RUB", "$": "USD", "€": "EUR", "£": "GBP", "долл": "USD", "долл.": "USD"}
_NUM = r"\d{1,3}(?:[ \u00a0]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
_SCALE_RU = {"тыс": ("тысяча", "тысячи", "тысяч"), "тысяч": ("тысяча", "тысячи", "тысяч"),
             "млн": ("миллион", "миллиона", "миллионов"), "млрд": ("миллиард", "миллиарда", "миллиардов"),
             "миллион": ("миллион", "миллиона", "миллионов"), "миллиона": ("миллион", "миллиона", "миллионов"),
             "миллионов": ("миллион", "миллиона", "миллионов"), "миллиард": ("миллиард", "миллиарда", "миллиардов"),
             "миллиарда": ("миллиард", "миллиарда", "миллиардов"), "миллиардов": ("миллиард", "миллиарда", "миллиардов")}
_SCALE_EN = {"thousand", "million", "billion", "trillion"}


def _ru_plural_unit(n: int, forms) -> str:
    """Plural form of a noun after the integer ``n``."""
    return nw.ru_plural(n, forms)


def _money(ctx: _Ctx, code: str, number: str, scale: str = "") -> str:
    """"5,50" + USD -> "пять долларов пятьдесят центов" / "five dollars fifty cents"."""
    ip, fp = _split_decimal(number)
    major, mg, minor, ng = _CURRENCY[code][ctx.lang]
    n = int(ip or "0")
    if ctx.lang == "ru":
        if scale:
            sc = _SCALE_RU.get(scale.lower().rstrip("."), ("тысяча", "тысячи", "тысяч"))
            return f"{nw.ru_cardinal(n, 'f' if sc[0] == 'тысяча' else 'm')} {nw.ru_plural(n, sc)} {major[2]}"
        if fp and len(fp) == 2 and int(fp):
            return (f"{nw.ru_cardinal(n, mg)} {_ru_plural_unit(n, major)} "
                    f"{nw.ru_cardinal(int(fp), ng)} {_ru_plural_unit(int(fp), minor)}")
        if fp:
            return f"{nw.ru_decimal(ip, fp)} {major[1] if code != 'EUR' else major[0]}"
        return f"{nw.ru_cardinal(n, mg)} {_ru_plural_unit(n, major)}"
    words = nw.en_decimal(ip, fp) if fp and len(fp) != 2 else nw.en_cardinal(n)
    if scale:
        return f"{words} {scale.lower()} {major[1]}"
    noun = major[0] if n == 1 and not fp else major[1]
    out = f"{words} {noun}"
    if fp and len(fp) == 2 and int(fp):
        k = int(fp)
        out += f" {nw.en_cardinal(k)} {minor[0] if k == 1 else minor[1]}"
    return out


# unit tables: written form -> (language -> (forms, gender))
_UNITS_RU = {
    "км/ч": (("километр в час", "километра в час", "километров в час"), "m"),
    "км": (("километр", "километра", "километров"), "m"), "кг": (("килограмм", "килограмма", "килограммов"), "m"),
    "см": (("сантиметр", "сантиметра", "сантиметров"), "m"), "мм": (("миллиметр", "миллиметра", "миллиметров"), "m"),
    "мг": (("миллиграмм", "миллиграмма", "миллиграммов"), "m"), "мл": (("миллилитр", "миллилитра", "миллилитров"), "m"),
    "мин": (("минута", "минуты", "минут"), "f"), "сек": (("секунда", "секунды", "секунд"), "f"),
    "м": (("метр", "метра", "метров"), "m"), "л": (("литр", "литра", "литров"), "m"),
    "ч": (("час", "часа", "часов"), "m"), "т": (("тонна", "тонны", "тонн"), "f"),
    "°C": (("градус Цельсия", "градуса Цельсия", "градусов Цельсия"), "m"),
    "°С": (("градус Цельсия", "градуса Цельсия", "градусов Цельсия"), "m"),
}
_UNITS_EN = {"km/h": ("kilometer per hour", "kilometers per hour"), "mph": ("mile per hour", "miles per hour"),
             "km": ("kilometer", "kilometers"), "kg": ("kilogram", "kilograms"), "cm": ("centimeter", "centimeters"),
             "mm": ("millimeter", "millimeters"), "mg": ("milligram", "milligrams"), "ml": ("milliliter", "milliliters"),
             "min": ("minute", "minutes"), "sec": ("second", "seconds"), "m": ("meter", "meters"), "l": ("liter", "liters"),
             "g": ("gram", "grams"), "h": ("hour", "hours"), "lb": ("pound", "pounds"), "oz": ("ounce", "ounces"),
             "°C": ("degree Celsius", "degrees Celsius"), "°F": ("degree Fahrenheit", "degrees Fahrenheit")}
_UNIT_ALT_RU = "|".join(re.escape(u) for u in sorted(_UNITS_RU, key=len, reverse=True))
_UNIT_ALT_EN = "|".join(re.escape(u) for u in sorted(_UNITS_EN, key=len, reverse=True))

_ORD_SUFFIX_RU = {  # suffix -> (case, gender); ``None`` case = decide by the preceding preposition
    "й": ("nom", "m"), "ый": ("nom", "m"), "ий": ("nom", "m"), "я": ("nom", "f"), "ая": ("nom", "f"), "ья": ("nom", "f"),
    "ю": ("acc", "f"), "ую": ("acc", "f"), "ью": ("acc", "f"), "е": ("nom", "n"), "ое": ("nom", "n"), "ье": ("nom", "n"),
    "ые": ("nom", "p"), "ие": ("nom", "p"), "ьи": ("nom", "p"), "го": ("gen", "m"), "ого": ("gen", "m"),
    "его": ("gen", "m"), "му": ("dat", "m"), "ому": ("dat", "m"), "ему": ("dat", "m"), "м": (None, "m"),
    "ым": ("ins", "m"), "им": ("ins", "m"), "ом": ("prep", "m"), "ем": ("prep", "m"), "ой": ("gen", "f"),
    "ей": ("gen", "f"), "х": ("gen", "p"), "ых": ("gen", "p"), "их": ("gen", "p"), "ми": ("ins", "p"),
    "ыми": ("ins", "p"), "ими": ("ins", "p"),
}
_ORD_SUFFIX_ALT = "|".join(sorted(_ORD_SUFFIX_RU, key=len, reverse=True))


def _numbers_ru(text: str, ctx: _Ctx) -> str:
    """Russian digits -> words (see the module docstring for the supported notations)."""
    C = STEP_NUMBERS
    text = ctx.sub(C, re.compile(r"№\s?(\d+)"), lambda m: "номер " + nw.ru_cardinal(int(m.group(1))), text)
    text = ctx.sub(C, re.compile(r"§\s?(\d+)"), lambda m: "параграф " + nw.ru_cardinal(int(m.group(1))), text)

    # --- dates: 12.05.2020 / 2020-05-12
    def numeric_date(d: int, mo: int, y: int) -> Optional[str]:
        """Spoken numeric date, or ``None`` if the numbers are not a valid date."""
        if not (1 <= d <= 31 and 1 <= mo <= 12 and 1000 <= y <= 2999):
            return None
        return f"{nw.ru_ordinal(d, 'gen', 'n')} {_RU_MONTH_BY_NUM[mo]} {nw.ru_ordinal(y, 'gen', 'm')} года"

    def date_dmy(m: "re.Match[str]") -> str:
        """Regex callback: DD.MM.YYYY."""
        return numeric_date(int(m.group(1)), int(m.group(2)), int(m.group(3))) or m.group(0)

    def date_iso(m: "re.Match[str]") -> str:
        """Regex callback: YYYY-MM-DD."""
        return numeric_date(int(m.group(3)), int(m.group(2)), int(m.group(1))) or m.group(0)
    text = ctx.sub(C, re.compile(r"(?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{4})(?![\d])"), date_dmy, text)
    text = ctx.sub(C, re.compile(r"(?<![\d-])(\d{4})-(\d{2})-(\d{2})(?![\d])"), date_iso, text)
    months = "|".join(_RU_MONTHS)

    def day_month(m: "re.Match[str]") -> str:
        """Regex callback: "12 мая [2020 [года|г.]]"."""
        day, month, year = int(m.group(1)), m.group(2), m.group(3)
        if not 1 <= day <= 31:
            return m.group(0)
        out = f"{nw.ru_ordinal(day, 'gen', 'n')} {month}"
        if year:
            out += f" {nw.ru_ordinal(int(year), 'gen', 'm')} года"
        return out
    text = ctx.sub(C, re.compile(rf"(?<![\d.])(\d{{1,2}})\s+({months})(?:\s+(\d{{4}})(?:\s*(?:года|г\.|г\b))?)?", re.I), day_month, text)
    text = ctx.sub(C, re.compile(rf"\b({months})\s+(\d{{4}})(?:\s*(?:года|г\.|г\b))?", re.I),
                   lambda m: f"{m.group(1)} {nw.ru_ordinal(int(m.group(2)), 'gen', 'm')} года", text)

    # --- years with the word год: "в 1999 году", "с 1999 года", "1999 г."
    def year_word(m: "re.Match[str]") -> str:
        """Regex callback for "<year> <год-form>"."""
        n = int(m.group(1))
        word = m.group(2).lower().rstrip(".")
        prev = _prev_word(text_ref[0], m.start())
        if word == "год":
            case, out = "nom", "год"
        elif word == "года":
            case, out = "gen", "года"
        elif word in ("году", "годе"):
            case = "prep" if (prev in _PREPS_PREP or word == "годе") else "dat"
            out = "году"
        elif word == "годом":
            case, out = "ins", "годом"
        else:                                                       # "г." / "г"
            case = "prep" if prev in _PREPS_PREP else "gen"
            out = "году" if case == "prep" else "года"
        return f"{nw.ru_ordinal(n, case, 'm', year=True)} {out}"
    text_ref = [text]
    rx_year = re.compile(r"(?<![\d.,])(\d{3,4})\s*(года|году|годе|годом|год|г\.|г)(?![\w])")
    text = _sub_with_ref(rx_year, year_word, text, text_ref, ctx, C)

    # --- year ranges: "в 1941—1945 гг."
    def year_range(m: "re.Match[str]") -> str:
        """Regex callback for "<year> — <year> гг."."""
        a, b = int(m.group(1)), int(m.group(2))
        prep = _prev_word(text_ref[0], m.start()) in _PREPS_PREP
        case = "prep" if prep else "nom"
        return f"{nw.ru_ordinal(a, case)} \u2014 {nw.ru_ordinal(b, case)} {'годах' if prep else 'годы'}"
    text_ref[0] = text
    text = _sub_with_ref(re.compile(r"(?<![\d.,])(\d{4})\s*[\u2014\u2013-]\s*(\d{4})\s*(?:гг\.|гг|годы|годах)(?![\w])"), year_range, text, text_ref, ctx, C)

    # --- Roman century: "XIX век", "в XX в."
    def century(m: "re.Match[str]") -> str:
        """Regex callback for "<roman> век-form"."""
        n = roman_to_int(m.group(1))
        if not n:
            return m.group(0)
        word = m.group(2).lower()
        prev = _prev_word(text_ref[0], m.start())
        forms = {"век": ("nom", "век"), "века": ("gen", "века"), "веку": ("dat", "веку"), "веке": ("prep", "веке"),
                 "веком": ("ins", "веком")}
        if word in forms:
            case, out = forms[word]
        else:                                                       # "в."
            case, out = ("prep", "веке") if prev in _PREPS_PREP else ("gen", "века")
        return f"{nw.ru_ordinal(n, case, 'm')} {out}"
    text_ref[0] = text
    text = _sub_with_ref(re.compile(r"(?<![\w])([IVXLCDM]{1,7})\s*(веком|веке|веку|века|век|в\.)(?![\w])"), century, text, text_ref, ctx, C)

    # --- currency (symbol before / after the amount)
    scale_alt = "|".join(sorted(_SCALE_RU, key=len, reverse=True))
    cur_alt = r"₽|руб\.?|\$|€|£|долл\.?"
    text = ctx.sub(C, re.compile(rf"({_NUM})(?:\s*({scale_alt})\.?)?\s?({cur_alt})(?![\w])"),
                   lambda m: _money(ctx, _CUR_SYMBOLS[m.group(3)], m.group(1), m.group(2) or ""), text)
    text = ctx.sub(C, re.compile(rf"([$€£])\s?({_NUM})(?:\s*({scale_alt})\.?(?![\w]))?"),
                   lambda m: _money(ctx, _CUR_SYMBOLS[m.group(1)], m.group(2), m.group(3) or ""), text)

    # --- percent
    def percent(m: "re.Match[str]") -> str:
        """Regex callback for "5 %"."""
        ip, fp = _split_decimal(m.group(1))
        if fp:
            return nw.ru_decimal(ip, fp) + " процента"
        n = int(ip)
        return f"{nw.ru_cardinal(n)} {nw.ru_plural(n, ('процент', 'процента', 'процентов'))}"
    text = ctx.sub(C, re.compile(rf"({_NUM})\s?%"), percent, text)

    # --- "5 тыс." / "3 млн" scales
    text = ctx.sub(C, re.compile(rf"({_NUM})\s*(тыс|млн|млрд)\.?(?![\w])"),
                   lambda m: _scaled_ru(m.group(1), m.group(2)), text)

    # --- units
    def unit(m: "re.Match[str]") -> str:
        """Regex callback for "5 км"."""
        forms, gender = _UNITS_RU[m.group(2)]
        ip, fp = _split_decimal(m.group(1))
        if fp:
            return f"{nw.ru_decimal(ip, fp)} {forms[1]}"
        n = int(ip)
        return f"{nw.ru_cardinal(n, gender)} {nw.ru_plural(n, forms)}"
    text = ctx.sub(C, re.compile(rf"(?<![\w.,])({_NUM})\s?({_UNIT_ALT_RU})(?:\.(?=\s*\d))?(?![\w])"), unit, text)

    # --- ordinals with a suffix: "5-му", "1-го", "2-я"
    def ordinal_suffix(m: "re.Match[str]") -> str:
        """Regex callback for "<n>-<suffix>"."""
        n = int(m.group(1))
        case, gender = _ORD_SUFFIX_RU[m.group(2).lower()]
        if (case, gender) == ("nom", "n") and _next_word(text_ref[0], m.end()).endswith(("ы", "и")):
            gender = "p"                                            # "90-е годы"
        if case is None:
            case = "prep" if _prev_word(text_ref[0], m.start()) in _PREPS_PREP else "ins"
        return nw.ru_ordinal(n, case, gender)
    text_ref[0] = text
    text = _sub_with_ref(re.compile(rf"(?<![\w.,])(\d{{1,4}})[-\u2011]({_ORD_SUFFIX_ALT})(?![\w])", re.I),
                         ordinal_suffix, text, text_ref, ctx, C)

    # --- decimals and plain integers
    text = ctx.sub(C, re.compile(r"(?<![\w.,])(\d{1,3}(?:[ \u00a0]\d{3})*|\d+),(\d+)(?![\w]|[.,]\d)"),
                   lambda m: nw.ru_decimal(re.sub(r"\D", "", m.group(1)), m.group(2)), text)

    def integer(m: "re.Match[str]") -> str:
        """Regex callback for plain integers; the gender comes from the next word."""
        n = _int(m.group(0))
        return nw.ru_cardinal(n, _ru_gender_for(n, _next_word(text_ref[0], m.end())))
    text_ref[0] = text
    return _sub_with_ref(re.compile(r"(?<![\w.,])(?:\d{1,3}(?:[ \u00a0]\d{3})+|\d+)(?![\w]|[.,]\d)"), integer, text, text_ref, ctx, C)


def _scaled_ru(number: str, scale: str) -> str:
    """"5" + "тыс" -> "пять тысяч"; "2,5" + "млн" -> "две целых пять десятых миллиона"."""
    forms = _SCALE_RU[scale]
    ip, fp = _split_decimal(number)
    if fp:
        return f"{nw.ru_decimal(ip, fp)} {forms[1]}"
    n = int(ip)
    return f"{nw.ru_cardinal(n, 'f' if forms[0] == 'тысяча' else 'm')} {nw.ru_plural(n, forms)}"


def _sub_with_ref(rx: "re.Pattern[str]", fn, text: str, ref: List[str], ctx: _Ctx, key: str) -> str:
    """``rx.sub`` whose callback can look at the *unmodified* text through ``ref[0]`` (for the surrounding words)."""
    ref[0] = text
    text, n = rx.subn(fn, text)
    if n:
        ctx.counts[key] += n
    ref[0] = text
    return text


def _numbers_en(text: str, ctx: _Ctx) -> str:
    """English digits -> words."""
    C = STEP_NUMBERS
    text = ctx.sub(C, re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+)(?![\d])"), lambda m: m.group(1).replace(",", ""), text)
    text = ctx.sub(C, re.compile(r"\b[Nn]o\.\s?(\d+)"), lambda m: "number " + nw.en_cardinal(int(m.group(1))), text)
    text = ctx.sub(C, re.compile(r"#(\d+)"), lambda m: "number " + nw.en_cardinal(int(m.group(1))), text)
    text = ctx.sub(C, re.compile(r"§\s?(\d+)"), lambda m: "section " + nw.en_cardinal(int(m.group(1))), text)
    months = "|".join(_EN_MONTHS)

    def month_day(m: "re.Match[str]") -> str:
        """Regex callback: "May 12, 2020" / "May 12th"."""
        out = f"{m.group(1)} {nw.en_ordinal(int(m.group(2)))}"
        if m.group(3):
            out += ", " + nw.en_year(int(m.group(3)))
        return out
    text = ctx.sub(C, re.compile(rf"\b({months})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?(?![\d])"), month_day, text)
    text = ctx.sub(C, re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({months})(?:,?\s+(\d{{4}}))?(?![\d])"),
                   lambda m: f"the {nw.en_ordinal(int(m.group(1)))} of {m.group(2)}" + (" " + nw.en_year(int(m.group(3))) if m.group(3) else ""), text)
    text = ctx.sub(C, re.compile(rf"\b({months})\s+(\d{{4}})\b"), lambda m: f"{m.group(1)} {nw.en_year(int(m.group(2)))}", text)
    text = ctx.sub(C, re.compile(r"(?<![\d-])(\d{4})-(\d{2})-(\d{2})(?![\d])"),
                   lambda m: (f"{_EN_MONTHS[int(m.group(2)) - 1]} {nw.en_ordinal(int(m.group(3)))}, {nw.en_year(int(m.group(1)))}"
                              if 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31 else m.group(0)), text)
    # a year after a typical preposition
    text = ctx.sub(C, re.compile(r"\b(in|since|until|till|by|from|of|year|before|after|circa)\s+(1[1-9]\d\d|20\d\d)\b(?![,.]\d)(?!\s?(?:%|km|kg|cm|mm|m\b))", re.I),
                   lambda m: f"{m.group(1)} {nw.en_year(int(m.group(2)))}", text)
    text = ctx.sub(C, re.compile(r"\b(1[1-9]\d\d|20\d\d)s\b"), lambda m: _plural_year_en(nw.en_year(int(m.group(1)))), text)

    scale_alt = "|".join(_SCALE_EN)
    text = ctx.sub(C, re.compile(rf"([$€£])\s?({_NUM})(?:\s+({scale_alt}))?"), lambda m: _money(ctx, _CUR_SYMBOLS[m.group(1)], m.group(2).replace(",", ""), m.group(3) or ""), text)
    text = ctx.sub(C, re.compile(rf"({_NUM})\s?(USD|EUR|GBP)\b"), lambda m: _money(ctx, m.group(2), m.group(1)), text)
    text = ctx.sub(C, re.compile(rf"({_NUM})\s?%"), lambda m: (nw.en_decimal(*_split_decimal(m.group(1)))) + " percent", text)

    def unit(m: "re.Match[str]") -> str:
        """Regex callback for "5 km"."""
        one, many = _UNITS_EN[m.group(2)]
        ip, fp = _split_decimal(m.group(1))
        return f"{nw.en_decimal(ip, fp)} {one if (int(ip) == 1 and not fp) else many}"
    text = ctx.sub(C, re.compile(rf"(?<![\w.,])({_NUM})\s?({_UNIT_ALT_EN})(?![\w])"), unit, text)
    text = ctx.sub(C, re.compile(r"(?<![\w.,])(\d+)(?:st|nd|rd|th)\b"), lambda m: nw.en_ordinal(int(m.group(1))), text)
    text = ctx.sub(C, re.compile(r"(?<![\w.,])(\d+)\.(\d+)(?![\w]|[.,]\d)"), lambda m: nw.en_decimal(m.group(1), m.group(2)), text)
    return ctx.sub(C, re.compile(r"(?<![\w.,])\d+(?![\w]|[.,]\d)"), lambda m: nw.en_cardinal(int(m.group(0))), text)


def _plural_year_en(words: str) -> str:
    """"nineteen ninety" -> "nineteen nineties" (decades)."""
    return words[:-1] + "ies" if words.endswith("y") else words + "s"


def step_numbers(text: str, ctx: _Ctx) -> str:
    """Spell out numbers, dates, years, ordinals, percents, currency and units (ru / en only)."""
    if ctx.lang == "ru":
        return _numbers_ru(text, ctx)
    if ctx.lang == "en":
        return _numbers_en(text, ctx)
    return text


# =========================================================================== abbreviations

_RU_ABBR: List[Tuple[str, str, bool]] = [        # (regex, replacement, may end a sentence)
    (r"\bт\.\s?е\.", "то есть", False), (r"\bт\.\s?к\.", "так как", False), (r"\bт\.\s?н\.", "так называемый", False),
    (r"\bт\.\s?ч\.", "том числе", False), (r"\bт\.\s?о\.", "таким образом", False),
    (r"\bт\.\s?д\.", "так далее", True), (r"\bт\.\s?п\.", "тому подобное", True),
    (r"\bи\s+др\.", "и другие", True), (r"\bи\s+пр\.", "и прочее", True), (r"\bи\s+т\.\s?д\.", "и так далее", True),
    (r"\bи\s+т\.\s?п\.", "и тому подобное", True), (r"\bи\s+т\.\s?ч\.", "и так далее", True),
    (r"\bн\.\s?э\.", "нашей эры", True), (r"\bдо\s+н\.\s?э\.", "до нашей эры", True),
    (r"\bим\.(?=\s)", "имени", False), (r"\bг-н\b", "господин", False), (r"\bг-жа\b", "госпожа", False),
    (r"\bтов\.(?=\s)", "товарищ", False), (r"\bпроф\.(?=\s)", "профессор", False), (r"\bакад\.(?=\s)", "академик", False),
    (r"\bд-р\b", "доктор", False), (r"\bнапр\.(?=\s)", "например", False), (r"\bсм\.(?=\s)", "смотри", False),
    (r"\bср\.(?=\s)", "сравни", False), (r"\bрис\.(?=\s)", "рисунок", False), (r"\bстр\.(?=\s)", "страница", False),
    (r"\bг\.(?=\s*[А-ЯЁ])", "город", False), (r"\bул\.(?=\s*[А-ЯЁ])", "улица", False), (r"\bпос\.(?=\s*[А-ЯЁ])", "посёлок", False),
    (r"\bтыс\.", "тысяч", False), (r"\bмлн\.?(?![\w])", "миллионов", False), (r"\bмлрд\.?(?![\w])", "миллиардов", False),
    (r"\bруб\.", "рублей", False), (r"\bкоп\.", "копеек", False), (r"\bгг\.", "годов", False),
]
_EN_ABBR: List[Tuple[str, str, bool]] = [
    (r"\bMr\.", "Mister", False), (r"\bMrs\.", "Missus", False), (r"\bMs\.", "Miss", False), (r"\bDr\.", "Doctor", False),
    (r"\bProf\.", "Professor", False), (r"\bJr\.", "Junior", False), (r"\bSr\.", "Senior", False),
    (r"\bCapt\.", "Captain", False), (r"\bGen\.", "General", False), (r"\bCol\.", "Colonel", False),
    (r"\bSgt\.", "Sergeant", False), (r"\bLt\.", "Lieutenant", False), (r"\bMt\.", "Mount", False),
    (r"\bSt\.(?=\s+[A-Z])", "Saint", False), (r"\bvs\.", "versus", False), (r"\be\.g\.", "for example", False),
    (r"\bi\.e\.", "that is", False), (r"\bapprox\.", "approximately", False), (r"\betc\.", "et cetera", True),
    (r"\bet al\.", "and others", True), (r"\bp\.m\.", "p m", False), (r"\ba\.m\.", "a m", False),
]
_RU_ABBR_RX = [(re.compile(p, re.I if not p.startswith(r"\bг-") else 0), r, e) for p, r, e in _RU_ABBR]
_EN_ABBR_RX = [(re.compile(p), r, e) for p, r, e in _EN_ABBR]


def step_abbrev(text: str, ctx: _Ctx) -> str:
    """Expand common abbreviations; a full stop that really ends the sentence is kept."""
    table = _RU_ABBR_RX if ctx.lang == "ru" else _EN_ABBR_RX if ctx.lang == "en" else []
    for rx, rep, may_end in table:
        if may_end:
            def fn(m: "re.Match[str]", rep=rep) -> str:
                """Keep the period if the abbreviation ends a sentence."""
                after = m.string[m.end():m.end() + 3]
                ends = (not after.strip()) or bool(re.match(r"\s+[A-ZА-ЯЁ\"«—]", after))
                return rep + ("." if ends else "")
            text = ctx.sub(STEP_ABBREV, rx, fn, text)
        else:
            text = ctx.sub(STEP_ABBREV, rx, rep, text)
    if ctx.lang == "en":
        text = ctx.sub(STEP_ABBREV, re.compile(r"(?<=\s)&(?=\s)"), "and", text)
    elif ctx.lang == "ru":
        text = ctx.sub(STEP_ABBREV, re.compile(r"(?<=\s)&(?=\s)"), "и", text)
    return text


# =========================================================================== headings

_HEADING_WORDS_RU = {"глава": "f", "часть": "f", "книга": "f", "песнь": "f", "сцена": "f", "раздел": "m", "том": "m",
                     "акт": "m", "урок": "m", "день": "m", "действие": "n", "явление": "n"}
_HEADING_WORDS_EN = {"chapter", "part", "book", "section", "volume", "act", "scene", "canto", "lesson", "day"}
_HEADING_RX = re.compile(r"^\s*([^\W\d_]+)\s+([IVXLCDM]+|\d+)\s*([.:)\u2014-]?)\s*(.*)$", re.I | re.S)
_BARE_ROMAN = re.compile(r"^\s*([IVXLCDM]+)\s*[.)]?\s*$")


def _sentence_case(s: str) -> str:
    """ALL CAPS heading -> "All caps heading" (only when it really is all caps and long enough to not be an acronym)."""
    letters = [c for c in s if c.isalpha()]
    if (len(letters) >= 5 or (len(letters) >= 4 and " " in s)) and s.upper() == s and s.lower() != s:
        low = s.lower()
        for i, ch in enumerate(low):
            if ch.isalpha():
                return low[:i] + ch.upper() + low[i + 1:]
    return s


def prepare_heading(line: str, ctx: _Ctx) -> str:
    """"ГЛАВА XII." -> "Глава двенадцатая"; "II" -> "два"; ALL CAPS -> sentence case (ru / en)."""
    s = line.strip()
    if not s:
        return s
    m = _HEADING_RX.match(s)
    if m and ctx.lang in ("ru", "en"):
        word, num, _sep, rest = m.groups()
        wl = word.lower()
        is_roman = bool(roman_to_int(num)) and not num.isdigit()
        value = int(num) if num.isdigit() else roman_to_int(num)
        if ctx.lang == "ru" and wl in _HEADING_WORDS_RU and value:
            ctx.counts[STEP_HEADINGS] += 1
            head = f"{_sentence_case(word)} {nw.ru_ordinal(value, 'nom', _HEADING_WORDS_RU[wl])}"
            return head + (". " + _sentence_case(rest.strip()) if rest.strip() else "")
        if ctx.lang == "en" and wl in _HEADING_WORDS_EN and is_roman and value:
            ctx.counts[STEP_HEADINGS] += 1
            head = f"{_sentence_case(word)} {nw.en_cardinal(value)}"
            return head + (". " + _sentence_case(rest.strip()) if rest.strip() else "")
    m = _BARE_ROMAN.match(s)
    if m and ctx.lang in ("ru", "en") and roman_to_int(m.group(1)) and len(m.group(1)) > 1:
        ctx.counts[STEP_HEADINGS] += 1
        return nw.cardinal(roman_to_int(m.group(1)), ctx.lang)
    if m and ctx.lang in ("ru", "en") and roman_to_int(m.group(1)):
        ctx.counts[STEP_HEADINGS] += 1
        return nw.cardinal(roman_to_int(m.group(1)), ctx.lang)
    fixed = _sentence_case(s)
    if fixed != s:
        ctx.counts[STEP_HEADINGS] += 1
    return fixed


def step_headings(text: str, ctx: _Ctx) -> str:
    """Heading-like paragraphs of the body: "ГЛАВА II", a bare "XIV", an ALL-CAPS line."""
    out: List[str] = []
    for p in text.split("\n\n"):
        s = p.strip()
        if s and "\n" not in s and len(s) <= 80 and not s.endswith(tuple(".!?…»”\"")) or _BARE_ROMAN.match(s or ""):
            heading_like = bool(_HEADING_RX.match(s)) or bool(_BARE_ROMAN.match(s)) or (s.upper() == s and s.lower() != s)
            if heading_like:
                out.append(prepare_heading(s, ctx))
                continue
        out.append(p)
    return "\n\n".join(out)


# =========================================================================== public API

_STEP_FUNCS: Dict[str, Callable[[str, _Ctx], str]] = {
    STEP_LAYOUT: step_layout, STEP_NOISE: step_noise, STEP_QUOTES: step_quotes, STEP_LINKS: step_links,
    STEP_HEADINGS: step_headings, STEP_NUMBERS: step_numbers, STEP_ABBREV: step_abbrev,
}


def resolve_language(book: Optional[Book] = None, hint: str = "", sample: str = "") -> str:
    """Language code (``ru`` / ``en`` / ``de`` / ``""``) of a text: the book's own tag, else the script of a sample, else ``hint``."""
    from core.text_utils import detect_language

    code = nw.lang_code(book.language.split("-")[0] if book and book.language else "")
    if code:
        return code
    text = sample or (" ".join(c.text[:2000] for c in book.chapters[:3]) if book else "")
    if text.strip():
        return nw.lang_code(detect_language(text)) or nw.lang_code(hint)
    return nw.lang_code(hint)


def prepare_text_block(text: str, lang: str, options: Optional[PrepOptions] = None, counts: Optional[Counter] = None) -> str:
    """Run the selected rule-based steps over a text (paragraphs separated by a blank line)."""
    options = options or PrepOptions()
    ctx = _Ctx(lang, counts)
    for key in _ORDER:
        if key not in options.steps:
            continue
        if key in LANGUAGE_STEPS and lang not in ("ru", "en"):
            continue
        text = _STEP_FUNCS[key](text, ctx)
    return text


def prepare_title(title: str, lang: str, options: Optional[PrepOptions] = None, counts: Optional[Counter] = None) -> str:
    """Prepare a chapter title (single line): headings first, then the other steps, trailing dots removed."""
    options = options or PrepOptions()
    ctx = _Ctx(lang, counts)
    text = re.sub(r"\s+", " ", title or "").strip()
    if STEP_LAYOUT in options.steps:
        text = step_layout(text, ctx)
    if STEP_HEADINGS in options.steps and lang in ("ru", "en"):
        text = prepare_heading(text, ctx)
    rest = PrepOptions(frozenset(options.steps - {STEP_LAYOUT, STEP_HEADINGS, STEP_NOISE}))
    text = prepare_text_block(text, lang, rest, ctx.counts)
    return text.rstrip(" .:;,") if text.rstrip(" .:;,") else text


def prepare_book(book: Book, options: Optional[PrepOptions] = None, language_hint: str = "") -> Tuple[Book, PrepReport]:
    """Return a prepared copy of ``book`` and a :class:`PrepReport`.  The original book is not modified."""
    options = options or PrepOptions()
    lang = resolve_language(book, language_hint)
    counts: Counter = Counter()
    chapters: List[Chapter] = []
    for ch in book.chapters:
        chapters.append(Chapter(prepare_title(ch.title, lang, options, counts) if ch.title else ch.title,
                                prepare_text_block(ch.text, lang, options, counts)))
    report = PrepReport(lang, [k for k in _ORDER if k in options.steps], dict(counts),
                        [k for k in _ORDER if k in options.steps and k in LANGUAGE_STEPS and lang not in ("ru", "en")])
    prepared = Book(book.title, book.author, book.language, chapters, book.cover, book.cover_ext)
    return prepared, report
