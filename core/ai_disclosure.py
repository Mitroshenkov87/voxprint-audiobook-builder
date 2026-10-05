"""Spoken AI disclosure at the start of an audiobook (opt-in, default OFF).

When switched on, the first thing the listener hears is e.g. "Эта книга озвучена с помощью искусственного интеллекта,
голос Анна, октябрь две тысячи двадцать шестого года." in the narrated language (ru / en / de; others fall back to English),
synthesized with the chosen voice.  Only month and year are said - no exact date.  It helps to meet AI-generated content
labelling duties (EU AI Act, Art. 50).  "ИИ"/"AI" is spelled out because a TTS model may read the abbreviation letter by
letter; numbers are written in words so the phrase does not depend on the number normalizer.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
from typing import List, Optional

from core import num_words as nw
from core.chunker import Chunk

PAUSE_AFTER_MS = 1200          # silence between the disclosure and the book

_MONTHS = {
    "ru": ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь",
           "декабрь"],
    "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"],
    "de": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November",
           "Dezember"],
}
_DE_UNITS = ["", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn", "elf", "zwölf",
             "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn"]
_DE_TENS = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"]


def _de_below_100(n: int) -> str:
    if n < 20:
        return _DE_UNITS[n]
    t, u = divmod(n, 10)
    return (("ein" if u == 1 else _DE_UNITS[u]) + "und" if u else "") + _DE_TENS[t]


def de_year(n: int) -> str:
    """German year in words: 2026 "zweitausendsechsundzwanzig", 1999 "neunzehnhundertneunundneunzig" (digits outside 1100-2999)."""
    if 2000 <= n <= 2999:
        hi, lo = divmod(n, 1000)
        rest, lo = divmod(lo, 100)
        return ("zwei" if hi == 2 else "") + "tausend" + (("ein" if rest == 1 else _de_below_100(rest)) + "hundert" if rest else "") + _de_below_100(lo)
    if 1100 <= n <= 1999:
        hi, lo = divmod(n, 100)
        return _de_below_100(hi) + "hundert" + _de_below_100(lo)
    return str(n)


def phrase(language: str, voice: str, when: Optional[_dt.date] = None) -> str:
    """The disclosure sentence in ``language`` (name or code) for the voice ``voice``, dated by month and year of ``when``."""
    when = when or _dt.date.today()
    lang = nw.lang_code(language)
    lang = lang if lang in _MONTHS else "en"
    month = _MONTHS[lang][when.month - 1]
    voice = (voice or "").strip()
    if lang == "ru":
        year = nw.ru_ordinal(when.year, "gen", year=True) + " года"
        who = f", голос {voice}" if voice else ""
        return f"Эта книга озвучена с помощью искусственного интеллекта{who}, {month} {year}."
    if lang == "de":
        who = f", Stimme {voice}" if voice else ""
        return f"Dieses Buch wurde mit Hilfe künstlicher Intelligenz vertont{who}, {month} {de_year(when.year)}."
    who = f", voice {voice}" if voice else ""
    return f"This book was narrated with the help of artificial intelligence{who}, {month} {nw.en_year(when.year)}."


def prepend(chunks: List[Chunk], text: str) -> List[Chunk]:
    """``chunks`` with the disclosure as the very first chunk (same chapter as the first one); indexes renumbered."""
    if not chunks or not text:
        return list(chunks)
    head = Chunk(0, chunks[0].chapter, text, PAUSE_AFTER_MS, "")
    return [head] + [dataclasses.replace(c, index=i + 1) for i, c in enumerate(chunks)]


# The switch is remembered like the pause switch (a tiny file in the state folder); never chosen = OFF.
def _file():
    from infra import paths

    return paths.state_dir() / "ai_disclosure_enabled.txt"


def load_enabled() -> bool:
    """Whether the user switched the spoken disclosure on (False when never chosen)."""
    try:
        return _file().read_text(encoding="utf-8").strip() == "1"
    except OSError:
        return False


def save_enabled(on: bool) -> None:
    """Remember the switch (a failure to write is not an error)."""
    try:
        p = _file()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("1" if on else "0", encoding="utf-8")
    except OSError:
        pass
