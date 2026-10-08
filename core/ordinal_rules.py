"""Data for :mod:`core.ordinals`: which words make the number next to them an ordinal, per narration language.

Pure data, no logic - extend a language by adding words here (and a test).  Format per language code:

* ``after`` - nouns followed by the number ("глава 2" -> "глава вторая").  ``{lemma: (gender, {form: case})}``; the number
  takes the gender and case of the form found in the text.  Genders: ``m`` / ``f`` / ``n`` (``p`` = plural); cases:
  ``nom gen dat acc ins prep``.  A form that is ambiguous (Russian "главе" = dative or prepositional) lists one case whose
  ordinal ending is the same; an inanimate nominative/accusative form lists ``nom`` (headings are the usual case).
* ``after_reading`` - ``"ordinal"`` (Russian: "глава вторая") or ``"cardinal"`` (English "Chapter 2" is read "chapter two",
  so only a Roman numeral there is turned into words: "Chapter IV" -> "Chapter four").
* ``before`` - nouns preceded by the number with a dot (German "3. Kapitel" -> "drittes Kapitel"): ``{noun: gender}``.
* ``suffix`` - numbers with a written ordinal ending ("1-го", "21st"): ``{ending: (gender, case)}``; ``None`` = take
  gender and case from the noun that follows (when it is listed under ``after``) or the default in ``suffix_default``.
* ``prepositional`` - Russian words before "N-м" that make it prepositional ("в 3-м томе"); otherwise instrumental.

Only the languages a Voxprint voice can narrate need an entry (Qwen3-TTS: Russian, English, German ...); a language without
an entry passes through unchanged.  Docs: docs/ORDINALS.md.
"""
from __future__ import annotations

from typing import Dict, Tuple


def _ru_m(nom: str, gen: str, dat: str, ins: str, prep: str) -> Tuple[str, Dict[str, str]]:
    return "m", {nom: "nom", gen: "gen", dat: "dat", ins: "ins", prep: "prep"}


def _ru_f(nom: str, gen: str, acc: str, ins: str, dat_prep: str = "") -> Tuple[str, Dict[str, str]]:
    # feminine ordinals have one ending ("-ой" / "-ей") for genitive, dative, instrumental and prepositional
    forms: Dict[str, str] = {}
    for form, case in ((nom, "nom"), (gen, "gen"), (acc, "acc"), (ins, "ins"), (dat_prep, "dat")):
        if form:
            forms.setdefault(form, case)          # "часть" (nominative = accusative) stays nominative
    return "f", forms


def _ru_n(nom: str, gen: str, dat: str, ins: str, prep: str) -> Tuple[str, Dict[str, str]]:
    return "n", {nom: "nom", gen: "gen", dat: "dat", ins: "ins", prep: "prep"}


RU_AFTER: Dict[str, Tuple[str, Dict[str, str]]] = {
    # time and structure of a book / scripture
    "день": _ru_m("день", "дня", "дню", "днём", "дне"),
    "стих": _ru_m("стих", "стиха", "стиху", "стихом", "стихе"),
    "том": _ru_m("том", "тома", "тому", "томом", "томе"),
    "раздел": _ru_m("раздел", "раздела", "разделу", "разделом", "разделе"),
    "параграф": _ru_m("параграф", "параграфа", "параграфу", "параграфом", "параграфе"),
    "псалом": _ru_m("псалом", "псалма", "псалму", "псалмом", "псалме"),
    "урок": _ru_m("урок", "урока", "уроку", "уроком", "уроке"),
    "эпизод": _ru_m("эпизод", "эпизода", "эпизоду", "эпизодом", "эпизоде"),
    "сезон": _ru_m("сезон", "сезона", "сезону", "сезоном", "сезоне"),
    "акт": _ru_m("акт", "акта", "акту", "актом", "акте"),
    "этап": _ru_m("этап", "этапа", "этапу", "этапом", "этапе"),
    "шаг": _ru_m("шаг", "шага", "шагу", "шагом", "шаге"),
    "пункт": _ru_m("пункт", "пункта", "пункту", "пунктом", "пункте"),
    "глава": _ru_f("глава", "главы", "главу", "главой", "главе"),
    "часть": _ru_f("часть", "части", "часть", "частью"),
    "книга": _ru_f("книга", "книги", "книгу", "книгой", "книге"),
    "серия": _ru_f("серия", "серии", "серию", "серией"),
    "сцена": _ru_f("сцена", "сцены", "сцену", "сценой", "сцене"),
    "песнь": _ru_f("песнь", "песни", "песнь", "песнью"),
    "неделя": _ru_f("неделя", "недели", "неделю", "неделей", "неделе"),
    "лекция": _ru_f("лекция", "лекции", "лекцию", "лекцией"),
    "статья": _ru_f("статья", "статьи", "статью", "статьёй", "статье"),
    "страница": _ru_f("страница", "страницы", "страницу", "страницей", "странице"),
    "явление": _ru_n("явление", "явления", "явлению", "явлением", "явлении"),
    "действие": _ru_n("действие", "действия", "действию", "действием", "действии"),
    "занятие": _ru_n("занятие", "занятия", "занятию", "занятием", "занятии"),
}

#: Written Russian ordinal endings: "1-й", "2-я", "3-го", "90-х" ...  ``None`` = decided by the next word.
RU_SUFFIX: Dict[str, object] = {
    "й": None, "ый": ("m", "nom"), "ий": ("m", "nom"), "ой": None,
    "я": ("f", "nom"), "ая": ("f", "nom"), "ья": ("f", "nom"),
    "е": ("n", "nom"), "ое": ("n", "nom"), "ье": ("n", "nom"), "ё": ("n", "nom"),
    "го": ("m", "gen"), "ого": ("m", "gen"), "его": ("m", "gen"),
    "му": ("m", "dat"), "ому": ("m", "dat"), "ему": ("m", "dat"),
    "м": None, "ом": ("m", "prep"), "ым": ("m", "ins"),
    "ю": ("f", "acc"), "ую": ("f", "acc"),
    "х": ("p", "gen"), "ых": ("p", "gen"), "ми": ("p", "ins"), "ыми": ("p", "ins"),
}
#: Gender/case when a ``None`` ending is not followed by a listed noun ("-й" -> masculine nominative, "-м" -> see below).
RU_SUFFIX_DEFAULT: Dict[str, Tuple[str, str]] = {"й": ("m", "nom"), "ой": ("m", "nom"), "м": ("m", "ins")}
RU_PREPOSITIONAL = ("в", "во", "о", "об", "обо", "на", "при")

EN_AFTER_ROMAN = ("chapter", "part", "book", "volume", "act", "scene", "psalm", "canto", "section", "lesson", "day")

DE_BEFORE: Dict[str, str] = {
    "kapitel": "n", "teil": "m", "buch": "n", "band": "m", "tag": "m", "abschnitt": "m", "akt": "m", "aufzug": "m",
    "szene": "f", "auftritt": "m", "psalm": "m", "vers": "m", "woche": "f", "lektion": "f", "stunde": "f", "mal": "n",
    "januar": "m", "februar": "m", "märz": "m", "april": "m", "mai": "m", "juni": "m", "juli": "m", "august": "m",
    "september": "m", "oktober": "m", "november": "m", "dezember": "m", "jahrhundert": "n", "platz": "m",
}

RULES: Dict[str, Dict[str, object]] = {
    "ru": {"after": RU_AFTER, "after_reading": "ordinal", "suffix": RU_SUFFIX, "suffix_default": RU_SUFFIX_DEFAULT,
           "prepositional": RU_PREPOSITIONAL},
    "en": {"after": {w: ("m", {w: "nom"}) for w in EN_AFTER_ROMAN}, "after_reading": "cardinal",
           "suffix": {"st": ("m", "nom"), "nd": ("m", "nom"), "rd": ("m", "nom"), "th": ("m", "nom")}},
    "de": {"before": DE_BEFORE},
}
