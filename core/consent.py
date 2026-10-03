"""Voice-owner consent: the spoken statement at the end of a recording, its automatic reading and the usage scope.

The speaker records one of three template sentences (see ``docs/voice-script-ru-v2.txt``) with their name and the date.
After the recording has been recognised, :func:`parse_statement` reads the *scope* from the words (rule based, Russian,
English and German) and falls back to the most restrictive level whenever the statement is unclear; the UI shows the
detected result for one-click confirmation or change.  The scope is a mark in ``voice.json`` (``consent`` block) and
maps onto the existing licence fields:

* ``commercial``           - output may be sold (``CC-BY-4.0``, commercial use allowed);
* ``public_noncommercial`` - output may be published for free but not sold (``CC-BY-NC-4.0``);
* ``private_only``         - the voice may narrate, but the results must stay on the user's computer (``custom/personal-only``).

This is a practical record of what the speaker said, not legal advice.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as _date
from typing import Any, Dict, Optional

from core import spoken_date

COMMERCIAL, PUBLIC_NC, PRIVATE = "commercial", "public_noncommercial", "private_only"
SCOPES = (COMMERCIAL, PUBLIC_NC, PRIVATE)
DEFAULT_SCOPE = PRIVATE                       # the most restrictive level wins whenever the statement is unclear
SCOPE_LICENSE = {COMMERCIAL: "CC-BY-4.0", PUBLIC_NC: "CC-BY-NC-4.0", PRIVATE: "custom/personal-only"}
CLIP_NAME = "consent_statement.wav"
METHODS = ("spoken", "spoken_confirmed", "manual", "none")

#: Template sentences the speaker can read (``{name}`` and ``{date}`` are filled in by the speaker); shown in the script and in docs.
TEMPLATES = {
    "ru": {
        COMMERCIAL: "Я, {name}, сегодня, {date}, разрешаю использовать эту запись и характеристики моего голоса для обучения нейросетей и синтеза речи, в том числе в коммерческих целях: озвучки можно продавать.",
        PUBLIC_NC: "Я, {name}, сегодня, {date}, разрешаю использовать эту запись и характеристики моего голоса для обучения нейросетей и синтеза речи в некоммерческих целях: озвучки можно публиковать в открытом доступе, но нельзя продавать.",
        PRIVATE: "Я, {name}, сегодня, {date}, разрешаю использовать эту запись и характеристики моего голоса для обучения нейросетей и синтеза речи только в личных целях: озвучки нельзя публиковать, они должны оставаться на компьютере пользователя.",
    },
    "en": {
        COMMERCIAL: "I, {name}, today, {date}, give permission to use this recording and the characteristics of my voice for training neural networks and for speech synthesis, including commercial purposes: the audio may be sold.",
        PUBLIC_NC: "I, {name}, today, {date}, give permission to use this recording and the characteristics of my voice for training neural networks and for speech synthesis for non-commercial purposes: the audio may be published freely but may not be sold.",
        PRIVATE: "I, {name}, today, {date}, give permission to use this recording and the characteristics of my voice for training neural networks and for speech synthesis for personal use only: the audio must not be published and must stay on the user's computer.",
    },
    "de": {
        COMMERCIAL: "Ich, {name}, erlaube heute, am {date}, diese Aufnahme und die Merkmale meiner Stimme für das Training neuronaler Netze und die Sprachsynthese zu verwenden, auch für kommerzielle Zwecke: die Hörbücher dürfen verkauft werden.",
        PUBLIC_NC: "Ich, {name}, erlaube heute, am {date}, diese Aufnahme und die Merkmale meiner Stimme für das Training neuronaler Netze und die Sprachsynthese zu verwenden, für nichtkommerzielle Zwecke: die Hörbücher dürfen frei veröffentlicht, aber nicht verkauft werden.",
        PRIVATE: "Ich, {name}, erlaube heute, am {date}, diese Aufnahme und die Merkmale meiner Stimme für das Training neuronaler Netze und die Sprachsynthese zu verwenden, nur für den persönlichen Gebrauch: die Hörbücher dürfen nicht veröffentlicht werden und bleiben auf dem Computer des Nutzers.",
    },
}

_PERMISSION = ("разреш", "даю согласие", "согласен", "согласна", "permission", "permit", "allow", "consent", "grant",
               "erlaub", "gestatt", "genehmig", "stimme zu", "einverstanden")
_PRIVATE = ("только в личных", "личных целях", "только для личного", "нельзя публиковать", "не публиковать", "не должны публиковаться",
            "оставаться на компьютере", "personal use only", "personal use", "private use", "must not be published",
            "not be published", "must stay on", "nur für den persönlichen", "nur für persönliche", "nur privat",
            "nicht veröffentlicht werden", "nicht veroffentlicht werden", "bleiben auf dem computer")
_PUBLIC = ("некоммерческ", "не коммерческ", "в открытом доступе", "нельзя продавать", "нельзя использовать в коммерческ",
           "бесплатно публиковать", "non-commercial", "noncommercial", "non commercial", "not be sold", "may not be sold",
           "not for sale", "nicht kommerziell", "nichtkommerziell", "nicht verkauft", "nicht kommerzielle", "nichtkommerzielle",
           "frei veröffentlicht", "frei veroffentlicht")
#: Commercial markers; a negation right before the word ("не-", "non-", "nicht") makes it NON-commercial, so those are excluded here.
_COMMERCIAL_RE = re.compile(
    r"(?<!не)(?<!не )(?<!некоммерческ)коммерческ|можно продавать|могут продаваться|"
    r"(?<!non-)(?<!non )(?<!non)commercial|may be sold|can be sold|"
    r"(?<!nicht )(?<!nicht)kommerzi|(?<!nicht )verkauft werden|verkaufen")

_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7, "августа": 8, "сентября": 9,
    "октября": 10, "ноября": 11, "декабря": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
    "januar": 1, "februar": 2, "märz": 3, "marz": 3, "mai": 5, "juni": 6, "juli": 7, "oktober": 10, "dezember": 12,
}


@dataclass
class Parsed:
    """What was read from the statement."""
    scope: str = DEFAULT_SCOPE
    name: str = ""
    date: str = ""            # ISO date if one was recognised in the statement, else ""
    confident: bool = False   # True only for an unambiguous scope together with a permission phrase
    text: str = ""            # the recognised statement


def _norm(text: str) -> str:
    t = text.lower().replace("\u0451", "\u0435")
    return re.sub(r"\s+", " ", t)


def _hits(t: str, words) -> bool:
    return any(w.replace("\u0451", "\u0435") in t for w in words)


def detect_scope(text: str) -> tuple:
    """(scope, confident).  The most restrictive level found wins; confident = exactly one level and a permission phrase."""
    t = _norm(text)
    priv, pub, com = _hits(t, _PRIVATE), _hits(t, _PUBLIC), bool(_COMMERCIAL_RE.search(t))
    found = [s for s, ok in ((PRIVATE, priv), (PUBLIC_NC, pub), (COMMERCIAL, com)) if ok]
    if len(found) == 1:
        return found[0], _hits(t, _PERMISSION)
    return (found[0] if found else DEFAULT_SCOPE), False     # none or contradictory: restrictive, ask the user to confirm


def _parse_date(t: str) -> str:
    m = re.search(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b", t)
    if m:
        d, mo, y = (int(g) for g in m.groups())
    else:
        m = re.search(r"\b(\d{1,2})\s*(?:\.|st|nd|rd|th|го)?\s+(?:of\s+)?([a-zа-яäöü]+)\s+(\d{4})\b", t)
        if not m or m.group(2) not in _MONTHS:
            m2 = re.search(r"\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", t)
            if not m2 or m2.group(1) not in _MONTHS:
                return spoken_date.find_date(t)     # the recogniser wrote the date in words

            mo, d, y = _MONTHS[m2.group(1)], int(m2.group(2)), int(m2.group(3))
        else:
            d, mo, y = int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3))
    try:
        return _date(y, mo, d).isoformat()
    except ValueError:
        return ""


#: Words that end the name in "I, <name>, today ..." (recognisers often drop the commas, so the end is found by these words).
_NAME_END = (r"(?:сегодня|heute|today|am\b|разрешаю|даю|согласен|согласна|erlaube|gestatte|give|grant|hereby|hiermit|permit|\d)")


def _parse_name(text: str) -> str:
    """The speaker's name from "I, <name>, today, <date>, ..." with or without commas (1-4 words); "" if not found."""
    t = re.sub(r"\s+", " ", text.strip())
    m = re.search(r"(?:^|[.!?]\s|\s)(?:я|ich|i)[\s,]+((?:[^\s,.]+[\s,]+){0,3}?[^\s,.]+)[\s,.]+" + _NAME_END, t, re.I)
    name = m.group(1).strip(" ,.") if m else ""
    if name and name == name.lower():
        name = name.title()
    return name if 2 <= len(name) <= 60 and not re.search(r"\d", name) else ""


def parse_statement(text: str) -> Parsed:
    """Read scope, name and date from the recognised statement (see the module docstring)."""
    scope, confident = detect_scope(text)
    return Parsed(scope=scope, name=_parse_name(text), date=_parse_date(_norm(text)), confident=confident, text=text.strip())


def build_consent(parsed: Optional[Parsed], *, scope: str, name: str = "", method: str, recorded: bool, confirmed: bool,
                  today: Optional[str] = None, clip: str = "") -> Dict[str, Any]:
    """The ``consent`` block of ``voice.json``."""
    p = parsed or Parsed()
    return {
        "scope": scope if scope in SCOPES else DEFAULT_SCOPE,
        "name": (name or p.name).strip(),
        "date": p.date or today or _date.today().isoformat(),
        "date_from_statement": bool(p.date),
        "recorded_statement": bool(recorded),
        "method": method if method in METHODS else "none",
        "confirmed": bool(confirmed),
        "statement": p.text[:1500],
        "clip": clip,
    }


def clean_consent(data: Any) -> Optional[Dict[str, Any]]:
    """Sanitised ``consent`` block from a (possibly hand-edited) voice.json; None when absent or not a dict."""
    if not isinstance(data, dict) or not data:
        return None
    scope = str(data.get("scope") or DEFAULT_SCOPE)
    method = str(data.get("method") or "none")
    return {
        "scope": scope if scope in SCOPES else DEFAULT_SCOPE,
        "name": re.sub(r"\s+", " ", str(data.get("name") or "")).strip()[:80],
        "date": str(data.get("date") or "")[:10],
        "date_from_statement": bool(data.get("date_from_statement", False)),
        "recorded_statement": bool(data.get("recorded_statement", False)),
        "method": method if method in METHODS else "none",
        "confirmed": bool(data.get("confirmed", False)),
        "statement": str(data.get("statement") or "")[:1500],
        "clip": str(data.get("clip") or "")[:80],
    }


def scope_of(info: Dict[str, Any]) -> str:
    """Scope of a voice: its consent mark, else derived from the licence (personal-only -> private, NC -> public, else commercial)."""
    c = clean_consent(info.get("consent"))
    if c:
        return c["scope"]
    lic = str(info.get("license") or "")
    if info.get("commercial_use"):
        return COMMERCIAL
    return PUBLIC_NC if "-NC" in lic else PRIVATE


def license_for_scope(scope: str) -> str:
    return SCOPE_LICENSE.get(scope, SCOPE_LICENSE[PRIVATE])


def example_statement(lang: str, scope: str, name: str = "", date: str = "") -> str:
    """A filled-in template sentence (for the docs and tests)."""
    return TEMPLATES.get(lang, TEMPLATES["en"])[scope].format(name=name or "...", date=date or "...")
