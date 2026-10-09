"""Localization: plain JSON catalogs ``locales/<code>.json`` with flat keys such as ``"ui.btn_lora"`` and
``{name}``-style parameters.

Languages: ``en`` (default), ``de``, ``ru``, ``uk``, ``lv``.  To add a language, copy ``locales/en.json`` to
``locales/<code>.json``, translate the values, add the code to ``LANGS`` and its native name to ``LANG_NAMES``
(the tests check that every catalog has exactly the same keys and placeholders as English).

The UI language is chosen in this order:

1. the ``VOXPRINT_LANG`` environment variable,
2. the user's saved choice: the shared ``ui_language`` of ``state/suite.json`` (every Voxprint program,
   :mod:`infra.suite_settings`), else this program's ``state/language``,
3. the OS language (Windows: ``GetUserDefaultLocaleName``; elsewhere ``LC_ALL``/``LC_MESSAGES``/``LANG``),
4. English.

``tr(key, **params)`` never raises: a missing translation falls back to English and then to the key itself.
"""
from __future__ import annotations

import json
import locale
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger("voxprint.i18n")

LANGS = ("en", "de", "ru", "uk", "lv")
DEFAULT_LANG = "en"
#: Language names are shown in their own language (in the language selector).
LANG_NAMES = {"en": "English", "de": "Deutsch", "ru": "Русский", "uk": "Українська", "lv": "Latviešu"}

_catalogs: Dict[str, Dict[str, str]] = {}
_current: Optional[str] = None


def locales_dir() -> Path:
    """Folder with the catalogs (next to the program, or the PyInstaller unpack dir when frozen)."""
    from infra.paths import resource_dir

    return resource_dir() / "locales"


def load_catalog(lang: str) -> Dict[str, str]:
    """Load and cache the catalog of ``lang``; a missing/broken file yields an empty dict (and a log warning)."""
    if lang not in _catalogs:
        data: Dict[str, str] = {}
        f = locales_dir() / f"{lang}.json"
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("locale %s not loaded: %s", lang, exc)
        _catalogs[lang] = data
    return _catalogs[lang]


def normalize_code(value: Optional[str]) -> Optional[str]:
    """'ru-RU' / 'de_AT.UTF-8' / 'EN' / 'uk_UA' / 'lv_LV' -> 'ru' / 'de' / 'en' / 'uk' / 'lv'; an unsupported language (incl. be) -> None."""
    if not value:
        return None
    code = value.strip().lower().replace("_", "-").split(".")[0].split("@")[0].split("-")[0]
    return code if code in LANGS else None


def system_language() -> Optional[str]:
    """UI language of the operating system, or ``None`` if it is not one of :data:`LANGS`."""
    if sys.platform == "win32":
        try:
            import ctypes

            buf = ctypes.create_unicode_buffer(85)
            if ctypes.windll.kernel32.GetUserDefaultLocaleName(buf, len(buf)):  # type: ignore[attr-defined]
                code = normalize_code(buf.value)
                if code:
                    return code
        except Exception:  # noqa: BLE001 - the UI language is not critical
            pass
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        code = normalize_code(os.environ.get(var))
        if code:
            return code
    try:
        return normalize_code(locale.getlocale()[0])
    except Exception:  # noqa: BLE001
        return None


def _state_file() -> Path:
    """File where the user's explicit language choice is stored."""
    from infra.paths import state_dir

    return state_dir() / "language"


def saved_language(include_suite: bool = True) -> Optional[str]:
    """The language the user picked earlier (shared ``suite.json`` first, then ``state/language``), or ``None``."""
    if include_suite:
        try:
            from infra import suite_settings

            shared = suite_settings.normalize_language(suite_settings.read_raw().get("ui_language"))
        except Exception:  # noqa: BLE001 - the UI language is not critical
            shared = None
        if shared:
            return shared
    try:
        return normalize_code(_state_file().read_text(encoding="utf-8-sig"))   # the installer may have written it
    except OSError:
        return None


def detect_language() -> str:
    """Resolve the language using the order described in the module docstring."""
    return (normalize_code(os.environ.get("VOXPRINT_LANG")) or saved_language() or system_language()
            or DEFAULT_LANG)


def get_language() -> str:
    """The active UI language code (detected lazily on first use)."""
    global _current
    if _current is None:
        _current = detect_language()
    return _current


def set_language(lang: str, *, persist: bool = False) -> str:
    """Switch the language.  ``persist=True`` remembers the user's choice in ``state/language`` and in the shared
    ``state/suite.json`` (so the other Voxprint programs follow it)."""
    global _current
    code = normalize_code(lang) or DEFAULT_LANG
    _current = code
    if persist:
        try:
            _state_file().write_text(code, encoding="utf-8")
        except OSError as exc:
            log.warning("language choice not saved: %s", exc)
        from infra import suite_settings

        suite_settings.set_quietly("ui_language", code)
    return code


def reset() -> None:
    """Forget the selected language (auto-detection runs again).  For tests."""
    global _current
    _current = None


def tr(key: str, **params: object) -> str:
    """Translate ``key`` into the active language and substitute ``{params}``.

    Falls back to English, then to the key.  A bad format string (missing parameter) returns the unformatted text
    instead of raising, so a translation typo can never crash the app.
    """
    text = load_catalog(get_language()).get(key)
    if text is None and get_language() != DEFAULT_LANG:
        text = load_catalog(DEFAULT_LANG).get(key)
    if text is None:
        text = key
    if params:
        try:
            return text.format(**params)
        except (KeyError, IndexError, ValueError):
            return text
    return text


def tr_lang(key: str, lang: str, **params: object) -> str:
    """Translate into the given language without changing the current one - for data from credits.json and for tests."""
    text = load_catalog(lang).get(key) or load_catalog(DEFAULT_LANG).get(key) or key
    try:
        return text.format(**params) if params else text
    except (KeyError, IndexError, ValueError):
        return text
