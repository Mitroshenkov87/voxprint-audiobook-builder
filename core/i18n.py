"""Локализация: простые JSON-каталоги locales/<код>.json (плоские ключи вида "ui.start", параметры {name}).

Языки: en (по умолчанию), de, ru. Как добавить язык - README, раздел «Adding a language». Порядок выбора языка:
  1. переменная окружения VOXPRINT_LANG;  2. сохранённый выбор (state/language);
  3. язык системы (Windows: GetUserDefaultLocaleName, иначе LC_ALL/LC_MESSAGES/LANG);  4. английский.
`tr(key, **params)` никогда не бросает исключений: нет перевода -> английский текст -> сам ключ.
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

LANGS = ("en", "de", "ru")
DEFAULT_LANG = "en"
#: Названия языков показываются на их собственном языке (в переключателе).
LANG_NAMES = {"en": "English", "de": "Deutsch", "ru": "Русский"}

_catalogs: Dict[str, Dict[str, str]] = {}
_current: Optional[str] = None


def locales_dir() -> Path:
    from infra.paths import resource_dir

    return resource_dir() / "locales"


def load_catalog(lang: str) -> Dict[str, str]:
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
    """'ru-RU' / 'de_AT.UTF-8' / 'EN' -> 'ru' / 'de' / 'en'; неподдерживаемый (в т.ч. uk, be) -> None."""
    if not value:
        return None
    code = value.strip().lower().replace("_", "-").split(".")[0].split("@")[0].split("-")[0]
    return code if code in LANGS else None


def system_language() -> Optional[str]:
    """Язык интерфейса ОС (Windows - через ctypes; прочие - переменные окружения/locale)."""
    if sys.platform == "win32":
        try:
            import ctypes

            buf = ctypes.create_unicode_buffer(85)
            if ctypes.windll.kernel32.GetUserDefaultLocaleName(buf, len(buf)):  # type: ignore[attr-defined]
                code = normalize_code(buf.value)
                if code:
                    return code
        except Exception:  # noqa: BLE001 - язык не критичен
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
    from infra.paths import state_dir

    return state_dir() / "language"


def saved_language() -> Optional[str]:
    try:
        return normalize_code(_state_file().read_text(encoding="utf-8"))
    except OSError:
        return None


def detect_language() -> str:
    return (normalize_code(os.environ.get("VOXPRINT_LANG")) or saved_language() or system_language()
            or DEFAULT_LANG)


def get_language() -> str:
    global _current
    if _current is None:
        _current = detect_language()
    return _current


def set_language(lang: str, *, persist: bool = False) -> str:
    """Переключает язык. persist=True запоминает выбор пользователя в state/language."""
    global _current
    code = normalize_code(lang) or DEFAULT_LANG
    _current = code
    if persist:
        try:
            _state_file().write_text(code, encoding="utf-8")
        except OSError as exc:
            log.warning("language choice not saved: %s", exc)
    return code


def reset() -> None:
    """Сбрасывает выбранный язык (повторное автоопределение). Для тестов."""
    global _current
    _current = None


def tr(key: str, **params: object) -> str:
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
    """Перевод на указанный язык (без смены текущего) - для данных из credits.json и тестов."""
    text = load_catalog(lang).get(key) or load_catalog(DEFAULT_LANG).get(key) or key
    try:
        return text.format(**params) if params else text
    except (KeyError, IndexError, ValueError):
        return text
