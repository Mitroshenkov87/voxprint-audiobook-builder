"""Сведения о приложении и сторонних компонентах: единый источник - credits.json в корне ресурсов.

REPO_URL (ссылка на репозиторий GitHub) тоже хранится в credits.json, ключ "repo_url". Пока там стоит
заглушка с «OWNER», ссылка в окне «О программе» скрыта. Чтобы показать её, замените OWNER в credits.json.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from infra.paths import resource_dir

REPO_PLACEHOLDER_MARK = "OWNER"
_FALLBACK = {"app": {"name": "Voxprint", "version": "0.1.0", "author": "Aleksandr Mitroshenkov"},
             "repo_url": "https://github.com/OWNER/voxprint", "components": []}


@lru_cache(maxsize=1)
def load_credits() -> Dict[str, Any]:
    try:
        return json.loads((resource_dir() / "credits.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(_FALLBACK)


def _app() -> Dict[str, str]:
    return load_credits().get("app", _FALLBACK["app"])


APP_NAME: str = _app().get("name", "Voxprint")
APP_VERSION: str = _app().get("version", "0.1.0")
APP_AUTHOR: str = _app().get("author", "Aleksandr Mitroshenkov")
#: Адрес репозитория; единственное место для правки - credits.json ("repo_url").
REPO_URL: str = str(load_credits().get("repo_url", _FALLBACK["repo_url"]))


def is_placeholder(url: str) -> bool:
    u = (url or "").strip()
    return not u or REPO_PLACEHOLDER_MARK in u


def public_repo_url(url: Optional[str] = None) -> Optional[str]:
    """Ссылка для показа пользователю или None, если адрес ещё не задан (заглушка)."""
    u = REPO_URL if url is None else url
    return None if is_placeholder(u) else u.strip()


def components() -> List[Dict[str, Any]]:
    return list(load_credits().get("components", []))


def localized(value: Any, lang: str) -> str:
    """Значение вида {"en": ..., "ru": ...} -> строка на нужном языке (иначе английская)."""
    if isinstance(value, dict):
        return str(value.get(lang) or value.get("en") or "")
    return str(value or "")


def notices_path() -> Path:
    return resource_dir() / "THIRD_PARTY_NOTICES.md"


def licenses_dir() -> Path:
    return resource_dir() / "licenses"
