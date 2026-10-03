"""Application metadata and third-party credits; the single source is ``credits.json`` in the resource root.

The GitHub repository URL (``repo_url`` in ``credits.json``) is a placeholder containing ``OWNER`` until the project
is published; while it is a placeholder the link is hidden in the About dialog.  Replace ``OWNER`` in
``credits.json`` to show it.
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
    """Read ``credits.json`` once (cached); fall back to built-in defaults if it is missing or broken."""
    try:
        return json.loads((resource_dir() / "credits.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(_FALLBACK)


def _app() -> Dict[str, str]:
    """The ``app`` section of the credits (name, version, author)."""
    return load_credits().get("app", _FALLBACK["app"])


APP_NAME: str = _app().get("name", "Voxprint")
APP_VERSION: str = _app().get("version", "0.1.0")
APP_AUTHOR: str = _app().get("author", "Aleksandr Mitroshenkov")
#: Repository URL; the only place to edit it is credits.json ("repo_url").
REPO_URL: str = str(load_credits().get("repo_url", _FALLBACK["repo_url"]))


def is_placeholder(url: str) -> bool:
    """True if ``url`` is empty or still contains the ``OWNER`` placeholder."""
    u = (url or "").strip()
    return not u or REPO_PLACEHOLDER_MARK in u


def public_repo_url(url: Optional[str] = None) -> Optional[str]:
    """The repository link to show the user, or None while the address is still the placeholder."""
    u = REPO_URL if url is None else url
    return None if is_placeholder(u) else u.strip()


def components() -> List[Dict[str, Any]]:
    """The list of open-source components / models credited in the About dialog."""
    return list(load_credits().get("components", []))


def localized(value: Any, lang: str) -> str:
    """A value like ``{"en": ..., "ru": ...}`` -> the string in the requested language (English as the fallback)."""
    if isinstance(value, dict):
        return str(value.get(lang) or value.get("en") or "")
    return str(value or "")


def notices_path() -> Path:
    """Path of ``THIRD_PARTY_NOTICES.md`` shipped with the program."""
    return resource_dir() / "THIRD_PARTY_NOTICES.md"


def licenses_dir() -> Path:
    """Folder with the full license texts shipped with the program."""
    return resource_dir() / "licenses"
