"""Application metadata and third-party credits; the single source is ``credits.json`` in the resource root.

The GitHub repository URL (``repo_url`` in ``credits.json``) is a placeholder containing ``OWNER`` until the project
is published; while it is a placeholder the link is hidden in the About dialog.  Replace ``OWNER`` in
``credits.json`` to show it.
"""
from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from core import build_stamp
from infra.paths import resource_dir

REPO_PLACEHOLDER_MARK = "OWNER"
_FALLBACK = {"app": {"name": "Voxprint", "display_name": "Voxprint AI Audiobook Builder", "version": "0.1.0", "author": "Aleksandr Mitroshenkov"},
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


APP_NAME: str = _app().get("name", "Voxprint")      # technical name (folders, exe, logs) - never localized
#: Product name shown to people (window titles, About, installer).  A brand: the same in every UI language.
APP_DISPLAY_NAME: str = _app().get("display_name", "Voxprint AI Audiobook Builder")
APP_VERSION: str = _app().get("version", "0.1.0")
APP_AUTHOR: str = _app().get("author", "Aleksandr Mitroshenkov")
#: CI build number (``tools/build_number.py`` stamps it into credits.json; 0 = a local / developer build) and its commit.
APP_BUILD: int = int(str(_app().get("build", 0) or 0)) if str(_app().get("build", 0) or 0).isdigit() else 0
APP_COMMIT: str = str(_app().get("commit", ""))
#: One Biblical Hebrew word (Latin transliteration) naming the build's changes, from BUILD.json (stamped with the number).
APP_CODENAME: str = str(_app().get("codename", ""))
APP_CHANNEL: str = str(_app().get("channel", ""))       # "beta" -> 0.2.4-beta; "rc" on 1.0.0-rc.2 is shown as 1.0.0 RC2


def apply_embedded_stamp(executable: Optional[str] = None) -> None:
    """Prefer the running executable's build stamp, then the constants baked in before PyInstaller.

    ``credits.json`` (often the installed build, left behind when only ``Voxprint.exe`` is replaced) is the fallback
    already stored in :data:`APP_BUILD`. A developer ``python`` has no trailer and ``build_stamp.BUILD`` is 0, so this
    leaves the credits values alone.
    """
    global APP_BUILD, APP_CODENAME
    trailer = build_stamp.read_trailer(sys.executable if executable is None else executable)
    if trailer is not None:
        build, name = trailer
        APP_BUILD = build
        if name:
            APP_CODENAME = name
        return
    if build_stamp.BUILD:
        APP_BUILD = int(build_stamp.BUILD)
        if build_stamp.CODENAME:
            APP_CODENAME = str(build_stamp.CODENAME)


apply_embedded_stamp()


def _has_channel(version: str, channel: str) -> bool:
    """True when ``version`` already ends in ``-<channel>`` or ``-<channel>.<n>`` (``1.0.0-rc``, ``1.0.0-rc.2``)."""
    return bool(channel) and re.search(rf"-{re.escape(channel)}(\.\d+)?$", str(version)) is not None


def release_version(version: Optional[str] = None, channel: Optional[str] = None) -> str:
    """``1.0.0-rc.2``, ``1.0.0-rc`` or ``0.2.4-beta``. A channel already present as a suffix is not repeated."""
    version = APP_VERSION if version is None else version
    channel = APP_CHANNEL if channel is None else channel
    if channel and not _has_channel(str(version), channel):
        return f"{version}-{channel}"
    return str(version)


def shown_version(version: Optional[str] = None, channel: Optional[str] = None) -> str:
    """Version as shown to people.

    ``1.0.0-rc`` is ``1.0.0 RC`` and ``1.0.0-rc.2`` is ``1.0.0 RC2``. A channel that is not already inside the
    version stays hyphenated, as in ``0.2.4-beta``.
    """
    version = APP_VERSION if version is None else version
    channel = APP_CHANNEL if channel is None else channel
    text = str(version)
    m = re.search(r"-rc(?:\.(\d+))?$", text)
    if m:
        return f"{text[:m.start()]} RC{m.group(1) or ''}"
    if channel and not _has_channel(text, channel):
        return f"{version}-{channel}"
    return text


def version_label() -> str:
    """``1.0.0 RC2 · build 1000 "Chazak"`` (an older beta: ``0.1.1-beta · build 665 "Tikkun"``).

    A local build with no CI number shows the version without ``build 0``.
    """
    label = shown_version()
    if APP_BUILD:
        label += f" \u00b7 build {APP_BUILD}"
    if APP_CODENAME:
        label += f' "{APP_CODENAME}"'
    return label
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
