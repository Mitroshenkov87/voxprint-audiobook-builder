"""Look-and-feel preferences: window transparency (``default`` / ``more`` / ``off``), stored in ``state/ui.json``."""
from __future__ import annotations

import json
import logging
from typing import Tuple

from infra import paths

log = logging.getLogger("voxprint.ui_prefs")

TRANSPARENCY_LEVELS: Tuple[str, ...] = ("default", "more", "off")


def _file():
    return paths.state_dir() / "ui.json"


def transparency() -> str:
    """The chosen level (``default`` when unset or unreadable)."""
    try:
        v = json.loads(_file().read_text(encoding="utf-8")).get("transparency", "default")
    except (OSError, ValueError, AttributeError):
        return "default"
    return v if v in TRANSPARENCY_LEVELS else "default"


def set_transparency(level: str) -> None:
    """Save the window transparency ``level``."""
    if level not in TRANSPARENCY_LEVELS:
        raise ValueError(level)
    f = _file()
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    d["transparency"] = level
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d), encoding="utf-8")
    log.info("window transparency: %s", level)
