"""Settings shared by every Voxprint program (Audiobook Builder, Movie Dubber): ``<app home>/state/suite.json``.

Agreed with the Movie Dubber (it reads and writes the same file).  Only four values are shared; everything else stays in each
program's own settings::

    {"schema": 1,
     "ui_language": "ru",            # ISO code; default: the OS language if supported, else "en"
     "theme": "glass-dark",          # unknown -> the program's default theme ("glass-dark" here, our only theme)
     "models_dir": null,             # absolute path, or null = the default folder; VOXPRINT_MODELS_DIR always wins
     "gpu": "auto"}                  # "auto" | "cuda:N"  (no CPU choice)

* UTF-8, written atomically (temp file + ``os.replace``); keys this program does not know are kept as they are.
* A missing key means its default; a missing or damaged file means all defaults (and is rewritten on the next change).
* Read at start (:mod:`core.i18n`, :mod:`infra.paths`, :mod:`core.tts_engine`), written when the user changes one of the
  values (UI language, models folder, ``voxprint settings set``).
* ``Voxprint.exe --sync-suite-settings`` (run by the installer after it wrote ``state/models_dir.txt``) copies this program's
  models folder into the file, and its UI language when the file has none yet.

Nothing here may fail the start of the program. The file format is the suite library's.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from voxprint_common import suite_settings as common

log = logging.getLogger("voxprint.suite")

SCHEMA = 1
FILENAME = "suite.json"
KEYS = ("ui_language", "theme", "models_dir", "gpu")
#: Our only look (dark glass, ui/main_window.py); a theme id we do not know falls back to it.
DEFAULT_THEME = "glass-dark"
THEMES = (DEFAULT_THEME,)
DEFAULT_GPU = "auto"


def path() -> Path:
    """``<app home>/state/suite.json``."""
    return common.file()


def read_raw() -> Dict[str, Any]:
    """The file as a dict (``{}`` when missing, unreadable or not an object; a BOM is accepted)."""
    return common.read()


def has(key: str) -> bool:
    """True when ``key`` is written in the file (``models_dir: null`` counts: it is an explicit "default folder")."""
    return key in read_raw()


# ------------------------------------------------------------------------------------------------- normalizers
def normalize_language(value: object) -> Optional[str]:
    """A supported UI language code, or ``None``."""
    from core.i18n import normalize_code

    return normalize_code(value) if isinstance(value, str) else None


def default_language() -> str:
    """The language to use when the shared file has none: the system language, else English."""
    from core.i18n import DEFAULT_LANG, system_language

    return system_language() or DEFAULT_LANG


def normalize_theme(value: object) -> str:
    """A theme this program knows; anything else -> :data:`DEFAULT_THEME`."""
    return common.normalize_theme(value)


def normalize_gpu(value: object) -> Optional[str]:
    """``auto`` / ``cuda:N`` (``cuda`` alone = ``cuda:0``), or ``None`` for anything else (including ``cpu``)."""
    return common.normalize_gpu(value)


def normalize_models_dir(value: object) -> Optional[Path]:
    """An absolute path, or ``None`` (null, empty or relative = the default folder)."""
    return common.normalize_models_dir(value)


# ------------------------------------------------------------------------------------------------- reading
def ui_language() -> str:
    """The shared interface language."""
    return normalize_language(read_raw().get("ui_language")) or default_language()


def theme() -> str:
    """The shared theme name."""
    return common.theme()


def models_dir() -> Optional[Path]:
    """The shared models folder from the file (``None`` = default or not set); the environment variable is NOT applied here
    (:func:`infra.paths.configured_models_dir` does that)."""
    return common.models_dir()


def gpu() -> str:
    """The shared GPU preference."""
    return common.gpu()


def values() -> Dict[str, Any]:
    """The four shared values with defaults applied (``models_dir`` as a string or ``None``)."""
    md = models_dir()
    return {"ui_language": ui_language(), "theme": theme(), "models_dir": str(md) if md else None, "gpu": gpu()}


# ------------------------------------------------------------------------------------------------- writing
def set_value(key: str, value: Any) -> Any:
    """Validate and store one shared value (other keys, known or not, stay).  Returns the stored value.

    ``ValueError`` for an unknown key or an invalid value. The UI language stays this program's list
    (``en`` / ``de`` / ``ru`` / ``uk`` / ``lv``); the file itself is the suite ``suite.json``.
    """
    if key == "ui_language":
        stored: Any = normalize_language(value)
        if stored is None:
            raise ValueError(f"unsupported UI language {value!r}")
        common.set_ui_language(stored)
        return stored
    if key == "theme":
        return common.set_theme(value if isinstance(value, str) else "")
    if key == "models_dir":
        if value is None or (isinstance(value, str) and not value.strip()):
            common.set_models_dir(None)
            return None
        common.set_models_dir(value)
        folder = common.models_dir()
        return str(folder) if folder else None
    if key == "gpu":
        return common.set_gpu(value if isinstance(value, str) else "")
    raise ValueError(f"not a shared setting: {key!r}")


def set_quietly(key: str, value: Any) -> None:
    """:func:`set_value` that logs instead of raising (called from code paths that must not fail)."""
    try:
        set_value(key, value)
    except Exception as exc:  # noqa: BLE001
        log.warning("shared setting %s not saved: %s", key, exc)


def sync_from_app() -> Dict[str, Any]:
    """Installer helper: our models folder (``state/models_dir.txt``, null when there is none) always, and our UI language
    (``state/language``) only when the file has none yet.  Returns the shared values afterwards."""
    from core import i18n
    from infra import paths

    md = paths.app_models_dir_file()
    set_value("models_dir", str(md) if md else None)
    if not has("ui_language"):
        lang = i18n.saved_language(include_suite=False)
        if lang:
            set_value("ui_language", lang)
    return values()


def sync_cli() -> int:
    """``--sync-suite-settings``: exit code 0 when the file was written, 1 on any error (the installer ignores the code)."""
    try:
        out = sync_from_app()
        print(json.dumps(out), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"suite settings: {exc}", flush=True)
        return 1


def migrate_quietly() -> None:
    """At start: copy this program's own models folder / UI language into the shared file when the file has none yet, so
    the other Voxprint programs follow choices made before 704.  Never raises."""
    try:
        from core import i18n
        from infra import paths

        if not has("models_dir"):
            md = paths.app_models_dir_file()
            if md is not None and paths.backup_root_of(md) is None:
                set_value("models_dir", str(md))
        if not has("ui_language"):
            lang = i18n.saved_language(include_suite=False)
            if lang:
                set_value("ui_language", lang)
    except Exception as exc:  # noqa: BLE001
        log.warning("shared settings not migrated: %s", exc)
