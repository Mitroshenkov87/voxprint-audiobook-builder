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

Stdlib only; nothing here may fail the start of the program.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("voxprint.suite")

SCHEMA = 1
FILENAME = "suite.json"
KEYS = ("ui_language", "theme", "models_dir", "gpu")
#: Our only look (dark glass, ui/main_window.py); a theme id we do not know falls back to it.
DEFAULT_THEME = "glass-dark"
THEMES = (DEFAULT_THEME,)
DEFAULT_GPU = "auto"
_GPU_RE = re.compile(r"^cuda:(\d{1,2})$")


def path() -> Path:
    """``<app home>/state/suite.json``."""
    from infra import paths

    return paths.app_home() / "state" / FILENAME


def read_raw() -> Dict[str, Any]:
    """The file as a dict (``{}`` when missing, unreadable or not an object; a BOM is accepted)."""
    try:
        data = json.loads(path().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


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
    return value if isinstance(value, str) and value in THEMES else DEFAULT_THEME


def normalize_gpu(value: object) -> Optional[str]:
    """``auto`` / ``cuda:N`` (``cuda`` alone = ``cuda:0``), or ``None`` for anything else (including ``cpu``)."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v == "auto":
        return v
    if v in ("cuda", "gpu"):
        return "cuda:0"
    m = _GPU_RE.match(v)
    return f"cuda:{int(m.group(1))}" if m else None


def normalize_models_dir(value: object) -> Optional[Path]:
    """An absolute path, or ``None`` (null, empty or relative = the default folder)."""
    if not isinstance(value, str) or not value.strip():
        return None
    p = Path(os.path.expandvars(os.path.expanduser(value.strip().strip('"'))))
    return p if p.is_absolute() else None


# ------------------------------------------------------------------------------------------------- reading
def ui_language() -> str:
    """The shared interface language."""
    return normalize_language(read_raw().get("ui_language")) or default_language()


def theme() -> str:
    """The shared theme name."""
    return normalize_theme(read_raw().get("theme"))


def models_dir() -> Optional[Path]:
    """The shared models folder from the file (``None`` = default or not set); the environment variable is NOT applied here
    (:func:`infra.paths.configured_models_dir` does that)."""
    return normalize_models_dir(read_raw().get("models_dir"))


def gpu() -> str:
    """The shared GPU preference."""
    return normalize_gpu(read_raw().get("gpu")) or DEFAULT_GPU


def values() -> Dict[str, Any]:
    """The four shared values with defaults applied (``models_dir`` as a string or ``None``)."""
    md = models_dir()
    return {"ui_language": ui_language(), "theme": theme(), "models_dir": str(md) if md else None, "gpu": gpu()}


# ------------------------------------------------------------------------------------------------- writing
def _write(data: Dict[str, Any]) -> None:
    f = path()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f"{f.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=True), encoding="utf-8")
    try:
        for i in range(10):
            try:
                os.replace(tmp, f)
                return
            except PermissionError:                  # the other program has the file open for a moment (Windows)
                time.sleep(0.1 * (i + 1))
        raise PermissionError(f"cannot replace {f}")
    finally:
        if tmp.exists():
            tmp.unlink()


def set_value(key: str, value: Any) -> Any:
    """Validate and store one shared value (other keys, known or not, stay).  Returns the stored value.

    ``ValueError`` for an unknown key or an invalid value."""
    if key == "ui_language":
        stored: Any = normalize_language(value)
        if stored is None:
            raise ValueError(f"unsupported UI language {value!r}")
    elif key == "theme":
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid theme {value!r}")
        stored = value.strip()                       # another program's theme id is kept; we read it as our default
    elif key == "models_dir":
        if value is None or (isinstance(value, str) and not value.strip()):
            stored = None
        else:
            p = normalize_models_dir(str(value))
            if p is None:
                raise ValueError(f"models_dir must be an absolute path or null, got {value!r}")
            stored = str(p)
    elif key == "gpu":
        stored = normalize_gpu(value)
        if stored is None:
            raise ValueError(f"gpu must be auto or cuda:N, got {value!r}")
    else:
        raise ValueError(f"not a shared setting: {key!r}")
    data = read_raw()
    if data.get(key, object()) == stored and data.get("schema") == SCHEMA:
        return stored
    data["schema"] = SCHEMA
    data[key] = stored
    _write(data)
    return stored


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
