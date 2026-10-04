"""Application directories (Windows: ``%LOCALAPPDATA%\\Voxprint``; Linux: ``~/.local/share/voxprint``;
overridable with ``VOXPRINT_HOME``).

Layout under :func:`app_home`::

    models/      downloaded Hugging Face / ModelScope snapshots
    packages/    updated Python packages, put on ``sys.path`` at start-up (before heavy imports)
    voices/      the voice library: one folder per voice (adapter + voice.json), see core/voice_library.py
    state/       small settings files (language, privacy acknowledgement, updater state, last adapter ...)
    logs/        log files
    .staging/    updates are installed here and checked for compatibility before being promoted

Every function creates its directory on demand, except :func:`packages_dir` (it is only read at start-up).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "Voxprint"


def app_home() -> Path:
    """Root data directory: ``$VOXPRINT_HOME``, else ``%LOCALAPPDATA%\\Voxprint`` (Windows) / ``$XDG_DATA_HOME/voxprint``
    (Linux and others, default ``~/.local/share/voxprint`` - lower case by XDG habit; the Linux installer keeps ``app/`` and
    ``venv/`` there too)."""
    env = os.environ.get("VOXPRINT_HOME")
    if env:
        p = Path(env)
    elif sys.platform == "win32":
        p = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / APP_NAME
    else:
        xdg = os.environ.get("XDG_DATA_HOME", "").strip()
        p = (Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".local" / "share") / APP_NAME.lower()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sub(name: str) -> Path:
    """Return (and create) the sub-directory ``name`` of :func:`app_home`."""
    p = app_home() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    """Directory with downloaded model snapshots."""
    return _sub("models")


def voices_dir() -> Path:
    """Directory of the voice library (``voices/<id>/`` = LoRA adapter + ``voice.json``)."""
    return _sub("voices")


def logs_dir() -> Path:
    """Directory for log files."""
    return _sub("logs")


def staging_dir() -> Path:
    """.staging/ - updates are installed here before their compatibility is checked."""
    return _sub(".staging")


def packages_dir() -> Path:
    """Directory with updated Python packages; added to ``sys.path`` at start-up (before the heavy libraries are imported)."""
    return app_home() / "packages"


def state_dir() -> Path:
    """Directory for small state files (language, privacy acknowledgement, updater state ...)."""
    return _sub("state")


def default_results_dir() -> Path:
    """Default parent folder for results: ``~/Documents/Voxprint``."""
    return Path.home() / "Documents" / APP_NAME


def resource_dir() -> Path:
    """Root of the resources shipped with the program (``locales/``, ``licenses/``, ``credits.json``).

    In a PyInstaller build this is the unpack directory (``sys._MEIPASS``), otherwise the project root.
    """
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base)
    return Path(__file__).resolve().parent.parent


def previous_homes() -> list:
    """Data folders of earlier Voxprint installations (read-only): models and settings can be found there.

    An installer-based upgrade keeps ``%LOCALAPPDATA%\\Voxprint`` (the same :func:`app_home`), which is the main case.
    Additionally the typical former locations and ``VOXPRINT_PREVIOUS_HOMES`` (a list separated by ``os.pathsep``) are
    checked.  A candidate counts only if it is a directory containing ``models``, ``state`` or ``packages``.
    """
    cur = None
    try:
        cur = app_home().resolve()
    except OSError:
        pass
    cands = []
    for part in os.environ.get("VOXPRINT_PREVIOUS_HOMES", "").split(os.pathsep):
        if part.strip():
            cands.append(Path(part.strip().strip('"')))
    if sys.platform == "win32":
        for var in ("APPDATA", "LOCALAPPDATA"):
            base = os.environ.get(var)
            if base:
                cands.append(Path(base) / APP_NAME)
                cands.append(Path(base) / APP_NAME.lower())
    try:
        home = Path.home()
        cands += [home / ".voxprint", home / ".local" / "share" / APP_NAME, home / ".local" / "share" / APP_NAME.lower()]
    except (RuntimeError, OSError):
        pass
    out = []
    for c in cands:
        try:
            r = c.resolve()
            if r == cur or r in [o.resolve() for o in out] or not c.is_dir():
                continue
            if any((c / n).is_dir() for n in ("models", "state", "packages")):
                out.append(c)
        except OSError:
            continue
    return out


#: Small settings files carried over from an earlier installation (copied only if they do not exist yet).
ADOPTED_STATE_FILES = ("language", "privacy_ack", "updater_state.json", "last_adapter.json")


def adopt_previous_settings() -> list:
    """Copy settings from an earlier Voxprint installation into the current state directory.

    The old folder is never modified and existing files are never overwritten.  Returns the names of the copied files.
    """
    done = []
    dst_dir = state_dir()
    for home in previous_homes():
        src_dir = home / "state"
        for name in ADOPTED_STATE_FILES:
            src, dst = src_dir / name, dst_dir / name
            try:
                if src.is_file() and not dst.exists():
                    dst.write_bytes(src.read_bytes())
                    done.append(name)
            except OSError:
                continue
    return done
