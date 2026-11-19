"""Application directories (Windows: ``%LOCALAPPDATA%\\VOXPrint AI Suite``; Linux: ``~/.local/share/voxprint-ai-suite``;
overridable with ``VOXPRINT_HOME``).

Layout under :func:`app_home`::

    shared/      suite-wide runtime and tools (``runtime/``, ``tools/``, ``manifest.json``).
                 Windows: ``%LOCALAPPDATA%\\VOXPrint AI Suite\\shared``. Linux: ``~/.local/share/voxprint-ai-suite/shared``.
                 Override with ``VOXPRINT_SHARED``. See :mod:`infra.suite_shared`.
    models/      the suite models root: :func:`voxprint_common.paths.models_dir` (``VOXPRINT_MODELS_DIR``, else the shared
                 ``suite.json`` value, else ``<home>/models``). The store is written only by VOXPrint MDL; this app reads
                 its models through the app view (see :mod:`infra.mdl_models`).
    packages/    updated Python packages, put on ``sys.path`` at start-up (before heavy imports)
    voices/      the voice library: one folder per voice (adapter + voice.json), see core/voice_library.py
    state/       small settings files (language, privacy acknowledgement, updater state, last adapter ...)
    logs/        log files
    .staging/    updates are installed here and checked for compatibility before they are promoted

Every function creates its directory on demand, except :func:`packages_dir` (it is only read at start-up).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from voxprint_common import paths as suite_paths

APP_NAME = "Voxprint"


def app_home() -> Path:
    """Root data directory: ``$VOXPRINT_HOME``, else the suite home.

    Windows: ``%LOCALAPPDATA%\\VOXPrint AI Suite``. Linux: ``$XDG_DATA_HOME/voxprint-ai-suite``
    (default ``~/.local/share/voxprint-ai-suite``). A legacy ``Voxprint`` / ``voxprint`` folder
    is moved there on first use.
    """
    p = suite_paths.home()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sub(name: str) -> Path:
    """Return (and create) the sub-directory ``name`` of :func:`app_home`."""
    p = app_home() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


MODELS_DIR_ENV = "VOXPRINT_MODELS_DIR"


def shared_root() -> Path:
    """Suite-wide folder: ``$VOXPRINT_SHARED``, else ``<app home>/shared``.

    Windows: ``%LOCALAPPDATA%\\VOXPrint AI Suite\\shared``. Linux: ``~/.local/share/voxprint-ai-suite/shared``.
    """
    env = os.environ.get("VOXPRINT_SHARED", "").strip()
    p = Path(env) if env else suite_paths.shared_dir()
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    """The shared suite models root (:func:`voxprint_common.paths.models_dir`).

    The resolution order (``VOXPRINT_MODELS_DIR``, the shared ``suite.json`` value, the legacy one-line
    ``state/models_dir.txt``, then ``<home>/models``) belongs to the suite library and is the same for every
    Voxprint program.  The store is written only by VOXPrint MDL.
    """
    return suite_paths.models_dir()


def voices_dir() -> Path:
    """Shared voice library (``<home>/voices/<id>/`` = LoRA adapter + ``voice.json``).

    This is :func:`voxprint_common.paths.home` plus ``voices``. The CLI and the GUI both
    read it. Voices found in an earlier data folder, or under ``shared/voices``, are
    moved here when that id is not already present.
    """
    target = _sub("voices")
    _migrate_voices(target)
    return target


def _migrate_voices(target: Path) -> None:
    """Move voice folders into ``target``. An id that is already there is left in place."""
    sources = [app_home() / "shared" / "voices"]
    for home in previous_homes():
        sources.append(home / "voices")
    seen: set[Path] = set()
    for src in sources:
        try:
            resolved = src.resolve()
        except OSError:
            continue
        if resolved in seen or resolved == target.resolve():
            continue
        seen.add(resolved)
        if not src.is_dir():
            continue
        try:
            children = list(src.iterdir())
        except OSError:
            continue
        for child in children:
            if not (child / "voice.json").is_file():
                continue
            dest = target / child.name
            if dest.exists():
                continue
            try:
                shutil.move(str(child), str(dest))
            except OSError:
                continue


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
    out: list[Path] = []
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
