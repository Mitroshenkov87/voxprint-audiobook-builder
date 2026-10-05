"""Application directories (Windows: ``%LOCALAPPDATA%\\Voxprint``; Linux: ``~/.local/share/voxprint``;
overridable with ``VOXPRINT_HOME``).

Layout under :func:`app_home`::

    models/      downloaded Hugging Face / ModelScope snapshots (the DEFAULT models folder; the installer's "Models folder" page
                 or ``VOXPRINT_MODELS_DIR`` may move the download target elsewhere - see :func:`models_dir`; a folder that
                 holds a Voxprint backup is never used as the models folder, it is restored into this one)
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
from typing import Optional

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


MODELS_DIR_ENV = "VOXPRINT_MODELS_DIR"
#: One line with the user's models folder (written by the installer's "Models folder" page, UTF-8 with or without BOM).
MODELS_DIR_FILE = "models_dir.txt"


def default_models_dir() -> Path:
    """The default models folder ``<app home>/models`` (``%LOCALAPPDATA%\\Voxprint\\models`` on Windows)."""
    return _sub("models")


def configured_models_dir() -> Optional[Path]:
    """The models folder chosen by the user (``$VOXPRINT_MODELS_DIR``, else ``state/models_dir.txt``), or ``None``.

    Only absolute paths count (a relative one would depend on the working directory of whoever starts the program)."""
    text = os.environ.get(MODELS_DIR_ENV, "").strip().strip('"')
    if not text:
        try:
            lines = (app_home() / "state" / MODELS_DIR_FILE).read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeDecodeError):
            lines = []
        text = next((ln.strip().strip('"') for ln in lines if ln.strip()), "")
    if not text:
        return None
    p = Path(os.path.expandvars(os.path.expanduser(text)))
    return p if p.is_absolute() else None


def set_models_dir(folder: Optional[Path]) -> None:
    """Remember ``folder`` as the models folder; ``None`` or the default folder forgets the choice."""
    f = state_dir() / MODELS_DIR_FILE
    if folder is None or _same(Path(folder), app_home() / "models"):
        f.unlink(missing_ok=True)
        return
    f.write_text(str(folder) + "\n", encoding="utf-8")


def _same(a: Path, b: Path) -> bool:
    try:
        return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))
    except OSError:
        return False


#: Marker file and folder name of a Voxprint backup (the same values as ``infra.backup.MANIFEST_NAME`` / ``BACKUP_DIRNAME``;
#: repeated here because this module must stay import-light - a test keeps them equal).
BACKUP_MANIFEST = "voxprint-backup.json"
BACKUP_DIRNAME = "Voxprint-backup"
#: Sub-folders of a backup a user may pick by mistake (``models`` is what the backup writes; ``model`` / ``voices`` / ``tools``
#: are accepted so that pointing at any folder inside the backup is still recognised).
BACKUP_SUBDIRS = ("models", "model", "voices", "tools")


def backup_root_of(folder: Optional[Path]) -> Optional[Path]:
    """The Voxprint backup behind a user-chosen folder, or ``None`` if it is no backup.

    Recognised: the folder holds ``voxprint-backup.json`` itself; it holds ``Voxprint-backup/voxprint-backup.json`` (the
    user picked the drive / parent folder); or it is a sub-folder (``models`` / ``voices`` ...) of a backup.  A backup is a
    portable archive and a RESTORE SOURCE - it must never become the live models folder (see :func:`models_dir`)."""
    if folder is None:
        return None
    p = Path(folder)
    try:
        if (p / BACKUP_MANIFEST).is_file():
            return p
        if (p / BACKUP_DIRNAME / BACKUP_MANIFEST).is_file():
            return p / BACKUP_DIRNAME
        if p.name.lower() in BACKUP_SUBDIRS and (p.parent / BACKUP_MANIFEST).is_file():
            return p.parent
    except OSError:              # an unplugged drive, no permission ...
        return None
    return None


def models_dir() -> Path:
    """Directory where models are downloaded: the user's models folder if one is configured and can be created, else
    :func:`default_models_dir` (a missing drive must not stop the program; it falls back and downloads there).

    A configured folder that is a Voxprint backup (:func:`backup_root_of`) is NOT used as the models folder: a backup is
    restored INTO the default folder (``infra.existing_models.adopt_backup_choice`` / ``restore_backup``), it is never the
    place the program downloads to or loads from."""
    p = configured_models_dir()
    if p is not None and backup_root_of(p) is not None:
        p = None
    if p is not None:
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except OSError:
            pass
    return default_models_dir()


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
