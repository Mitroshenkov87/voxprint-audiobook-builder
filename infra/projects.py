"""The projects folder: where the heavy working data of every project lives (book narrations, voice trainings, Re-voice).

Best practice for big, re-creatable data on Windows: a local folder that is NOT synced by OneDrive.

* default: ``<app home>/Projects`` (``%LOCALAPPDATA%\\Voxprint\\Projects`` on Windows, ``~/.local/share/voxprint/Projects``
  on Linux); one sub-folder per kind and one consolidated folder per project inside it: ``Audiobooks/<book>/``,
  ``Voices/<voice>_Voxprint/``, ``Re-voice/``;
* the user can choose another folder in Settings (remembered in ``state/projects_folder.txt``); a folder inside OneDrive
  gets a warning (hundreds of GB would be uploaded);
* a "Voxprint Projects" shortcut is put into the user's real Documents folder, found through the Known Folder API
  (``SHGetKnownFolderPath(FOLDERID_Documents)``), so it also works when Documents is redirected to OneDrive.  The
  shortcut is a small ``.lnk`` file (Windows) or a symlink (Linux): the data itself never goes to OneDrive.

Qt-free; everything is injectable for tests.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("voxprint.projects")

STATE_FILE = "projects_folder.txt"
SHORTCUT_NAME = "Voxprint Projects"
AUDIOBOOKS, VOICES, REVOICE = "Audiobooks", "Voices", "Re-voice"


def _state_file() -> Path:
    from infra import paths

    return paths.state_dir() / STATE_FILE


def default_projects_dir() -> Path:
    """``<app home>/Projects`` - local, outside every synced folder."""
    from infra import paths

    return paths.app_home() / "Projects"


def configured_projects_dir() -> Optional[Path]:
    """The folder chosen in Settings, or None."""
    try:
        text = _state_file().read_text(encoding="utf-8-sig").strip()
    except OSError:
        return None
    return Path(text) if text else None


def projects_dir() -> Path:
    """The projects folder (created); the default one when the chosen folder cannot be created (e.g. a missing drive)."""
    for p in (configured_projects_dir(), default_projects_dir()):
        if p is None:
            continue
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except OSError:
            log.warning("projects folder %s not usable - using the default", p)
    return default_projects_dir()


def set_projects_dir(folder: Optional[Path]) -> Path:
    """Remember ``folder`` (None = back to the default); returns the folder now in use."""
    f = _state_file()
    if folder is None or Path(folder) == default_projects_dir():
        f.unlink(missing_ok=True)
    else:
        f.write_text(str(Path(folder)), encoding="utf-8")
    return projects_dir()


def sub(kind: str) -> Path:
    """``<projects>/<kind>`` (Audiobooks, Voices, Re-voice), created."""
    p = projects_dir() / kind
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------------------------------------ Documents, OneDrive
def documents_dir(known_folder: Optional[Callable[[], Optional[str]]] = None) -> Path:
    """The user's real Documents folder: the Known Folder API on Windows (follows a OneDrive redirect),
    ``xdg-user-dir DOCUMENTS`` on Linux, else ``~/Documents``."""
    if known_folder is None:
        if sys.platform == "win32":
            from infra import platform_win

            known_folder = platform_win.documents_folder
        else:
            known_folder = _xdg_documents
    try:
        p = known_folder()
    except Exception:  # noqa: BLE001 - never fatal
        p = None
    return Path(p) if p else Path.home() / "Documents"


def _xdg_documents() -> Optional[str]:
    try:
        r = subprocess.run(["xdg-user-dir", "DOCUMENTS"], capture_output=True, text=True, timeout=5)  # noqa: S607
        out = r.stdout.strip()
        return out if r.returncode == 0 and out and out != str(Path.home()) else None
    except (OSError, subprocess.SubprocessError):
        return None


def onedrive_roots() -> list:
    """Folders synced by OneDrive (from its environment variables)."""
    roots = []
    for var in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        v = os.environ.get(var, "").strip()
        if v and Path(v) not in roots:
            roots.append(Path(v))
    return roots


def in_onedrive(folder: Path) -> bool:
    """True when ``folder`` lies inside a OneDrive-synced folder (environment variables, else a ``OneDrive`` path part)."""
    p = Path(os.path.abspath(folder))
    for root in onedrive_roots():
        try:
            p.relative_to(Path(os.path.abspath(root)))
            return True
        except ValueError:
            continue
    return any(part.lower().startswith("onedrive") for part in p.parts[1:])


# ------------------------------------------------------------------------------------------------ the shortcut
def shortcut_path(docs: Optional[Path] = None) -> Path:
    docs = docs or documents_dir()
    return docs / (SHORTCUT_NAME + (".lnk" if sys.platform == "win32" else ""))


def ensure_shortcut(target: Optional[Path] = None, docs: Optional[Path] = None,
                    make_link: Optional[Callable[[Path, Path], None]] = None) -> Optional[Path]:
    """Put (or update) the "Voxprint Projects" shortcut into Documents; returns its path or None.  Never raises."""
    target = target or projects_dir()
    try:
        link = shortcut_path(docs)
        if not link.parent.is_dir():
            return None
        from infra import paths

        done = paths.state_dir() / "projects_shortcut.txt"         # the target the shortcut was last made for
        try:
            same = done.read_text(encoding="utf-8").strip() == str(target)
        except OSError:
            same = False
        if same and (link.exists() or link.is_symlink()):
            return link
        if make_link is None:
            make_link = _make_lnk if sys.platform == "win32" else _make_symlink
        make_link(link, target)
        done.write_text(str(target), encoding="utf-8")
        return link
    except Exception as exc:  # noqa: BLE001 - a convenience only
        log.info("projects shortcut not created: %s", exc)
        return None


def _make_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        if Path(os.readlink(link)) == target:
            return
        link.unlink()
    elif link.exists():
        return                                   # something of the user's with that name: never replace it
    link.symlink_to(target, target_is_directory=True)


def _make_lnk(link: Path, target: Path) -> None:
    """A Windows ``.lnk`` through the Windows Script Host (PowerShell, no window); paths are passed as environment
    variables, never pasted into the script."""
    script = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:VX_LNK); $s.TargetPath = $env:VX_TARGET; "
              "$s.Description = 'Voxprint projects (local, not synced)'; $s.Save()")
    env = dict(os.environ, VX_LNK=str(link), VX_TARGET=str(target))
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],  # noqa: S603,S607
                   env=env, capture_output=True, timeout=30, check=True,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
