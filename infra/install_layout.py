"""Where this program is installed, and how another VOXPrint app finds it.

Windows installs under ``C:\\Program Files\\voxprint-ai-suite\\audiobook-builder``.
The windowed file is ``voxprint-audiobook-builder.exe``. Scripts call
``voxprint-audiobook-cli.exe``. The Start menu folder is ``VOXPrint AI Suite`` and the
shortcut is ``VOXPrint AI Audiobook Builder`` (display names only). Linux installs the
package at ``/opt/voxprint-ai-suite/audiobook-builder`` and puts
``voxprint-audiobook-cli`` on ``PATH``. Voices and settings stay in
``%LOCALAPPDATA%\\Voxprint`` and ``~/.local/share/voxprint``.

Peers read the shared manifest key ``installs`` (not ``apps``, which counted
resources), then the Windows registry ``Software\\voxprint\\Apps\\audiobook-builder``,
then the default path. voxprint-common 0.1.1 has no install registry; the extra
manifest key is the stand-in until the library grows one.

    from infra.install_layout import locate_cli, register_install

    register_install(install_dir, cli_path)
    path = locate_cli()
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path, PureWindowsPath
from typing import Optional

from voxprint_common.atomic import exclusive_lock
from voxprint_common.runtime_manifest import lock_path

from infra import platform_win, suite_shared

log = logging.getLogger("voxprint.install")

SUITE_DIR = "voxprint-ai-suite"
APP_FOLDER = "audiobook-builder"
GUI_EXE = "voxprint-audiobook-builder.exe"
CLI_EXE = "voxprint-audiobook-cli.exe"
LINUX_CLI = "voxprint-audiobook-cli"
LINUX_CLI_COMPAT = "voxprint"
LINUX_INSTALL = "/opt/voxprint-ai-suite/audiobook-builder"
LINUX_LAUNCHER = "/usr/local/bin/voxprint-audiobook-cli"
LINUX_DESKTOP = "/usr/share/applications/voxprint-audiobook-builder.desktop"
#: Per-user package from before the /opt layout. Data next to it is kept.
OLD_LINUX_APP = "~/.local/share/voxprint/app"
LEGACY_DIR = "Voxprint"
LEGACY_GUI = "Voxprint.exe"
LEGACY_CLI = "voxprint-cli.exe"
#: Earlier suite folders. On Windows the two spellings are one directory, and neither is the kebab-case folder.
LEGACY_SUITE_DIRS = ("Voxprint AI Suite", "VOXPrint AI Suite")
LEGACY_APP_FOLDER = "Audiobook Builder"
LEGACY_SUITE_GUI = "VOXPrint AI Audiobook Builder.exe"
START_MENU = "VOXPrint AI Suite"
SHORTCUT = "VOXPrint AI Audiobook Builder"
INSTALLS_KEY = "installs"
APP_ID = suite_shared.APP_ID
#: Silent migration removes the old program folder. It never runs the old uninstaller
#: (that drops model and runtime refcounts) and it never deletes this data folder.
USER_DATA_DIR_NAME = "Voxprint"


def windows_install_dir(program_files: str = r"C:\Program Files") -> str:
    """Default Windows install directory ({autopf}\\voxprint-ai-suite\\audiobook-builder)."""
    return str(PureWindowsPath(program_files) / SUITE_DIR / APP_FOLDER)


def windows_cli_path(program_files: str = r"C:\Program Files") -> str:
    """Default Windows console program."""
    return str(PureWindowsPath(windows_install_dir(program_files)) / CLI_EXE)


def legacy_program_dir(program_files: str = r"C:\Program Files") -> str:
    """The previous install folder (``C:\\Program Files\\Voxprint``)."""
    return str(PureWindowsPath(program_files) / LEGACY_DIR)


def user_data_dir(local_app_data: str) -> str:
    """Voices, settings, the shared runtime and models. Migration leaves this in place."""
    return str(PureWindowsPath(local_app_data) / USER_DATA_DIR_NAME)


def legacy_suite_app_dirs(program_files: str = r"C:\Program Files") -> list[str]:
    """Earlier suite install folders (``Voxprint AI Suite`` and ``VOXPrint AI Suite``)."""
    return [str(PureWindowsPath(program_files) / name / LEGACY_APP_FOLDER) for name in LEGACY_SUITE_DIRS]


def migration_removes_program_dir(path: str, program_files: str = r"C:\Program Files") -> bool:
    """True when silent migration may delete ``path`` (an old program folder only)."""
    candidate = PureWindowsPath(path)
    roots = [PureWindowsPath(legacy_program_dir(program_files))]
    roots.extend(PureWindowsPath(item) for item in legacy_suite_app_dirs(program_files))
    return any(candidate == root or root in candidate.parents for root in roots)


def case_only_rename(actual: str, wanted: str) -> bool:
    """True when ``actual`` and ``wanted`` are the same name apart from case.

    The Start menu folder ``Voxprint AI Suite`` is renamed to ``VOXPrint AI Suite``.
    The program folder ``voxprint-ai-suite`` is a different directory, so it is not a case rename.
    """
    return actual.casefold() == wanted.casefold() and actual != wanted


def legacy_suite_dir(program_files: str = r"C:\Program Files") -> str:
    """The earlier suite install folder (``Audiobook Builder`` under the mixed-case suite name)."""
    return legacy_suite_app_dirs(program_files)[0]


def default_install_dir() -> Path:
    """The well-known install directory on this operating system."""
    if sys.platform == "win32":
        root = os.environ.get("ProgramFiles", r"C:\Program Files")
        return Path(root) / SUITE_DIR / APP_FOLDER
    return Path(LINUX_INSTALL)


def default_cli_path() -> Path:
    """The well-known console program, whether or not it is installed yet."""
    if sys.platform == "win32":
        return default_install_dir() / CLI_EXE
    return Path(LINUX_LAUNCHER)


def cli_file_name() -> str:
    """File name of the console program on this operating system."""
    return CLI_EXE if sys.platform == "win32" else LINUX_CLI


def _blank_manifest() -> dict:
    return {"schema": suite_shared.SCHEMA, "components": {}}


def _read_manifest_file(path: Path) -> dict:
    if not path.is_file():
        return _blank_manifest()
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return _blank_manifest()
    if not isinstance(data, dict):
        return _blank_manifest()
    return data


def _update_manifest(mutate) -> None:
    """Read and write the manifest under its lock. Does not call ``runtime_manifest.load`` (that lock is not re-entrant)."""
    path = suite_shared.manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(lock_path(path), timeout=30):
        data = _read_manifest_file(path)
        mutate(data)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)


def register_install(install_path: Path, cli_path: Path) -> None:
    """Record this install in the shared manifest and, on Windows, in the current-user registry."""
    install = str(Path(install_path))
    cli = str(Path(cli_path))

    def mutate(data: dict) -> None:
        data["schema"] = suite_shared.SCHEMA
        if not isinstance(data.get("components"), dict):
            data["components"] = {}
        installs = data.get(INSTALLS_KEY)
        if not isinstance(installs, dict):
            installs = {}
        installs[APP_ID] = {"install_path": install, "cli_path": cli}
        data[INSTALLS_KEY] = installs

    _update_manifest(mutate)
    platform_win.write_install_registry(APP_ID, install, cli)


def unregister_install() -> None:
    """Drop this app's install record. Shared components, models and user data stay."""
    path = suite_shared.manifest_path()
    if path.is_file():

        def mutate(data: dict) -> None:
            installs = data.get(INSTALLS_KEY)
            if isinstance(installs, dict) and APP_ID in installs:
                del installs[APP_ID]
                if installs:
                    data[INSTALLS_KEY] = installs
                else:
                    data.pop(INSTALLS_KEY, None)

        _update_manifest(mutate)
    platform_win.delete_install_registry(APP_ID)


def read_install_record() -> Optional[dict]:
    """The manifest record for this app, or None."""
    installs = _read_manifest_file(suite_shared.manifest_path()).get(INSTALLS_KEY)
    if not isinstance(installs, dict):
        return None
    row = installs.get(APP_ID)
    return row if isinstance(row, dict) else None


def _existing(path: Path) -> Optional[Path]:
    try:
        return path if path.is_file() else None
    except OSError:
        return None


def locate_cli() -> Optional[Path]:
    """Console program another app should call.

    Order: manifest ``installs``, Windows registry, the default path, then ``PATH``.
    The old ``C:\\Program Files\\Voxprint`` folder is not a result.
    """
    row = read_install_record()
    if row:
        found = _existing(Path(str(row.get("cli_path") or "")))
        if found is not None:
            return found
        install = Path(str(row.get("install_path") or ""))
        found = _existing(install / cli_file_name())
        if found is not None:
            return found
    reg = platform_win.read_install_registry(APP_ID)
    if reg is not None:
        found = _existing(Path(reg[1]))
        if found is not None:
            return found
        found = _existing(Path(reg[0]) / cli_file_name())
        if found is not None:
            return found
    found = _existing(default_cli_path())
    if found is not None:
        return found
    which = shutil.which(LINUX_CLI)
    return Path(which) if which else None


def register_current() -> None:
    """Record the directory of this executable. Used by ``--register-install`` and a frozen start."""
    exe = Path(sys.executable)
    folder = exe.parent
    cli = folder / cli_file_name()
    if not cli.is_file():
        cli = exe
    register_install(folder, cli)


def register_quietly() -> None:
    """A frozen program records itself. A source checkout does not (that would point peers at a dev tree)."""
    if not getattr(sys, "frozen", False):
        return
    try:
        register_current()
    except Exception:  # noqa: BLE001 - discovery is a convenience; the program still runs
        log.debug("install was not registered", exc_info=True)
