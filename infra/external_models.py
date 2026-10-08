"""Models used in place from a folder the user chose ("don't copy"), usually on an external drive.

The path is ``<app home>/state/external_models_dir.txt`` (one line, UTF-8). ``VOXPRINT_EXTERNAL_MODELS`` overrides
the file. It points at the backup's ``models`` directory (``Owner--Name`` snapshots plus ``llm``, ``deepfilternet``
and ``dnsmos``), not at the backup root. The model locator searches it even when ``VOXPRINT_NO_EXTERNAL_MODELS``
is set: this folder was chosen, it is not a guess. If the drive is unplugged the path is ignored and the normal
download runs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from infra import backup, paths

ENV_VAR = "VOXPRINT_EXTERNAL_MODELS"
CONFIG_NAME = "external_models_dir.txt"
KIND = "linked"


def config_file() -> Path:
    """The file that remembers the chosen folder."""
    return paths.state_dir() / CONFIG_NAME


def configured_text() -> str:
    """The configured path as written (even if the folder is missing), ``""`` if none."""
    env = os.environ.get(ENV_VAR, "").strip().strip('"')
    if env:
        return env
    try:
        for line in config_file().read_text(encoding="utf-8-sig").splitlines():
            line = line.strip().strip('"')
            if line:
                return line
    except OSError:
        pass
    return ""


def configured() -> Optional[Path]:
    """The external models folder when one is configured and currently reachable."""
    text = configured_text()
    if not text:
        return None
    p = Path(os.path.expandvars(os.path.expanduser(text)))
    try:
        return p if p.is_dir() else None
    except OSError:
        return None


def set_folder(path: Optional[Path]) -> None:
    """Remember ``path`` (``None`` forgets it). A missing folder is still stored so the UI can show it."""
    f = config_file()
    if not path or not str(path).strip():
        try:
            f.unlink()
        except OSError:
            pass
        return
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(str(path).strip() + "\n", encoding="utf-8")


def verified_file(rel: str) -> Optional[Path]:
    """A file under the external models folder whose size and SHA-256 match the backup manifest (and any pin).

    ``rel`` is relative to the models folder (``llm/note.txt``, ``dnsmos/sig_bak_ovr.onnx``). ``None`` when the
    drive is absent, the file is not listed, or the bytes do not match."""
    root = configured()
    if root is None:
        return None
    rel = rel.replace("\\", "/").lstrip("/")
    path = root.joinpath(*rel.split("/"))
    broot = backup._backup_root_of_file(path)
    if broot is None:
        return None
    try:
        key = path.resolve().relative_to(broot.resolve()).as_posix()
        idx = backup.manifest_index(broot)
    except (OSError, ValueError, backup.BackupError):
        return None
    meta = idx.get(key)
    if meta is None:
        return None
    size, sha = meta
    try:
        if not path.is_file() or path.stat().st_size != size:
            return None
        digest = backup.sha256_file(path) if sha else ""
        if sha and digest != sha:
            return None
    except OSError:
        return None
    pin = backup.default_pins().get(key)
    if pin and backup._pinned_bad(key, size, digest or (backup.sha256_file(path) if pin[1] else ""), {key: pin}):
        return None
    return path


def verified_dir(name: str) -> Optional[Path]:
    """``<external>/<name>`` when every file the manifest lists inside it matches. Otherwise ``None``."""
    root = configured()
    if root is None:
        return None
    folder = root / name
    try:
        if not folder.is_dir():
            return None
    except OSError:
        return None
    broot = backup._backup_root_of_file(folder)
    if broot is None:
        return None
    try:
        prefix = folder.resolve().relative_to(broot.resolve()).as_posix()
        idx = backup.manifest_index(broot)
    except (OSError, ValueError, backup.BackupError):
        return None
    listed = [k for k in idx if k.startswith(prefix + "/")]
    if not listed:
        return None
    for key in listed:
        under = key[len(prefix) + 1:]
        if verified_file(f"{name}/{under}") is None:
            return None
    return folder
