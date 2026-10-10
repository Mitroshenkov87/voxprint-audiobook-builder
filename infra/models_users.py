"""Which Voxprint programs use the shared models folder: ``<models>/.users.json``.

Voxprint AI Audiobook Builder and Voxprint AI Movie Dubber keep their models (Qwen3-TTS, Qwen3-ASR, Opus-MT ...) in ONE folder
(:func:`infra.paths.models_dir`, by default ``%LOCALAPPDATA%\\Voxprint\\models``), so a model is downloaded once.  This small file
records which programs rely on it, so uninstalling one never deletes models the other still needs.  The format was agreed
with the Movie Dubber (its ``dubber/infra/model_store.py`` reads and writes the same file):

    {"audiobook-builder": true, "movie-dubber": true}

* The app adds its own key at every start (:func:`register`, idempotent), and ``Voxprint.exe --register-models-user`` does
  the same from a script.
* The uninstaller runs ``Voxprint.exe --unregister-models-user --out FILE`` (:func:`unregister_cli`): our key is removed and
  FILE gets two lines, the number of OTHER programs still using the folder and the folder itself.  Only when that number is
  0 does the uninstaller offer to delete the models, and the default answer is "keep" (a silent uninstall keeps them).
* The same file format is used for the runtime folder (``<app home>/runtime/.users.json``, :func:`runtime_root`):
  ``--register-runtime-user`` / ``--unregister-runtime-user --out FILE``.  A GUI start always writes our key, so the file
  exists after the first launch (the folder is created then).  Only the bookkeeping exists so far; the runtime is not
  shared between the programs yet.
* Unknown keys are kept as they are; a damaged file counts as "no users" for reading and is rewritten on the next write.

Stdlib only; nothing here may fail the start of the app.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

USER_KEY = "audiobook-builder"
USERS_FILE = ".users.json"


def _root(root: Optional[Path] = None) -> Path:
    if root is not None:
        return Path(root)
    from infra import paths

    return paths.models_dir()


def runtime_root() -> Path:
    """``<app home>/runtime`` - the downloaded Python runtime modules (thin builds)."""
    from infra import paths

    return paths.app_home() / "runtime"


def users_path(root: Optional[Path] = None) -> Path:
    """``<models>/.users.json``."""
    return _root(root) / USERS_FILE


def read_users(root: Optional[Path] = None) -> Dict[str, bool]:
    """The users file as ``{program: True/False}``; ``{}`` when it is missing or unreadable (a BOM is accepted)."""
    try:
        data = json.loads(users_path(root).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return {str(k): bool(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _write_users(users: Dict[str, bool], root: Optional[Path] = None) -> None:
    """Atomic write (temp file + replace); retried briefly while the other program has the file open (Windows)."""
    path = users_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(users, indent=1, sort_keys=True), encoding="utf-8")
    try:
        for i in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.1 * (i + 1))
        raise PermissionError(f"cannot replace {path}")
    finally:
        if tmp.exists():
            tmp.unlink()


def register(key: str = USER_KEY, root: Optional[Path] = None) -> Dict[str, bool]:
    """Add ``key`` (no write when it is already there).  Returns the users after the change."""
    users = read_users(root)
    if users.get(key) is True:
        return users
    users[key] = True
    _write_users(users, root)
    return users


def unregister(key: str = USER_KEY, root: Optional[Path] = None) -> List[str]:
    """Remove ``key``.  Returns the OTHER programs still using the folder (empty = the models may go, if the user agrees)."""
    users = read_users(root)
    if key in users:
        users.pop(key)
        _write_users(users, root)
    return sorted(k for k, v in users.items() if v)


def register_quietly(root: Optional[Path] = None) -> None:
    """:func:`register` for the app start: any error is logged, never raised."""
    try:
        register(root=root)
    except Exception as exc:  # noqa: BLE001 - a read-only or missing models drive must not stop the program
        log.warning("users file not updated: %s", exc)


def register_runtime_quietly() -> None:
    """Add our key to ``<app home>/runtime/.users.json`` (created on the first GUI start); never raises."""
    try:
        register(root=runtime_root())
    except Exception as exc:  # noqa: BLE001
        log.warning("runtime users file not updated: %s", exc)


def unregister_cli(out: Optional[str], root: Optional[Path] = None) -> int:
    """``--unregister-models-user --out FILE`` for the uninstaller: FILE gets ``<number of other users>\\n<models folder>\\n``.

    Exit code 0 when the file was written; on any error 1 and no FILE, so the uninstaller keeps the models."""
    try:
        others = unregister(root=root)
        if out:
            Path(out).write_text(f"{len(others)}\n{_root(root)}\n", encoding="utf-8")
        what = "the runtime" if root is not None else "the models"
        print(f"other programs using {what}: {', '.join(others) or 'none'}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"users file: {exc}", flush=True)
        return 1
