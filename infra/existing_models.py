"""An "existing models folder": models downloaded by a previous Voxprint install (or a backup made by the program).

The user (or the installer, see ``installer/Voxprint.iss``) can name one folder that already holds the models.  Before
anything is downloaded, :func:`infra.model_downloader.ensure_model` looks there (:func:`find`, the validation and revision
rules of :mod:`core.model_locator` apply) and **imports** a complete copy into Voxprint's own models folder
(:func:`import_model`):

* hard link when the folder is on the same drive (no extra space, instant), a plain copy otherwise;
* every file is verified by SHA-256 - against the manifest when the folder is a Voxprint backup
  (``Voxprint-backup/voxprint-backup.json``), else by reading the copy back and comparing it with the source;
* the import goes through ``<name>.importing`` and is renamed only when the model is complete, so a crash leaves nothing
  that looks like a finished model, and the next start continues;
* the folder itself is never modified.

Where the path is stored: ``<app home>/state/existing_models_dir.txt`` (UTF-8, one line; the installer writes it, the
Settings dialog edits it).  ``VOXPRINT_EXISTING_MODELS`` overrides the file.  Accepted layouts below the folder: the folder
itself, ``models/``, ``Voxprint-backup/models/`` (folders named ``Owner--Name``), or any Hugging Face / ModelScope cache
layout the model locator understands.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from core.errors import BackupError, CancelledByUser
from core.events import CancelToken
from core.i18n import tr
from infra import backup, paths

log = logging.getLogger("voxprint.existing_models")

ENV_VAR = "VOXPRINT_EXISTING_MODELS"
CONFIG_NAME = "existing_models_dir.txt"
KIND = "existing"


def config_file() -> Path:
    """The file that remembers the chosen folder."""
    return paths.state_dir() / CONFIG_NAME


def configured_text() -> str:
    """The configured path as written (even if the folder does not exist), ``""`` if none."""
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
    """The existing-models folder if one is configured and exists."""
    t = configured_text()
    if not t:
        return None
    p = Path(os.path.expandvars(os.path.expanduser(t)))
    try:
        return p if p.is_dir() else None
    except OSError:
        return None


def set_folder(path: Optional[Path]) -> None:
    """Remember ``path`` (``None`` / empty forgets it)."""
    f = config_file()
    if not path or not str(path).strip():
        try:
            f.unlink()
        except OSError:
            pass
        return
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(str(path).strip() + "\n", encoding="utf-8")


def roots() -> List[Tuple[str, Path]]:
    """Folders to search below the configured one (existing, de-duplicated)."""
    e = configured()
    if e is None:
        return []
    out, seen = [], set()
    for p in (e, e / "models", e / backup.BACKUP_DIRNAME / "models", e / backup.BACKUP_DIRNAME):
        try:
            if p.is_dir() and p.resolve() not in seen:
                seen.add(p.resolve())
                out.append((KIND, p))
        except OSError:
            continue
    return out


def find(repo_id: str, pinned_revision: Optional[str] = None):
    """A complete copy of ``repo_id`` in the configured folder (:class:`core.model_locator.FoundModel`), or ``None``."""
    rs = roots()
    if not rs:
        return None
    from core import model_locator

    return model_locator.find_model(repo_id, pinned_revision, rs, ignore_disabled=True)


def manifest_hashes(model_dir: Path) -> Tuple[Dict[str, str], str]:
    """``({relative path: sha256}, revision)`` from the backup manifest above ``model_dir``; empty if it is no backup."""
    p = Path(model_dir)
    for up in (p.parent.parent, p.parent.parent.parent):
        if (up / backup.MANIFEST_NAME).is_file():
            try:
                for it in backup.items_from_manifest(backup.read_manifest(up)):
                    if it.kind == backup.KIND_MODEL and it.complete and it.name == p.name:
                        return {f.p: f.sha256 for f in it.files if f.sha256}, it.revision
            except BackupError:
                pass
    return {}, ""


def _same_drive(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def import_model(found, progress: Callable[[float, str], None] = lambda f, m="": None,
                 cancel: Optional[CancelToken] = None, usage: Optional[Callable[[Path], object]] = None,
                 allow_link: bool = True, dest_root: Optional[Path] = None) -> Path:
    """Bring the model of ``found`` into Voxprint's models folder (link or copy, hash-verified); returns its folder."""
    from infra import model_downloader as md

    cancel = cancel or CancelToken()
    src_dir = Path(found.path)
    dest = md.local_dir_for(found.repo_id, dest_root)
    stage = dest.with_name(dest.name + ".importing")
    hashes, manifest_rev = manifest_hashes(src_dir)
    files = backup.list_files(src_dir)
    total = max(1, sum(f.size for f in files))
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    linking = allow_link and _same_drive(src_dir, stage)
    # a hard link needs no extra space; a copy needs the whole model
    if not linking:
        backup.check_space(sum(f.size for f in files), dest.parent, usage)
    done = [0]
    short = found.repo_id.split("/")[-1]
    try:
        for f in files:
            cancel.check()
            src, dst = src_dir / f.p, stage / f.p
            want = hashes.get(f.p, "")
            if dst.is_file() and dst.stat().st_size == f.size and (not want or backup.sha256_file(dst, cancel) == want):
                done[0] += f.size
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            real = Path(os.path.realpath(src))
            linked = False
            if linking:
                if want and backup.sha256_file(real, cancel) != want:
                    raise BackupError(tr("backup.err_hash", name=f.p), code="hash", details=str(real))
                try:
                    if dst.exists():
                        dst.unlink()
                    os.link(real, dst)
                    linked = True
                except OSError:
                    linking = False                      # e.g. a file system without hard links: copy instead
            if not linked:
                digest = backup.copy_file(real, dst, cancel, lambda n: progress(
                    min(1.0, (done[0]) / total), tr("backup.importing", name=short)), verify=True)
                if want and digest != want:
                    dst.unlink()
                    raise BackupError(tr("backup.err_hash", name=f.p), code="hash", details=str(real))
            done[0] += f.size
            progress(min(1.0, done[0] / total), tr("backup.importing", name=short))
        rev = found.revision or manifest_rev
        if rev and not (stage / ".revision").exists():
            (stage / ".revision").write_text(rev, encoding="utf-8")
        if not md.verify_local_model(stage):
            raise BackupError(tr("backup.err_incomplete", name=short), code="io", details=str(stage))
    except CancelledByUser:
        raise                                            # the stage folder stays: the next start continues
    except BackupError:
        shutil.rmtree(stage, ignore_errors=True)         # unusable, start over next time
        raise
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    os.replace(stage, dest)
    return dest


def pending(repos: Iterable[str]) -> bool:
    """True if a folder is configured, it holds a usable copy of a required model and that model is not imported yet.

    The start-up code then runs the first-run model step, which imports it."""
    if configured() is None:
        return False
    from infra import model_downloader as md

    for repo in repos:
        if md.verify_local_model(md.local_dir_for(repo)):
            continue
        try:
            if find(repo, md.pinned_revision(repo)) is not None:
                return True
        except Exception as exc:  # noqa: BLE001 - never block start-up
            log.warning("existing models folder check failed for %s: %s", repo, exc)
    return False


def import_available(repos: Iterable[str], progress: Callable[[float, str], None] = lambda f, m="": None,
                     cancel: Optional[CancelToken] = None, **kw) -> "backup.Report":
    """Import every listed model that is missing in Voxprint's folder and present in the configured folder."""
    from infra import model_downloader as md

    cancel = cancel or CancelToken()
    report = backup.Report(configured() or Path("."))
    todo = [r for r in repos if not md.verify_local_model(md.local_dir_for(r))]
    for i, repo in enumerate(todo):
        cancel.check()
        found = find(repo, md.pinned_revision(repo))
        if found is None:
            continue
        path = import_model(found, lambda f, m="", i=i: progress((i + f) / max(1, len(todo)), m), cancel, **kw)
        files = backup.list_files(path)
        report.items += 1
        report.copied_files += len(files)
        report.copied_bytes += sum(f.size for f in files)
    progress(1.0, tr("backup.imported_msg"))
    return report
