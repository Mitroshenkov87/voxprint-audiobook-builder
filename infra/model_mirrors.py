"""Hugging Face backup mirror of the permissively licensed models (last download fallback).

Order of sources for a model (see :func:`infra.model_downloader.ensure_model`): local cache / existing copies ->
the original repository (Hugging Face, then ModelScope) -> **this mirror**.  The mirror lives under the project
owner's Hugging Face account (``Mitroshenkov87/voxprint-mirror-*``); its files are byte-identical to the original
commit.  ``model_mirrors.json`` (next to this file) lists every file with its SHA-256; a file is accepted only
if the hash matches, otherwise it is deleted and the download counts as failed (the caller then reports the error).

Only models with permissive licences (Apache-2.0, MIT, CC0, CC-BY ...) are mirrored.  ``README.md`` of a mirror is a
model card written for the mirror (flag ``card``), so it is not downloaded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Optional

log = logging.getLogger("voxprint.models")

MANIFEST_PATH = Path(__file__).with_name("model_mirrors.json")
SCHEMA = 1
#: Set to 1 to disable only the Hugging Face backup mirror (``VOXPRINT_NO_MIRROR=1`` disables all mirrors).
ENV_DISABLE = "VOXPRINT_NO_HF_MIRROR"
#: ``fetch(mirror_repo, filename, revision, local_dir)`` downloads one file into ``local_dir`` (injectable in tests).
Fetch = Callable[[str, str, str, Path], Any]


class MirrorError(Exception):
    """The mirror cannot supply a complete, hash-verified copy."""


@dataclass(frozen=True)
class MirrorEntry:
    """One mirrored model: where it comes from, where the copy is and the expected hash of each file."""
    source_repo: str
    source_revision: str
    license: str
    mirror_repo: str
    mirror_revision: str
    files: Dict[str, Dict[str, Any]]      # relative posix path -> {"size", "sha256", ["card"]}

    def downloadable(self) -> Dict[str, Dict[str, Any]]:
        """Files that make up the model (the mirror's own model card is left out)."""
        return {n: m for n, m in self.files.items() if not m.get("card")}


def enabled() -> bool:
    """True unless the backup mirror (or all mirrors) is disabled by an environment variable."""
    off = ("1", "true", "yes", "on")
    return (os.environ.get(ENV_DISABLE, "").strip().lower() not in off
            and os.environ.get("VOXPRINT_NO_MIRROR", "").strip().lower() not in off)


def _safe_name(name: str) -> bool:
    """A manifest file name must be a plain relative path (no absolute path, no ``..``)."""
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and ".." not in p.parts and "\\" not in name and ":" not in name


def load(path: Optional[Path] = None) -> Dict[str, MirrorEntry]:
    """``{source_repo: MirrorEntry}`` from the manifest; an empty dict if it is missing or invalid (never raises)."""
    try:
        data = json.loads((path or MANIFEST_PATH).read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise ValueError("unsupported schema")
        out: Dict[str, MirrorEntry] = {}
        for src, e in data["models"].items():
            files = e["files"]
            if not files or not all(_safe_name(n) and len(m["sha256"]) == 64 for n, m in files.items()):
                raise ValueError(f"bad file list for {src}")
            out[src] = MirrorEntry(src, e["source_revision"], e.get("license", ""), e["mirror_repo"],
                                   e["mirror_revision"], files)
        return out
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        log.warning("model mirror manifest unusable: %s", exc)
        return {}


def entry_for(repo_id: str, path: Optional[Path] = None) -> Optional[MirrorEntry]:
    """Mirror entry of a model, or None (not mirrored / mirror disabled / manifest unusable)."""
    return load(path).get(repo_id) if enabled() else None


def sha256_file(path: Path) -> str:
    """SHA-256 (hex) of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def file_ok(path: Path, meta: Dict[str, Any]) -> bool:
    """True if the file exists with the expected size and SHA-256."""
    try:
        return path.is_file() and path.stat().st_size == int(meta["size"]) and sha256_file(path) == meta["sha256"]
    except OSError:
        return False


def _default_fetch(mirror_repo: str, filename: str, revision: str, local_dir: Path) -> Any:
    """Download one file of the mirror repository with huggingface_hub (resumable)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=mirror_repo, filename=filename, revision=revision, local_dir=str(local_dir))


def download(entry: MirrorEntry, dest: Path, progress: Callable[[float], None] = lambda f: None,
             fetch: Optional[Fetch] = None) -> str:
    """Download the mirror copy into ``dest`` and verify every file against the manifest.

    Files already in ``dest`` with the right size and hash are kept (resume).  A file with a wrong hash is deleted
    and ``MirrorError`` is raised; nothing unverified stays behind.  Returns the original commit sha the files
    are identical to.
    """
    fetch = fetch or _default_fetch
    files = entry.downloadable()
    total = sum(int(m["size"]) for m in files.values()) or 1
    done = 0
    dest.mkdir(parents=True, exist_ok=True)
    for name, meta in sorted(files.items()):
        target = dest.joinpath(*PurePosixPath(name).parts)
        if not file_ok(target, meta):
            try:
                if target.exists():
                    target.unlink()
                target.parent.mkdir(parents=True, exist_ok=True)
                fetch(entry.mirror_repo, name, entry.mirror_revision, dest)
            except Exception as exc:  # noqa: BLE001 - network / auth / missing file: the caller falls back or reports
                raise MirrorError(f"{entry.mirror_repo}: {name}: {type(exc).__name__}: {exc}") from exc
            if not file_ok(target, meta):
                try:
                    target.unlink()
                except OSError:
                    pass
                raise MirrorError(f"{entry.mirror_repo}: {name}: SHA-256 / size mismatch with the manifest")
        done += int(meta["size"])
        progress(min(1.0, done / total))
    return entry.source_revision
