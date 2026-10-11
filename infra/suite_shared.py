"""Shared suite resources for every Voxprint app.

One tree, one ``manifest.json`` (schema 1, :mod:`voxprint_common.runtime_manifest`)::

    %LOCALAPPDATA%\\Voxprint\\shared\\          Windows
    ~/.local/share/voxprint/shared/         Linux

    runtime/py314-torch2.11-cu130   torch 2.11.0+cu130 (reused, not downloaded again)
    wheels/audiobook-builder        this app's other wheels (CPU onnxruntime never sits beside torch)
    tools/                          ffmpeg and other pinned tools
    manifest.json

Model files stay in the suite models root. VOXPrint MDL records each file. This module
never registers that root, ``shared/runtime``, the voices root or a cache root as one
component. ``refs`` lists the apps that still use a component. Uninstalling drops only
this app's ref and deletes a directory only when this app was the last ref and the path
is not one of those roots. ``VOXPRINT_SHARED`` overrides the shared folder (tests).

    from infra.suite_shared import register, unregister

    register("audiobook-builder", ["runtime", "tools"])
    left = unregister("audiobook-builder")

Licence: Apache-2.0
Credit: VOXPrint AI Audiobook Builder https://github.com/Mitroshenkov87/voxprint-audiobook-builder
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple

from voxprint_common import paths as suite_paths
from voxprint_common import runtime_manifest
from voxprint_common.hashing import sha256_file

from infra import paths

log = logging.getLogger("voxprint.suite")

APP_ID = "audiobook-builder"
MANIFEST_NAME = "manifest.json"
SCHEMA = 1
RESOURCES = ("runtime", "tools")
TORCH_ID = "torch-2.11.0-cu130"
TORCH_VERSION = "2.11.0+cu130"
TORCH_REL = "runtime/py314-torch2.11-cu130"
WHEELS_ID = "audiobook-wheels"
WHEELS_REL = "wheels/audiobook-builder"
_BOOKKEEPING = {".users.json", "reuse.json"}


class _Spec(NamedTuple):
    """One resource name and the component it claims."""

    id: str
    kind: str
    version: str
    path: str
    sha_label: str


_SPECS = {
    "runtime": _Spec(TORCH_ID, "runtime", TORCH_VERSION, TORCH_REL, ""),
    "tools": _Spec("ffmpeg-8", "tool", "8", "tools", "ffmpeg-8"),
}


def root() -> Path:
    """The shared suite folder (created on demand)."""
    return paths.shared_root()


def manifest_path() -> Path:
    """Path of ``manifest.json`` (created on the first :func:`register`)."""
    return root() / MANIFEST_NAME


def torch_dir() -> Path:
    """Directory of the shared torch 2.11.0+cu130 component."""
    return root() / TORCH_REL


def wheels_dir() -> Path:
    """This app's wheels, kept out of the shared torch tree."""
    return root() / WHEELS_REL


def component_dir(name: str) -> Path:
    """Directory of a resource. Does not create it."""
    spec = _SPECS.get(name)
    if spec is None:
        raise ValueError(name)
    return root() / spec.path


def resource_dir(name: str) -> Path:
    """``runtime`` or ``tools`` (created on demand). Models are not a component."""
    folder = component_dir(name)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def torch_sha() -> str:
    """SHA-256 of ``runtime_lock.json``, the archive this torch component was installed from."""
    lock = Path(__file__).resolve().with_name("runtime_lock.json")
    if lock.is_file():
        return sha256_file(lock)
    return _label_sha(TORCH_VERSION)


def _label_sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sha_of(spec: _Spec) -> str:
    return torch_sha() if spec.id == TORCH_ID else _label_sha(spec.sha_label)


def _manifest():
    return runtime_manifest.load(manifest_path())


def load_manifest() -> Dict:
    """Schema 1 document. A missing file is empty and is not created."""
    _migrate_apps_manifest()
    path = manifest_path()
    if not path.is_file():
        return {"schema": SCHEMA, "components": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": SCHEMA, "components": {}}
    if not isinstance(data, dict):
        return {"schema": SCHEMA, "components": {}}
    data["schema"] = SCHEMA
    if not isinstance(data.get("components"), dict):
        data["components"] = {}
    return data


def _add(spec: _Spec, app: str) -> None:
    runtime_manifest.add_ref(
        spec.id, app, kind=spec.kind, version=spec.version, path=spec.path,
        sha256=_sha_of(spec), manifest=manifest_path(),
    )


def register(app: str, resources: Iterable[str]) -> Dict:
    """Remember that ``app`` uses these resources. Unknown names are ignored."""
    _migrate_apps_manifest()
    names = [name for name in resources if name in _SPECS]
    for name in names:
        _add(_SPECS[name], str(app))
        resource_dir(name)
    return load_manifest()


def apps_using(name: str, data: Dict | None = None) -> List[str]:
    """App ids that still reference ``name``."""
    spec = _SPECS.get(name)
    if spec is None:
        return []
    if data is None:
        current = _manifest().components.get(spec.id)
        return list(current.refs) if current is not None else []
    components = data.get("components") if isinstance(data.get("components"), dict) else {}
    row = components.get(spec.id) if isinstance(components, dict) else None
    if isinstance(row, dict):
        return [str(ref) for ref in row.get("refs") or []]
    return []


def unregister(app: str) -> List[str]:
    """Drop ``app`` from every component.

    Returns the component ids whose ref list became empty. A directory is removed
    only when this app was the last ref, the component is not a model (MDL owns
    those files) and the path is not a shared root.
    """
    _migrate_apps_manifest()
    app_id = str(app)
    emptied: List[str] = []
    for cid, kind, rel in _component_rows():
        folder = _folder_of(rel)
        try:
            removed = runtime_manifest.remove_ref(cid, app_id, manifest=manifest_path())
        except runtime_manifest.ManifestError:
            log.warning("could not drop %s from %s", app_id, cid, exc_info=True)
            continue
        if not removed:
            continue
        emptied.append(cid)
        if kind == "model" or folder is None:
            continue
        _delete_owned(folder)
    return emptied


def release_runtime(app: str = APP_ID) -> List[str]:
    """Drop this app's runtime ref and delete the component when nobody else uses it."""
    return unregister_names(app, ("runtime",))


def unregister_names(app: str, names: Iterable[str]) -> List[str]:
    """Drop ``app`` from the named resources and delete directories that are now unused."""
    _migrate_apps_manifest()
    gone: List[str] = []
    for name in names:
        if name not in _SPECS:
            continue
        if runtime_manifest.remove_ref(_SPECS[name].id, str(app), manifest=manifest_path()):
            gone.append(name)
    if "runtime" in names and runtime_manifest.remove_ref(WHEELS_ID, str(app), manifest=manifest_path()):
        _delete_owned(wheels_dir())
    delete_resources(gone)
    return gone


def delete_resources(names: Iterable[str]) -> List[str]:
    """Delete component directories that no app still references.

    A shared root is left in place even when its ref list is empty.
    """
    removed = []
    for name in names:
        if name not in _SPECS or apps_using(name):
            continue
        if _delete_owned(component_dir(name)):
            removed.append(name)
        if name == "runtime":
            current = _manifest().components.get(WHEELS_ID)
            if current is None or not current.refs:
                _delete_owned(wheels_dir())
    return removed


def populated(folder: Path) -> bool:
    """True when ``folder`` exists and is not empty."""
    try:
        return folder.is_dir() and any(folder.iterdir())
    except OSError:
        return False


def foreign_torch_path() -> Path | None:
    """A torch 2.11.0+cu130 tree another app already refs, or None when we may write our own."""
    _migrate_apps_manifest()
    if not manifest_path().is_file():
        return None
    current = _manifest().components.get(TORCH_ID)
    if current is None or current.version != TORCH_VERSION:
        return None
    if not any(ref != APP_ID for ref in current.refs):
        return None
    folder = root() / current.path
    if not _has_torch(folder):
        return None
    return folder


def claim_torch_if_present() -> bool:
    """Add our ref when the shared torch tree is already on disk.

    A different sha held by another app is left untouched (:class:`runtime_manifest.ManifestConflict`).
    """
    folder = torch_dir()
    if not _has_torch(folder):
        borrowed = foreign_torch_path()
        if borrowed is None:
            return False
    try:
        _add(_SPECS["runtime"], APP_ID)
    except runtime_manifest.ManifestConflict:
        log.info("shared torch %s is held at a different sha; using it in place", TORCH_ID)
        return False
    return True


def quarantine_cpu_onnxruntime(folder: Path) -> bool:
    """Move a CPU ``onnxruntime`` install out of the shared torch tree. The GPU build stays."""
    if not folder.is_dir():
        return False
    moved = False
    gpu = any(_is_gpu_ort(child.name) for child in folder.iterdir())
    for child in list(folder.iterdir()):
        if _keep_in_torch(child.name, gpu_ort=gpu):
            continue
        if not _is_cpu_ort(child.name):
            continue
        wheels_dir().mkdir(parents=True, exist_ok=True)
        dest = wheels_dir() / child.name
        if dest.exists():
            continue
        shutil.move(str(child), str(dest))
        moved = True
    if moved:
        try:
            runtime_manifest.add_ref(
                WHEELS_ID, APP_ID, kind="wheels", version="1", path=WHEELS_REL,
                sha256=_label_sha(WHEELS_ID), manifest=manifest_path(),
            )
        except runtime_manifest.ManifestConflict:
            log.info("wheels component is held by another record; CPU onnxruntime stays in %s", wheels_dir())
    return moved


def migrate_installed_runtime() -> None:
    """Turn an rc.4 apps manifest and a per-app ``runtime/`` into schema 1 components."""
    _migrate_apps_manifest()
    dest = torch_dir()
    legacy = paths.app_home() / "runtime"
    flat = root() / "runtime"
    if _has_torch(legacy) and not _has_torch(dest):
        _split_move(legacy, dest, wheels_dir())
    if flat.is_dir() and _has_torch(flat) and not _has_torch(dest):
        _split_move(flat, dest, wheels_dir())
    if _has_torch(dest):
        claim_torch_if_present()
        quarantine_cpu_onnxruntime(dest)


def _migrate_apps_manifest() -> None:
    """Copy an ``apps`` document (rc.4) into components. Components that already exist win."""
    path = manifest_path()
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    apps = data.get("apps")
    if not isinstance(apps, dict):
        return
    components = data.get("components")
    if isinstance(components, dict) and components:
        return
    for name in sorted(RESOURCES):
        holders = []
        for app, info in apps.items():
            if isinstance(info, dict) and name in (info.get("resources") or []):
                holders.append(str(app))
        for app in holders:
            try:
                _add(_SPECS[name], app)
            except runtime_manifest.ManifestError:
                log.warning("could not migrate %s for %s", name, app, exc_info=True)


def _split_move(src: Path, torch_dest: Path, wheels_dest: Path) -> None:
    children = []
    for child in list(src.iterdir()):
        if child.name in _BOOKKEEPING or _same(child, torch_dest) or _same(child, wheels_dest):
            continue
        if _contains(child, torch_dest) or _contains(child, wheels_dest):
            continue
        children.append(child)
    gpu = any(_is_gpu_ort(child.name) for child in children)
    for child in children:
        target_root = torch_dest if _keep_in_torch(child.name, gpu_ort=gpu) else wheels_dest
        target_root.mkdir(parents=True, exist_ok=True)
        dest = target_root / child.name
        if dest.exists():
            continue
        shutil.move(str(child), str(dest))


def _has_torch(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    if (folder / "torch").is_dir() or (folder / "voxprint-components.json").is_file():
        return True
    try:
        return any(folder.glob("torch-*.dist-info")) or (folder / "Lib" / "site-packages" / "torch").is_dir()
    except OSError:
        return False


def _is_gpu_ort(name: str) -> bool:
    low = name.lower()
    return "onnxruntime_gpu" in low or "onnxruntime-gpu" in low


def _is_cpu_ort(name: str) -> bool:
    low = name.lower()
    if _is_gpu_ort(low):
        return False
    return low == "onnxruntime" or low.startswith("onnxruntime-")


def _keep_in_torch(name: str, *, gpu_ort: bool) -> bool:
    low = name.lower()
    if low in _BOOKKEEPING:
        return False
    if low == "torch" or low.startswith("torch-") or low.startswith("torchaudio"):
        return True
    if low.startswith("nvidia"):
        return True
    if _is_gpu_ort(low):
        return True
    if low == "onnxruntime" and gpu_ort:
        return True
    return False


def _same(child: Path, dest: Path) -> bool:
    try:
        return child.resolve() == dest.resolve()
    except OSError:
        return False


def _contains(child: Path, dest: Path) -> bool:
    try:
        dest.resolve().relative_to(child.resolve())
    except (OSError, ValueError):
        return False
    return True


class SharedRootError(RuntimeError):
    """Deleting this path would remove a shared suite root or a folder that contains one."""


def _protected_roots() -> List[Path]:
    """Models, runtime, voices, cache, and the suite home. Not their private subdirectories."""
    home = paths.app_home()
    library = suite_paths.home()
    shared = root()
    found = [
        home,
        library,
        home / "models",
        library / "models",
        shared / "models",
        shared / "runtime",
        home / "runtime",
        home / "voices",
        library / "voices",
        shared / "voices",
        home / "cache",
        library / "cache",
        home / "setup-cache",
        library / "setup-cache",
        home / "trt-cache",
        library / "trt-cache",
        suite_paths.trt_cache_dir(),
    ]
    try:
        found.append(paths.models_dir())
    except OSError:
        pass
    cache = os.environ.get("VOXPRINT_MODEL_CACHE", "").strip()
    if cache:
        found.append(Path(os.path.expandvars(os.path.expanduser(cache))))
    return found


def _norm(folder: Path) -> Path:
    try:
        text = str(Path(folder).resolve(strict=False))
    except OSError:
        text = str(Path(folder).absolute())
    return Path(os.path.normcase(text))


def refuse_shared_root(folder: Path) -> None:
    """Raise when ``folder`` is a shared root or a parent of one.

    The models root, ``shared/runtime``, the voices root, the cache roots and
    the suite home (``%LOCALAPPDATA%\\Voxprint`` on Windows) are never deleted,
    even when no app still lists them. A version directory inside ``shared/runtime``
    is not a root.
    """
    candidate = _norm(folder)
    for protected in _protected_roots():
        try:
            _norm(protected).relative_to(candidate)
        except ValueError:
            continue
        raise SharedRootError(f"refusing to delete {folder}: it is a shared root or contains one")


def _is_shared_root(folder: Path) -> bool:
    try:
        refuse_shared_root(folder)
    except SharedRootError:
        return True
    return False


def _component_rows() -> List[tuple]:
    """``(id, kind, path)`` from the manifest. An unreadable file is empty."""
    path = manifest_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    components = data.get("components") if isinstance(data, dict) else None
    if not isinstance(components, dict):
        return []
    rows = []
    for cid, row in components.items():
        if not isinstance(row, dict):
            continue
        rows.append((str(cid), str(row.get("kind") or ""), str(row.get("path") or "")))
    return rows


def _folder_of(rel: str) -> Path | None:
    text = (rel or "").strip()
    if not text:
        return None
    candidate = Path(text)
    return candidate if candidate.is_absolute() else root() / candidate


def rollback(app: str = APP_ID) -> List[str]:
    """Drop ``app`` after a failed setup or an uninstall.

    Same rules as :func:`unregister`: our ref goes, and a directory is removed
    only when this app exclusively owned it and it is not a shared root.
    """
    return unregister(app)


def _delete_owned(folder: Path) -> bool:
    """Delete ``folder`` when it is not a shared root. Returns whether it was removed."""
    if _is_shared_root(folder):
        log.error("not deleting %s: it is a shared root", folder)
        return False
    _delete_tree(folder)
    return True


def _delete_tree(folder: Path) -> None:
    refuse_shared_root(folder)
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)
