"""Shared suite resources for every Voxprint app.

One tree, versioned by the manifest, reused instead of each app keeping its own copy::

    %LOCALAPPDATA%\\Voxprint\\shared\\          Windows
    ~/.local/share/voxprint/shared/         Linux

    runtime/   Python packages (PyTorch and the rest)
    models/    required base models
    tools/     ffmpeg and other pinned tools
    manifest.json

``manifest.json`` lists which app still uses each resource. Uninstalling one app drops only
that app's references and deletes a resource when no app remains.

    from infra.suite_shared import register, unregister

    register("audiobook-builder", ["runtime", "models", "tools"])
    left = unregister("audiobook-builder")

``VOXPRINT_SHARED`` overrides the folder (tests). Depends only on :mod:`infra.paths`.

Licence: Apache-2.0
Credit: Voxprint AI Audiobook Builder https://github.com/Mitroshenkov87/voxprint-audiobook-builder
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

from infra import paths

APP_ID = "audiobook-builder"
MANIFEST_NAME = "manifest.json"
SCHEMA = 1
RESOURCES = ("runtime", "models", "tools")


def root() -> Path:
    """The shared suite folder (created on demand)."""
    return paths.shared_root()


def resource_dir(name: str) -> Path:
    """``runtime``, ``models`` or ``tools`` under :func:`root` (created on demand)."""
    if name not in RESOURCES:
        raise ValueError(name)
    folder = root() / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def manifest_path() -> Path:
    """Path of ``manifest.json`` (the file is created on the first :func:`register`)."""
    return root() / MANIFEST_NAME


def load_manifest() -> Dict:
    """The manifest, or an empty schema-1 document when the file is missing or damaged."""
    try:
        data = json.loads(manifest_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": SCHEMA, "apps": {}}
    if not isinstance(data, dict):
        return {"schema": SCHEMA, "apps": {}}
    apps = data.get("apps")
    if not isinstance(apps, dict):
        data["apps"] = {}
    data["schema"] = SCHEMA
    return data


def _save(data: Dict) -> None:
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def register(app: str, resources: Iterable[str]) -> Dict:
    """Remember that ``app`` uses these resources. Unknown names are ignored."""
    names = [name for name in resources if name in RESOURCES]
    data = load_manifest()
    data["apps"][str(app)] = {"resources": sorted(set(names))}
    _save(data)
    for name in names:
        resource_dir(name)
    return data


def apps_using(name: str, data: Dict | None = None) -> List[str]:
    """App ids that still list ``name``."""
    data = load_manifest() if data is None else data
    found = []
    for app, info in (data.get("apps") or {}).items():
        if isinstance(info, dict) and name in (info.get("resources") or []):
            found.append(str(app))
    return found


def unregister(app: str) -> List[str]:
    """Drop ``app``. Returns resource names that no remaining app still lists."""
    data = load_manifest()
    info = (data.get("apps") or {}).pop(str(app), None)
    _save(data)
    had = list((info or {}).get("resources") or []) if isinstance(info, dict) else []
    return [name for name in had if name in RESOURCES and not apps_using(name, data)]


def delete_resources(names: Iterable[str]) -> List[str]:
    """Delete resource directories that no app still lists. Returns the names removed."""
    removed = []
    data = load_manifest()
    for name in names:
        if name not in RESOURCES or apps_using(name, data):
            continue
        folder = root() / name
        if folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)
        removed.append(name)
    return removed


def populated(folder: Path) -> bool:
    """True when ``folder`` exists and is not empty."""
    try:
        return folder.is_dir() and any(folder.iterdir())
    except OSError:
        return False
