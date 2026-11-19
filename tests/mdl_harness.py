"""Shared helpers for the MDL store tests: build blobs + ``.verified`` markers and a real app view (sparse files, no real weights).

The views are built with the same voxprint-common code VOXPrint MDL uses, so ``infra.mdl_models`` sees exactly what it would
see after a real download.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

from voxprint_common.mdl import views as mdl_views
from voxprint_common.paths import models_dir

APP_ID = "audiobook-builder"


def row_maps(model_id: str, manifest: Optional[Path] = None) -> Tuple[Dict[str, str], Dict[str, int]]:
    """``({path: sha256}, {path: size})`` of the first manifest row with ``model_id``."""
    from voxprint_common.mdl.catalog import load_manifest

    from infra.mdl_gate import manifest_path

    doc = load_manifest(manifest if manifest is not None else manifest_path())
    row = next(m for m in doc.models if m.id == model_id)
    return {f.path: f.sha256.lower() for f in row.files}, {f.path: f.size for f in row.files}


def digest12(files: Dict[str, str]) -> str:
    """The view digest: the first 12 hex of the SHA-256 of the sorted ``path<TAB>sha256`` lines."""
    listing = "\n".join(f"{name}\t{digest.lower()}" for name, digest in sorted(files.items()))
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()[:12]


def build_store(model_id: str, root: Optional[Path] = None, manifest: Optional[Path] = None) -> Path:
    """Create the row's blobs (sparse, ``.verified`` marked) and build the app view; returns the view folder."""
    root = Path(root) if root is not None else models_dir()
    files, sizes = row_maps(model_id, manifest)
    for path, sha in files.items():
        blob = root / "blobs" / sha
        blob.parent.mkdir(parents=True, exist_ok=True)
        with open(blob, "wb") as fh:                       # sparse: the right size, no real bytes
            fh.truncate(sizes.get(path, 0))
        blob.with_name(blob.name + ".verified").write_text(sha, encoding="ascii")
    mdl_views.repair_view(root, model_id, files, APP_ID, sizes=sizes)
    return mdl_views.view_directory(root, model_id, files, app=APP_ID)


def fake_runner(monkeypatch, mdl_models, model_ids, root=None, code: int = 0, calls=None):
    """Patch ``mdl_models.runner`` with a fake that "downloads" ``model_ids`` by building their store views.

    Returns the list that collects the argv of every call."""
    calls = calls if calls is not None else []

    def _fake(argv, progress=None):
        calls.append(list(argv))
        if code == 0:
            for mid in model_ids:
                build_store(mid, root=root)
        return code

    monkeypatch.setattr(mdl_models, "runner", _fake)
    return calls
