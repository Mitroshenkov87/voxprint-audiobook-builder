"""Gate the program on VOXPrint MDL and import the stock voice zips from the store.

The installer runs ``voxprint-mdl download``. This module asks
:func:`voxprint_common.mdl.ready` for this app, which checks that the required
files are present and the right size. It does not download; on-demand models
are fetched through :mod:`infra.mdl_models` (a subprocess running
``voxprint-mdl ensure``).

    from infra.mdl_gate import models_ready, import_stock_voices

    if models_ready():
        import_stock_voices()

``VOXPRINT_SKIP_SUITE=1`` (the unit tests) reports ready and does not touch disk.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from voxprint_common.mdl import CatalogError, load_manifest, ready
from voxprint_common.mdl.views import inspect_view

from core.voice_library import VoiceLibrary
from infra import paths

APP_ID = "audiobook-builder"
MDL_HINT = "voxprint-mdl download --app audiobook-builder --json"
MANIFEST_NAME = "voxprint-app-models.json"


def skipped() -> bool:
    """True when suite downloads must not run (the same rule as :func:`infra.suite_install.skipped`)."""
    return str(os.environ.get("VOXPRINT_SKIP_SUITE", "")).strip().lower() not in ("", "0", "false", "no")


def models_ready() -> bool:
    """True when the required models are installed, or the suite check is skipped.

    ``ready(app)`` in voxprint-common checks the files for this tier,
    including size, and does not trust the status file.
    """
    if skipped():
        return True
    return _reported_ready()


def _reported_ready() -> bool:
    try:
        return bool(ready(APP_ID))
    except (CatalogError, OSError):
        return False


def manifest_path() -> Path:
    """The app manifest shipped next to the program."""
    bundled = paths.resource_dir() / MANIFEST_NAME
    if bundled.is_file():
        return bundled
    beside = Path(sys.argv[0]).resolve().parent / MANIFEST_NAME
    return beside if beside.is_file() else bundled


def manage_argv() -> list[str]:
    """``voxprint-mdl --manifest <ours> --gui --app audiobook-builder`` through this program.

    That opens the localized Manage models window on this app's models.
    """
    tail = ["--manifest", str(manifest_path()), "--gui", "--app", APP_ID]
    if getattr(sys, "frozen", False):
        return [sys.executable, "--run-mdl", *tail]
    main = Path(__file__).resolve().parent.parent / "main.py"
    return [sys.executable, str(main), "--run-mdl", *tail]


def launch_manage() -> int:
    """Open the model manager and wait until it exits. Returns its exit code."""
    completed = subprocess.run(manage_argv(), check=False)
    return int(completed.returncode)


def import_stock_voices() -> None:
    """Import every stock voice zip (manifest rows ``voice-*``) whose view is ready, once.

    The zip bytes live in the shared store; MDL hardlinks them into
    ``views/audiobook-builder/<id>-<digest12>/voices/<id>.zip`` and the voice is
    unpacked into :func:`paths.voices_dir` (the same import the old downloader did).
    """
    if skipped() or not _reported_ready():
        return
    from voxprint_common.paths import models_dir

    from infra import mdl_models

    path = manifest_path()
    if not path.is_file():
        return
    try:
        document = load_manifest(path)
    except Exception:  # noqa: BLE001 - a damaged manifest must not stop a later repair
        return
    mdl_models.opt_in()
    root = models_dir()
    library: VoiceLibrary | None = None
    for model in document.models:
        if not model.id.startswith("voice-") or not model.files:
            continue
        files = {f.path: f.sha256.lower() for f in model.files}
        sizes = {f.path: f.size for f in model.files}
        if inspect_view(root, model.id, files, APP_ID, sizes=sizes) != "ok":
            continue
        view = mdl_models.view_dir(model.id)
        for rel in files:
            if not (rel.startswith("voices/") and rel.endswith(".zip")):
                continue
            if library is None:
                library = VoiceLibrary()
            _import_voice(library, view / rel, Path(rel).stem)


def _import_voice(library: VoiceLibrary, zip_path: Path, voice_id: str) -> None:
    folder = paths.voices_dir() / voice_id
    if (folder / "voice.json").is_file():
        return
    try:
        library.import_zip(zip_path)
    except Exception:  # noqa: BLE001 - one bad zip must not hide the other voices
        return
