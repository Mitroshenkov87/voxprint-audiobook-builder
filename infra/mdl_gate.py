"""Gate the program on VOXPrint MDL and place downloaded files where the app loads them.

The installer runs ``voxprint-mdl download``. This module only checks
:func:`voxprint_common.mdl.ready` and links the shared blobs into the folders
the loaders already use. It does not download.

    from infra.mdl_gate import models_ready, materialize

    if models_ready():
        materialize()

``VOXPRINT_SKIP_SUITE=1`` (the unit tests) reports ready and does not touch disk.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from voxprint_common.mdl import load_manifest, ready
from voxprint_common.paths import models_dir as suite_models_dir

from core.voice_library import VoiceLibrary
from infra import paths

APP_ID = "audiobook-builder"
MDL_HINT = "voxprint-mdl download --app audiobook-builder --json"
MANIFEST_NAME = "voxprint-app-models.json"


def skipped() -> bool:
    """True when suite downloads must not run (the same rule as :func:`infra.suite_install.skipped`)."""
    return str(os.environ.get("VOXPRINT_SKIP_SUITE", "")).strip().lower() not in ("", "0", "false", "no")


def models_ready() -> bool:
    """True when the required models are installed, or the suite check is skipped."""
    if skipped():
        return True
    return bool(ready())


def manifest_path() -> Path:
    """The app manifest shipped next to the program."""
    bundled = paths.resource_dir() / MANIFEST_NAME
    if bundled.is_file():
        return bundled
    beside = Path(sys.argv[0]).resolve().parent / MANIFEST_NAME
    return beside if beside.is_file() else bundled


def manage_argv() -> list[str]:
    """``voxprint-mdl download --gui --app audiobook-builder`` through this program.

    ``--gui`` is a flag of ``download`` (voxprint-common 0.2.1). A bare
    ``voxprint-mdl --gui`` is a usage error.
    """
    tail = ["download", "--gui", "--app", APP_ID, "--manifest", str(manifest_path())]
    if getattr(sys, "frozen", False):
        return [sys.executable, "--run-mdl", *tail]
    main = Path(__file__).resolve().parent.parent / "main.py"
    return [sys.executable, str(main), "--run-mdl", *tail]


def launch_manage() -> int:
    """Open the model manager and wait until it exits. Returns its exit code."""
    completed = subprocess.run(manage_argv(), check=False)
    return int(completed.returncode)


def materialize() -> None:
    """Hardlink verified blobs into the model folders and import stock voice zips.

    MDL stores bytes at ``<models>/blobs/<sha256>``. The loaders read
    ``Qwen--Name/...``, ``llm/``, ``ace-step-1.5/`` and the shared voices folder.
    A voice zip (``voices/<id>.zip``) is imported once into :func:`paths.voices_dir`.
    """
    if skipped() or not ready():
        return
    path = manifest_path()
    if not path.is_file():
        return
    try:
        document = load_manifest(path)
    except Exception:  # noqa: BLE001 - a damaged manifest must not stop a later repair
        return
    blobs = suite_models_dir() / "blobs"
    root = paths.models_dir()
    library = None
    for model in document.models:
        for item in model.files:
            blob = blobs / item.sha256
            if not blob.is_file():
                continue
            if item.path.startswith("voices/") and item.path.endswith(".zip"):
                if library is None:
                    library = VoiceLibrary()
                _import_voice(library, blob, Path(item.path).stem)
                continue
            _link(blob, root / item.path, item.size)


def _import_voice(library: VoiceLibrary, blob: Path, voice_id: str) -> None:
    folder = paths.voices_dir() / voice_id
    if (folder / "voice.json").is_file():
        return
    try:
        library.import_zip(blob)
    except Exception:  # noqa: BLE001 - one bad zip must not hide the other voices
        return


def _link(blob: Path, dest: Path, size: int) -> None:
    try:
        if dest.is_file() and dest.stat().st_size == size:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        try:
            os.link(blob, dest)
        except OSError:
            shutil.copy2(blob, dest)
    except OSError:
        return
