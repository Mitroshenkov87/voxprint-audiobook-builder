"""Gate the program on VOXPrint MDL and place downloaded files where the app loads them.

The installer runs ``voxprint-mdl download``. This module checks
:func:`voxprint_common.mdl.ready` and that the required files are on disk, then
links the shared blobs into the folders the loaders already use. It does not download.

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
from infra import paths, suite_shared

APP_ID = "audiobook-builder"
MDL_HINT = "voxprint-mdl download --app audiobook-builder --json"
MANIFEST_NAME = "voxprint-app-models.json"


def skipped() -> bool:
    """True when suite downloads must not run (the same rule as :func:`infra.suite_install.skipped`)."""
    return str(os.environ.get("VOXPRINT_SKIP_SUITE", "")).strip().lower() not in ("", "0", "false", "no")


def models_ready() -> bool:
    """True when the required models are installed, or the suite check is skipped.

    voxprint-common 0.2.1 ``ready()`` only reads the status file. A file that
    says ready can outlive a deleted weight. Until that function checks the
    files itself, this also requires each required manifest item on disk and a
    torch package in the shared runtime.
    """
    if skipped():
        return True
    if not ready():
        return False
    # TODO: drop this extra check once voxprint-common ready() verifies presence and size.
    return _required_present() and _torch_present()


def _required_present() -> bool:
    path = manifest_path()
    if not path.is_file():
        return False
    try:
        document = load_manifest(path)
    except Exception:  # noqa: BLE001 - a damaged manifest is not a ready install
        return False
    groups: dict[str, list] = {}
    for model in document.models:
        if model.on_demand or model.unavailable:
            continue
        groups.setdefault(model.id, []).append(model)
    if not groups:
        return False
    return all(any(_variant_present(model) for model in variants) for variants in groups.values())


def _variant_present(model) -> bool:
    return bool(model.files) and all(_file_present(item) for item in model.files)


def _file_present(item) -> bool:
    """A required file is present as the MDL blob or at the path the manifest names.

    MDL stores bytes at ``blobs/<sha256>`` and :func:`materialize` links them to
    the named path afterwards. Either non-empty file counts. The declared size
    is not checked here.
    """
    blob = suite_models_dir() / "blobs" / item.sha256
    named = paths.models_dir() / item.path
    return _nonempty(blob) or _nonempty(named)


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _torch_present() -> bool:
    """True when the shared torch runtime directory contains a torch package."""
    folder = suite_shared.torch_dir()
    if not folder.is_dir():
        return False
    if (folder / "torch").is_dir() or (folder / "Lib" / "site-packages" / "torch").is_dir():
        return True
    try:
        return any(folder.glob("torch-*.dist-info"))
    except OSError:
        return False


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
