"""VOXPrint MDL is the only model manager: this module is the app's one door to the shared store.

Every loader gets its model through here: the manifest row is resolved to a
store view (``views/audiobook-builder/<id>-<digest12>``) and, when the view is
missing, ``voxprint-mdl ensure`` runs in a subprocess through this program's own
interpreter (``--run-mdl``).  Nothing in this app downloads a model itself.

    from infra import mdl_models

    folder = mdl_models.ensure("qwen3-forcedaligner") / "Qwen--Qwen3-ForcedAligner-0.6B"

``VOXPRINT_SKIP_SUITE=1`` (the unit tests) never spawns MDL: tests inject a fake
:func:`runner` and use a temporary ``VOXPRINT_MODELS_DIR`` / ``VOXPRINT_HOME``.
Licence: Apache-2.0.

Credit: VOXPrint AI Audiobook Builder https://github.com/Mitroshenkov87/voxprint-audiobook-builder
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from voxprint_common.mdl import repair_store as _repair_store
from voxprint_common.mdl.catalog import CatalogError, load_manifest
from voxprint_common.mdl.job import detect_tier
from voxprint_common.mdl.views import inspect_view, view_directory
from voxprint_common.paths import derived_dir as _suite_derived_dir
from voxprint_common.paths import models_dir

from core.errors import ModelDownloadError
from core.i18n import tr

from infra.mdl_gate import manifest_path, skipped

log = logging.getLogger("voxprint.mdl_models")

APP_ID = "audiobook-builder"

#: ``progress(fraction, text)`` while MDL fetches a model.
ProgressFn = Callable[[float, str], None]

_TIER_RANK = {"any": 0, "8gb": 1, "16gb": 2, "24gb": 3}


def _root() -> Path:
    """The shared models store (never creates it: views are opt-in, see :func:`opt_in`)."""
    return models_dir()


def _document():
    """The parsed app manifest (``voxprint-app-models.json``)."""
    return load_manifest(manifest_path())


def _files_of_row(row) -> Dict[str, str]:
    """``{repo-relative path: sha256}`` of one manifest row."""
    return {f.path: f.sha256.lower() for f in row.files}


def _sizes_of_row(row) -> Dict[str, int]:
    """``{repo-relative path: size}`` of one manifest row (stub detection checks sizes)."""
    return {f.path: f.size for f in row.files}


def _row_rank(row) -> int:
    return _TIER_RANK.get(str(getattr(row, "tier", "any")), 0)


def _rows(model_id: str):
    """Every manifest row with this id (one per VRAM tier)."""
    return [m for m in _document().models if m.id == model_id]


def _planned_row(model_id: str):
    """The manifest row this machine plans: the complete view wins, else the highest tier the card reaches.

    Rows may share an id across VRAM tiers (the qwen3-tts pair).  A card at or
    below 8 GB plans the 8 GB row; a bigger card the biggest row that fits.  No
    GPU (or a probe failure) plans as 8 GB, the same rule MDL's planner uses.
    """
    rows = _rows(model_id)
    if not rows:
        raise ModelDownloadError(tr("err.mdl_not_ready", model=model_id), details=f"no manifest row: {model_id}")
    root = _root()
    for row in rows:
        if inspect_view(root, model_id, _files_of_row(row), APP_ID, sizes=_sizes_of_row(row)) == "ok":
            return row
    try:
        card = str(detect_tier())          # "8gb" | "16gb" | "24gb" | "unknown"
    except Exception:  # noqa: BLE001 - no nvidia-smi in tests / on odd boxes
        card = "unknown"
    rank = _TIER_RANK.get(card, 1)
    fitting = [r for r in rows if _row_rank(r) <= rank]
    if not fitting:
        fitting = rows
    return max(fitting, key=_row_rank)


def files_of(model_id: str) -> Dict[str, str]:
    """``{path: sha256}`` of the row :func:`_planned_row` picks for this machine."""
    return _files_of_row(_planned_row(model_id))


def model_bytes(model_id: str) -> int:
    """Total size in bytes of the planned row (what MDL would fetch for this model)."""
    return sum(f.size for f in _planned_row(model_id).files)


def opt_in() -> None:
    """Create ``views/audiobook-builder``: MDL builds views only for apps whose folder exists."""
    (_root() / "views" / APP_ID).mkdir(parents=True, exist_ok=True)


def view_dir(model_id: str) -> Path:
    """``<store>/views/audiobook-builder/<id>-<digest12>`` for the planned row."""
    return view_directory(_root(), model_id, files_of(model_id), app=APP_ID)


def is_ready(model_id: str) -> bool:
    """True when the store holds a complete, verified view of the model (a stub counts as missing)."""
    try:
        row = _planned_row(model_id)
    except (ModelDownloadError, CatalogError):
        return False
    try:
        return inspect_view(_root(), model_id, _files_of_row(row), APP_ID, sizes=_sizes_of_row(row)) == "ok"
    except OSError:
        return False


def state(model_id: str) -> str:
    """``installed`` | ``missing`` for UI status (no network, no downloads)."""
    return "installed" if is_ready(model_id) else "missing"


def _argv(model_id: Optional[str] = None, command: str = "ensure") -> List[str]:
    """``python main.py --run-mdl <command> --app-manifest <manifest> [--model <id>] --json`` through this program."""
    tail: List[str] = [command, "--app-manifest", str(manifest_path())]
    if model_id is not None:
        tail += ["--model", model_id]
    tail.append("--json")
    if getattr(sys, "frozen", False):
        return [sys.executable, "--run-mdl", *tail]
    return [sys.executable, str(Path(__file__).resolve().parent.parent / "main.py"), "--run-mdl", *tail]


def _subprocess_runner(argv: List[str], progress: Optional[ProgressFn] = None) -> int:
    """Run MDL in a subprocess, feeding its ``--json`` progress lines to ``progress``; returns the exit code.

    ``VOXPRINT_SKIP_SUITE=1`` (the unit tests) never spawns MDL: the call fails with exit code 2 instead.
    Tests inject :data:`runner` with a fake, which is honoured regardless of the flag."""
    if skipped():
        log.info("VOXPRINT_SKIP_SUITE: not starting VOXPrint MDL (%s)", " ".join(argv))
        return 2
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                                errors="replace")
    except OSError as exc:
        log.warning("could not start VOXPrint MDL (%s): %s", " ".join(argv), exc)
        return 2
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line or progress is None:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if payload.get("type") == "progress":
            fraction = payload.get("fraction")
            try:
                progress(float(fraction) if fraction is not None else 0.0, str(payload.get("message") or ""))
            except Exception:  # noqa: BLE001 - a cancelled UI must not kill the download
                pass
    return int(proc.wait())


#: Test hook: ``runner(argv, progress) -> exit_code``.  The unit tests replace it
#: with a fake so ``ensure`` never spawns MDL (VOXPRINT_SKIP_SUITE=1 also refuses).
runner = _subprocess_runner


def _manage_models_error(model_id: str, code: int) -> ModelDownloadError:
    """The error that points the user at Manage models (:func:`infra.mdl_gate.launch_manage`)."""
    return ModelDownloadError(tr("err.mdl_not_ready", model=model_id), details=f"voxprint-mdl exited with {code}")


def ensure(model_id: str, progress: Optional[ProgressFn] = None) -> Path:
    """The view folder of ``model_id``, fetching it through MDL when it is missing.

    Returns ``views/audiobook-builder/<id>-<digest12>``.  Raises :class:`core.errors.ModelDownloadError`
    (its message points at Manage models) when the store cannot serve the model.
    """
    opt_in()
    if is_ready(model_id):
        return view_dir(model_id)
    code = runner(_argv(model_id), progress)
    if code == 0 and is_ready(model_id):
        return view_dir(model_id)
    if not is_ready(model_id):
        try:   # a stub view / a dropped ref: MDL relinks it from the blobs, once
            _repair_store.repair_store(manifests=[manifest_path()], only=(model_id,))
        except Exception as exc:  # noqa: BLE001 - the error below is the one the user acts on
            log.info("repair-store after ensure(%s) failed: %s", model_id, exc)
    if is_ready(model_id):
        return view_dir(model_id)
    raise _manage_models_error(model_id, code)


def model_id_for_repo(repo_id: str) -> str:
    """The manifest row whose file paths live under ``Org--Name/`` for ``repo_id`` (derived from the manifest)."""
    folder = repo_id.replace("/", "--") + "/"
    for row in _document().models:
        if any(str(f.path).startswith(folder) for f in row.files):
            return row.id
    raise ModelDownloadError(tr("err.mdl_not_ready", model=repo_id), details=f"no manifest row for {repo_id}")


def path_for_repo(repo_id: str, progress: Optional[ProgressFn] = None) -> Path:
    """The ``Org--Name`` folder of ``repo_id`` inside this app's view (fetching the row through MDL when missing)."""
    return ensure(model_id_for_repo(repo_id), progress) / repo_id.replace("/", "--")


def derived_dir(model_id: str) -> Path:
    """App-owned folder for files built from a model (unpacked archives, executable copies).

    ``<store>/derived/audiobook-builder/<id>/<sha12 of the main (largest) file>`` - never inside the view:
    the store blobs are read-only for this app.
    """
    row = _planned_row(model_id)
    main = max(row.files, key=lambda f: f.size)
    return _suite_derived_dir(APP_ID, model_id, main.sha256)


def verify_repair(progress: Optional[ProgressFn] = None) -> bool:
    """``voxprint-mdl verify --repair`` for this app (Check & repair -> models); True when every model is ready."""
    opt_in()
    if skipped():
        return False
    return runner(_argv(command="verify"), progress) == 0


def repair_store(model_ids: Tuple[str, ...] = ()) -> bool:
    """``repair-store`` in-process: adopt on-disk copies into the store and rebuild stub views; True when ready."""
    opt_in()
    try:
        report = _repair_store.repair_store(manifests=[manifest_path()], only=tuple(model_ids) or None)
    except (CatalogError, OSError) as exc:
        log.info("repair-store failed: %s", exc)
        return False
    if model_ids:
        return all(is_ready(m) for m in model_ids)
    return bool(report.ready)
