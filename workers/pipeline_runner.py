"""Application scenarios without Qt: "dataset", "voice (LoRA)", "universal model" and the first-run model prefetch.

One call runs the whole chain of stages.  Failures are :class:`DatasetMakerError` exceptions with a user-readable
message (the UI shows them).  Everything here is plain Python so it can be driven from the GUI worker, the CLI and tests.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from core.aligner import make_default_aligner
from core.dataset_builder import BuildConfig, DatasetBuilder
from core.events import CancelToken, ProgressCallback, Stage, noop_progress
from core.i18n import tr
from infra import model_downloader as md, paths
from infra.updater import Updater

log = logging.getLogger("voxprint.runner")

KIND_DATASET = "dataset"
KIND_LORA = "lora"
KIND_MERGE = "merge"   # universal (merged) model from an already trained adapter (button only)

#: Stages of each scenario, used to turn per-stage progress into one overall percentage.
PLAN_DATASET: List[Stage] = [Stage.UPDATES, Stage.MODEL, Stage.ALIGN, Stage.SLICE, Stage.SAVE]
PLAN_LORA: List[Stage] = [Stage.UPDATES, Stage.MODEL, Stage.ALIGN, Stage.SLICE, Stage.TRAIN, Stage.SAVE]
PLAN_MERGE: List[Stage] = [Stage.MODEL, Stage.SAVE]


def plan_for(kind: str) -> List[Stage]:
    """The ordered list of stages a task kind goes through (for the overall progress percentage)."""
    if kind == KIND_MERGE:
        return PLAN_MERGE
    return PLAN_LORA if kind == KIND_LORA else PLAN_DATASET


def safe_name(value: str) -> str:
    """Make ``value`` safe as a folder name (letters, digits, ``-._`` and spaces are kept; falls back to ``voice``)."""
    return re.sub(r"[^\w\-. ]", "_", value).strip() or "voice"


@dataclass
class TaskRequest:
    """What the user asked for: the task kind, input files, output folder, CPU switch and optional voice metadata."""
    kind: str
    audio: Optional[Path] = None
    text: Optional[Path] = None
    out_root: Optional[Path] = None
    force_cpu: bool = False
    adapter_dir: Optional[Path] = None   # KIND_MERGE only: folder of the trained adapter
    voice_type: str = ""                 # optional, goes to voice.json: male / female / child / other
    voice_description: str = ""          # optional free text, goes to voice.json

    def voice_name(self) -> str:
        """Voice name = the recording's file name (or the adapter folder's name); also the result folder name in ``output/``."""
        if self.audio:
            return safe_name(Path(self.audio).stem)
        if self.adapter_dir:
            return safe_name(Path(self.adapter_dir).name)
        return "voice"

    def resolved_root(self) -> Path:
        """Result folder: ``out_root`` if given, else ``<audio folder>/<voice>_Voxprint``."""
        if self.out_root:
            return Path(self.out_root)
        if self.audio is None:
            raise ValueError("audio is required")
        return Path(self.audio).resolve().parent / f"{self.voice_name()}_Voxprint"


@dataclass
class TaskResult:
    """What a finished task produced: folders, number of segments, warnings and (for LoRA/merge) the artifact paths."""
    kind: str
    root_dir: Path
    dataset_dir: Path
    adapter_path: Optional[Path] = None
    n_segments: int = 0
    warnings: List[str] = field(default_factory=list)
    update_summary: str = ""
    merged_path: Optional[Path] = None   # folder of the universal model (KIND_MERGE)
    speaker: str = ""                   # voice (speaker) name inside the model

    @property
    def open_dir(self) -> Path:
        """Folder the UI opens when the task is done."""
        return self.root_dir


def safe_auto_update(progress: ProgressCallback, updater: Optional[Updater] = None) -> str:
    """Weekly automatic update check.  Any error is swallowed: updates must never get in the way of the work."""
    try:
        u = updater or Updater()
        if not u.should_autocheck():
            progress(Stage.UPDATES, 1.0, tr("upd.recent"))
            return ""
        _, res = u.check_and_apply(progress)
        return res.summary() if (res.after or res.models_updated or res.rolled_back) else ""
    except Exception as exc:  # noqa: BLE001
        log.warning("auto-update failed: %s", exc)
        progress(Stage.UPDATES, 1.0, tr("upd.skipped"))
        return ""


# --------------------------------------------------------------------------- last adapter


def _last_adapter_file() -> Path:
    """State file remembering the most recently trained adapter."""
    return paths.state_dir() / "last_adapter.json"


def remember_adapter(adapter_dir: Path, voice_name: str = "") -> None:
    """Remember the last trained adapter - for the "universal model" button."""
    try:
        _last_adapter_file().write_text(json.dumps(
            {"adapter_dir": str(adapter_dir), "voice_name": voice_name, "time": int(time.time())},
            ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        log.warning("cannot remember adapter: %s", exc)


def last_adapter() -> Optional[Path]:
    """Folder of the last trained adapter; None if there was none or the folder was deleted."""
    try:
        d = Path(json.loads(_last_adapter_file().read_text(encoding="utf-8"))["adapter_dir"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return d if (d / "adapter_model.safetensors").exists() else None


def _write_voice_json(req: TaskRequest, adapter_dir: Path, language: str, speech_seconds: float) -> None:
    """Write voice.json next to the adapter. A failure here must not fail a finished training run."""
    from core import voice_info

    epochs, base_model = 0, ""
    try:   # epochs and base model are recorded by the trainer in training_meta.json
        meta = json.loads((Path(adapter_dir) / "training_meta.json").read_text(encoding="utf-8"))
        epochs, base_model = int(meta.get("epochs", 0)), str(meta.get("model_name", ""))
    except (OSError, ValueError, TypeError):
        log.warning("training_meta.json unreadable; voice.json will lack epochs/base model")
    info = voice_info.build_voice_info(req.voice_name(), language, speech_seconds, epochs, base_model,
                                       req.voice_type, req.voice_description)
    try:
        voice_info.write_voice_json(adapter_dir, info)
    except OSError as exc:
        log.warning("cannot write voice.json: %s", exc)


def _run_merge(req: TaskRequest, progress: ProgressCallback, cancel: CancelToken) -> TaskResult:
    """Export the adapter merged into the base model as a universal model (``KIND_MERGE``)."""
    from core.errors import ExportError
    from core.model_export import MERGED_DIRNAME, export_merged_model, speaker_name

    if not req.adapter_dir:
        raise ExportError(tr("err.export_no_adapter"))
    adapter = Path(req.adapter_dir)
    out = adapter / MERGED_DIRNAME
    name = req.voice_name()
    export_merged_model(adapter, out, progress=progress, cancel=cancel, voice_name=name)
    return TaskResult(KIND_MERGE, out, adapter, adapter_path=adapter, merged_path=out, speaker=speaker_name(name))


def run_task(req: TaskRequest, progress: ProgressCallback = noop_progress, cancel: Optional[CancelToken] = None,
             updater: Optional[Updater] = None,
             aligner_factory: Optional[Callable[[], object]] = None) -> TaskResult:
    """Run one scenario end to end and return its :class:`TaskResult`.

    Dataset: check updates -> load aligner -> align -> slice -> quality filter -> save.  LoRA additionally trains the
    adapter, remembers it and writes ``voice.json`` next to it.  ``updater`` and ``aligner_factory`` are injectable for
    tests; ``cancel`` is checked between stages and inside the long loops.
    """
    cancel = cancel or CancelToken()
    if req.kind == KIND_MERGE:
        return _run_merge(req, progress, cancel)
    root = req.resolved_root()
    dataset_dir, output_dir = root / "dataset", root / "output" / req.voice_name()
    lora = req.kind == KIND_LORA

    progress(Stage.UPDATES, 0.0, tr("upd.checking"))
    summary = safe_auto_update(progress, updater)
    cancel.check()

    if aligner_factory is not None:
        aligner = aligner_factory()
    else:
        path = md.ensure_aligner_model(progress)
        aligner = make_default_aligner(str(path), "cpu" if req.force_cpu else "auto")
    cfg = BuildConfig()
    try:
        build = DatasetBuilder(aligner, cfg, save_stage=Stage.SLICE if lora else Stage.SAVE).run(
            req.audio, req.text, dataset_dir, progress, cancel)
    finally:
        try:
            aligner.unload()
        except Exception:  # noqa: BLE001
            pass
    res = TaskResult(req.kind, root, dataset_dir, n_segments=build.n_segments, warnings=list(build.warnings),
                     update_summary=summary)
    if lora:
        from core.lora_trainer import train_lora_from_dataset

        res.adapter_path = train_lora_from_dataset(dataset_dir, output_dir, progress, cancel, req.force_cpu,
                                                   language=build.training_language, warnings_out=res.warnings)
        remember_adapter(res.adapter_path, req.voice_name())
        _write_voice_json(req, res.adapter_path, build.language, build.total_seconds)
    return res


# --------------------------------------------------------------------------- first run


def required_model_repos() -> List[str]:
    """Models needed for the full scenario on this computer (the TTS base depends on the available VRAM)."""
    from infra.vram_optimizer import detect_gpu, plan_training

    plan = plan_training(detect_gpu(), 100)
    return [md.ALIGNER_REPO, plan.base_model]


def models_missing(repos: Optional[List[str]] = None) -> List[str]:
    """Models that exist neither in Voxprint's folder nor in another program's copy (they must be downloaded)."""
    repos = repos if repos is not None else required_model_repos()
    return [r for r in repos
            if not md.verify_local_model(md.local_dir_for(r)) and md.external_model(r) is None]


def prefetch_models(progress: ProgressCallback = noop_progress, repos: Optional[List[str]] = None,
                    ensure=None) -> List[str]:
    """First run: download all required models automatically.  Returns the list of repositories that were downloaded."""
    ensure = ensure or md.ensure_model
    repos = repos if repos is not None else required_model_repos()
    todo = models_missing(repos)
    for repo in repos:   # copies of other programs: instant, only the "found, using it" message is shown
        if repo not in todo and not md.verify_local_model(md.local_dir_for(repo)):
            ensure(repo, lambda s, f, m: progress(Stage.MODEL, 0.0, m))
    for i, repo in enumerate(todo):
        ensure(repo, lambda s, f, m, i=i: progress(Stage.MODEL, (i + f) / max(1, len(todo)), m))
    if ensure is md.ensure_model and sys.platform == "win32":
        from infra import assets   # system ffmpeg, else the pinned LGPL build (best effort, never blocks)

        assets.ensure_ffmpeg_tool(lambda f, m: progress(Stage.MODEL, 0.0, m))
    return todo
