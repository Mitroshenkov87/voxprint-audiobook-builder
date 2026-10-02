"""Сценарии работы приложения без Qt: «датасет», «голос (LoRA)», «обновления».

Один вызов = вся цепочка этапов. Ошибки - исключения DatasetMakerError с понятным текстом (UI их показывает).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from core.aligner import make_default_aligner
from core.dataset_builder import BuildConfig, DatasetBuilder
from core.events import CancelToken, ProgressCallback, Stage, noop_progress
from infra import model_downloader as md
from infra.updater import Updater

log = logging.getLogger("voxprint.runner")

KIND_DATASET = "dataset"
KIND_LORA = "lora"

PLAN_DATASET: List[Stage] = [Stage.UPDATES, Stage.MODEL, Stage.ALIGN, Stage.SLICE, Stage.SAVE]
PLAN_LORA: List[Stage] = [Stage.UPDATES, Stage.MODEL, Stage.ALIGN, Stage.SLICE, Stage.TRAIN, Stage.SAVE]


def plan_for(kind: str) -> List[Stage]:
    return PLAN_LORA if kind == KIND_LORA else PLAN_DATASET


@dataclass
class TaskRequest:
    kind: str
    audio: Path
    text: Path
    out_root: Optional[Path] = None
    force_cpu: bool = False

    def resolved_root(self) -> Path:
        if self.out_root:
            return Path(self.out_root)
        safe = re.sub(r"[^\w\-. ]", "_", Path(self.audio).stem).strip() or "voice"
        return Path(self.audio).resolve().parent / f"{safe}_Voxprint"


@dataclass
class TaskResult:
    kind: str
    root_dir: Path
    dataset_dir: Path
    adapter_path: Optional[Path] = None
    n_segments: int = 0
    warnings: List[str] = field(default_factory=list)
    update_summary: str = ""

    @property
    def open_dir(self) -> Path:
        return self.root_dir


def safe_auto_update(progress: ProgressCallback, updater: Optional[Updater] = None) -> str:
    """Еженедельная авто-проверка. Любая ошибка проглатывается: обновления не должны мешать работе."""
    try:
        u = updater or Updater()
        if not u.should_autocheck():
            progress(Stage.UPDATES, 1.0, "Обновления проверялись недавно")
            return ""
        _, res = u.check_and_apply(progress)
        return res.summary_ru() if (res.after or res.models_updated or res.rolled_back) else ""
    except Exception as exc:  # noqa: BLE001
        log.warning("auto-update failed: %s", exc)
        progress(Stage.UPDATES, 1.0, "Проверка обновлений пропущена")
        return ""


def run_task(req: TaskRequest, progress: ProgressCallback = noop_progress, cancel: Optional[CancelToken] = None,
             updater: Optional[Updater] = None,
             aligner_factory: Optional[Callable[[], object]] = None) -> TaskResult:
    cancel = cancel or CancelToken()
    root = req.resolved_root()
    dataset_dir, output_dir = root / "dataset", root / "output"
    lora = req.kind == KIND_LORA

    progress(Stage.UPDATES, 0.0, "Проверяю обновления…")
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
    res = TaskResult(req.kind, root, dataset_dir, None, build.n_segments, list(build.warnings), summary)
    if lora:
        from core.lora_trainer import train_lora_from_dataset

        res.adapter_path = train_lora_from_dataset(dataset_dir, output_dir, progress, cancel, req.force_cpu,
                                                   language=build.training_language, warnings_out=res.warnings)
    return res


# --------------------------------------------------------------------------- первый запуск


def required_model_repos() -> List[str]:
    """Модели, нужные для полного сценария на этом компьютере (база TTS зависит от VRAM)."""
    from infra.vram_optimizer import detect_gpu, plan_training

    plan = plan_training(detect_gpu(), 100)
    return [md.ALIGNER_REPO, plan.base_model]


def models_missing(repos: Optional[List[str]] = None) -> List[str]:
    repos = repos if repos is not None else required_model_repos()
    return [r for r in repos if not md.verify_local_model(md.local_dir_for(r))]


def prefetch_models(progress: ProgressCallback = noop_progress, repos: Optional[List[str]] = None,
                    ensure=None) -> List[str]:
    """Первый запуск: автоматически докачивает все нужные модели. Возвращает список скачанных."""
    ensure = ensure or md.ensure_model
    repos = repos if repos is not None else required_model_repos()
    todo = models_missing(repos)
    for i, repo in enumerate(todo):
        ensure(repo, lambda s, f, m, i=i: progress(Stage.MODEL, (i + f) / max(1, len(todo)), m))
    return todo
