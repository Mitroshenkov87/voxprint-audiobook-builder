"""Фоновые потоки Qt (QThread) для тяжёлых операций. Интерфейс с UI - только сигналы."""
from __future__ import annotations

from core.i18n import tr
import logging
import traceback
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal

from core.errors import CancelledByUser, DatasetMakerError, ModelDownloadError, OutOfMemoryError_
from core.events import CancelToken, Stage, overall_percent
from workers.pipeline_runner import TaskRequest, plan_for, run_task

log = logging.getLogger("voxprint.worker")


def classify_exception(exc: BaseException) -> tuple:
    """-> (kind, user_message, details, url). Неизвестные исключения становятся «other» с общим текстом."""
    if isinstance(exc, DatasetMakerError):
        url = getattr(exc, "url", "")
        return exc.kind, exc.user_message, exc.details, url
    name = type(exc).__name__
    if "OutOfMemory" in name or "out of memory" in str(exc).lower():
        return "oom", OutOfMemoryError_().user_message, str(exc), ""
    return ("other", tr("err.other"), "".join(traceback.format_exception_only(type(exc), exc)).strip(), "")


class ProcessWorker(QThread):
    """Выполняет TaskRequest. Сигналы: progress(percent, stage_label, message), done(TaskResult),
    failed(kind, message, details, url), cancelled()."""

    progress = Signal(int, str, str)
    done = Signal(object)
    failed = Signal(str, str, str, str)
    cancelled = Signal()

    def __init__(self, request: TaskRequest, runner: Callable[..., Any] = run_task, parent=None) -> None:
        super().__init__(parent)
        self.request = request
        self.runner = runner
        self.token = CancelToken()
        self._plan = plan_for(request.kind)
        self._last_pct = 0

    def cancel(self) -> None:
        self.token.cancel()

    def _on_progress(self, stage: Stage, frac: float, msg: str) -> None:
        pct = max(self._last_pct, overall_percent(self._plan, stage, frac))  # прогресс не идёт назад
        self._last_pct = pct
        self.progress.emit(pct, stage.label, msg)

    def run(self) -> None:  # noqa: D401 - QThread
        try:
            result = self.runner(self.request, self._on_progress, self.token)
        except CancelledByUser:
            self.cancelled.emit()
            return
        except BaseException as exc:  # noqa: BLE001 - поток не должен падать молча
            log.exception("task failed")
            self.failed.emit(*classify_exception(exc))
            return
        self.progress.emit(100, Stage.SAVE.label, tr("ui.ready"))
        self.done.emit(result)


class UpdateWorker(QThread):
    """Ручная/автоматическая проверка обновлений. done(summary_text, changed: bool)."""

    progress = Signal(int, str, str)
    done = Signal(str, bool)
    failed = Signal(str, str, str, str)

    def __init__(self, updater_factory: Optional[Callable[[], Any]] = None, silent: bool = False,
                 only_if_due: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.updater_factory = updater_factory
        self.silent = silent
        self.only_if_due = only_if_due

    def run(self) -> None:
        try:
            if self.updater_factory:
                u = self.updater_factory()
            else:
                from infra.updater import Updater

                u = Updater()
            if self.only_if_due and not u.should_autocheck():
                self.done.emit("", False)
                return
            _, res = u.check_and_apply(
                lambda s, f, m: self.progress.emit(int(f * 100), s.label, m))
            changed = bool(res.after or res.models_updated)
            self.done.emit(res.summary(), changed)
        except BaseException as exc:  # noqa: BLE001
            log.exception("update failed")
            self.failed.emit(*classify_exception(exc))


class PrefetchWorker(QThread):
    """Первый запуск: скачивание моделей. progress(percent, message); done(list_of_downloaded); failed(message)."""

    progress = Signal(int, str)
    done = Signal(list)
    failed = Signal(str, str)

    def __init__(self, prefetch_fn: Optional[Callable[..., Any]] = None, parent=None) -> None:
        super().__init__(parent)
        from workers.pipeline_runner import prefetch_models

        self.prefetch_fn = prefetch_fn or prefetch_models

    def run(self) -> None:
        try:
            got = self.prefetch_fn(lambda s, f, m: self.progress.emit(int(f * 100), m))
            self.done.emit(list(got))
        except BaseException as exc:  # noqa: BLE001
            log.exception("prefetch failed")
            kind, msg, details, url = classify_exception(exc)
            self.failed.emit(msg, url)
