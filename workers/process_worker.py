"""Фоновые потоки Qt (QThread) для тяжёлых операций. Интерфейс с UI - только сигналы."""
from __future__ import annotations

from core.i18n import tr
import inspect
import logging
import sys
import threading
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

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
    """Ручная/автоматическая проверка обновлений. done(summary_text, changed: bool).

    Устаревший компонент в окружении пользователя молча не меняется: worker шлёт ``offer`` (список словарей
    name/installed/target/compatible/env) и ждёт ответа GUI через ``answer(names)`` (таймаут -> отказ).
    ``auto_answer(offers) -> names`` - ответ без GUI (тесты, фоновый режим)."""

    progress = Signal(int, str, str)
    done = Signal(str, bool)
    failed = Signal(str, str, str, str)
    offer = Signal(list)
    ANSWER_TIMEOUT_S = 600.0

    def __init__(self, updater_factory: Optional[Callable[[], Any]] = None, silent: bool = False,
                 only_if_due: bool = False, parent=None,
                 auto_answer: Optional[Callable[[list], Any]] = None) -> None:
        super().__init__(parent)
        self.updater_factory = updater_factory
        self.silent = silent
        self.only_if_due = only_if_due
        self.auto_answer = auto_answer
        self._answered = threading.Event()
        self._accepted: Set[str] = set()

    def answer(self, accepted_names: Iterable[str] = ()) -> None:
        """Ответ пользователя на ``offer``: имена, которые он разрешил обновить (пусто = отказ)."""
        self._accepted = set(accepted_names)
        self._answered.set()

    def _ask(self, offers) -> Set[str]:
        items = [dict(name=o.name, installed=o.installed, target=o.target, compatible=o.compatible, env=o.env)
                 for o in offers]
        if self.auto_answer is not None:
            return set(self.auto_answer(items))
        self._answered.clear()
        self._accepted = set()
        self.offer.emit(items)
        if not self._answered.wait(self.ANSWER_TIMEOUT_S):
            log.info("no answer to the upgrade offer - treated as declined")
            return set()
        return set(self._accepted)

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
            cb = lambda s, f, m: self.progress.emit(int(f * 100), s.label, m)  # noqa: E731
            if "ask" in inspect.signature(u.check_and_apply).parameters:
                _, res = u.check_and_apply(cb, ask=self._ask)
            else:
                _, res = u.check_and_apply(cb)
            changed = bool(res.after or res.models_updated)
            self.done.emit(res.summary(), changed)
        except BaseException as exc:  # noqa: BLE001
            log.exception("update failed")
            self.failed.emit(*classify_exception(exc))


class RepairWorker(QThread):
    """Кнопка «Исправить»: пересобирает собственное окружение Voxprint (uv). done(exit_code, text)."""

    progress = Signal(int, str)
    done = Signal(int, str)

    def __init__(self, repair_fn: Optional[Callable[..., Tuple[int, str]]] = None, parent=None) -> None:
        super().__init__(parent)
        self.repair_fn = repair_fn

    def run(self) -> None:
        try:
            fn = self.repair_fn
            if fn is None:
                import shutil

                from infra import install_state
                from infra.updater import run_subprocess

                def fn(progress):  # noqa: E306
                    return install_state.repair_install(run_subprocess, shutil.which, progress)
            rc, text = fn(lambda f, m: self.progress.emit(int(f * 100), m))
            self.done.emit(int(rc), str(text))
        except BaseException as exc:  # noqa: BLE001
            log.exception("repair failed")
            self.done.emit(1, classify_exception(exc)[1])


class StatusWorker(QThread):
    """Фоновая проверка состояния: целостность установки и состояние моделей (импорт torch не блокирует GUI).
    done(health_reasons: list[str] локализованные, model_states: dict repo -> missing|partial|ready)."""

    done = Signal(list, dict)

    def __init__(self, health_fn: Optional[Callable[[], List[str]]] = None,
                 model_states_fn: Optional[Callable[[], Dict[str, str]]] = None, parent=None) -> None:
        super().__init__(parent)
        self.health_fn = health_fn or default_health
        self.model_states_fn = model_states_fn or default_model_states

    def run(self) -> None:
        reasons: List[str] = []
        states: Dict[str, str] = {}
        try:
            reasons = list(self.health_fn())
        except Exception:  # noqa: BLE001
            log.exception("health check failed")
        try:
            states = dict(self.model_states_fn())
        except Exception:  # noqa: BLE001
            log.exception("model states failed")
        self.done.emit(reasons, states)


def default_health() -> List[str]:
    """Локализованные причины неисправности собственной установки Voxprint ([] = всё хорошо или установки нет:
    пакет .exe и чужие окружения не проверяются)."""
    if getattr(sys, "frozen", False):
        return []
    from infra import install_state

    if not (install_state.manifest_path().exists() or install_state.venv_dir().exists()):
        return []
    return install_state.describe_reasons(install_state.verify_install(require_manifest=True))


def default_model_states() -> Dict[str, str]:
    from infra import model_downloader as md
    from workers.pipeline_runner import required_model_repos

    return md.model_states(required_model_repos())


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
