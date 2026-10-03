"""Qt background threads for the voice library (online repository) and the narrator.  Interface to the UI: signals only."""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from PySide6.QtCore import QThread, Signal

from core import narration as nr
from core.errors import CancelledByUser, DatasetMakerError
from core.events import CancelToken
from core.i18n import tr
from infra import voice_repository as repo
from workers.narration_runner import NarrationJob, run_narration
from workers.process_worker import classify_exception

log = logging.getLogger("voxprint.workers")


class NarrateWorker(QThread):
    """Runs one narration job.  Signals: ``progress(NarrationProgress)``, ``done(NarrationResult)``,
    ``failed(kind, message, details)``, ``cancelled()``.  ``pause()``/``resume()``/``cancel()`` are thread-safe."""

    progress = Signal(object)
    done = Signal(object)
    failed = Signal(str, str, str)
    cancelled = Signal()
    plan = Signal(list)        # ordered chunk files of the job (for the live player)

    def __init__(self, job: NarrationJob, runner: Callable[..., Any] = run_narration, parent=None) -> None:
        """``runner(job, progress, cancel, pause)`` is injectable (tests)."""
        super().__init__(parent)
        self.job, self.runner = job, runner
        if getattr(job, "on_plan", None) is None:
            job.on_plan = lambda paths: self.plan.emit(list(paths))
        self.cancel_token = CancelToken()
        self.pause_token = nr.PauseToken()

    def pause(self) -> None:
        """Hold the job before the next chunk."""
        self.pause_token.pause()

    def resume(self) -> None:
        """Continue after :meth:`pause`."""
        self.pause_token.resume()

    def cancel(self) -> None:
        """Stop after the current chunk (finished chunks stay cached for a later resume)."""
        self.cancel_token.cancel()
        self.pause_token.resume()

    def run(self) -> None:  # noqa: D401 - QThread
        """Thread body."""
        try:
            result = self.runner(self.job, self.progress.emit, self.cancel_token, self.pause_token)
        except CancelledByUser:
            self.cancelled.emit()
        except DatasetMakerError as exc:
            log.error("narration failed: %s (%s)", exc.user_message, exc.details)
            self.failed.emit(exc.kind, exc.user_message, exc.details)
        except Exception as exc:  # noqa: BLE001
            kind, msg, details, _url = classify_exception(exc)
            log.exception("narration crashed")
            self.failed.emit(kind, msg, details)
        else:
            self.done.emit(result)


class RepoIndexWorker(QThread):
    """Fetches the voice index in the background.  Signal ``done(IndexResult)``."""

    done = Signal(object)

    def __init__(self, url: Optional[str] = None, fetch: Callable[..., Any] = repo.fetch_index, parent=None) -> None:
        """``fetch`` is injectable (tests)."""
        super().__init__(parent)
        self.url, self.fetch = url, fetch

    def run(self) -> None:  # noqa: D401
        """Thread body."""
        try:
            self.done.emit(self.fetch(self.url))
        except Exception as exc:  # noqa: BLE001 - fetch_index should not raise, but never kill the thread silently
            self.done.emit(repo.IndexResult(error="unreachable", detail=str(exc)))


class RepoDownloadWorker(QThread):
    """Downloads (and verifies) the selected repository voices one after another.

    Signals: ``progress(fraction, name)``, ``voice_done(voice_id)``, ``finished_all(list of ids)``, ``failed(message)``."""

    progress = Signal(float, str)
    voice_done = Signal(str)
    finished_all = Signal(list)
    failed = Signal(str)

    def __init__(self, entries: List[repo.RepoVoice], library, download: Callable[..., Any] = repo.download_voice,
                 parent=None) -> None:
        """``download(entry, library, progress=..., cancel=...)`` is injectable (tests)."""
        super().__init__(parent)
        self.entries, self.library, self.download = entries, library, download
        self.cancel_token = CancelToken()

    def cancel(self) -> None:
        """Abort the current download."""
        self.cancel_token.cancel()

    def run(self) -> None:  # noqa: D401
        """Thread body: stops at the first failure and reports it."""
        ids: List[str] = []
        for e in self.entries:
            try:
                rec = self.download(e, self.library, progress=lambda f, n: self.progress.emit(f, n),
                                    cancel=self.cancel_token)
            except CancelledByUser:
                break
            except DatasetMakerError as exc:
                self.failed.emit(f"{e.name}: {exc.user_message}")
                break
            except Exception as exc:  # noqa: BLE001
                self.failed.emit(f"{e.name}: {tr('err.voice_repo_download')} ({exc})")
                break
            ids.append(rec.id)
            self.voice_done.emit(rec.id)
        self.finished_all.emit(ids)


class TextModelDownloadWorker(QThread):
    """Downloads an optional text model (e.g. the clean-up model) in the background.

    Signals: ``progress(fraction)``, ``done(key)``, ``failed(message)``."""

    progress = Signal(float)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, model, ensure: Callable[..., Any], parent=None) -> None:
        """``ensure(model, progress)`` is :func:`infra.text_models.ensure` (injectable for tests)."""
        super().__init__(parent)
        self.model, self.ensure = model, ensure

    def run(self) -> None:  # noqa: D401
        """Thread body."""
        try:
            self.ensure(self.model, lambda stage, f, msg="": self.progress.emit(float(f)))
        except DatasetMakerError as exc:
            self.failed.emit(exc.user_message)
            return
        except Exception as exc:  # noqa: BLE001 - network / disk errors of any kind
            log.exception("text model download failed")
            self.failed.emit(str(exc))
            return
        self.done.emit(self.model.key)
