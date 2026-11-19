"""Qt background thread for Settings -> "Auto-repair" (:func:`infra.auto_repair.run`)."""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal

from core.errors import CancelledByUser
from core.events import CancelToken

log = logging.getLogger("voxprint.workers")


class AutoRepairWorker(QThread):
    """Runs ``job(progress, cancel)`` (default :func:`infra.auto_repair.run`).

    Signals: ``progress(fraction, message)``, ``done(report)``, ``failed(message)``, ``cancelled()``."""

    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, job: Optional[Callable[..., Any]] = None, parent=None) -> None:
        """``job`` is injectable (tests)."""
        super().__init__(parent)
        self.job = job
        self.cancel_token = CancelToken()

    def cancel(self) -> None:
        """Stop after the current file / item."""
        self.cancel_token.cancel()

    def run(self) -> None:  # noqa: D401
        """Thread body."""
        job = self.job
        if job is None:
            from infra import auto_repair

            job = auto_repair.run

        def report(fraction: float, message: str = "") -> None:
            self.progress.emit(float(fraction), str(message))

        try:
            rep = job(report, self.cancel_token)
        except CancelledByUser:
            self.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001 - shown to the user, details in the log
            log.exception("auto-repair failed")
            self.failed.emit(getattr(exc, "user_message", "") or str(exc))
            return
        self.done.emit(rep)
