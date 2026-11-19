"""Qt background thread for backup / restore.  Interface to the UI: signals only."""
from __future__ import annotations

import logging
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal

from core.errors import CancelledByUser, DatasetMakerError
from core.events import CancelToken
from core.i18n import tr

log = logging.getLogger("voxprint.workers")


class BackupWorker(QThread):
    """Runs ``job(progress, cancel)`` (a backup or a restore) in the background.

    Signals: ``progress(fraction, message)``, ``done(Report)``, ``failed(message)``, ``cancelled()``."""

    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, job: Callable[..., Any], parent=None) -> None:
        """``job(progress, cancel)`` returns an :class:`infra.backup.Report`."""
        super().__init__(parent)
        self.job = job
        self.cancel_token = CancelToken()

    def cancel(self) -> None:
        """Stop after the current chunk; finished files stay (the next run continues)."""
        self.cancel_token.cancel()

    def run(self) -> None:  # noqa: D401
        """Thread body."""
        try:
            result = self.job(lambda f, m="": self.progress.emit(float(f), m), self.cancel_token)
        except CancelledByUser:
            self.cancelled.emit()
        except DatasetMakerError as exc:
            self.failed.emit(exc.user_message)
        except Exception as exc:  # noqa: BLE001
            log.exception("backup job crashed")
            self.failed.emit(tr("backup.err_io", path="", error=str(exc)))
        else:
            self.done.emit(result)
