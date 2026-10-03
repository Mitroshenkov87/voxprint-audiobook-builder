"""Qt background thread for "Maximum quality (auto)": downloads the models the automatic steps need."""
from __future__ import annotations

import logging
from typing import Any, Callable, List

from PySide6.QtCore import QThread, Signal

from core.errors import DatasetMakerError

log = logging.getLogger("voxprint.workers")


class AutoQualityWorker(QThread):
    """Runs ``job(needs, progress)`` (default :func:`infra.auto_steps.download_models`) for the missing models.

    Signals: ``progress(fraction, message)``, ``done(count)``, ``failed(message)``."""

    progress = Signal(float, str)
    done = Signal(int)
    failed = Signal(str)

    def __init__(self, needs: List[Any], job: Callable[..., int], parent=None) -> None:
        """``needs`` is a list of :class:`infra.auto_steps.ModelNeed`."""
        super().__init__(parent)
        self.needs, self.job = needs, job

    def run(self) -> None:  # noqa: D401
        """Thread body."""
        try:
            n = self.job(self.needs, lambda f, m="": self.progress.emit(float(f), m))
        except DatasetMakerError as exc:
            self.failed.emit(exc.user_message)
            return
        except Exception as exc:  # noqa: BLE001 - network / disk errors of any kind
            log.exception("maximum quality: model download failed")
            self.failed.emit(str(exc))
            return
        self.done.emit(int(n))
