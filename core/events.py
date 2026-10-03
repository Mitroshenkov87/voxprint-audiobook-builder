"""Pipeline stages, the progress callback protocol and the cancellation token (no Qt dependency).

Long-running code reports progress as ``progress(stage, fraction_inside_stage, message)``.  The GUI turns that into a
progress bar (``overall_percent``) and "chips" for the stages; the CLI prints it.  Cancellation is cooperative:
workers call ``CancelToken.check()`` at safe points.
"""
from __future__ import annotations

from core.i18n import tr
import enum
import threading
from typing import Callable, Dict, Iterable, List

from core.errors import CancelledByUser


class Stage(enum.Enum):
    """The user-visible phases of a task, in the order they usually run."""
    UPDATES = "updates"
    MODEL = "model"
    ALIGN = "align"
    SLICE = "slice"
    TRAIN = "train"
    SAVE = "save"

    @property
    def label(self) -> str:
        """Name of the stage in the UI language (key ``stage.<value>`` in ``locales/*.json``)."""
        return tr("stage." + self.value)


#: progress(stage, fraction_inside_stage 0..1, message)
ProgressCallback = Callable[[Stage, float, str], None]

#: Относительные «веса» этапов для общего процента.
STAGE_WEIGHTS: Dict[Stage, float] = {
    Stage.UPDATES: 2,
    Stage.MODEL: 8,
    Stage.ALIGN: 25,
    Stage.SLICE: 5,
    Stage.TRAIN: 55,
    Stage.SAVE: 5,
}


def noop_progress(stage: Stage, fraction: float, message: str = "") -> None:  # pragma: no cover
    """Progress callback that ignores everything (default argument for headless calls)."""
    return None


def overall_percent(plan: Iterable[Stage], stage: Stage, fraction: float) -> int:
    """Overall progress 0..100 for ``stage`` at ``fraction`` (0..1) of a task that runs the stages in ``plan``.

    Stages are weighted by ``STAGE_WEIGHTS`` (training dominates), so the bar moves roughly in proportion to wall time.
    """
    plan_list: List[Stage] = list(plan)
    if stage not in plan_list:
        return 0
    total = sum(STAGE_WEIGHTS[s] for s in plan_list)
    done = 0.0
    for s in plan_list:
        if s is stage:
            break
        done += STAGE_WEIGHTS[s]
    fraction = min(1.0, max(0.0, fraction))
    return int(round(100.0 * (done + STAGE_WEIGHTS[stage] * fraction) / total))


class CancelToken:
    """Thread-safe cancellation flag shared between the GUI thread and the worker."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation (idempotent)."""
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """True once ``cancel()`` was called."""
        return self._event.is_set()

    def check(self) -> None:
        """Raise :class:`CancelledByUser` if cancellation was requested; call this at safe points."""
        if self._event.is_set():
            raise CancelledByUser()
