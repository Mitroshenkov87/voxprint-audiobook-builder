"""Этапы работы, колбэки прогресса и токен отмены (без зависимостей от Qt)."""
from __future__ import annotations

from core.i18n import tr
import enum
import threading
from typing import Callable, Dict, Iterable, List

from core.errors import CancelledByUser


class Stage(enum.Enum):
    UPDATES = "updates"
    MODEL = "model"
    ALIGN = "align"
    SLICE = "slice"
    TRAIN = "train"
    SAVE = "save"

    @property
    def label(self) -> str:
        """Название этапа на языке интерфейса (ключ stage.<значение> в locales/*.json)."""
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
    return None


def overall_percent(plan: Iterable[Stage], stage: Stage, fraction: float) -> int:
    """Общий процент 0..100 для заданного плана этапов."""
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
    """Потокобезопасный флаг отмены."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise CancelledByUser()
