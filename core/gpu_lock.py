"""Cross-process GPU lock shared with the other Voxprint apps.

The file is the suite lock (:func:`voxprint_common.gpu_lock.lock_path`): ``%TEMP%/voxprint-gpu.lock``,
or ``VOXPRINT_GPU_LOCK``. Records use ``app`` and ``stage``. This program's app id is
``audiobook-builder``. :func:`hold` is the narration API; it waits, honours cancel, and
:meth:`HeldLock.note` keeps ``eta`` at least two hours ahead.

    with gpu_lock.hold(job_name, on_busy=on_busy, cancel=token) as held:
        held.note(seconds_left, stage=phase)
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

from voxprint_common.gpu_lock import GpuLockTimeout
from voxprint_common.gpu_lock import gpu_job
from voxprint_common.gpu_lock import lock_path as _suite_lock_path
from voxprint_common.gpu_lock import refresh

log = logging.getLogger("voxprint.gpu_lock")

OWNER = "audiobook-builder"
LOCK_NAME = "voxprint-gpu.lock"
#: A refreshed eta never sits closer than this, so a batch with no progress events is not stolen.
ETA_FLOOR_S = 7200.0

__all__ = ["GpuLockTimeout", "HeldLock", "OWNER", "hold", "lock_path"]


def lock_path() -> Path:
    """The suite GPU lock file. Tests replace this; :func:`hold` calls it when no path is passed."""
    return _suite_lock_path()


class HeldLock:
    """The lock this process already owns. :meth:`note` moves ``eta`` forward."""

    def __init__(self, path: Path, job: str) -> None:
        """Remember the lock file and the stage name this process acquired."""
        self.path = path
        self.job = job

    def note(self, seconds_left: Optional[float] = None, stage: Optional[str] = None) -> None:
        """Keep ``eta`` at least two hours ahead, or further when the job reports a longer estimate."""
        del stage  # the suite record keeps the stage from acquire; eta is what narration refreshes
        try:
            seconds = float(seconds_left) if seconds_left is not None else 0.0
        except (TypeError, ValueError):
            seconds = 0.0
        if seconds != seconds:  # NaN
            seconds = 0.0
        refresh(max(ETA_FLOOR_S, max(0.0, seconds)), path=self.path, app=OWNER)


def _stage_name(job: str) -> str:
    text = str(job or "").replace("\n", " ").replace("\r", " ").strip() or "audiobook"
    return text[:200]


@contextmanager
def hold(job: str, on_busy: Optional[Callable[[str, str], None]] = None, cancel: Optional[object] = None, *,
         path: Optional[Path] = None, poll_s: float = 0.25, timeout: Optional[float] = None,
         **_ignored: object) -> Iterator[HeldLock]:
    """Hold the GPU lock until the block ends. Re-entrant in this process; the outermost block releases it.

    ``cancel.check()`` runs while another app holds the lock. ``on_busy(app, stage)`` is the wait notice.
    ``path`` overrides the file; otherwise :func:`lock_path` (so a test can replace it).
    """
    target = path if path is not None else lock_path()
    stage = _stage_name(job)

    def _cancel() -> bool:
        if cancel is None:
            return False
        cancel.check()  # type: ignore[attr-defined]
        return False

    def _wait(holder: object, _waited: float) -> None:
        if on_busy is None:
            return
        on_busy(str(getattr(holder, "app", "")), str(getattr(holder, "stage", "")))

    with gpu_job(OWNER, stage, eta=ETA_FLOOR_S, timeout=timeout, poll=poll_s, path=target,
                 cancel=_cancel if cancel is not None else None, on_wait=_wait if on_busy is not None else None):
        yield HeldLock(target, stage)
