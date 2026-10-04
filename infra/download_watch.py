"""Watches a model download: real progress (MB done / total, speed) and a stall watchdog.

The download itself runs in a helper thread (:func:`run_watched`); the calling thread samples the size of the target folder
once a second.  If no byte arrives for ``STALL_SECONDS`` the source is abandoned (:class:`Stalled`), so that the caller can
switch to the next one - the partial files stay and are resumed.  Standard library only.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("voxprint.models")

#: No growth of the target folder for this long = the source is stalled (env ``VOXPRINT_STALL_SECONDS``).
STALL_SECONDS = float(os.environ.get("VOXPRINT_STALL_SECONDS", "45") or 45)
#: Seconds the helper thread gets to finish after an abort before the next source starts.
ABORT_GRACE = 10.0
#: Sampling interval of the watchdog / progress line (seconds).
POLL = 1.0


class Stalled(OSError):
    """No data arrived for too long; the partial files are kept for the next source / the next run."""


def dir_bytes(folder: Path) -> int:
    """Total size of all files below ``folder`` (``.incomplete`` files of huggingface_hub included); 0 if it is missing."""
    total = 0
    stack = [str(folder)]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        else:
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def fmt_bytes(n: float) -> str:
    """``1.4 GB`` / ``230 MB`` / ``12 KB``."""
    n = float(max(0.0, n))
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.0f} MB"
    return f"{n / 1024:.0f} KB"


class Meter:
    """Bytes done, smoothed speed and the source label of one download."""

    def __init__(self, source: str, total: Callable[[], int], clock: Callable[[], float] = time.monotonic) -> None:
        self.source = source
        self.total_fn = total
        self.clock = clock
        self.done = 0
        self.speed = 0.0                    # bytes per second, smoothed over about 8 s
        self._last: Optional[tuple] = None
        self.fraction = 0.0                 # the source's own progress report (0..1)

    def sample(self, done: int) -> None:
        now = self.clock()
        if self._last is not None:
            t0, d0 = self._last
            dt = now - t0
            if dt > 0:
                inst = max(0, done - d0) / dt
                alpha = min(1.0, dt / 8.0)
                self.speed = inst if self.speed == 0 else self.speed + alpha * (inst - self.speed)
        self._last = (now, done)
        self.done = done

    @property
    def total(self) -> int:
        try:
            return int(self.total_fn() or 0)
        except Exception:  # noqa: BLE001
            return 0

    def text(self, tr: Callable[..., str], short: str, plain_pct: int) -> str:
        """The status line: detailed once something was measured, otherwise the plain percentage line."""
        if self.done <= 0:
            return tr("progress.downloading", short=short, pct=plain_pct)
        total = self.total
        speed = f"{fmt_bytes(self.speed)}/s" if self.speed > 0 else "..."
        if total > 0:
            pct = max(plain_pct, min(99, int(self.done * 100 / total)))
            return tr("progress.detail", short=short, pct=pct, done=fmt_bytes(self.done), total=fmt_bytes(total),
                      speed=speed, source=self.source)
        return tr("progress.detail_unknown", short=short, done=fmt_bytes(self.done), speed=speed, source=self.source)


def run_watched(fn: Callable[[], Any], folder: Path, meter: Meter, cancel: threading.Event,
                on_tick: Optional[Callable[[Meter], None]] = None, stall: Optional[float] = None, poll: Optional[float] = None,
                clock: Callable[[], float] = time.monotonic, grace: Optional[float] = None, log_every: float = 15.0) -> Any:
    """Run ``fn`` in a helper thread and watch ``folder``.  Returns its result, re-raises its exception, or raises
    :class:`Stalled` (after setting ``cancel`` so that the helper stops at its next progress call)."""
    stall = STALL_SECONDS if stall is None else stall
    grace = ABORT_GRACE if grace is None else grace
    poll = POLL if poll is None else poll
    box: dict = {}

    def target() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - handed to the caller's thread
            box["error"] = exc

    th = threading.Thread(target=target, daemon=True, name="vx-model-download")
    th.start()
    base = dir_bytes(folder)
    last_size, last_move, last_log = base, clock(), clock()
    meter.sample(base)
    while True:
        th.join(poll)
        if not th.is_alive():
            break
        size = dir_bytes(folder)
        now = clock()
        if size != last_size:
            last_size, last_move = size, now
        meter.sample(size)
        if on_tick is not None:
            try:
                on_tick(meter)
            except Exception:  # noqa: BLE001
                pass
        if now - last_log >= log_every:
            last_log = now
            log.info("download via %s: %s%s, %s/s", meter.source, fmt_bytes(meter.done),
                     f" of {fmt_bytes(meter.total)}" if meter.total else "", fmt_bytes(meter.speed))
        if now - last_move >= stall:
            cancel.set()
            th.join(grace)
            raise Stalled(f"no data from {meter.source} for {int(stall)} s")
    if "error" in box:
        raise box["error"]
    return box.get("result")
