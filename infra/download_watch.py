"""Watches a model download: real progress (MB done / total, speed) and a stall watchdog.

The download itself runs in a helper thread (:func:`run_watched`); the calling thread samples the size of the target folder
once a second.  If no byte arrives for ``STALL_SECONDS`` the source is abandoned (:class:`Stalled`), so that the caller can
switch to the next one - the partial files stay and are resumed.  Standard library only.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

log = logging.getLogger("voxprint.models")

#: No growth of the target folder for this long = the source is stalled (env ``VOXPRINT_STALL_SECONDS``).
STALL_SECONDS = float(os.environ.get("VOXPRINT_STALL_SECONDS", "45") or 45)
#: Seconds the helper thread gets to finish after an abort before the next source starts.
ABORT_GRACE = 10.0
#: Sampling interval of the watchdog / progress line (seconds).
POLL = 1.0
#: Longest wait for an abandoned helper thread to let go of the partial files before the next download starts in the same
#: folder (it may sit in a socket read: read timeout 30 s + margin).
RELEASE_WAIT = 45.0

_abandoned: Dict[str, threading.Thread] = {}      # folder -> helper thread that was given up on while still running
_abandoned_lock = threading.Lock()


class Stalled(OSError):
    """No data arrived for too long; the partial files are kept for the next source / the next run.

    ``progressed`` tells whether the abandoned attempt had received data before it stalled (worth resuming the same source)."""

    def __init__(self, *args: Any, progressed: bool = False) -> None:
        """Store the OS error and whether any bytes had arrived before the stall."""
        super().__init__(*args)
        self.progressed = progressed


def _key(folder: Path) -> str:
    return os.path.normcase(os.path.abspath(str(folder)))


def _abandon(folder: Path, th: threading.Thread) -> None:
    """Remember a helper thread that is still running after the abort grace (it keeps files open, e.g. on Windows)."""
    if th.is_alive():
        log.info("the abandoned download thread is still running (blocked in a read); the next download waits for it")
        with _abandoned_lock:
            _abandoned[_key(folder)] = th


def busy(folder: Path) -> bool:
    """True while a helper thread that was given up on still runs for ``folder`` (it may hold files open there, e.g. the
    lock files of huggingface_hub in ``.cache``)."""
    with _abandoned_lock:
        th = _abandoned.get(_key(folder))
    return th is not None and th.is_alive()


def wait_released(folder: Path, timeout: Optional[float] = None) -> bool:
    """Wait until an abandoned helper thread of ``folder`` has ended (two downloads must never write the same partial
    files: Windows refuses with WinError 32 / 5).  Returns False if it is still running after ``timeout``."""
    key = _key(folder)
    with _abandoned_lock:
        th = _abandoned.get(key)
    if th is None:
        return True
    if th.is_alive():
        log.info("waiting for the previous download to release files in %s", Path(folder).name)
        th.join(RELEASE_WAIT if timeout is None else timeout)
    if th.is_alive():
        log.warning("the previous download of %s is still running - continuing anyway", Path(folder).name)
        return False
    with _abandoned_lock:
        if _abandoned.get(key) is th:
            del _abandoned[key]
    return True


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


_PARTS = ".incomplete.parts"
_OURS_INCOMPLETE = ".incomplete"


def progress_bytes(folder: Path) -> int:
    """Bytes of the download in ``folder`` for the progress line, each file counted ONCE.

    :func:`dir_bytes` counts everything (right for the stall watchdog: any growth is activity), but the same file can be on
    disk several times while it is assembled or re-fetched: the multi-connection parts (``X.incomplete.parts/``), the joined
    ``X.incomplete`` and the finished ``X``; the temporary ``*.assembling`` copy; the lock and metadata files of
    huggingface_hub.  Here a finished file wins over its ``.incomplete`` and parts, ``*.assembling`` is ignored and only the
    ``*.incomplete`` downloads of the hub's ``.cache`` count."""
    total = 0
    stack = [(str(folder), False)]
    while stack:
        cur, in_cache = stack.pop()
        try:
            with os.scandir(cur) as it:
                entries = list(it)
        except OSError:
            continue
        names = {e.name for e in entries}
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    if e.name.endswith(_PARTS):
                        base = e.name[: -len(_PARTS)]
                        if base in names or base + _OURS_INCOMPLETE in names:
                            continue                     # already joined / finished: the parts are a duplicate
                    stack.append((e.path, in_cache or e.name == ".cache"))
                    continue
                n = e.name
                if n.endswith(".assembling"):
                    continue
                if in_cache:
                    if n.endswith(_OURS_INCOMPLETE):     # the hub's running download (lock / metadata files are not data)
                        total += e.stat(follow_symlinks=False).st_size
                    continue
                if n.endswith(_OURS_INCOMPLETE) and n[: -len(_OURS_INCOMPLETE)] in names:
                    continue                             # finished file next to a stale partial copy
                total += e.stat(follow_symlinks=False).st_size
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


def fmt_duration(seconds: float) -> str:
    """Coarse remaining time that does not flicker: ``45 s`` / ``3 min 10 s`` / ``1 h 20 min`` (rounded to 5 s / 10 s / 5 min)."""
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{max(5, int(round(s / 5.0)) * 5)} s"
    if s < 3600:
        s = int(round(s / 10.0)) * 10
        m, sec = divmod(s, 60)
        return f"{m} min {sec} s" if sec else f"{m} min"
    s = int(round(s / 300.0)) * 300
    h, rest = divmod(s, 3600)
    return f"{h} h {rest // 60} min" if rest else f"{h} h"


class Meter:
    """Bytes done, smoothed speed, remaining time and the source label of ONE model download (all its sources).

    * ``done`` never decreases (a source switch or a retry that re-reads a file must not move the counter backwards);
    * ``total`` is the size of the whole download known up front (:meth:`set_total`; it is not re-estimated while
      downloading, so "N MB of TOTAL" does not jump); without it the line shows only what was downloaded;
    * the speed is an exponential moving average (time constant about ``TAU`` seconds) and the remaining time is taken
      from it, shown only after ``ETA_AFTER`` seconds of measurement and kept steady (it counts down between updates and
      is replaced only when the new estimate differs clearly).
    """

    TAU = 12.0
    ETA_AFTER = 6.0

    def __init__(self, source: str, total: Optional[Callable[[], int]] = None, clock: Callable[[], float] = time.monotonic) -> None:
        """Watch one download. ``total`` reports the expected size when it is not fixed yet."""
        self.source = source
        self.total_fn = total
        self.clock = clock
        self.done = 0
        self.speed = 0.0                    # bytes per second, moving average
        self._last: Optional[tuple] = None
        self._t_first: Optional[float] = None
        self._fixed_total = 0
        self._eta: Optional[tuple] = None   # (shown seconds, clock time of that value)
        self.fraction = 0.0                 # the source's own progress report (0..1)

    def set_source(self, source: str) -> None:
        """A new source continues the same download: counter, speed and total are kept."""
        self.source = source

    def set_total(self, total: int) -> None:
        """Total size of the whole download (sum of the file sizes known up front); the largest value given wins."""
        if total and int(total) > self._fixed_total:
            self._fixed_total = int(total)

    def sample(self, done: int) -> None:
        """Record that ``done`` bytes are on disk and update the speed."""
        now = self.clock()
        done = max(self.done, int(done))                    # monotonic
        if self._t_first is None:
            self._t_first = now
        if self._last is not None:
            t0, d0 = self._last
            dt = now - t0
            if dt > 0:
                inst = max(0, done - d0) / dt
                alpha = 1.0 - math.exp(-dt / self.TAU)
                self.speed = inst if self.speed == 0 else self.speed + alpha * (inst - self.speed)
        self._last = (now, done)
        self.done = done

    @property
    def total(self) -> int:
        """The expected size in bytes, or 0 when it is not known yet."""
        if self._fixed_total:
            return self._fixed_total
        if self.total_fn is None:
            return 0
        try:
            return int(self.total_fn() or 0)
        except Exception:  # noqa: BLE001
            return 0

    def eta(self) -> Optional[float]:
        """Seconds left (steady value) or None while it cannot be told yet."""
        total, now = self.total, self.clock()
        if total <= 0 or self.speed <= 1024 or self._t_first is None or now - self._t_first < self.ETA_AFTER or self.done >= total:
            return None
        new = (total - self.done) / self.speed
        if self._eta is not None:
            shown = max(0.0, self._eta[0] - (now - self._eta[1]))     # counts down by itself
            if abs(new - shown) <= max(10.0, 0.25 * new):
                self._eta = (shown, now)
                return shown
        self._eta = (new, now)
        return new

    def text(self, tr: Callable[..., str], short: str, plain_pct: int) -> str:
        """The status line: detailed once something was measured, otherwise the plain percentage line."""
        if self.done <= 0:
            return tr("progress.downloading", short=short, pct=plain_pct)
        total = self.total
        speed = f"{fmt_bytes(self.speed)}/s" if self.speed > 0 else "..."
        if total > 0:
            done = min(self.done, total)
            pct = max(plain_pct, min(99, int(done * 100 / total)))
            line = tr("progress.detail", short=short, pct=pct, done=fmt_bytes(done), total=fmt_bytes(total),
                      speed=speed, source=self.source)
            eta = self.eta()
            if eta is not None:
                line += tr("progress.eta", eta=fmt_duration(eta))
            return line
        return tr("progress.detail_unknown", short=short, done=fmt_bytes(self.done), speed=speed, source=self.source)


def run_watched(fn: Callable[[], Any], folder: Path, meter: Meter, cancel: threading.Event,
                on_tick: Optional[Callable[[Meter], None]] = None, idle_ok: Optional[Callable[[], bool]] = None, stall: Optional[float] = None, poll: Optional[float] = None,
                clock: Callable[[], float] = time.monotonic, grace: Optional[float] = None, log_every: float = 15.0) -> Any:
    """Run ``fn`` in a helper thread and watch ``folder``.  Returns its result, re-raises its exception, or raises
    :class:`Stalled` (after setting ``cancel`` so that the helper stops at its next progress call)."""
    stall = STALL_SECONDS if stall is None else stall
    grace = ABORT_GRACE if grace is None else grace
    poll = POLL if poll is None else poll
    box: dict = {}
    wait_released(folder)

    def target() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - handed to the caller's thread
            box["error"] = exc

    th = threading.Thread(target=target, daemon=True, name="vx-model-download")
    base = dir_bytes(folder)                  # before the helper runs, so its first write still counts as progress
    th.start()
    last_size, last_move, last_log = base, clock(), clock()
    meter.sample(progress_bytes(folder))      # each file once (progress line)
    while True:
        th.join(poll)
        if not th.is_alive():
            break
        size = dir_bytes(folder)
        now = clock()
        if size != last_size:
            last_size, last_move = size, now
        meter.sample(progress_bytes(folder))
        if on_tick is not None:
            try:
                on_tick(meter)
            except Exception:  # noqa: BLE001
                pass
        if now - last_log >= log_every:
            last_log = now
            total = meter.total
            log.info("download via %s: %s%s, %s/s", meter.source, fmt_bytes(min(meter.done, total) if total else meter.done),
                     f" of {fmt_bytes(total)}" if total else "", fmt_bytes(meter.speed))
        if now - last_move >= stall:
            if idle_ok is not None and idle_ok():
                # nothing is left to download (everything is on disk): that is not a stall - finish with what we have
                log.info("download via %s: all files are present, finishing", meter.source)
                cancel.set()
                th.join(grace)
                _abandon(folder, th)
                return box.get("result")
            cancel.set()
            th.join(grace)
            _abandon(folder, th)
            raise Stalled(f"no data from {meter.source} for {int(stall)} s", progressed=last_size > base)
    if "error" in box:
        raise box["error"]
    return box.get("result")
