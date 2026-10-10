"""Cross-process GPU lock shared with Voxprint AI Movie Dubber.

One file, ``voxprint-gpu.lock``, in the system temp directory (``%TEMP%`` on Windows, :func:`tempfile.gettempdir`
elsewhere). The holder writes this JSON, atomically (a temp file in the same directory, then ``os.replace``)::

    {"owner": "audiobook-builder", "pid": 1234, "job": "Book", "started": "2026-10-10T04:23:00+00:00",
     "eta": "2026-10-10T08:23:00+00:00"}

``owner`` for this program is ``audiobook-builder``; the Movie Dubber uses its own owner string and the same keys.
``started`` and ``eta`` are ISO 8601 in UTC. The lock is held for the whole narration job and removed in ``finally``.
The same process may enter again (a nested job does not wait and does not delete the file early).

Another live process: wait and poll, and report ``GPU busy: <owner> <job>``. The file is stale, and may be taken, when
its pid is not running, or when ``eta`` is more than two hours in the past (a crashed holder whose pid was reused).
While this process holds the lock, ``eta`` is refreshed and kept at least two hours ahead, so a long batch is not
treated as abandoned.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

from infra.platform_win import process_exists

log = logging.getLogger("voxprint.gpu_lock")

OWNER = "audiobook-builder"
LOCK_NAME = "voxprint-gpu.lock"
#: A lock whose eta is this far in the past belongs to a job that is no longer running.
STALE_AFTER = timedelta(hours=2)
#: Refreshed eta never sits closer than this, so a batch with no progress events is not stolen mid-flight.
ETA_FLOOR = timedelta(hours=2)
POLL_S = 2.0

_guard = threading.Lock()
_depth = 0
_held: Optional["HeldLock"] = None


def lock_path() -> Path:
    """``%TEMP%\\voxprint-gpu.lock`` on Windows, the same name under the temp directory elsewhere."""
    return Path(tempfile.gettempdir()) / LOCK_NAME


def pid_alive(pid: int) -> bool:
    """True if ``pid`` is a running process. On Windows this does not use ``os.kill`` (that terminates the process)."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        return process_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    try:
        text = stat.read_text(encoding="utf-8", errors="replace")
        state = text.rsplit(")", 1)[-1].split()[0]
        return state != "Z"
    except OSError:
        return True


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _parse_eta(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        when = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc)


def _read(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "pid" not in data or "owner" not in data:
        return None
    return data


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _payload(job: str, started: str, eta: datetime) -> dict:
    return {"owner": OWNER, "pid": os.getpid(), "job": job, "started": started, "eta": _iso(eta)}


def is_stale(info: dict, now: datetime, alive: Callable[[int], bool] = pid_alive) -> bool:
    """True when the holder is gone, or its eta is more than two hours ago."""
    try:
        pid = int(info.get("pid", 0))
    except (TypeError, ValueError):
        return True
    if not alive(pid):
        return True
    eta = _parse_eta(info.get("eta"))
    if eta is None:
        return False
    return now > eta + STALE_AFTER


def _job_name(job: str) -> str:
    text = str(job or "").replace("\n", " ").replace("\r", " ").strip() or "audiobook"
    return text[:200]


class HeldLock:
    """The lock this process already owns. :meth:`note` refreshes ``eta``."""

    def __init__(self, path: Path, job: str, started: str, eta: datetime,
                 now: Callable[[], datetime]) -> None:
        """Remember the lock file, the job name and the clock this process owns."""
        self.path = path
        self.job = job
        self.started = started
        self.eta = eta
        self._now = now

    def note(self, seconds_left: Optional[float] = None) -> None:
        """Move ``eta`` to the estimated finish, and never closer than :data:`ETA_FLOOR`."""
        try:
            seconds = float(seconds_left) if seconds_left is not None else 0.0
        except (TypeError, ValueError):
            seconds = 0.0
        if seconds != seconds:  # NaN
            seconds = 0.0
        when = self._now() + max(ETA_FLOOR, timedelta(seconds=max(0.0, seconds)))
        current = _read(self.path)
        try:
            owner_pid = int(current.get("pid", -1)) if current else -1
        except (TypeError, ValueError):
            owner_pid = -1
        if owner_pid != os.getpid():
            return
        self.eta = when
        try:
            _atomic_write(self.path, _payload(self.job, self.started, when))
        except OSError:
            log.warning("could not refresh the GPU lock", exc_info=True)


def _release(path: Path) -> None:
    info = _read(path)
    try:
        pid = int(info.get("pid", -1)) if info else -1
    except (TypeError, ValueError):
        pid = -1
    if pid != os.getpid():
        return
    try:
        path.unlink()
    except OSError:
        log.warning("could not remove the GPU lock", exc_info=True)


def _sleep_poll(seconds: float, cancel: Optional[object], sleep: Callable[[float], None]) -> None:
    if seconds <= 0:
        sleep(0)
        return
    end = time.monotonic() + seconds
    while True:
        if cancel is not None:
            cancel.check()  # type: ignore[attr-defined]
        left = end - time.monotonic()
        if left <= 0:
            return
        sleep(min(0.2, left))


def _acquire(path: Path, job: str, on_busy: Optional[Callable[[str, str], None]], cancel: Optional[object],
             poll_s: float, now: Callable[[], datetime], sleep: Callable[[float], None],
             alive: Callable[[int], bool]) -> HeldLock:
    announced: Optional[tuple] = None
    while True:
        if cancel is not None:
            cancel.check()  # type: ignore[attr-defined]
        info = _read(path)
        moment = now()
        try:
            holder = int(info.get("pid", -1)) if info else -1
        except (TypeError, ValueError):
            holder = -1
        if info is None or holder == os.getpid() or is_stale(info, moment, alive):
            started = _iso(moment)
            eta = moment + ETA_FLOOR
            try:
                _atomic_write(path, _payload(job, started, eta))
            except OSError:
                log.warning("could not write the GPU lock", exc_info=True)
                _sleep_poll(poll_s, cancel, sleep)
                continue
            got = _read(path)
            try:
                got_pid = int(got.get("pid", -1)) if got else -1
            except (TypeError, ValueError):
                got_pid = -1
            if got_pid == os.getpid():
                return HeldLock(path, job, str((got or {}).get("started") or started), eta, now)
            continue
        owner = str(info.get("owner", "")).strip()
        other = str(info.get("job", "")).strip()
        key = (owner, other)
        if key != announced:
            log.info("GPU busy: %s %s", owner, other)
            announced = key
        if on_busy is not None:
            on_busy(owner, other)
        _sleep_poll(poll_s, cancel, sleep)


@contextmanager
def hold(job: str, on_busy: Optional[Callable[[str, str], None]] = None, cancel: Optional[object] = None, *,
         path: Optional[Path] = None, poll_s: float = POLL_S, now: Optional[Callable[[], datetime]] = None,
         sleep: Optional[Callable[[float], None]] = None,
         pid_alive_fn: Optional[Callable[[int], bool]] = None) -> Iterator[HeldLock]:
    """Hold the GPU lock until the block ends. Re-entrant in this process; always released by the outermost block."""
    global _depth, _held
    target = path if path is not None else lock_path()
    clock = now or (lambda: datetime.now(timezone.utc))
    sleeper = sleep or time.sleep
    alive = pid_alive_fn or pid_alive
    with _guard:
        _depth += 1
        nested = _depth > 1
        current = _held
    try:
        if nested:
            if current is None:
                raise RuntimeError("GPU lock depth is nested without a holder")
            yield current
        else:
            held = _acquire(target, _job_name(job), on_busy, cancel, poll_s, clock, sleeper, alive)
            with _guard:
                _held = held
            try:
                yield held
            finally:
                with _guard:
                    if _held is held:
                        _held = None
    finally:
        with _guard:
            _depth -= 1
            last = _depth == 0
        if last:
            _release(target)
