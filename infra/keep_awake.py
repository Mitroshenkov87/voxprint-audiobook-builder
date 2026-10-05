"""Keep the computer from going to sleep while a long job runs (training, narration, translation).

On Windows this is ``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`` for the calling thread; the flag is
cleared again when the block ends (or the thread exits).  The display may still turn off.  Other systems: a no-op.
"""
from __future__ import annotations

import contextlib
import logging
import sys
from typing import Iterator

log = logging.getLogger("voxprint.keep_awake")

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _set(flags: int) -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except (AttributeError, OSError) as exc:  # pragma: no cover - Windows only
        log.debug("SetThreadExecutionState failed: %s", exc)
        return False


@contextlib.contextmanager
def keep_awake() -> Iterator[bool]:
    """Context manager; yields True if sleep is actually blocked."""
    active = _set(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield active
    finally:
        if active:
            _set(ES_CONTINUOUS)
