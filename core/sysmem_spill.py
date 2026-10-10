"""Windows guard against CUDA silently spilling into system memory.

After the voice model has loaded, a background thread samples the process's GPU shared memory
(``\\GPU Process Memory(pid_<pid>*)\\Shared Usage``, PDH with a typeperf fallback; see
:func:`infra.platform_win.gpu_shared_usage_bytes`). The first sample is the baseline. If that counter later grows by
more than 256 MB, narration treats it like an out-of-memory error: it empties the CUDA cache and halves the batch.

The sampler is best-effort and off the GPU thread. Off Windows, or when the counter cannot be read, it stays quiet.
The diagnostic text names the NVIDIA Control Panel switch that stops the driver from using system memory in the first
place.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Callable, Optional

from infra.platform_win import gpu_shared_usage_bytes

log = logging.getLogger("voxprint.sysmem")

#: Growth over the post-load baseline that is treated as running out of video memory.
SPILL_BYTES = 256 * 1024 * 1024
SAMPLE_EVERY_S = 5.0

DIAGNOSTIC_NOTE = (
    "NVIDIA Control Panel: Manage 3D settings, set 'CUDA - Sysmem Fallback Policy' to "
    "'Prefer No Sysmem Fallback'. Otherwise the driver may place CUDA allocations in system RAM "
    "(GPU shared memory) and narration slows down as if the card had run out of video memory."
)


def _sample_shared() -> Optional[int]:
    if sys.platform != "win32":
        return None
    return gpu_shared_usage_bytes(os.getpid())


class Monitor:
    """Background shared-memory sampler. :meth:`note` is what the thread (and the tests) record."""

    def __init__(self, sample: Callable[[], Optional[int]] = _sample_shared, interval: float = SAMPLE_EVERY_S) -> None:
        self._sample = sample
        self.interval = interval
        self.baseline: Optional[int] = None
        self.latest: Optional[int] = None
        self._reacted = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "Monitor":
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="voxprint-sysmem", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)

    def note(self, value: Optional[int]) -> None:
        """Record one reading. The first reading is the baseline taken after the model load."""
        if value is None:
            return
        with self._lock:
            reading = int(value)
            if self.baseline is None:
                self.baseline = reading
            self.latest = reading

    def growth(self) -> int:
        """Bytes of shared GPU memory above the post-load baseline (0 until both readings exist)."""
        with self._lock:
            if self.baseline is None or self.latest is None:
                return 0
            return max(0, self.latest - self.baseline)

    def tripped(self) -> bool:
        """True when growth has crossed another 256 MB step that narration has not reacted to yet."""
        grown = self.growth()
        return grown > SPILL_BYTES and grown > self._reacted

    def mark(self) -> None:
        """Remember that the current growth was already handled (so the batch is not halved every group)."""
        self._reacted = max(self._reacted, self.growth())

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.note(self._sample())
            except Exception:  # noqa: BLE001 - a failed counter must not kill the sampler
                log.debug("GPU shared-memory sample failed", exc_info=True)
            if self._stop.wait(self.interval):
                return


def start_after_load() -> Monitor:
    """A sampler whose baseline is the first reading after the model is on the GPU. Inert off Windows."""
    monitor = Monitor()
    if sys.platform == "win32":
        monitor.start()
    return monitor
