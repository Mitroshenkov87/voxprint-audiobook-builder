"""Automatic GPU cooling during a long narration. There is no user setting.

The first 2.5 hours of a job run at full speed. After that, if ``nvidia-smi`` reports ``temperature.gpu`` staying at
or above 83 °C — the median of the samples from the last five minutes — narration pauses between batch groups until
the median is back at or below 75 °C (:func:`voxprint_common.thermal.wait_until_ready`). The progress line says
"Cooling GPU".

``nvidia-smi`` and the clock are injectable so tests never call the real tool or wait out the pause.
"""
from __future__ import annotations

import logging
import shutil
import statistics
import subprocess
import threading
import time
from typing import Callable, List, Optional, Tuple

from voxprint_common.thermal import FULL_SPEED_S, PAUSE_C, RESUME_C, Action, decide, wait_until_ready

log = logging.getLogger("voxprint.gpu_thermal")

#: Sustained temperature at or above this (degrees C) starts a cooling pause. The suite rule is 83 °C.
HOT_C = PAUSE_C
#: Samples older than this are ignored when judging "sustained".
WINDOW_S = 5 * 60
#: A single hot reading is not "sustained"; the median needs a few samples in the window.
MIN_SAMPLES = 3
SAMPLE_EVERY_S = 30.0


def query_temperature(index: int = 0, run: Optional[Callable] = None) -> Optional[float]:
    """``temperature.gpu`` from ``nvidia-smi`` for GPU ``index``, or None when the tool is missing or unreadable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    cmd = [exe, "-i", str(int(index)), "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]
    try:
        result = (run or subprocess.run)(
            cmd, capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(result, "returncode", 1) != 0:
        return None
    line = (getattr(result, "stdout", "") or "").strip().splitlines()
    if not line:
        return None
    try:
        return float(line[0].strip().split(",")[0])
    except ValueError:
        return None


class Monitor:
    """Samples GPU temperature on a side thread and decides when a batch boundary should pause."""

    def __init__(self, started: float, now: Callable[[], float] = time.monotonic,
                 read_temp: Optional[Callable[[], Optional[float]]] = None,
                 sleep: Callable[[float], None] = time.sleep, gpu_index: int = 0) -> None:
        """Remember the job start time, the clock and the temperature reader."""
        self.started = float(started)
        self.now = now
        self.sleep = sleep
        self.gpu_index = int(gpu_index)
        self._read = read_temp
        self.samples: List[Tuple[float, float]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "Monitor":
        """Start the sampler. With no injected reader and no ``nvidia-smi``, this does nothing."""
        if self._read is None and shutil.which("nvidia-smi") is None:
            return self
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="voxprint-gpu-temp", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the sampler thread and wait for it to finish."""
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)

    def retarget(self, device: object) -> None:
        """Follow the engine's ``cuda:N`` (samples from the previous GPU are dropped)."""
        text = str(device or "")
        index = 0
        if text.startswith("cuda:"):
            try:
                index = int(text.split(":", 1)[1])
            except ValueError:
                index = 0
        if index == self.gpu_index:
            return
        self.gpu_index = index
        with self._lock:
            self.samples.clear()

    def observe(self, temp: float, at: Optional[float] = None) -> None:
        """Record one sample (the sampler thread, or a test)."""
        stamp = self.now() if at is None else float(at)
        with self._lock:
            self.samples.append((stamp, float(temp)))
            cutoff = stamp - 15 * 60
            self.samples = [item for item in self.samples if item[0] >= cutoff]

    def median_c(self) -> Optional[float]:
        """Median °C of the samples in the last five minutes, or None when there are fewer than :data:`MIN_SAMPLES`."""
        moment = self.now()
        with self._lock:
            window = [temp for stamp, temp in self.samples if moment - WINDOW_S <= stamp <= moment + 1e-6]
        if len(window) < MIN_SAMPLES:
            return None
        return float(statistics.median(window))

    def should_cool(self) -> bool:
        """True after 2.5 h when the recent median is at least 83 °C."""
        decision = decide(self.now() - self.started, self.median_c(), 0.0, paused=False)
        return decision.action is Action.PAUSE

    def cool(self, cancel: Optional[object] = None, pause: Optional[object] = None) -> None:
        """Wait until the GPU is at or below 75 °C. Cancel and pause are still honoured."""
        if cancel is not None:
            cancel.check()  # type: ignore[attr-defined]
        log.info("Cooling GPU until it is at or below %.0f C", RESUME_C)

        def sample() -> tuple:
            return self.median_c(), 0.0

        def sleeper(seconds: float) -> None:
            end = self.now() + seconds
            while True:
                if cancel is not None:
                    cancel.check()  # type: ignore[attr-defined]
                if pause is not None:
                    pause.wait(cancel)  # type: ignore[attr-defined]
                left = end - self.now()
                if left <= 0:
                    return
                self.sleep(min(0.25, left))

        wait_until_ready(sample, started=self.started, paused=True, sleep=sleeper, now=self.now, interval=5.0)

    def _read_temp(self) -> Optional[float]:
        if self._read is not None:
            return self._read()
        return query_temperature(self.gpu_index)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                temp = self._read_temp()
            except Exception:  # noqa: BLE001 - a failed probe must not kill the sampler
                log.debug("GPU temperature read failed", exc_info=True)
                temp = None
            if temp is not None:
                self.observe(temp)
            if self._stop.wait(SAMPLE_EVERY_S):
                return
