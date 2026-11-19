"""CPU cores and RAM of this computer (Qt-free).  psutil when present, else the operating system directly; never raises."""
from __future__ import annotations

import os
import sys
from typing import Optional


def physical_cores() -> int:
    """Physical CPU cores (hyper-threads not counted); at least 1."""
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:  # noqa: BLE001 - psutil missing or unsupported platform
        pass
    logical = os.cpu_count() or 2
    return max(1, logical // 2 if logical >= 4 else logical)     # no better source: assume 2 threads per core


def _meminfo() -> dict:
    out = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                key, _, rest = line.partition(":")
                out[key] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return out


def _windows_memory() -> Optional[tuple]:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):   # type: ignore[attr-defined]
        return int(st.ullTotalPhys), int(st.ullAvailPhys)
    return None


def memory() -> tuple:
    """``(total, available)`` physical RAM in bytes; ``(0, 0)`` if unknown."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:  # noqa: BLE001
        pass
    try:
        if sys.platform == "win32":
            got = _windows_memory()
            if got:
                return got
        info = _meminfo()
        if info:
            return info.get("MemTotal", 0), info.get("MemAvailable", info.get("MemFree", 0))
    except Exception:  # noqa: BLE001
        pass
    return 0, 0


def lower_thread_priority() -> bool:
    """Make the CALLING thread low-priority (background loading must not slow down the UI); False if not possible.

    Linux applies nice values per thread (the native thread id works with ``setpriority``); Windows has thread priorities.
    """
    try:
        if sys.platform == "win32":
            import ctypes

            k32 = ctypes.windll.kernel32                                # type: ignore[attr-defined]
            return bool(k32.SetThreadPriority(k32.GetCurrentThread(), -2))      # THREAD_PRIORITY_LOWEST
        import threading

        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
        return True
    except Exception:  # noqa: BLE001 - macOS without per-thread nice, missing permissions ...
        return False
