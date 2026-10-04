"""Closing the main window ends *everything* at once.

Workers are threads in the main process, but they start helper processes (ffmpeg, pip/uv, a second Voxprint for the
background quality steps ...) and a download thread can sit in a socket read for minutes; waiting for all of that left
"zombie" Voxprint processes that still held model folders open (Windows then refused to rename them).  So when the user closes the
main window with X we kill the whole process tree and leave:

* :func:`arm` is called once by ``main.py`` (never in tests / self-checks, where closing a window must not kill the runner);
* :func:`fire` kills all descendants (psutil; ``taskkill /T`` or ``/proc`` when psutil is absent) and ends this process
  without waiting for threads;
* on Windows :func:`arm` also puts the process in a *job object* with KILL_ON_JOB_CLOSE, so even a crash or a kill of the
  main process from the task manager takes every helper with it.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from typing import List

log = logging.getLogger("voxprint.exit")

_armed = False
_job = None            # keeps the Windows job handle alive for the lifetime of the process


def armed() -> bool:
    return _armed


def arm() -> None:
    """Enable :func:`fire` (and the Windows kill-on-close job)."""
    global _armed
    _armed = True
    if sys.platform == "win32":
        _assign_kill_on_close_job()


def _assign_kill_on_close_job() -> None:
    global _job
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class _Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class _IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("a", "b", "c", "d", "e", "f")]

        class _Ext(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _Basic), ("IoInfo", _IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        job = k32.CreateJobObjectW(None, None)
        info = _Ext()
        info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not job or not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject")
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject")   # e.g. already in a job on old Windows
        _job = job
    except Exception as exc:  # noqa: BLE001 - only a safety net; fire() still kills the tree
        log.info("job object not available: %s", exc)


def descendants(pid: int = 0) -> List[int]:
    """PIDs of all child / grandchild ... processes of ``pid`` (default: this process)."""
    pid = pid or os.getpid()
    try:
        import psutil

        return [c.pid for c in psutil.Process(pid).children(recursive=True)]
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        log.debug("psutil: %s", exc)
    out: List[int] = []
    if sys.platform.startswith("linux"):             # /proc fallback
        parent_of = {}
        for name in os.listdir("/proc"):
            if name.isdigit():
                try:
                    with open(f"/proc/{name}/stat", "rb") as fh:
                        parent_of[int(name)] = int(fh.read().rsplit(b")", 1)[1].split()[1])
                except (OSError, ValueError, IndexError):
                    continue
        todo = [pid]
        while todo:
            cur = todo.pop()
            for child, parent in parent_of.items():
                if parent == cur and child not in out:
                    out.append(child)
                    todo.append(child)
    return out


def kill_descendants(pid: int = 0) -> List[int]:
    """Kill every descendant of ``pid`` (children first so none can respawn through its parent); return the PIDs."""
    pids = descendants(pid)
    for p in reversed(pids):
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p)], capture_output=True, timeout=10,
                               creationflags=0x08000000)   # CREATE_NO_WINDOW
            else:
                os.kill(p, signal.SIGKILL)
        except Exception:  # noqa: BLE001 - already gone
            pass
    return pids


def fire(code: int = 0) -> None:
    """Kill all helper processes and end this process now, without waiting for threads.  No-op unless :func:`arm` was called."""
    if not _armed:
        return
    exit_now(code)


def exit_now(code: int = 0) -> None:
    try:
        killed = kill_descendants()
        log.info("window closed - ended %d helper process(es), exiting", len(killed))
    except Exception:  # noqa: BLE001
        pass
    try:
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)                                   # (on Windows the kill-on-close job takes whatever was missed)


def wait_gone(pids: List[int], timeout: float = 5.0) -> bool:
    """Test helper: True when none of ``pids`` exists any more."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        alive = []
        for p in pids:
            try:
                os.kill(p, 0)
                if sys.platform.startswith("linux"):                 # a killed child nobody has reaped yet is a zombie: gone
                    with open(f"/proc/{p}/stat", "rb") as fh:
                        if fh.read().rsplit(b")", 1)[1].split()[0] == b"Z":
                            continue
                alive.append(p)
            except OSError:
                pass
        if not alive:
            return True
        time.sleep(0.05)
    return False
