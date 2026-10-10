"""Windows-specific code (everything is guarded by ``sys.platform``): OS version check, dark title bar, Acrylic.

Target OS: Windows 11 26H2 (build 26300) and newer builds of the same branch (24H2/25H2 = 26100+); the minimum is
26100.  On other platforms every function safely does nothing, so a platform module for Linux
(``infra/platform_linux.py``) with the same set of functions can be added later.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
from ctypes import wintypes
from dataclasses import dataclass
from typing import Optional

from core.i18n import tr

MIN_BUILD = 26100          # Windows 11 24H2 and newer (26H2 = 26300)

IS_WINDOWS = sys.platform == "win32"

# DWM constants - checked against the Microsoft Learn documentation (dwmapi.h, 2026-10-02):
#   DWMWA_USE_IMMERSIVE_DARK_MODE = 20 (Win10 20H1+/Win11), DWMWA_WINDOW_CORNER_PREFERENCE = 33,
#   DWMWA_SYSTEMBACKDROP_TYPE = 38 (Win11 build 22621+);
#   DWM_SYSTEMBACKDROP_TYPE: AUTO=0, NONE=1, MAINWINDOW=2 (Mica), TRANSIENTWINDOW=3 (Acrylic),
#   TABBEDWINDOW=4 (Mica Alt); DWM_WINDOW_CORNER_PREFERENCE: DEFAULT=0, DONOTROUND=1, ROUND=2, ROUNDSMALL=3.
# Only the native DWM through ctypes is used: GPL libraries (PyQt-Frameless-Window, PySide6-Fluent-Widgets) are avoided.
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWA_SYSTEMBACKDROP_TYPE = 38
DWMSBT_AUTO, DWMSBT_NONE, DWMSBT_MAINWINDOW, DWMSBT_TRANSIENTWINDOW, DWMSBT_TABBEDWINDOW = 0, 1, 2, 3, 4
DWMWCP_ROUND = 2


@dataclass(frozen=True)
class OsCheck:
    """Result of the OS check: ``ok`` flag, the detected build (None if unknown) and a user-facing message (empty when ok)."""
    ok: bool
    build: Optional[int]
    message: str


def windows_build() -> Optional[int]:
    """Return the Windows build number, or None on other platforms / when it cannot be determined."""
    if not IS_WINDOWS:
        return None
    try:
        return int(sys.getwindowsversion().build)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return None


def check_os(build: Optional[int] = None, is_windows: Optional[bool] = None) -> OsCheck:
    """Soft OS check: returns a message for the user and never raises."""
    win = IS_WINDOWS if is_windows is None else is_windows
    if not win:
        return OsCheck(False, None, tr("os.unsupported"))
    b = build if build is not None else windows_build()
    if b is None:
        return OsCheck(True, None, "")
    if b < MIN_BUILD:
        return OsCheck(False, b, tr("os.old_build", min=MIN_BUILD, build=b))
    return OsCheck(True, b, "")


def apply_backdrop(hwnd: int, dark: bool = True) -> str:
    """Enable Acrylic through ``DwmSetWindowAttribute``.  Returns ``'acrylic'`` or ``'plain'``.

    Supported Windows is 11 24H2 (build 26100) or newer, which already has the backdrop API.
    On any error the window stays plainly dark.
    """
    if not IS_WINDOWS:
        return "plain"
    try:
        import ctypes
        from ctypes import wintypes

        dwm = ctypes.windll.dwmapi  # type: ignore[attr-defined]
        h = wintypes.HWND(hwnd)

        def _set(attr: int, value: int) -> int:
            v = ctypes.c_int(value)
            return dwm.DwmSetWindowAttribute(h, ctypes.c_uint(attr), ctypes.byref(v), ctypes.sizeof(v))

        _set(DWMWA_USE_IMMERSIVE_DARK_MODE, 1 if dark else 0)
        _set(DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND)

        class MARGINS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_int), ("r", ctypes.c_int), ("t", ctypes.c_int), ("b", ctypes.c_int)]

        m = MARGINS(-1, -1, -1, -1)  # extend the "glass" frame over the whole client area
        dwm.DwmExtendFrameIntoClientArea(h, ctypes.byref(m))
        hr = _set(DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_TRANSIENTWINDOW)
        return "acrylic" if hr == 0 else "plain"
    except Exception:  # noqa: BLE001
        return "plain"


def documents_folder() -> Optional[str]:
    """The user's real Documents folder from the Known Folder API (``SHGetKnownFolderPath(FOLDERID_Documents)``): follows a
    redirect to OneDrive or another drive.  None off Windows or on any error."""
    if not IS_WINDOWS:
        return None
    try:
        import ctypes
        import uuid
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                        ("Data4", ctypes.c_ubyte * 8)]

        u = uuid.UUID("{FDD39AD0-238F-46AF-ADB4-6C85480369C7}")       # FOLDERID_Documents
        g = GUID(u.fields[0], u.fields[1], u.fields[2], (ctypes.c_ubyte * 8)(*u.bytes[8:]))
        out = ctypes.c_wchar_p()
        shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
        hr = shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(out))
        try:
            return out.value if hr == 0 and out.value else None
        finally:
            ctypes.windll.ole32.CoTaskMemFree(out)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------------------------- GPU process memory
#: Growth of this counter means CUDA allocations spilled into system RAM (see core/sysmem_spill.py).
_SHARED_COUNTER = r"\GPU Process Memory(pid_{pid}*)\Shared Usage"
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_PDH_FMT_DOUBLE = 0x00000200
_cached_shared_paths: dict = {}


class _PdhFmtDouble(ctypes.Structure):
    """``PDH_FMT_COUNTERVALUE`` laid out for ``PDH_FMT_DOUBLE`` (status, padding, double)."""
    _fields_ = [("CStatus", wintypes.DWORD), ("_pad", wintypes.DWORD), ("doubleValue", ctypes.c_double)]


def process_exists(pid: int) -> bool:
    """True if ``pid`` is a running process. False off Windows or when the process cannot be opened.

    ``os.kill(pid, 0)`` on Windows terminates the process, so the check goes through ``OpenProcess``.
    """
    if not IS_WINDOWS or pid <= 0:
        return False
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    except Exception:  # noqa: BLE001
        return False


def parse_typeperf_csv(text: str) -> Optional[int]:
    """Sum the numeric columns of the last ``typeperf`` data row (bytes). None when the text has no sample."""
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith(("exiting", "error")):
            continue
        if stripped.startswith('"') and "," in stripped:
            rows.append(stripped)
    if len(rows) < 2:
        return None
    total = 0.0
    found = False
    for part in _csv_fields(rows[-1])[1:]:
        try:
            total += float(part)
        except ValueError:
            continue
        found = True
    return int(total) if found else None


def _csv_fields(line: str) -> list:
    fields: list = []
    current: list = []
    quoted = False
    for ch in line:
        if ch == '"':
            quoted = not quoted
            continue
        if ch == "," and not quoted:
            fields.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    fields.append("".join(current).strip())
    return fields


def gpu_shared_usage_bytes(pid: int) -> Optional[int]:
    """Bytes of GPU shared (system) memory for ``pid``, or None. Best effort; off Windows always None.

    The counter is ``\\GPU Process Memory(pid_<pid>*)\\Shared Usage``. PDH is tried first (in process); ``typeperf``
    is the fallback. A missing counter or a failed read returns None and never raises.
    """
    if not IS_WINDOWS or pid <= 0:
        return None
    try:
        paths = _cached_shared_paths.get(pid)
        if not paths:
            paths = _discover_shared_counters(pid)
            if paths:
                _cached_shared_paths[pid] = paths
        if not paths:
            return None
        value = _pdh_sum(paths)
        if value is None:
            value = _typeperf_read(paths)
        return value
    except Exception:  # noqa: BLE001 - the counter is optional
        return None


def _discover_shared_counters(pid: int) -> list:
    wildcard = _SHARED_COUNTER.format(pid=int(pid))
    try:
        found = _pdh_expand(wildcard)
    except Exception:  # noqa: BLE001
        found = []
    if found:
        return found
    try:
        return _typeperf_list(pid)
    except Exception:  # noqa: BLE001
        return []


def _pdh_api():
    pdh = ctypes.WinDLL("pdh", use_last_error=True)
    pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    pdh.PdhOpenQueryW.restype = ctypes.c_long
    pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    pdh.PdhAddEnglishCounterW.restype = ctypes.c_long
    pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
    pdh.PdhCollectQueryData.restype = ctypes.c_long
    pdh.PdhGetFormattedCounterValue.argtypes = [
        ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_PdhFmtDouble)]
    pdh.PdhGetFormattedCounterValue.restype = ctypes.c_long
    pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
    pdh.PdhCloseQuery.restype = ctypes.c_long
    pdh.PdhExpandWildCardPathW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
    pdh.PdhExpandWildCardPathW.restype = ctypes.c_long
    return pdh


def _pdh_expand(wildcard: str) -> list:
    pdh = _pdh_api()
    needed = wintypes.DWORD(0)
    pdh.PdhExpandWildCardPathW(None, wildcard, None, ctypes.byref(needed), 0)
    if needed.value <= 1:
        return []
    buf = ctypes.create_unicode_buffer(needed.value)
    status = pdh.PdhExpandWildCardPathW(None, wildcard, buf, ctypes.byref(needed), 0)
    if status != 0:
        return []
    paths: list = []
    current: list = []
    for index in range(needed.value):
        ch = buf[index]
        if ch == "\x00":
            if not current:
                break
            paths.append("".join(current))
            current = []
        else:
            current.append(ch)
    return [p for p in paths if p.lower().endswith("shared usage")]


def _pdh_sum(paths: list) -> Optional[int]:
    if not paths:
        return None
    pdh = _pdh_api()
    query = ctypes.c_void_p()
    if pdh.PdhOpenQueryW(None, None, ctypes.byref(query)) != 0:
        return None
    try:
        counters = []
        for path in paths:
            counter = ctypes.c_void_p()
            if pdh.PdhAddEnglishCounterW(query, path, None, ctypes.byref(counter)) == 0:
                counters.append(counter)
        if not counters:
            return None
        pdh.PdhCollectQueryData(query)
        if pdh.PdhCollectQueryData(query) != 0:
            return None
        total = 0.0
        found = False
        for counter in counters:
            value = _PdhFmtDouble()
            if pdh.PdhGetFormattedCounterValue(counter, _PDH_FMT_DOUBLE, None, ctypes.byref(value)) == 0 and value.CStatus == 0:
                total += float(value.doubleValue)
                found = True
        return int(total) if found else None
    finally:
        pdh.PdhCloseQuery(query)


def _typeperf_list(pid: int) -> list:
    result = subprocess.run(
        ["typeperf", "-qx", r"\GPU Process Memory"], capture_output=True, text=True, timeout=8,
        creationflags=_CREATE_NO_WINDOW)
    needle = f"pid_{int(pid)}".lower()
    found = []
    for line in (result.stdout or "").splitlines():
        text = line.strip()
        low = text.lower()
        if needle in low and "shared usage" in low:
            found.append(text)
    return found


def _typeperf_read(paths: list) -> Optional[int]:
    if not paths:
        return None
    result = subprocess.run(
        ["typeperf", *paths, "-sc", "1"], capture_output=True, text=True, timeout=8, creationflags=_CREATE_NO_WINDOW)
    return parse_typeperf_csv(result.stdout or "")
