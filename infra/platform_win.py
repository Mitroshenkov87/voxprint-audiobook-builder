"""Windows-specific code (everything is guarded by ``sys.platform``): OS version check, dark title bar, Acrylic.

Target OS: Windows 11 26H2 (build 26300) and newer builds of the same branch (24H2/25H2 = 26100+); the minimum is
26100.  On other platforms every function safely does nothing, so a platform module for Linux
(``infra/platform_linux.py``) with the same set of functions can be added later.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

from core.i18n import tr

MIN_BUILD = 26100          # Windows 11 24H2 and newer (26H2 = 26300)
BACKDROP_MIN_BUILD = 22621  # DWMWA_SYSTEMBACKDROP_TYPE appeared in 22H2

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
    """Мягкая проверка ОС: сообщение для пользователя, исключений не бросает."""
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
    """Включает Acrylic через DwmSetWindowAttribute. Возвращает 'acrylic' или 'plain'.

    Без Windows 11 22H2+ (или при любой ошибке) возвращает 'plain' - окно останется просто тёмным.
    """
    if not IS_WINDOWS:
        return "plain"
    build = windows_build() or 0
    if build < BACKDROP_MIN_BUILD:
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
