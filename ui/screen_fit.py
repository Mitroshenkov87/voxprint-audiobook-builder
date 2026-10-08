"""Window geometry from the screen's work area (``QScreen.availableGeometry()``, logical pixels: already divided by the
Windows display scale).  Every window and dialog is clamped to it and centred, so nothing opens off-screen or under the
taskbar - e.g. a 2560x1600 laptop at 150 % gives 1707x1067 logical pixels, ~1707x1027 without the taskbar.

:func:`fit` sizes a window from its content; :class:`ScreenClamp` (installed on the application) re-checks every
top-level window/dialog when it is shown, which also covers the dialogs that size themselves."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QObject, QRect, Qt
from PySide6.QtWidgets import QApplication, QWidget

#: Share of the work area a window takes at most when it opens (the rest stays visible around it).
MAX_W, MAX_H = 0.94, 0.90


def available(win: QWidget) -> Optional[QRect]:
    """The work area of the window's screen (the primary screen before it is shown); None without a screen."""
    screen = win.screen() or QApplication.primaryScreen()
    return screen.availableGeometry() if screen is not None else None


def fit(win: QWidget, want_w: int, want_h: int, avail: Optional[QRect] = None) -> None:
    """Resize to ``want_w`` x ``want_h`` limited to :data:`MAX_W` x :data:`MAX_H` of the work area and centre it."""
    avail = avail or available(win)
    if avail is None:
        win.resize(want_w, want_h)
        return
    w = max(min(want_w, int(avail.width() * MAX_W)), min(win.minimumWidth(), avail.width()))
    h = max(min(want_h, int(avail.height() * MAX_H)), min(win.minimumHeight(), avail.height()))
    win.resize(w, h)
    win.move(avail.x() + (avail.width() - w) // 2, avail.y() + (avail.height() - h) // 2)


def clamp(win: QWidget, avail: Optional[QRect] = None) -> None:
    """Shrink a window that is larger than the work area and move it fully inside (no-op when it already fits)."""
    avail = avail or available(win)
    if avail is None:
        return
    if win.minimumWidth() > avail.width() or win.minimumHeight() > avail.height():   # a hard minimum must not win
        win.setMinimumSize(min(win.minimumWidth(), avail.width()), min(win.minimumHeight(), avail.height()))
    fg = win.frameGeometry()                       # the title bar and borders count too
    extra_w, extra_h = fg.width() - win.width(), fg.height() - win.height()
    w, h = min(win.width(), avail.width() - extra_w), min(win.height(), avail.height() - extra_h)
    if (w, h) != (win.width(), win.height()):
        win.resize(w, h)
    g = win.frameGeometry()
    x = min(max(g.x(), avail.x()), avail.right() - g.width() + 1)
    y = min(max(g.y(), avail.y()), avail.bottom() - g.height() + 1)
    if (x, y) != (g.x(), g.y()):
        win.move(x, y)


class ScreenClamp(QObject):
    """Application event filter: every top-level window or dialog is clamped to its screen when it is shown."""

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt naming
        if event.type() == QEvent.Type.Show and isinstance(obj, QWidget) and obj.isWindow() \
                and obj.windowType() not in (Qt.WindowType.Popup, Qt.WindowType.ToolTip, Qt.WindowType.SplashScreen):
            clamp(obj)                                     # (combo-box lists and menus are popups: Qt places those)
        return False


def install(app: QApplication) -> ScreenClamp:
    """Install :class:`ScreenClamp` on ``app`` (kept as a child of the app so it lives as long as it)."""
    f = ScreenClamp(app)
    app.installEventFilter(f)
    return f
