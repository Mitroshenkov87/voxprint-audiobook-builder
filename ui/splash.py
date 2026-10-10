"""Start-up splash screen: shown by ``main`` right after the QApplication exists, before the heavy UI modules are
imported, so something appears within about a second.  Only Qt and small modules are imported here.

The artwork (``assets/splash.jpg``, the author's own) is square, about 480 logical pixels (smaller on a small work area),
rendered at the screen's scale; a translucent strip at the bottom carries the name, version, a thin progress bar and the
status line.

Six clicks inside two seconds (the same rule as Voxprint AI Movie Dubber) paint ``assets/easter-egg.jpg`` and a caption
on top of the splash for three seconds, or until another click, once per launch.  Loading is not delayed: the picture is
only a repaint, and :meth:`Splash.finish` returns at once, closing the splash later only if that picture is still up.
"""
from __future__ import annotations

import time

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen, QWidget

from core import appinfo
from core.i18n import tr
from infra import paths

SIDE = 480                                   # logical px; at most 60 % of the shorter side of the work area
STRIP = 92                                   # height of the bottom strip
EGG_STRIP = 48                               # caption band on the easter-egg picture
TEXT, MUTED, BAR = "#f2f2f5", "#d0d0da", "#f5d76e"
EGG_CLICKS = 6                               # clicks inside EGG_WINDOW_S, same rule as Movie Dubber
EGG_WINDOW_S = 2.0
EGG_HOLD_S = 3.0
EGG_FILE = "easter-egg.jpg"


def _side() -> int:
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return SIDE
    a = screen.availableGeometry()
    return max(240, min(SIDE, int(min(a.width(), a.height()) * 0.6)))


def _pixmap(side: int) -> QPixmap:
    screen = QGuiApplication.primaryScreen()
    dpr = screen.devicePixelRatio() if screen is not None else 1.0
    art = QPixmap(str(paths.resource_dir() / "assets" / "splash.jpg"))
    if art.isNull():
        art = QPixmap(int(side * dpr), int(side * dpr))
        art.fill(QColor("#17171c"))
    pm = art.scaled(int(side * dpr), int(side * dpr), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation)
    pm.setDevicePixelRatio(dpr)                # drawn in logical pixels, sharp at 150 %
    return pm


def record_click(stamps: list[float], now: float, need: int = EGG_CLICKS,
                 window: float = EGG_WINDOW_S) -> tuple[list[float], bool]:
    """Keep the clicks that are still inside ``window`` seconds of ``now`` (the edge counts) and say if ``need`` landed.

    ``stamps`` are the earlier click times; ``now`` is this click.  Six clicks inside 2.0 seconds is the Movie Dubber rule.
    """
    kept = [t for t in stamps if now - t <= window]
    kept.append(now)
    return kept, len(kept) >= need


class Splash(QSplashScreen):
    """The splash; :meth:`status` sets the line and the bar and repaints at once (the event loop is not running yet)."""

    def __init__(self) -> None:
        self.side = _side()
        super().__init__(_pixmap(self.side))
        self._text, self._fraction = "", 0.0
        self._clicks: list[float] = []
        self._egg = False
        self._egg_done = False                   # once per launch
        self._egg_timer: QTimer | None = None
        self._egg_pm: QPixmap | None = None
        self._pending: QWidget | None = None     # main window ready while the picture is still up

    def drawContents(self, p: QPainter) -> None:  # noqa: N802 - Qt naming
        if self._egg:
            self._paint_egg(p)
            return
        s = self.side
        top = s - STRIP
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(QRect(0, top, s, STRIP), QColor(10, 12, 18, 170))
        f = QFont(self.font())
        f.setPixelSize(19)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(TEXT))
        p.drawText(QRect(18, top + 10, s - 36, 26), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   appinfo.APP_DISPLAY_NAME)
        f.setPixelSize(12)
        f.setBold(False)
        p.setFont(f)
        p.setPen(QColor(MUTED))
        # the version line (0.1.1-beta · build 665 "Tikkun") sits on the status row, right-aligned: too long for the title row
        p.drawText(QRect(18, top + 54, s - 36, 26), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"v{appinfo.version_label()}")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 50))
        p.drawRoundedRect(QRectF(18, top + 44, s - 36, 4), 2, 2)
        p.setBrush(QColor(BAR))
        p.drawRoundedRect(QRectF(18, top + 44, (s - 36) * self._fraction, 4), 2, 2)
        p.setPen(QColor(MUTED))
        p.drawText(QRect(18, top + 54, s - 36, 26), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._text)

    def message(self) -> str:  # noqa: D102 - the status line (QSplashScreen.message() stays empty: we paint it ourselves)
        return self._text

    def status(self, text: str, fraction: float = -1.0) -> None:
        self._text = text
        if fraction >= 0:
            self._fraction = max(0.0, min(1.0, fraction))
        self.repaint()
        QApplication.processEvents()

    def loading_ui(self) -> None:
        self.status(tr("splash.loading_ui"), 0.25)

    def checking(self) -> None:
        self.status(tr("splash.checking"), 0.7)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # the base implementation hides the splash on the first click, which would make the sequence impossible
        if self._egg:
            self.dismiss_egg()
        elif not self._egg_done:
            self._clicks, hit = record_click(self._clicks, time.monotonic())
            if hit:
                self._egg_done = True
                self._clicks = []
                self._show_egg()
        event.accept()

    def egg_visible(self) -> bool:
        """True while the easter-egg picture is painted on the splash."""
        return self._egg

    def egg_caption(self) -> str:
        """The caption painted with the picture; empty when the picture is not up."""
        return tr("easter_egg.caption") if self._egg else ""

    def dismiss_egg(self) -> None:
        """Hide the picture.  If the main window was already ready, close the splash now."""
        if self._egg_timer is not None:
            self._egg_timer.stop()
        was = self._egg
        self._egg = False
        pending = self._pending
        self._pending = None
        if was:
            self.repaint()
        if pending is not None:
            super().finish(pending)

    def finish(self, widget: QWidget) -> None:  # noqa: N802 - Qt naming
        if self._egg:
            self._pending = widget               # loading is done; the splash stays only until the picture closes
            return
        super().finish(widget)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._egg_timer is not None:
            self._egg_timer.stop()
        super().closeEvent(event)

    def _show_egg(self) -> None:
        self._egg = True
        if self._egg_timer is None:
            self._egg_timer = QTimer(self)
            self._egg_timer.setSingleShot(True)
            self._egg_timer.timeout.connect(self.dismiss_egg)
        self._egg_timer.start(int(EGG_HOLD_S * 1000))
        self.repaint()
        QApplication.processEvents()             # visible during start-up, before the event loop; does not wait

    def _egg_pixmap(self) -> QPixmap:
        if self._egg_pm is not None:
            return self._egg_pm
        screen = QGuiApplication.primaryScreen()
        dpr = screen.devicePixelRatio() if screen is not None else 1.0
        src = QPixmap(str(paths.resource_dir() / "assets" / EGG_FILE))
        if src.isNull():
            self._egg_pm = QPixmap()
            return self._egg_pm
        px = max(1, int(self.side * dpr))
        pm = src.scaled(px, px, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        pm.setDevicePixelRatio(dpr)
        self._egg_pm = pm
        return pm

    def _paint_egg(self, p: QPainter) -> None:
        s = self.side
        p.fillRect(QRect(0, 0, s, s), QColor("#17171c"))
        art = self._egg_pixmap()
        if not art.isNull():
            dpr = art.devicePixelRatio() or 1.0
            w, h = art.width() / dpr, art.height() / dpr
            p.drawPixmap(QPointF((s - w) / 2, (s - h) / 2), art)
        top = s - EGG_STRIP
        p.fillRect(QRect(0, top, s, EGG_STRIP), QColor(10, 12, 18, 200))
        f = QFont(self.font())
        f.setPixelSize(18)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(TEXT))
        p.drawText(QRect(12, top, s - 24, EGG_STRIP), Qt.AlignmentFlag.AlignCenter, self.egg_caption())


def show() -> Splash:
    s = Splash()
    s.show()
    QApplication.processEvents()
    return s
