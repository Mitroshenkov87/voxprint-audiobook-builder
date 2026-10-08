"""Start-up splash screen: shown by ``main`` right after the QApplication exists, before the heavy UI modules are
imported, so something appears within about a second.  Only Qt and small modules are imported here."""
from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from core import appinfo
from core.i18n import tr
from infra import paths

#: logical size (a 150 % screen draws it at 1.5x; the pixmap is rendered at the device pixel ratio, so it stays sharp)
W, H = 480, 260
BG, TEXT, MUTED, ACCENT = "#17171c", "#f2f2f5", "#c4c4d0", "#f5d76e"


def _pixmap() -> QPixmap:
    screen = QGuiApplication.primaryScreen()
    dpr = screen.devicePixelRatio() if screen is not None else 1.0
    pm = QPixmap(int(W * dpr), int(H * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(QColor(BG))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    p.setPen(QColor(ACCENT))
    p.drawRect(0, 0, W - 1, H - 1)
    logo = QPixmap(str(paths.resource_dir() / "assets" / "voxprint.png"))
    if not logo.isNull():
        p.drawPixmap(QRect(32, 56, 96, 96), logo)
    f = QFont()
    f.setPixelSize(26)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor(TEXT))
    p.drawText(QRect(148, 56, W - 168, 70), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
               | Qt.TextFlag.TextWordWrap, appinfo.APP_DISPLAY_NAME)
    f.setPixelSize(14)
    f.setBold(False)
    p.setFont(f)
    p.setPen(QColor(MUTED))
    p.drawText(QRect(148, 126, W - 168, 24), Qt.AlignmentFlag.AlignLeft, f"v{appinfo.APP_VERSION}")
    p.end()
    return pm


class Splash(QSplashScreen):
    """The splash with one status line at the bottom; :meth:`status` repaints right away (the event loop is not running)."""

    def __init__(self) -> None:
        super().__init__(_pixmap())
        f = self.font()
        f.setPixelSize(13)
        self.setFont(f)

    def loading_ui(self) -> None:
        self.status(tr("splash.loading_ui"))

    def checking(self) -> None:
        self.status(tr("splash.checking"))

    def status(self, text: str) -> None:
        self.showMessage(text, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, QColor(MUTED))
        QApplication.processEvents()


def show() -> Splash:
    s = Splash()
    s.show()
    QApplication.processEvents()
    return s
