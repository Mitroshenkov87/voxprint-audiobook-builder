"""Start-up splash: the suite window from :mod:`voxprint_common.ui.splash`.

``main`` calls :func:`show` as soon as the QApplication exists. The artwork is ``assets/splash.jpg``
(16:9). The identity colour is ``#f5d76e``. The build line comes from :func:`voxprint_common.buildinfo.load`
on ``BUILD.json``, with the product version from ``credits.json`` (``BUILD.json`` has no version field).

Six clicks inside two seconds call :attr:`voxprint_common.ui.splash.Splash.on_click`. That shows
``assets/easter-egg.jpg`` in our own window for three seconds, or until another click, once per launch.
The suite window stays the splash; this module does not paint on the private window.
"""
from __future__ import annotations

import os
import time
from dataclasses import replace

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

from voxprint_common.buildinfo import load, parse
from voxprint_common.ui.splash import HEIGHT, WIDTH
from voxprint_common.ui.splash import Splash as SuiteSplash
from voxprint_common.ui.splash import SplashEvent

from core import appinfo
from core.i18n import tr
from infra import paths

IDENTITY = "#f5d76e"
EGG_CLICKS = 6
EGG_WINDOW_S = 2.0
EGG_HOLD_S = 3.0
EGG_FILE = "easter-egg.jpg"
__all__ = ["EGG_CLICKS", "EGG_WINDOW_S", "HEIGHT", "IDENTITY", "WIDTH", "Splash", "record_click", "show"]


def record_click(stamps: list[float], now: float, need: int = EGG_CLICKS,
                 window: float = EGG_WINDOW_S) -> tuple[list[float], bool]:
    """Keep clicks still inside ``window`` seconds of ``now`` and say whether ``need`` of them landed."""
    kept = [stamp for stamp in stamps if now - stamp <= window]
    kept.append(now)
    return kept, len(kept) >= need


def build_info():
    """Identity drawn on the splash. ``GITHUB_RUN_NUMBER`` is not added to the recorded build.

    ``load`` reads ``BUILD.json`` (build and codename). The version string stays in ``credits.json``.
    ``dump`` is not called.
    """
    env = {key: value for key, value in os.environ.items() if key != "GITHUB_RUN_NUMBER"}
    catalog = paths.resource_dir() / "BUILD.json"
    try:
        loaded = load(catalog, app="audiobook-builder", env=env)
    except (OSError, ValueError):
        loaded = None
    payload: dict = {"app": "audiobook-builder", "version": appinfo.APP_VERSION}
    if loaded is not None:
        payload["app"] = loaded.app or "audiobook-builder"
        payload["codename"] = loaded.codename or appinfo.APP_CODENAME
        if loaded.build is not None:
            payload["build"] = loaded.build
    elif appinfo.APP_CODENAME:
        payload["codename"] = appinfo.APP_CODENAME
    if appinfo.APP_BUILD:
        payload["build"] = appinfo.APP_BUILD
    info = parse(payload, app="audiobook-builder", env=env)
    # voxprint-common 0.1.1 draws the id audiobook-builder as "Voxprint AI Audiobook Builder".
    # A name it does not know is drawn as written, so the splash uses the product name.
    return replace(info, app=appinfo.APP_DISPLAY_NAME)


class _Egg(QWidget):
    """Our picture. It is not the suite splash window."""

    def __init__(self, art: QPixmap, caption: str, on_dismiss) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._art = art
        self._caption = caption
        self._on_dismiss = on_dismiss
        self.resize(WIDTH, HEIGHT)
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            self.move(area.center().x() - WIDTH // 2, area.center().y() - HEIGHT // 2)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(on_dismiss)
        self._timer.start(int(EGG_HOLD_S * 1000))
        self.show()
        QApplication.processEvents()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._on_dismiss()
        event.accept()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#17171c"))
        band = 48
        art = self._art
        if not art.isNull():
            scaled = art.scaled(self.width(), max(1, self.height() - band), Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap((self.width() - scaled.width()) // 2,
                               (self.height() - band - scaled.height()) // 2, scaled)
        top = self.height() - band
        painter.fillRect(QRect(0, top, self.width(), band), QColor(10, 12, 18, 200))
        font = QFont(self.font())
        font.setPixelSize(18)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#f2f2f5"))
        painter.drawText(QRect(12, top, self.width() - 24, band), Qt.AlignmentFlag.AlignCenter, self._caption)
        painter.end()


class Splash:
    """Thin wrapper so ``main`` can set a stage, count clicks, and close the suite splash."""

    def __init__(self) -> None:
        art = paths.resource_dir() / "assets" / "splash.jpg"
        self._clicks: list[float] = []
        self._egg_done = False
        self._egg: _Egg | None = None
        self._pending_finish = False
        self._inner = SuiteSplash(art, IDENTITY, build_info(), on_click=self._on_click)
        self._text = ""
        self._fraction = -1.0

    def _on_click(self, event: SplashEvent) -> None:
        if self._egg is not None:
            self._dismiss_egg()
            return
        if self._egg_done:
            return
        self._clicks, hit = record_click(self._clicks, time.monotonic())
        if hit:
            self._egg_done = True
            self._clicks = []
            self._show_egg()

    def _show_egg(self) -> None:
        art = QPixmap(str(paths.resource_dir() / "assets" / EGG_FILE))
        self._egg = _Egg(art, tr("easter_egg.caption"), self._dismiss_egg)

    def _dismiss_egg(self) -> None:
        egg = self._egg
        self._egg = None
        if egg is not None:
            egg.close()
        if self._pending_finish:
            self._pending_finish = False
            self._inner.finish()

    def egg_visible(self) -> bool:
        """True while the easter-egg window is up."""
        return self._egg is not None

    def message(self) -> str:
        """The stage line currently shown."""
        return self._text

    def status(self, text: str, fraction: float = -1.0) -> None:
        """Set the stage and, when ``fraction`` is given, the determinate bar."""
        self._text = text
        self._inner.set_stage(text)
        if fraction >= 0:
            self._fraction = max(0.0, min(1.0, fraction))
            self._inner.set_progress(self._fraction)

    def loading_ui(self) -> None:
        """The line shown while the studio window is imported."""
        self.status(tr("splash.loading_ui"), 0.25)

    def checking(self) -> None:
        """The line shown while models and the thin runtime are checked."""
        self.status(tr("splash.checking"), 0.7)

    def isVisible(self) -> bool:  # noqa: N802 - Qt naming, used by the screen-fit test
        return bool(self._inner._window.isVisible())

    def width(self) -> int:
        """Window width in pixels, including the drop shadow."""
        return int(self._inner._window.width())

    def height(self) -> int:
        """Window height in pixels, including the drop shadow."""
        return int(self._inner._window.height())

    def grab(self):
        """A snapshot of the splash window (the screen-fit test paints it)."""
        return self._inner._window.grab()

    def finish(self, widget: object = None) -> None:
        """Keep the splash up for the suite minimum, then fade it.

        While the egg is up, the fade waits until the picture closes. ``widget`` is unused.
        """
        del widget
        if self._egg is not None:
            self._pending_finish = True
            return
        self._inner.finish()


def show() -> Splash:
    """Show the suite splash and return it."""
    return Splash()
