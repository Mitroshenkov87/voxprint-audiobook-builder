"""Start-up splash: the suite window from :mod:`voxprint_common.ui.splash`.

``main`` calls :func:`show` as soon as the QApplication exists. The artwork is ``assets/splash.jpg``
(16:9). The identity colour is ``#f5d76e``. The build line comes from ``BUILD.json`` and ``credits.json``.
"""
from __future__ import annotations

import json
import os

from voxprint_common.buildinfo import parse
from voxprint_common.ui.splash import HEIGHT, WIDTH
from voxprint_common.ui.splash import Splash as SuiteSplash

from core import appinfo
from core.i18n import tr
from infra import paths

IDENTITY = "#f5d76e"
__all__ = ["HEIGHT", "IDENTITY", "WIDTH", "Splash", "show"]


def build_info():
    """Identity drawn on the splash. ``GITHUB_RUN_NUMBER`` is not added to the explicit build."""
    data: dict = {}
    try:
        loaded = json.loads((paths.resource_dir() / "BUILD.json").read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data.update(loaded)
    except (OSError, ValueError):
        data = {}
    data["app"] = "audiobook-builder"
    data["version"] = appinfo.APP_VERSION
    if appinfo.APP_BUILD:
        data["build"] = appinfo.APP_BUILD
    if appinfo.APP_CODENAME:
        data["codename"] = appinfo.APP_CODENAME
    env = {key: value for key, value in os.environ.items() if key != "GITHUB_RUN_NUMBER"}
    return parse(data, app="audiobook-builder", env=env)


class Splash:
    """Thin wrapper so ``main`` can set a stage and close the suite splash."""

    def __init__(self) -> None:
        art = paths.resource_dir() / "assets" / "splash.jpg"
        self._inner = SuiteSplash(art, IDENTITY, build_info())
        self._text = ""
        self._fraction = -1.0

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
        """Keep the splash up for the suite minimum, then fade it. ``widget`` is unused."""
        del widget
        self._inner.finish()


def show() -> Splash:
    """Show the suite splash and return it."""
    return Splash()
