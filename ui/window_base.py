"""Shared shell of the Studio's windows: dark/Acrylic look, scroll area, header with "back to Studio" button.

``MainWindow`` (the training window) predates this module and keeps its own copy of the same shell; the new windows
(Studio home, My voices, Narrate a book) derive from :class:`SubWindow`.

Navigation is request-based: a window never shows another one itself, it emits ``go("studio" | "train" | "voices" |
"narrate")`` (optionally with a payload via the Studio's own signals) and the Studio switches windows.
"""
from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout,
                               QWidget)

from core.i18n import tr
from infra import platform_win
from ui.main_window import APP_TITLE, MIN_WINDOW_H, MIN_WINDOW_W, build_style


def fit_to_screen(win: QWidget, content: QWidget, min_w: int = 820, min_h: int = 560) -> None:
    """Initial size: the content's natural size (at least ``min_w`` x ``min_h``) limited to ~94 % x ~90 % of the work area."""
    hint = content.sizeHint()
    w, h = max(min_w, hint.width()), max(min_h, hint.height() + 8)
    try:
        avail = (win.screen() or QApplication.primaryScreen()).availableGeometry()
        w, h = min(w, int(avail.width() * 0.94)), min(h, int(avail.height() * 0.90))
    except Exception:  # noqa: BLE001 - no screen (tests)
        pass
    win.setMinimumSize(MIN_WINDOW_W, MIN_WINDOW_H)
    win.resize(max(w, MIN_WINDOW_W), max(h, MIN_WINDOW_H))


def card_frame() -> QFrame:
    """A rounded panel (``QFrame#card``)."""
    f = QFrame()
    f.setObjectName("card")
    return f


def hint_label(wrap: bool = True) -> QLabel:
    """A small secondary-text label."""
    lbl = QLabel()
    lbl.setObjectName("hint")
    lbl.setWordWrap(wrap)
    return lbl


class SubWindow(QWidget):
    """Base class: translucent root + scroll area + header (back button, title, optional gear)."""

    go = Signal(str)          # navigation request
    closing = Signal()        # the user closed this window

    def __init__(self, with_back: bool = True, with_gear: bool = False) -> None:
        """Build the empty shell; subclasses add widgets to ``self.body`` and implement :meth:`retranslate`."""
        super().__init__()
        self.backdrop = "plain"
        self.setObjectName("root")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.content = QWidget()
        self.content.setObjectName("content")
        self.scroll.setWidget(self.content)
        self.scroll.viewport().setAutoFillBackground(False)
        outer.addWidget(self.scroll)
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(28, 24, 28, 24)
        self.body.setSpacing(14)

        head = QHBoxLayout()
        self.btn_back = QPushButton()
        self.btn_back.setObjectName("back")
        self.btn_back.setVisible(with_back)
        self.btn_back.clicked.connect(lambda: self.go.emit("studio"))
        head.addWidget(self.btn_back)
        self.lbl_title = QLabel()
        self.lbl_title.setObjectName("title")
        head.addWidget(self.lbl_title)
        head.addStretch(1)
        self.btn_gear: Optional[QPushButton] = None
        if with_gear:
            self.btn_gear = QPushButton("\u2699")
            self.btn_gear.setObjectName("gear")
            head.addWidget(self.btn_gear)
        self.body.addLayout(head)
        self.setStyleSheet(build_style(False))

    # ------------------------------------------------------------------ to override
    def window_title(self) -> str:
        """Title shown in the header and the title bar (subclasses return a localized string)."""
        return APP_TITLE

    def retranslate(self) -> None:
        """Apply the current UI language (subclasses extend this and call ``super().retranslate()``)."""
        self.btn_back.setText(tr("nav.back"))
        self.lbl_title.setText(self.window_title())
        self.setWindowTitle(APP_TITLE if self.window_title() == APP_TITLE else f"{APP_TITLE} - {self.window_title()}")

    # ------------------------------------------------------------------ window effects
    def showEvent(self, e) -> None:  # noqa: N802
        """Enable the Acrylic backdrop once the native window exists (Windows 11), else stay on the plain dark look."""
        super().showEvent(e)
        if sys.platform == "win32" and self.backdrop == "plain":
            self.backdrop = platform_win.apply_backdrop(int(self.winId()))
            self.setStyleSheet(build_style(self.backdrop == "acrylic"))
            if self.backdrop != "acrylic":
                self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

    def closeEvent(self, e) -> None:  # noqa: N802
        """Tell the Studio (it shuts the application down)."""
        super().closeEvent(e)
        self.closing.emit()

    def apply_style_from(self, other: QWidget) -> None:
        """Copy the stylesheet of another window (Acrylic vs. plain is decided per top-level window)."""
        self.setStyleSheet(other.styleSheet())
