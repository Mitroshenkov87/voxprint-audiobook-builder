"""Shared shell of the Studio's windows: dark/Acrylic look, scroll area, header with "back to Studio" button.

``MainWindow`` (the training window) predates this module and keeps its own copy of the same shell; the new windows
(Studio home, My voices, Narrate a book) derive from :class:`SubWindow`.

Navigation is request-based: a window never shows another one itself, it emits ``go("studio" | "train" | "voices" |
"narrate")`` (optionally with a payload via the Studio's own signals) and the Studio switches windows.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout,
                               QWidget)

from core.i18n import tr
from ui import screen_fit
from ui.main_window import APP_TITLE, MIN_WINDOW_H, MIN_WINDOW_W, apply_look, build_style


def fit_to_screen(win: QWidget, content: QWidget, min_w: int = 820, min_h: int = 560) -> None:
    """Initial size: the content's natural size (at least ``min_w`` x ``min_h``) limited to the work area, centred
    (:func:`ui.screen_fit.fit`); the content scrolls below that."""
    hint = content.sizeHint()
    win.setMinimumSize(MIN_WINDOW_W, MIN_WINDOW_H)
    screen_fit.fit(win, max(min_w, hint.width()), max(min_h, hint.height() + 8))


class ColumnFlow(QWidget):
    """Cards in one column, or in two when the window is wide enough (height-first: a short 16:10 laptop screen at 150 %
    shows the whole form with little scrolling).  The order is kept: the first cards fill the left column up to about
    half of the total height, the rest go right.  Cards hidden at that moment count as zero height."""

    #: never two columns below this width, however narrow the cards are
    TWO_COLUMNS_MIN_W = 1000

    def __init__(self, parent: Optional[QWidget] = None, min_two: int = TWO_COLUMNS_MIN_W) -> None:
        super().__init__(parent)
        self.min_two = min_two
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        self._cols = (QVBoxLayout(), QVBoxLayout())
        for col in self._cols:
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(10)
            row.addLayout(col, 1)
        self._cards: list = []
        self.two = False

    def add(self, card: QWidget) -> None:
        self._cards.append(card)
        self._place(self.two)

    def _split(self) -> int:
        """Index of the first right-hand card: the split that makes the taller column as short as possible."""
        heights = [0 if c.isHidden() else c.sizeHint().height() for c in self._cards]
        total, acc = sum(heights), 0
        best = (total, len(self._cards))
        for i, hgt in enumerate(heights):
            acc += hgt
            best = min(best, (max(acc, total - acc), i + 1))
        return best[1]

    def two_column_width(self) -> int:
        """Width the two columns need (long check-box texts do not wrap, so the widest card of each column counts)."""
        split = self._split()
        left = max((c.minimumSizeHint().width() for c in self._cards[:split]), default=0)
        right = max((c.minimumSizeHint().width() for c in self._cards[split:]), default=0)
        return left + right + 10

    def _place(self, two: bool) -> None:
        self.two = two
        for col in self._cols:                       # empty both columns (the cards stay children of this widget)
            while col.count():
                col.takeAt(0)
        split = self._split() if two else len(self._cards)
        for i, card in enumerate(self._cards):
            self._cols[0 if i < split else 1].addWidget(card)
        for col in self._cols:
            col.addStretch(1)

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        two = self.width() >= max(self.min_two, self.two_column_width())
        if two != self.two:
            self._place(two)


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
        self.body.setContentsMargins(20, 14, 20, 14)
        self.body.setSpacing(10)

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
        apply_look(self)

    def closeEvent(self, e) -> None:  # noqa: N802
        """Tell the Studio (it shuts the application down)."""
        super().closeEvent(e)
        self.closing.emit()

    def apply_style_from(self, other: QWidget) -> None:
        """Copy the stylesheet of another window (Acrylic vs. plain is decided per top-level window)."""
        self.setStyleSheet(other.styleSheet())
