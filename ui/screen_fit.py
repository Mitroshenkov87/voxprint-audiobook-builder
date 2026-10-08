"""Window geometry from the screen's work area (``QScreen.availableGeometry()``, logical pixels: already divided by the
Windows display scale).  Every window and dialog is clamped to it and centred, so nothing opens off-screen or under the
taskbar - e.g. a 2560x1600 laptop at 150 % gives 1707x1067 logical pixels, ~1707x1027 without the taskbar.

:func:`fit` sizes a window from its content; :class:`ScreenClamp` (installed on the application) re-checks every
top-level window/dialog when it is shown, which also covers the dialogs that size themselves."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QObject, QRect, QSize, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QWidget,
)

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


def nobreak(text: str) -> str:
    """``text`` that never wraps inside (a version like ``0.1.1-beta · build 665``): word joiners around the hyphens,
    no-break spaces for the spaces - the visible characters stay the same."""
    return text.replace("-", "\u2060-\u2060").replace(" ", "\u00a0")


class ElidedCombo(QComboBox):
    """A combo box that elides a current text too long for its width ("Download both (use…") instead of cutting it
    off; the full text is its tooltip and the open list is as wide as the longest entry."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.currentTextChanged.connect(self.setToolTip)

    def showPopup(self) -> None:
        fm = self.view().fontMetrics()
        widest = max((fm.horizontalAdvance(self.itemText(i)) for i in range(self.count())), default=0)
        self.view().setMinimumWidth(widest + 40)            # (room for the scroll bar and the item padding)
        super().showPopup()

    def elided_text(self) -> str:
        """The current text as painted: elided to the text field of the box."""
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        field = self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, opt,
                                            QStyle.SubControl.SC_ComboBoxEditField, self)
        return self.fontMetrics().elidedText(self.currentText(), Qt.TextElideMode.ElideRight, field.width())

    def paintEvent(self, event) -> None:
        p = QStylePainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        p.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt)
        opt.currentText = self.elided_text()
        p.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt)


class BoundLabel(QLabel):
    """A word-wrapped label whose minimum width stays small.

    A plain word-wrapped ``QLabel`` reports the full line as its width, so a long Check & repair status squeezes the
    other column until buttons paint on top of each other. This label wraps inside the width it is given.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWordWrap(True)
        self.setMinimumWidth(140)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt naming
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt naming
        return self.fontMetrics().boundingRect(0, 0, max(width, 1), 10000, int(Qt.TextFlag.TextWordWrap), self.text()).height() + 4

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(140, max(self.heightForWidth(140), 1))

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(220, max(self.heightForWidth(220), self.minimumSizeHint().height()))


class _ToggleLabel(QLabel):
    def __init__(self, box: QCheckBox) -> None:
        super().__init__()
        self._box = box

    def mouseReleaseEvent(self, event) -> None:
        if self._box.isEnabled() and event.button() == Qt.MouseButton.LeftButton:
            self._box.toggle()
        super().mouseReleaseEvent(event)


def wrapped_check(box: QCheckBox) -> tuple[QWidget, QLabel]:
    """A check box whose text wraps (a QCheckBox text never does): ``box`` without text plus a word-wrapped label
    that toggles it.  Returns the row to lay out and the label to set the text on."""
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    label = _ToggleLabel(box)
    label.setWordWrap(True)
    lay.addWidget(box, 0, Qt.AlignmentFlag.AlignTop)
    lay.addWidget(label, 1)
    return row, label
