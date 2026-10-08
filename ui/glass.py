"""Dialogs with the main window's look: the same Acrylic backdrop and transparency level (Settings -> Window transparency).

:class:`GlassDialog` is a ``QDialog`` with a translucent root (object name ``root``, like the windows); on every show it
calls :func:`ui.main_window.apply_look`, which asks Windows 11 for the Acrylic backdrop once and applies the stylesheet of
the current transparency level ("off" = the solid colours).  :func:`fit_height` grows a dialog whose word-wrapped texts
need more height than it has (e.g. at 150 % display scale), within the screen's work area.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QScrollArea, QWidget


class GlassDialog(QDialog):
    """A dialog that looks like the main window (translucent, Acrylic where available, same transparency level)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("root")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def showEvent(self, e) -> None:  # noqa: N802 - Qt naming
        super().showEvent(e)
        from ui.main_window import (
            apply_look,  # local: ui.main_window imports the dialogs (circular at load time)
        )

        clear_scroll_fills(self)
        apply_look(self)
        fit_height(self)


def clear_scroll_fills(dlg: QWidget) -> None:
    """Let the translucent root show through every plain scroll area of ``dlg``: ``QScrollArea.setWidget`` switches on
    the opaque palette fill of the scrolled widget (a solid grey block over the Acrylic, the 0.1.1 Settings bug); the
    windows avoid it with the transparent ``QWidget#content``.  Lists, text fields and combo boxes keep their own
    (styled) backgrounds."""
    for sa in dlg.findChildren(QScrollArea):
        sa.viewport().setAutoFillBackground(False)
        inner = sa.widget()
        if inner is not None:
            inner.setAutoFillBackground(False)


def fit_height(dlg: QWidget) -> None:
    """Grow ``dlg`` to the height its layout needs at its current width (word-wrapped labels), capped by the screen."""
    lay = dlg.layout()
    if lay is None:
        return
    need = lay.totalHeightForWidth(dlg.width()) if lay.hasHeightForWidth() else lay.sizeHint().height()
    need = max(need, lay.totalMinimumSize().height())
    if need <= dlg.height():
        return
    from ui import screen_fit

    avail = screen_fit.available(dlg)
    if avail is not None:
        need = min(need, int(avail.height() * screen_fit.MAX_H))
    dlg.resize(dlg.width(), need)
