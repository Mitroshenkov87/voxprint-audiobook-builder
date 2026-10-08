"""Settings right column: a long Check & repair line must not make the controls overlap."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication

from tests.test_studio import app, lib, make_studio  # noqa: F401
from ui.window_base import ColumnFlow


def _rect(dialog, widget) -> QRect:
    origin = widget.mapTo(dialog, widget.rect().topLeft())
    return QRect(origin, widget.rect().size())


def test_check_and_repair_does_not_overlap_the_right_column(app, lib):
    s = make_studio(lib)
    d = s.settings_dialog()
    d.show()
    d.lbl_autorepair_status.setText("Checking " + ("models/component.bin " * 40))
    d.bar_autorepair.setVisible(True)
    flow = d.findChild(ColumnFlow, "content")
    d.resize(2400, 900)
    QApplication.processEvents()
    d.layout().activate()
    QApplication.processEvents()
    assert flow.two
    assert d.lbl_autorepair_status.minimumSizeHint().width() <= 160
    assert d.lbl_autorepair_status.height() > 30          # the long line wraps instead of widening the column
    widgets = [d.btn_projects_open, d.btn_projects_change, d.btn_projects_default, d.btn_backup, d.btn_restore,
               d.btn_existing, d.btn_existing_clear, d.btn_autorepair, d.btn_repair_restore,
               d.lbl_autorepair_status, d.bar_autorepair, d.lbl_repair_desc]
    rects = [_rect(d, w) for w in widgets if w.isVisible() and w.width() > 0 and w.height() > 0]
    assert len(rects) == len(widgets)
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            hit = a.intersected(b)
            assert hit.width() <= 2 or hit.height() <= 2, (a, b, hit)
    assert abs(d.btn_projects_open.y() - d.btn_projects_change.y()) > 8
    s.shutdown()
