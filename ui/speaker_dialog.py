"""Editable preview of speaker marks (narrator / male / female) before narration starts."""
from __future__ import annotations

from typing import List, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout

from core.i18n import tr
from core.speakers import SpeakerLine
from ui.glass import GlassDialog
from ui.window_base import hint_label


class SpeakerDialog(GlassDialog):
    """One row per paragraph: role, optional name, and the paragraph text."""

    def __init__(self, lines: Sequence[SpeakerLine], paragraphs: Sequence[str], parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("root")
        self.setWindowTitle(tr("spk.title"))
        self.resize(760, 480)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(8)
        self.note = hint_label()
        self.note.setText(tr("spk.note"))
        lay.addWidget(self.note)
        self.table = QTableWidget(len(lines), 3)
        self.table.setHorizontalHeaderLabels([tr("spk.col_role"), tr("spk.col_name"), tr("spk.col_text")])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(False)
        roles = (("narrator", tr("spk.role_narrator")), ("male", tr("spk.role_male")), ("female", tr("spk.role_female")))
        for row, (line, text) in enumerate(zip(lines, paragraphs)):
            combo = QComboBox()
            for role, label in roles:
                combo.addItem(label, role)
            combo.setCurrentIndex(max(0, combo.findData(line.role)))
            self.table.setCellWidget(row, 0, combo)
            self.table.setItem(row, 1, QTableWidgetItem(line.name))
            item = QTableWidgetItem(" ".join(text.split()))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 2, item)
        self.table.resizeColumnsToContents()
        lay.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.btn_ok = QPushButton(tr("ui.ready"))
        buttons.addWidget(self.btn_ok)
        lay.addLayout(buttons)
        self.btn_ok.clicked.connect(self.accept)

    def lines(self) -> List[SpeakerLine]:
        """The marks as edited in the table."""
        out: List[SpeakerLine] = []
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 0)
            name_item = self.table.item(row, 1)
            role = str(combo.currentData() or "narrator") if isinstance(combo, QComboBox) else "narrator"
            out.append(SpeakerLine(role, name_item.text() if name_item is not None else ""))
        return out
