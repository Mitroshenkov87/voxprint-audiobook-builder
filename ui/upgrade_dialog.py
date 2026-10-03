"""One-click prompt: «component X in your environment is outdated - upgrade to the newest stable version?»

Shown only for components that live in the USER's environment (system Python, conda, Pinokio...).  Nothing is changed
before the user presses «Upgrade»; «Not now» keeps the component as it is (or, when it is too old to work, Voxprint
uses its own copy and the user's environment stays untouched).  Non-blocking: ``open()`` + ``decided`` signal.
"""
from __future__ import annotations

import html
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from core.i18n import tr


class UpgradeOfferDialog(QDialog):
    decided = Signal(list)     # names the user agreed to upgrade ([] = declined)

    def __init__(self, offers: List[Dict], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.offers = list(offers)
        self._emitted = False
        self.setWindowTitle(tr("upg.title"))
        self.setObjectName("root")
        self.setMinimumWidth(520)
        if parent is not None:
            self.setStyleSheet(parent.styleSheet())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(10)

        self.lbl_intro = QLabel(tr("upg.intro"))
        self.lbl_intro.setWordWrap(True)
        lay.addWidget(self.lbl_intro)

        self.lbl_items = QLabel(self._items_html())
        self.lbl_items.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_items.setWordWrap(True)
        lay.addWidget(self.lbl_items)

        env = self.offers[0].get("env", "") if self.offers else ""
        self.lbl_changes = QLabel(tr("upg.what_changes", env=env))
        self.lbl_changes.setWordWrap(True)
        self.lbl_changes.setObjectName("hint")
        lay.addWidget(self.lbl_changes)

        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_later = QPushButton(tr("upg.btn_later"))
        self.btn_upgrade = QPushButton(tr("upg.btn_upgrade"))
        self.btn_upgrade.setDefault(True)
        row.addWidget(self.btn_later)
        row.addWidget(self.btn_upgrade)
        lay.addLayout(row)
        self.btn_upgrade.clicked.connect(self.accept_upgrade)
        self.btn_later.clicked.connect(self.reject)

    def _items_html(self) -> str:
        parts = []
        for o in self.offers:
            line = html.escape(tr("upg.item", name=o["name"], old=o.get("installed") or "-", new=o["target"]))
            note = tr("upg.note_compatible") if o.get("compatible", True) else tr("upg.note_incompatible")
            parts.append(f"<b>{line}</b><br><span style='color:#9aa0aa'>{html.escape(note)}</span>")
        return "<br><br>".join(parts)

    def accepted_names(self) -> List[str]:
        return [o["name"] for o in self.offers]

    def _finish(self, names: List[str]) -> None:
        if not self._emitted:
            self._emitted = True
            self.decided.emit(names)

    def accept_upgrade(self) -> None:
        self._finish(self.accepted_names())
        self.accept()

    def reject(self) -> None:           # «Not now», Esc, closing the window
        self._finish([])
        super().reject()
