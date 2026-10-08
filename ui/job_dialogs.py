"""Dialogs around a narration job's working folder (see :mod:`core.workspace`): may the book file be copied/moved into the
job folder, and what to keep when the job is finished.

Both functions return a default instead of opening a modal dialog on the ``offscreen`` Qt platform (test-suite,
``--selftest``): a modal ``exec()`` there would wait forever for a click nobody can make.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout, QWidget

from core import workspace as ws
from core.i18n import tr
from ui.window_base import hint_label


def _headless() -> bool:
    return QApplication.platformName() == "offscreen"


def ask_book_place(parent: Optional[QWidget], book: Path, job_dir: Path) -> str:
    """Ask whether the book file goes into the job folder: ``copy`` | ``move`` | ``leave`` (default when closed)."""
    if _headless():
        return ws.PLACE_LEAVE
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(tr("work.place_title"))
    box.setText(tr("work.place_text", name=Path(book).name, folder=str(job_dir)))
    copy = box.addButton(tr("work.place_copy"), QMessageBox.ButtonRole.AcceptRole)
    move = box.addButton(tr("work.place_move"), QMessageBox.ButtonRole.ActionRole)
    leave = box.addButton(tr("work.place_leave"), QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(copy)
    box.setEscapeButton(leave)
    box.exec()
    return {copy: ws.PLACE_COPY, move: ws.PLACE_MOVE}.get(box.clickedButton(), ws.PLACE_LEAVE)


def _mb(n: int) -> str:
    return f"{max(n, 0) / 1024 ** 2:.1f}"


class CleanupDialog(QDialog):
    """After a finished job: tick what to keep (result audio, original book, prepared text); temporary parts always go."""

    def __init__(self, files: ws.JobFiles, parent: Optional[QWidget] = None) -> None:
        """One check box per group that has files (all ticked)."""
        super().__init__(parent)
        self.setObjectName("root")
        self.setMinimumWidth(460)
        if parent is not None:
            self.setStyleSheet(parent.styleSheet())
        self.setWindowTitle(tr("work.clean_title"))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(10)
        intro = hint_label()
        intro.setText(tr("work.clean_text", folder=str(files.job_dir), mb=_mb(files.size(files.temp))))
        lay.addWidget(intro)
        self.chk_results = QCheckBox(tr("work.keep_results", mb=_mb(files.size(files.results))))
        self.chk_original = QCheckBox(tr("work.keep_original", mb=_mb(files.size(files.original))))
        self.chk_prepared = QCheckBox(tr("work.keep_prepared", mb=_mb(files.size(files.prepared))))
        for chk, items in ((self.chk_results, files.results), (self.chk_original, files.original),
                           (self.chk_prepared, files.prepared)):
            chk.setChecked(True)
            chk.setVisible(bool(items))
            lay.addWidget(chk)
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_cancel = QPushButton(tr("work.clean_later"))
        self.btn_ok = QPushButton(tr("work.clean_ok"))
        self.btn_ok.setObjectName("primary")
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_ok)
        lay.addLayout(row)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.accept)

    def choice(self) -> dict:
        """Keyword arguments for :func:`core.workspace.clean_job`."""
        return {"keep_results": self.chk_results.isChecked(), "keep_original": self.chk_original.isChecked(),
                "keep_prepared": self.chk_prepared.isChecked()}


def ask_cleanup(parent: Optional[QWidget], files: ws.JobFiles) -> Optional[dict]:
    """The user's choice for :func:`core.workspace.clean_job`; ``None`` = "later" (nothing is deleted).
    Headless: keep everything, drop only the temporary parts."""
    if _headless():
        return {"keep_results": True, "keep_original": True, "keep_prepared": True}
    dlg = CleanupDialog(files, parent)
    return dlg.choice() if dlg.exec() == QDialog.DialogCode.Accepted else None
