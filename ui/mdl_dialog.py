"""The window shown when VOXPrint MDL has not finished the required models."""
from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

from core.i18n import tr
from infra import mdl_gate


def prompt() -> bool:
    """Ask the user to open Manage models. True when the models are ready afterwards."""
    box = QMessageBox()
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(tr("mdl.not_ready_title"))
    box.setText(tr("mdl.not_ready"))
    manage = box.addButton(tr("mdl.manage"), QMessageBox.ButtonRole.ActionRole)
    box.addButton(tr("mdl.quit"), QMessageBox.ButtonRole.RejectRole)
    box.exec()
    if box.clickedButton() is not manage:
        return False
    mdl_gate.launch_manage()
    return mdl_gate.models_ready()
