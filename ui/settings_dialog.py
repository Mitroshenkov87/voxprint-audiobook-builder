"""Settings dialog (opened with the gear button): everything that is *not* part of the core workflow.

The main window only keeps "choose audio / choose text / create voice".  Service items live here:

* interface language,
* "Check for updates",
* shortcuts to the models folder and the data/log folder,
* "Repair installation",
* "About".

The dialog owns no business logic.  Every button calls back into the :class:`ui.main_window.MainWindow` that created
it, so the same code paths (and tests) are used whether an action starts from here or from the main screen.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6.QtWidgets import QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from core import i18n
from core.i18n import tr

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers (avoids a circular import at runtime)
    from ui.main_window import MainWindow


class SettingsDialog(QDialog):
    """Non-blocking-friendly settings window; one instance is kept by the main window and re-shown on demand."""

    def __init__(self, window: "MainWindow", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent or window)
        self._win = window
        self.setObjectName("root")
        self.setMinimumWidth(460)
        self.setStyleSheet(window.styleSheet())     # same dark / acrylic look as the main window

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(12)

        self.lbl_title = QLabel()
        self.lbl_title.setObjectName("title")
        lay.addWidget(self.lbl_title)

        # --- language ---
        row = QHBoxLayout()
        self.lbl_language = QLabel()
        self.cmb_lang = QComboBox()
        for code in i18n.LANGS:
            self.cmb_lang.addItem(i18n.LANG_NAMES[code], code)
        self.cmb_lang.setCurrentIndex(max(0, self.cmb_lang.findData(i18n.get_language())))
        row.addWidget(self.lbl_language)
        row.addStretch(1)
        row.addWidget(self.cmb_lang)
        lay.addLayout(row)

        # --- service buttons ---
        self.btn_update = QPushButton()
        self.btn_models = QPushButton()
        self.btn_data = QPushButton()
        self.btn_repair = QPushButton()
        self.btn_about = QPushButton()
        for b in (self.btn_update, self.btn_models, self.btn_data, self.btn_repair, self.btn_about):
            lay.addWidget(b)
        lay.addStretch(1)
        self.btn_close = QPushButton()
        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_row.addWidget(self.btn_close)
        lay.addLayout(close_row)

        self.cmb_lang.currentIndexChanged.connect(self._on_language_changed)
        self.btn_update.clicked.connect(window.check_updates)
        self.btn_models.clicked.connect(window.open_models_folder)
        self.btn_data.clicked.connect(window.open_data_folder)
        self.btn_repair.clicked.connect(window.start_repair)
        self.btn_about.clicked.connect(window.open_about)
        self.btn_close.clicked.connect(self.accept)
        self.retranslate()
        self.refresh()

    # ------------------------------------------------------------------ texts and state
    def retranslate(self) -> None:
        """Apply the current UI language to every label and button (called on language change)."""
        self.setWindowTitle(tr("ui.settings_title"))
        self.lbl_title.setText(tr("ui.settings_title"))
        self.lbl_language.setText(tr("ui.language"))
        self.btn_update.setText(tr("ui.btn_update"))
        self.btn_models.setText(tr("ui.settings_models_folder"))
        self.btn_data.setText(tr("ui.settings_data_folder"))
        self.btn_repair.setText(tr("ui.settings_repair"))
        self.btn_repair.setToolTip(tr("ui.settings_repair_tip"))
        self.btn_about.setText(tr("ui.about"))
        self.btn_about.setToolTip(tr("ui.about_tip"))
        self.btn_close.setText(tr("about.btn_close"))

    def refresh(self) -> None:
        """Enable/disable controls according to the main window's busy state."""
        busy = self._win.busy
        self.cmb_lang.setEnabled(not busy)           # switching language mid-task would relabel a running job
        self.btn_update.setEnabled(not busy and not self._win.updating)
        self.btn_repair.setEnabled(not busy and not self._win.repairing)

    def sync_language(self) -> None:
        """Make the combo box show the active language without re-triggering a language change."""
        self.cmb_lang.blockSignals(True)
        self.cmb_lang.setCurrentIndex(max(0, self.cmb_lang.findData(i18n.get_language())))
        self.cmb_lang.blockSignals(False)

    def _on_language_changed(self, _index: int = 0) -> None:
        code = self.cmb_lang.currentData()
        if code:
            self._win.set_language(str(code))
