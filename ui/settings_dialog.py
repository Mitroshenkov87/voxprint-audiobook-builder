"""Settings dialog (opened with the gear button): everything that is *not* part of the core workflow.

The main window only keeps "choose audio / choose text / create voice".  Service items live here:

* interface language,
* "Check for updates",
* shortcuts to the models folder and the data/log folder,
* "Repair installation" and "Auto-repair" (every component and model checked by hash, missing / broken parts fetched
  again: :mod:`infra.auto_repair`),
* backup / restore of the models and voices to any folder or drive, and the "existing models folder" that is imported
  before anything is downloaded (:mod:`infra.backup`, :mod:`infra.existing_models`),
* "About".

The dialog owns no business logic.  Every button calls back into the :class:`ui.main_window.MainWindow` that created
it, so the same code paths (and tests) are used whether an action starts from here or from the main screen.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QProgressBar,
                               QPushButton, QVBoxLayout, QWidget)

from core import i18n
from core.errors import BackupError
from core.i18n import tr
from infra import backup, existing_models, modules as runtime_modules, netroute
from workers import backup_runner
from workers.auto_repair_worker import AutoRepairWorker
from workers.backup_worker import BackupWorker

log = logging.getLogger("voxprint.ui.settings")

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers (avoids a circular import at runtime)
    from ui.main_window import MainWindow


class SettingsDialog(QDialog):
    """Non-blocking-friendly settings window; one instance is kept by the main window and re-shown on demand."""

    def __init__(self, window: "MainWindow", parent: Optional[QWidget] = None) -> None:
        """Build the dialog for ``window``; the buttons call the matching :class:`MainWindow` methods."""
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

        # --- network interface (infra/netroute.py): Auto / system default / a specific adapter ---
        nrow = QHBoxLayout()
        self.lbl_net = QLabel()
        self.cmb_net = QComboBox()
        self.cmb_net.setMinimumWidth(220)
        nrow.addWidget(self.lbl_net)
        nrow.addStretch(1)
        nrow.addWidget(self.cmb_net)
        lay.addLayout(nrow)
        self._fill_net()
        self.cmb_net.currentIndexChanged.connect(self._on_net_changed)

        # --- service buttons ---
        self.btn_update = QPushButton()
        self.btn_models = QPushButton()
        self.btn_data = QPushButton()
        self.btn_repair = QPushButton()
        self.btn_about = QPushButton()
        self.btn_components = QPushButton()          # thin build only: the runtime modules (infra/modules.py)
        for b in (self.btn_update, self.btn_models, self.btn_data, self.btn_components):
            lay.addWidget(b)
        self.btn_components.setVisible(runtime_modules.is_thin())
        self.btn_components.clicked.connect(self.open_components)

        # (No "Maximum quality (auto)" button any more: every recommended option is pre-selected and the first start downloads
        # ALL models automatically - see main._offer_components / workers.pipeline_runner.prefetch_models.)

        # --- backup / restore and the existing models folder ---
        self.pick_folder: Callable[[str], str] = self._default_pick_folder      # injectable (tests)
        self.confirm: Callable[[str, str], bool] = self._default_confirm
        self.notify: Callable[[str, str], None] = self._default_notify
        self.backup_job: Callable[..., Any] = backup_runner.run_backup_job
        self.restore_job: Callable[..., Any] = backup_runner.run_restore_job
        self.import_job: Callable[..., Any] = backup_runner.run_import_job
        self.collect_items: Callable[[bool], list] = backup_runner.collect
        self.backup_worker: Optional[BackupWorker] = None
        self.lbl_backup_title = QLabel()
        self.lbl_backup_title.setObjectName("sectiontitle")
        lay.addWidget(self.lbl_backup_title)
        self.chk_voices = QCheckBox()
        self.chk_voices.setChecked(True)
        lay.addWidget(self.chk_voices)
        brow = QHBoxLayout()
        self.btn_backup = QPushButton()
        self.btn_restore = QPushButton()
        brow.addWidget(self.btn_backup)
        brow.addWidget(self.btn_restore)
        lay.addLayout(brow)
        self.bar_backup = QProgressBar()
        self.bar_backup.setRange(0, 100)
        self.bar_backup.setVisible(False)
        self.lbl_backup_status = QLabel()
        self.lbl_backup_status.setWordWrap(True)
        self.btn_backup_cancel = QPushButton()
        self.btn_backup_cancel.setVisible(False)
        lay.addWidget(self.bar_backup)
        lay.addWidget(self.lbl_backup_status)
        lay.addWidget(self.btn_backup_cancel)
        self.lbl_existing_title = QLabel()
        self.lbl_existing_title.setObjectName("sectiontitle")
        lay.addWidget(self.lbl_existing_title)
        self.lbl_existing_hint = QLabel()
        self.lbl_existing_hint.setWordWrap(True)
        self.lbl_existing_hint.setObjectName("cardnote")
        lay.addWidget(self.lbl_existing_hint)
        erow = QHBoxLayout()
        self.lbl_existing = QLabel()
        self.lbl_existing.setWordWrap(True)
        self.btn_existing = QPushButton()
        self.btn_existing_clear = QPushButton()
        erow.addWidget(self.lbl_existing, 1)
        erow.addWidget(self.btn_existing)
        erow.addWidget(self.btn_existing_clear)
        lay.addLayout(erow)

        rrow = QHBoxLayout()                      # Repair (environment only) | Auto-repair (everything, by hash)
        self.btn_autorepair = QPushButton()
        rrow.addWidget(self.btn_repair)
        rrow.addWidget(self.btn_autorepair)
        lay.addLayout(rrow)
        self.autorepair_job: Optional[Callable[..., Any]] = None          # injectable (tests); None = infra.auto_repair.run
        self.autorepair_worker: Optional[AutoRepairWorker] = None
        self.bar_autorepair = QProgressBar()
        self.bar_autorepair.setRange(0, 100)
        self.bar_autorepair.setVisible(False)
        self.lbl_autorepair_status = QLabel()
        self.lbl_autorepair_status.setWordWrap(True)
        lay.addWidget(self.bar_autorepair)
        lay.addWidget(self.lbl_autorepair_status)
        lay.addWidget(self.btn_about)
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
        self.btn_autorepair.clicked.connect(self.toggle_autorepair)
        self.btn_about.clicked.connect(window.open_about)
        self.btn_backup.clicked.connect(self.start_backup)
        self.btn_restore.clicked.connect(self.start_restore)
        self.btn_backup_cancel.clicked.connect(self.cancel_backup)
        self.btn_existing.clicked.connect(self.choose_existing_folder)
        self.btn_existing_clear.clicked.connect(self.clear_existing_folder)
        self.btn_close.clicked.connect(self.accept)
        self.retranslate()
        self.refresh()

    # ------------------------------------------------------------------ texts and state
    def retranslate(self) -> None:
        """Apply the current UI language to every label and button (called on language change)."""
        self.setWindowTitle(tr("ui.settings_title"))
        self.lbl_title.setText(tr("ui.settings_title"))
        self.lbl_language.setText(tr("ui.language"))
        self.lbl_net.setText(tr("ui.net_iface"))
        self.cmb_net.setToolTip(tr("ui.net_iface_tip"))
        self.lbl_net.setToolTip(tr("ui.net_iface_tip"))
        self._fill_net()
        self.btn_update.setText(tr("ui.btn_update"))
        self.btn_components.setText(tr("modules.title"))
        self.btn_models.setText(tr("ui.settings_models_folder"))
        self.btn_data.setText(tr("ui.settings_data_folder"))
        self.btn_repair.setText(tr("ui.settings_repair"))
        self.btn_repair.setToolTip(tr("ui.settings_repair_tip"))
        self.btn_autorepair.setText(tr("autorepair.stop") if self.autorepair_running else tr("autorepair.button"))
        self.btn_autorepair.setToolTip(tr("autorepair.tip"))
        self.btn_about.setText(tr("ui.about"))
        self.btn_about.setToolTip(tr("ui.about_tip"))
        self.btn_close.setText(tr("about.btn_close"))
        self.lbl_backup_title.setText(tr("backup.title"))
        self.chk_voices.setText(tr("backup.include_voices"))
        self.btn_backup.setText(tr("backup.btn_backup"))
        self.btn_restore.setText(tr("backup.btn_restore"))
        self.btn_backup_cancel.setText(tr("ui.cancel"))
        self.lbl_existing_title.setText(tr("existing.title"))
        self.lbl_existing_hint.setText(tr("existing.hint"))
        self.btn_existing.setText(tr("existing.choose"))
        self.btn_existing_clear.setText(tr("existing.clear"))
        self._render_existing()

    # ------------------------------------------------------------------ runtime modules (thin build)
    def open_components(self) -> None:
        """Show the Components window (download / check the runtime modules)."""
        from ui.modules_dialog import ModulesDialog, default_extras

        dlg = ModulesDialog(self.styleSheet(), self, extras_fn=default_extras)      # SAGE comes with the components
        self._components = dlg
        dlg.refresh()
        dlg.open()

    # ------------------------------------------------------------------ network interface
    def _fill_net(self) -> None:
        """Items: Automatic, System default only, then every local interface/address (the saved choice stays selectable)."""
        if not hasattr(self, "cmb_net"):
            return
        pref = netroute.preference()
        self.cmb_net.blockSignals(True)
        self.cmb_net.clear()
        self.cmb_net.addItem(tr("ui.net_iface_auto"), "auto")
        self.cmb_net.addItem(tr("ui.net_iface_default"), "default")
        seen = set()
        try:
            addrs = netroute.local_addresses()
        except Exception:  # noqa: BLE001
            addrs = []
        for name, ip in addrs:
            if name not in seen:
                seen.add(name)
                self.cmb_net.addItem(f"{name} ({ip})" if name != ip else ip, name)
        i = self.cmb_net.findData(pref)
        if i < 0 and pref not in ("auto", "default"):
            self.cmb_net.addItem(pref, pref)             # a saved adapter that is not up right now
            i = self.cmb_net.count() - 1
        self.cmb_net.setCurrentIndex(max(0, i))
        self.cmb_net.blockSignals(False)

    def _on_net_changed(self, _index: int = 0) -> None:
        """Save the choice (``net_iface.txt`` in the state folder; the next download uses it)."""
        value = self.cmb_net.currentData()
        try:
            netroute.set_preference(str(value or "auto"))
            netroute.forget()
        except OSError:
            log.warning("could not save the network interface choice")

    # ------------------------------------------------------------------ Auto-repair
    @property
    def autorepair_running(self) -> bool:
        """True while the auto-repair runs."""
        return bool(self.autorepair_worker and self.autorepair_worker.isRunning())

    def toggle_autorepair(self) -> bool:
        """Start the auto-repair (or stop the running one).  Returns True when a run was started."""
        if self.autorepair_running:
            self.autorepair_worker.cancel()
            self.lbl_autorepair_status.setText(tr("ui.stopping"))
            return False
        if self._win.busy or self._win.repairing:
            return False
        w = AutoRepairWorker(self.autorepair_job, parent=self)
        w.progress.connect(self._on_autorepair_progress)
        w.done.connect(self._on_autorepair_done)
        w.failed.connect(lambda m: self._autorepair_finished(tr("autorepair.error", error=m)))
        w.cancelled.connect(lambda: self._autorepair_finished(tr("autorepair.cancelled")))
        self.autorepair_worker = w
        self.bar_autorepair.setValue(0)
        self.bar_autorepair.setVisible(True)
        self.lbl_autorepair_status.setText(tr("autorepair.running"))
        w.start()
        self.btn_autorepair.setText(tr("autorepair.stop"))
        self.refresh()
        return True

    def _on_autorepair_progress(self, fraction: float, message: str) -> None:
        self.bar_autorepair.setValue(int(max(0.0, min(1.0, fraction)) * 100))
        if message:
            self.lbl_autorepair_status.setText(message)

    def _on_autorepair_done(self, report) -> None:
        self.bar_autorepair.setValue(100)
        self._autorepair_finished(report.summary())

    def _autorepair_finished(self, text: str) -> None:
        if self.autorepair_worker:
            self.autorepair_worker.wait(2000)
        self.bar_autorepair.setVisible(False)
        self.lbl_autorepair_status.setText(text)
        self.btn_autorepair.setText(tr("autorepair.button"))
        self.refresh()

    # ------------------------------------------------------------------ backup / restore
    def _default_pick_folder(self, title: str) -> str:
        """Folder chooser (replaced in tests)."""
        return QFileDialog.getExistingDirectory(self, title)

    def _default_confirm(self, title: str, text: str) -> bool:
        """Yes / no question (replaced in tests)."""
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def _default_notify(self, title: str, text: str) -> None:
        """Information box (replaced in tests)."""
        QMessageBox.information(self, title, text)

    @property
    def backing_up(self) -> bool:
        """True while a backup / restore runs."""
        return bool(self.backup_worker and self.backup_worker.isRunning())

    def _render_existing(self) -> None:
        """Show the configured existing-models folder."""
        t = existing_models.configured_text()
        self.lbl_existing.setText(tr("existing.current", path=t) if t else tr("existing.none"))
        self.btn_existing_clear.setEnabled(bool(t) and not self.backing_up)

    def _set_status(self, text: str, busy: bool) -> None:
        """Status line, progress bar and Cancel button of a running backup / restore."""
        self.lbl_backup_status.setText(text)
        self.bar_backup.setVisible(busy)
        self.btn_backup_cancel.setVisible(busy)
        self.refresh()

    def start_backup(self) -> bool:
        """Ask for a target, show what will be copied and start the backup in the background."""
        if self._win.busy or self.backing_up:
            self.notify(tr("backup.title"), tr("backup.busy"))
            return False
        target = self.pick_folder(tr("backup.choose_target"))
        if not target:
            return False
        include = self.chk_voices.isChecked()
        try:
            items = self.collect_items(include)
        except Exception as exc:  # noqa: BLE001 - unreadable folders must not crash the dialog
            log.exception("collecting the backup items failed")
            self._set_status(tr("backup.err_io", path="", error=str(exc)), False)
            return False
        if not items:
            self.notify(tr("backup.title"), tr("backup.empty"))
            return False
        total = sum(i.size for i in items)
        files = sum(len(i.files) for i in items)
        try:
            free = backup.format_size(backup.free_bytes(Path(target)))
        except OSError:
            free = "?"
        if not self.confirm(tr("backup.btn_backup"), tr("backup.confirm_backup", size=backup.format_size(total), files=files,
                                                        target=str(Path(target)), free=free)):
            return False
        self._run(lambda prog, cancel: self.backup_job(Path(target), include, prog, cancel, items=items))
        return True

    def start_restore(self) -> bool:
        """Ask for a backup folder, show what it holds and restore it in the background."""
        if self._win.busy or self.backing_up:
            self.notify(tr("backup.title"), tr("backup.busy"))
            return False
        src = self.pick_folder(tr("backup.choose_source"))
        if not src:
            return False
        include = self.chk_voices.isChecked()
        root = backup.find_backup(Path(src))
        if root is None:
            self._set_status(tr("backup.err_no_manifest", path=str(src)), False)
            self.notify(tr("backup.btn_restore"), tr("backup.err_no_manifest", path=str(src)))
            return False
        try:
            items = [i for i in backup.items_from_manifest(backup.read_manifest(root)) if i.complete
                     and (include or i.kind != backup.KIND_VOICES)]
        except BackupError as exc:
            self._set_status(exc.user_message, False)
            return False
        size = backup.format_size(sum(i.size for i in items))
        if not self.confirm(tr("backup.btn_restore"), tr("backup.confirm_restore", size=size, source=str(root))):
            return False
        self._run(lambda prog, cancel: self.restore_job(root, include, prog, cancel))
        return True

    def _run(self, job: Callable[..., Any]) -> None:
        """Start ``job(progress, cancel)`` in a :class:`BackupWorker`."""
        w = BackupWorker(job, parent=self)
        w.progress.connect(self._on_backup_progress)
        w.done.connect(self._on_backup_done)
        w.failed.connect(self._on_backup_failed)
        w.cancelled.connect(self._on_backup_cancelled)
        self.backup_worker = w
        self.bar_backup.setValue(0)
        w.start()
        self._set_status(tr("backup.scanning"), True)

    def cancel_backup(self) -> None:
        """Stop the running backup / restore (finished files are kept)."""
        if self.backup_worker and self.backup_worker.isRunning():
            self.backup_worker.cancel()

    def _on_backup_progress(self, fraction: float, message: str) -> None:
        self.bar_backup.setValue(int(max(0.0, min(1.0, fraction)) * 100))
        if message:
            self.lbl_backup_status.setText(message)

    def _finish(self, text: str) -> None:
        if self.backup_worker:
            self.backup_worker.wait(2000)
        self._set_status(text, False)
        self._render_existing()

    def _on_backup_done(self, report) -> None:
        text = tr("backup.done", copied=report.copied_files, skipped=report.skipped_files,
                  size=backup.format_size(report.copied_bytes))
        if report.conflicts:
            text += " " + tr("backup.done_conflicts", names=", ".join(report.conflicts))
        self.bar_backup.setValue(100)
        self._finish(text)

    def _on_backup_failed(self, message: str) -> None:
        self._finish(message)

    def _on_backup_cancelled(self) -> None:
        self._finish(tr("backup.cancelled"))

    # ------------------------------------------------------------------ existing models folder
    def choose_existing_folder(self) -> bool:
        """Remember a folder with models from a previous install; they are imported before anything is downloaded."""
        folder = self.pick_folder(tr("existing.pick"))
        if not folder:
            return False
        existing_models.set_folder(Path(folder))
        self._render_existing()
        self.lbl_backup_status.setText(tr("existing.saved"))
        if not self._win.busy and not self.backing_up and self.confirm(tr("existing.title"), tr("existing.import_now")):
            self._run(lambda prog, cancel: self.import_job(prog, cancel))      # copies / links now; nothing is downloaded
        return True

    def clear_existing_folder(self) -> None:
        """Forget the folder (already imported models stay)."""
        existing_models.set_folder(None)
        self._render_existing()

    def refresh(self) -> None:
        """Enable/disable controls according to the main window's busy state."""
        busy = self._win.busy
        self.cmb_lang.setEnabled(not busy)           # switching language mid-task would relabel a running job
        self.btn_update.setEnabled(not busy and not self._win.updating)
        self.btn_repair.setEnabled(not busy and not self._win.repairing and not self.autorepair_running)
        self.btn_autorepair.setEnabled(self.autorepair_running or (not busy and not self._win.repairing))
        idle = not busy and not self.backing_up
        for b in (self.btn_backup, self.btn_restore, self.btn_existing, self.chk_voices):
            b.setEnabled(idle)
        self.btn_existing_clear.setEnabled(idle and bool(existing_models.configured_text()))

    def sync_language(self) -> None:
        """Make the combo box show the active language without re-triggering a language change."""
        self.cmb_lang.blockSignals(True)
        self.cmb_lang.setCurrentIndex(max(0, self.cmb_lang.findData(i18n.get_language())))
        self.cmb_lang.blockSignals(False)

    def _on_language_changed(self, _index: int = 0) -> None:
        """The combo box selection changed: switch the whole UI to that language."""
        code = self.cmb_lang.currentData()
        if code:
            self._win.set_language(str(code))
