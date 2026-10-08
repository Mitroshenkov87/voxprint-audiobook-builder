"""Settings dialog (opened with the gear button): everything that is *not* part of the core workflow.

The main window only keeps "choose audio / choose text / create voice".  Service items live here:

* interface language and window transparency (:mod:`infra.ui_prefs`),
* "Check for updates",
* shortcuts to the models folder and the data/log folder,
* "Check & repair" (the program, every component and model checked by hash, missing / broken parts fetched again:
  :mod:`infra.auto_repair`),
* backup / restore of the models and voices to any folder or drive, and the "existing models folder" that is imported
  before anything is downloaded (:mod:`infra.backup`, :mod:`infra.existing_models`),
* "Preload models into memory at startup" (:mod:`infra.preload`; off by default, offered only with enough RAM),
* the speech recognition model (:mod:`infra.asr_choice`: automatic by VRAM, 0.6B, 1.7B or both downloaded),
* "About".

The dialog owns no business logic.  Every button calls back into the :class:`ui.main_window.MainWindow` that created
it, so the same code paths (and tests) are used whether an action starts from here or from the main screen.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core import i18n
from core import pace as pc
from core import pauses as pz
from core.errors import BackupError
from core.i18n import tr
from infra import (
    asr_choice,
    backup,
    existing_models,
    netroute,
    preload,
    sysinfo,
    ui_prefs,
)
from infra import modules as runtime_modules
from ui import screen_fit
from ui.glass import GlassDialog
from workers import backup_runner
from workers.auto_repair_worker import AutoRepairWorker
from workers.backup_worker import BackupWorker

log = logging.getLogger("voxprint.ui.settings")

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers (avoids a circular import at runtime)
    from ui.main_window import MainWindow


class SettingsDialog(GlassDialog):
    """Non-blocking-friendly settings window; one instance is kept by the main window and re-shown on demand."""

    def __init__(self, window: "MainWindow", parent: Optional[QWidget] = None) -> None:
        """Build the dialog for ``window``; the buttons call the matching :class:`MainWindow` methods."""
        super().__init__(parent or window)
        self._win = window
        self.setObjectName("root")
        self.setMinimumWidth(460)
        # the look (Acrylic + transparency level) is set by GlassDialog on every show, for this dialog's own native
        # window - not copied from the main window, which may be hidden (opened from the Studio) and still plain

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(8)

        self.lbl_title = QLabel()
        self.lbl_title.setObjectName("title")
        lay.addWidget(self.lbl_title)

        from ui.window_base import (
            ColumnFlow,  # local: ui.main_window imports this module (circular at load time)
        )

        # two columns that fold into one on a narrow dialog, inside a scroll area: the dialog never grows past the screen
        left_w, right_w = QWidget(), QWidget()
        left, right = QVBoxLayout(left_w), QVBoxLayout(right_w)
        for col in (left, right):
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(8)
        flow = ColumnFlow(min_two=0)
        flow.setObjectName("content")           # QWidget#content: transparent, like the windows' scroll content
        scroll = QScrollArea()
        scroll.setObjectName("settingsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(flow)                  # (setWidget turns on the palette fill: GlassDialog clears it on show)
        scroll.viewport().setAutoFillBackground(False)
        flow.setAutoFillBackground(False)
        lay.addWidget(scroll, 1)

        # --- the four choices (language, transparency, network, speech recognition): one aligned grid, label | combo; the
        # combos share the column, grow with the dialog and never cut their text (they are as wide as the longest entry)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        left.addLayout(grid)

        def add_row(label: QLabel, combo: QComboBox) -> None:
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            r = grid.rowCount()
            grid.addWidget(label, r, 0, Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(combo, r, 1)

        # --- language ---
        self.lbl_language = QLabel()
        self.cmb_lang = screen_fit.ElidedCombo()   # elided + tooltip when too narrow (150 %)
        for code in i18n.LANGS:
            self.cmb_lang.addItem(i18n.LANG_NAMES[code], code)
        self.cmb_lang.setCurrentIndex(max(0, self.cmb_lang.findData(i18n.get_language())))
        add_row(self.lbl_language, self.cmb_lang)

        # --- window transparency (infra/ui_prefs.py): default / more / off ---
        self.lbl_transparency = QLabel()
        self.cmb_transparency = screen_fit.ElidedCombo()   # elided + tooltip when too narrow (150 %)
        for level in ui_prefs.TRANSPARENCY_LEVELS:
            self.cmb_transparency.addItem("", level)
        self.cmb_transparency.setCurrentIndex(max(0, self.cmb_transparency.findData(ui_prefs.transparency())))
        add_row(self.lbl_transparency, self.cmb_transparency)
        self.cmb_transparency.currentIndexChanged.connect(self._on_transparency_changed)

        # --- network interface (infra/netroute.py): Auto / system default / a specific adapter ---
        self.lbl_net = QLabel()
        self.cmb_net = screen_fit.ElidedCombo()   # elided + tooltip when too narrow (150 %)
        add_row(self.lbl_net, self.cmb_net)
        self._fill_net()
        self.cmb_net.currentIndexChanged.connect(self._on_net_changed)

        # --- preload models into RAM at startup: checkbox + whether this PC has the memory for it (infra/preload.py) ---
        self.memory: Callable[[], tuple] = sysinfo.memory                       # injectable (tests)
        self.preload_targets: Callable[[str, bool], list] = preload.find_targets
        prow = QHBoxLayout()
        self.chk_preload = QCheckBox()
        self.lbl_preload = QLabel()
        self.lbl_preload.setObjectName("cardnote")
        self.lbl_preload.setWordWrap(True)
        prow.addWidget(self.chk_preload)
        prow.addWidget(self.lbl_preload, 1)
        left.addLayout(prow)
        self.chk_preload.toggled.connect(self._on_preload_toggled)

        # --- speech recognition model (infra/asr_choice.py): automatic by VRAM / 0.6B / 1.7B / both downloaded ---
        self.asr_resolve: Callable[[str], str] = lambda choice: asr_choice.preferred_repo(choice)   # injectable (tests)
        self.asr_missing: Callable[[], list] = self._default_asr_missing                            # injectable (tests)
        self.lbl_asr = QLabel()
        self.cmb_asr = screen_fit.ElidedCombo()   # elided + tooltip when too narrow (150 %)
        for choice in asr_choice.CHOICES:
            self.cmb_asr.addItem("", choice)
        add_row(self.lbl_asr, self.cmb_asr)
        self.lbl_asr_note = QLabel()
        self.lbl_asr_note.setObjectName("cardnote")
        self.lbl_asr_note.setWordWrap(True)
        left.addWidget(self.lbl_asr_note)
        self.cmb_asr.currentIndexChanged.connect(self._on_asr_changed)

        # --- narration: pause lengths (core/pauses.PauseLengths), reading speed and style (core/pace.py) ---
        self.lbl_narr_title = QLabel()
        self.lbl_narr_title.setObjectName("sectiontitle")
        left.addWidget(self.lbl_narr_title)
        ngrid = QGridLayout()
        ngrid.setHorizontalSpacing(10)
        ngrid.setColumnStretch(1, 1)
        left.addLayout(ngrid)
        self.lbl_pause: dict = {}
        self.spn_pause: dict = {}
        lengths = pz.load_lengths()
        for kind in pz.DEFAULT_LENGTHS_MS:
            lbl, spn = QLabel(), QDoubleSpinBox()
            spn.setRange(0.0, pz.MAX_PAUSE_MS / 1000)
            spn.setSingleStep(0.05)
            spn.setDecimals(2)
            spn.setValue(getattr(lengths, kind) / 1000)
            spn.valueChanged.connect(self._on_pause_changed)
            r = ngrid.rowCount()
            ngrid.addWidget(lbl, r, 0)
            ngrid.addWidget(spn, r, 1)
            self.lbl_pause[kind], self.spn_pause[kind] = lbl, spn
        pace = pc.load()
        self.lbl_speed = QLabel()
        self.sld_speed = QSlider(Qt.Orientation.Horizontal)
        self.sld_speed.setRange(int(pc.MIN_SPEED * 100), int(pc.MAX_SPEED * 100))
        self.sld_speed.setSingleStep(5)
        self.sld_speed.setPageStep(5)
        self.sld_speed.setValue(int(round(pace.speed * 100)))
        self.lbl_speed_value = QLabel()
        srow = QHBoxLayout()
        srow.addWidget(self.sld_speed, 1)
        srow.addWidget(self.lbl_speed_value)
        r = ngrid.rowCount()
        ngrid.addWidget(self.lbl_speed, r, 0)
        ngrid.addLayout(srow, r, 1)
        self.lbl_style = QLabel()
        self.cmb_style = screen_fit.ElidedCombo()
        for style in pc.STYLES:
            self.cmb_style.addItem("", style)
        self.cmb_style.setCurrentIndex(max(0, self.cmb_style.findData(pace.style)))
        r = ngrid.rowCount()
        ngrid.addWidget(self.lbl_style, r, 0)
        ngrid.addWidget(self.cmb_style, r, 1)
        self.lbl_narr_hint = QLabel()
        self.lbl_narr_hint.setObjectName("cardnote")
        self.lbl_narr_hint.setWordWrap(True)
        left.addWidget(self.lbl_narr_hint)
        self.btn_narr_defaults = QPushButton()
        left.addWidget(self.btn_narr_defaults)
        self.sld_speed.valueChanged.connect(self._on_pace_changed)
        self.cmb_style.currentIndexChanged.connect(self._on_pace_changed)
        self.btn_narr_defaults.clicked.connect(self.reset_narration)

        # --- service buttons ---
        self.btn_update = QPushButton()
        self.btn_models = QPushButton()
        self.btn_data = QPushButton()
        self.btn_about = QPushButton()
        self.btn_components = QPushButton()          # thin build only: the runtime modules (infra/modules.py)
        self.btn_diag = QPushButton()                # logs + system information as one zip (infra/diagnostics.py)
        for b in (self.btn_update, self.btn_models, self.btn_data, self.btn_diag, self.btn_components):
            left.addWidget(b)
        self.btn_diag.clicked.connect(self.save_diagnostics)
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
        self.collect_items: Callable[..., list] = backup_runner.collect
        self.backup_worker: Optional[BackupWorker] = None
        # --- projects folder (infra/projects.py): local, not synced; a shortcut in Documents ---
        self.lbl_projects_title = QLabel()
        self.lbl_projects_title.setObjectName("sectiontitle")
        right.addWidget(self.lbl_projects_title)
        self.lbl_projects = QLabel()
        self.lbl_projects.setWordWrap(True)
        self.lbl_projects.setObjectName("cardnote")
        right.addWidget(self.lbl_projects)
        self.lbl_projects_warn = QLabel()
        self.lbl_projects_warn.setWordWrap(True)
        self.lbl_projects_warn.setObjectName("warn")
        right.addWidget(self.lbl_projects_warn)
        prj = QHBoxLayout()
        self.btn_projects_open = QPushButton()
        self.btn_projects_change = QPushButton()
        self.btn_projects_default = QPushButton()
        for b in (self.btn_projects_open, self.btn_projects_change, self.btn_projects_default):
            prj.addWidget(b)
        right.addLayout(prj)
        self.btn_projects_open.clicked.connect(self.open_projects_folder)
        self.btn_projects_change.clicked.connect(self.choose_projects_folder)
        self.btn_projects_default.clicked.connect(lambda: self.set_projects_folder(None))

        self.lbl_backup_title = QLabel()
        self.lbl_backup_title.setObjectName("sectiontitle")
        right.addWidget(self.lbl_backup_title)
        self.chk_models = QCheckBox()
        self.chk_models.setChecked(True)
        self.chk_voices = QCheckBox()
        self.chk_voices.setChecked(True)
        self.chk_link = QCheckBox()
        self.lbl_link_warn = QLabel()
        self.lbl_link_warn.setWordWrap(True)
        self.lbl_link_warn.setObjectName("warn")
        self.lbl_link_warn.setVisible(False)
        self.chk_link.toggled.connect(self.lbl_link_warn.setVisible)
        right.addWidget(self.chk_models)
        right.addWidget(self.chk_voices)
        right.addWidget(self.chk_link)
        right.addWidget(self.lbl_link_warn)
        brow = QHBoxLayout()
        self.btn_backup = QPushButton()
        self.btn_restore = QPushButton()
        brow.addWidget(self.btn_backup)
        brow.addWidget(self.btn_restore)
        right.addLayout(brow)
        self.bar_backup = QProgressBar()
        self.bar_backup.setRange(0, 100)
        self.bar_backup.setVisible(False)
        self.lbl_backup_status = QLabel()
        self.lbl_backup_status.setWordWrap(True)
        self.btn_backup_cancel = QPushButton()
        self.btn_backup_cancel.setVisible(False)
        right.addWidget(self.bar_backup)
        right.addWidget(self.lbl_backup_status)
        right.addWidget(self.btn_backup_cancel)
        self.lbl_existing_title = QLabel()
        self.lbl_existing_title.setObjectName("sectiontitle")
        right.addWidget(self.lbl_existing_title)
        self.lbl_existing_hint = QLabel()
        self.lbl_existing_hint.setWordWrap(True)
        self.lbl_existing_hint.setObjectName("cardnote")
        right.addWidget(self.lbl_existing_hint)
        erow = QHBoxLayout()
        self.lbl_existing = QLabel()
        self.lbl_existing.setWordWrap(True)
        self.btn_existing = QPushButton()
        self.btn_existing_clear = QPushButton()
        erow.addWidget(self.lbl_existing, 1)
        erow.addWidget(self.btn_existing)
        erow.addWidget(self.btn_existing_clear)
        right.addLayout(erow)

        # ONE "Check & repair" button (was "Repair installation" + "Auto-repair"): program, components and models checked by
        # hash, missing / broken parts downloaded again (infra/auto_repair.py); a description and a live per-step line
        self.lbl_repair_title = QLabel()
        self.lbl_repair_title.setObjectName("sectiontitle")
        right.addWidget(self.lbl_repair_title)
        self.lbl_repair_desc = QLabel()
        self.lbl_repair_desc.setObjectName("cardnote")
        self.lbl_repair_desc.setWordWrap(True)
        right.addWidget(self.lbl_repair_desc)
        self.btn_repair_restore = QPushButton()
        right.addWidget(self.btn_repair_restore)
        self.btn_autorepair = QPushButton()
        self.btn_repair = self.btn_autorepair        # old name kept for callers
        right.addWidget(self.btn_autorepair)
        self.autorepair_job: Optional[Callable[..., Any]] = None          # injectable (tests); None = infra.auto_repair.run
        self.autorepair_worker: Optional[AutoRepairWorker] = None
        self.bar_autorepair = QProgressBar()
        self.bar_autorepair.setRange(0, 100)
        self.bar_autorepair.setVisible(False)
        self.lbl_autorepair_status = QLabel()
        self.lbl_autorepair_status.setWordWrap(True)
        right.addWidget(self.bar_autorepair)
        right.addWidget(self.lbl_autorepair_status)
        right.addWidget(self.btn_about)
        flow.add(left_w)
        flow.add(right_w)
        self.btn_close = QPushButton()
        close_row = QHBoxLayout()
        from core import appinfo

        self.lbl_version = QLabel(f"{appinfo.APP_DISPLAY_NAME} {appinfo.version_label()}")   # 0.1.1-beta · build 665 "Tikkun"
        self.lbl_version.setObjectName("hint")
        self.lbl_version.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        close_row.addWidget(self.lbl_version)
        close_row.addStretch(1)
        close_row.addWidget(self.btn_close)
        lay.addLayout(close_row)

        self.cmb_lang.currentIndexChanged.connect(self._on_language_changed)
        self.btn_update.clicked.connect(window.check_updates)
        self.btn_models.clicked.connect(window.open_models_folder)
        self.btn_data.clicked.connect(window.open_data_folder)
        self.btn_autorepair.clicked.connect(self.toggle_autorepair)
        self.btn_about.clicked.connect(window.open_about)
        self.btn_backup.clicked.connect(self.start_backup)
        self.btn_restore.clicked.connect(self.start_restore)
        self.btn_repair_restore.clicked.connect(self.start_restore)
        self.btn_backup_cancel.clicked.connect(self.cancel_backup)
        self.btn_existing.clicked.connect(self.choose_existing_folder)
        self.btn_existing_clear.clicked.connect(self.clear_existing_folder)
        self.btn_close.clicked.connect(self.accept)
        self.retranslate()
        self.refresh()
        hint = max(left_w.sizeHint().height(), right_w.sizeHint().height())
        screen_fit.fit(self, left_w.sizeHint().width() + right_w.sizeHint().width() + 60, hint + 140)

    def _on_transparency_changed(self, _i: int = 0) -> None:
        """Store the level and restyle the open windows (the Studio refreshes all its pages) and this dialog."""
        from ui.main_window import apply_look  # local: circular at load time

        ui_prefs.set_transparency(self.cmb_transparency.currentData())
        refresh = getattr(self._win, "apply_look_all", None)
        if refresh is not None:
            refresh()
        else:
            apply_look(self._win)
        apply_look(self)

    # ------------------------------------------------------------------ narration pauses and pace
    def _on_pause_changed(self, _v: float = 0.0) -> None:
        """Save the pause lengths (used by the next narration)."""
        pz.save_lengths(pz.PauseLengths(**{k: round(s.value() * 1000) for k, s in self.spn_pause.items()}))

    def _on_pace_changed(self, _v: int = 0) -> None:
        """Save speed and style; show the speed in per cent."""
        pc.save(pc.Pace(self.sld_speed.value() / 100, self.cmb_style.currentData()))
        self.lbl_speed_value.setText(tr("narrset.speed_value", pct=self.sld_speed.value()))

    def reset_narration(self) -> None:
        """Back to the default pause lengths, speed 100 % and automatic style."""
        for kind, ms in pz.DEFAULT_LENGTHS_MS.items():
            self.spn_pause[kind].setValue(ms / 1000)
        self.sld_speed.setValue(100)
        self.cmb_style.setCurrentIndex(max(0, self.cmb_style.findData(pc.AUTO)))
        self._on_pause_changed()
        self._on_pace_changed()

    # ------------------------------------------------------------------ texts and state
    def retranslate(self) -> None:
        """Apply the current UI language to every label and button (called on language change)."""
        self.setWindowTitle(tr("ui.settings_title"))
        self.lbl_title.setText(tr("ui.settings_title"))
        self.lbl_language.setText(tr("ui.language"))
        self.lbl_transparency.setText(tr("ui.transparency"))
        for i, text in enumerate((tr("ui.transparency_default"), tr("ui.transparency_more"), tr("ui.transparency_off"))):
            self.cmb_transparency.setItemText(i, text)              # same order as ui_prefs.TRANSPARENCY_LEVELS
        self.lbl_net.setText(tr("ui.net_iface"))
        self.cmb_net.setToolTip(tr("ui.net_iface_tip"))
        self.lbl_net.setToolTip(tr("ui.net_iface_tip"))
        self._fill_net()
        self.btn_update.setText(tr("ui.btn_update"))
        self.btn_components.setText(tr("modules.title"))
        self.btn_models.setText(tr("ui.settings_models_folder"))
        self.btn_data.setText(tr("ui.settings_data_folder"))
        self.btn_diag.setText(tr("diag.button"))
        self.btn_diag.setToolTip(tr("diag.tip"))
        self.lbl_repair_title.setText(tr("autorepair.title"))
        self.lbl_repair_desc.setText(tr("autorepair.desc"))
        self.btn_autorepair.setText(tr("autorepair.stop") if self.autorepair_running else tr("autorepair.button"))
        self.btn_autorepair.setToolTip(tr("autorepair.tip"))
        self.btn_about.setText(tr("ui.about"))
        self.btn_about.setToolTip(tr("ui.about_tip"))
        self.btn_close.setText(tr("about.btn_close"))
        self.lbl_projects_title.setText(tr("projects.title"))
        self.btn_projects_open.setText(tr("projects.open"))
        self.btn_projects_change.setText(tr("projects.change"))
        self.btn_projects_default.setText(tr("projects.default"))
        self.btn_projects_change.setToolTip(tr("projects.tip"))
        self.render_projects()
        self.lbl_backup_title.setText(tr("backup.title"))
        self.chk_models.setText(tr("backup.include_models"))
        self.chk_voices.setText(tr("backup.include_voices"))
        self.chk_link.setText(tr("backup.link_models"))
        self.lbl_link_warn.setText(tr("backup.link_warn"))
        self.btn_backup.setText(tr("backup.btn_backup"))
        self.btn_restore.setText(tr("backup.btn_restore"))
        self.btn_repair_restore.setText(tr("backup.btn_restore_folder"))
        self.btn_backup_cancel.setText(tr("ui.cancel"))
        self.lbl_existing_title.setText(tr("existing.title"))
        self.lbl_existing_hint.setText(tr("existing.hint"))
        self.btn_existing.setText(tr("existing.choose"))
        self.btn_existing_clear.setText(tr("existing.clear"))
        self._render_existing()
        self.chk_preload.setText(tr("preload.option"))
        self.refresh_preload()
        self.lbl_asr.setText(tr("asrmodel.label"))
        for i, text in enumerate((tr("asrmodel.auto"), tr("asrmodel.small"), tr("asrmodel.large"), tr("asrmodel.both"))):
            self.cmb_asr.setItemText(i, text)              # same order as asr_choice.CHOICES
        self.cmb_asr.setToolTip(tr("asrmodel.tip"))
        self.lbl_asr.setToolTip(tr("asrmodel.tip"))
        self.refresh_asr()
        self.lbl_narr_title.setText(tr("narrset.title"))
        names = {"comma": tr("narrset.comma"), "mid": tr("narrset.mid"), "sentence": tr("narrset.sentence"),
                 "paragraph": tr("narrset.paragraph"), "chapter": tr("narrset.chapter")}
        for kind, lbl in self.lbl_pause.items():
            lbl.setText(names[kind])
            self.spn_pause[kind].setSuffix(tr("narrset.seconds"))
        self.lbl_speed.setText(tr("narrset.speed"))
        self.lbl_speed_value.setText(tr("narrset.speed_value", pct=self.sld_speed.value()))
        self.lbl_style.setText(tr("narrset.style"))
        for i, text in enumerate((tr("narrset.style_auto"), tr("narrset.style_scripture"), tr("narrset.style_fiction"),
                                  tr("narrset.style_dialogue"))):
            self.cmb_style.setItemText(i, text)              # same order as pace.STYLES
        self.lbl_narr_hint.setText(tr("narrset.hint"))
        self.btn_narr_defaults.setText(tr("narrset.defaults"))

    # ------------------------------------------------------------------ projects folder
    def render_projects(self) -> None:
        """Path of the projects folder and the OneDrive warning."""
        from infra import projects

        folder = projects.projects_dir()
        self.lbl_projects.setText(tr("projects.current", folder=str(folder)))
        warn = projects.in_onedrive(folder)
        self.lbl_projects_warn.setText(tr("projects.onedrive_warn") if warn else "")
        self.lbl_projects_warn.setVisible(warn)
        self.btn_projects_default.setEnabled(projects.configured_projects_dir() is not None)

    def open_projects_folder(self) -> None:
        from infra import projects
        from ui.main_window import open_folder  # local: circular at load time

        open_folder(projects.projects_dir())

    def choose_projects_folder(self) -> None:
        folder = self.pick_folder(tr("projects.choose"))
        if folder:
            self.set_projects_folder(Path(folder))

    def set_projects_folder(self, folder: Optional[Path]) -> None:
        """Use ``folder`` (None = the default) for new projects; existing projects stay where they are.  The Documents
        shortcut is re-pointed in the background."""
        import threading

        from infra import projects

        used = projects.set_projects_dir(folder)
        threading.Thread(target=projects.ensure_shortcut, args=(used,), name="projects-shortcut", daemon=True).start()
        self.render_projects()

    # ------------------------------------------------------------------ preload models at startup
    def preload_availability(self) -> preload.Availability:
        """What preloading would need on this PC right now (installed models of the current voice, RAM)."""
        pre = getattr(self._win, "preloader", None)
        voice = pre.voice_fn() if pre is not None else None
        try:
            targets = self.preload_targets(getattr(voice, "base_model", "") or "", preload.cuda_likely())
        except Exception:  # noqa: BLE001 - a broken models folder must not break the dialog
            log.warning("could not list the models to preload", exc_info=True)
            targets = []
        total, available = self.memory()
        return preload.availability(targets, total, available, pre.held_bytes() if pre is not None else 0)

    def refresh_preload(self) -> None:
        """Checkbox state and the availability note next to it."""
        av = self.preload_availability()
        need, total = preload.gb(av.need, up=True), preload.gb(av.total)
        if av.status == preload.OK:
            note = tr("preload.available", total=total)
        elif av.status == preload.TOO_SMALL:
            note = tr("preload.too_small", need=need, total=total)
        elif av.status == preload.LOW_NOW:
            note = tr("preload.low_now", free=preload.gb(av.available), need=preload.gb(av.models, up=True))
        elif av.status == preload.NOTHING:
            note = tr("preload.no_models")
        else:
            note = tr("preload.unknown")
        on = preload.enabled()
        self.chk_preload.blockSignals(True)
        self.chk_preload.setChecked(on)
        self.chk_preload.setEnabled(av.offered or on)        # a ticked box can always be unticked
        self.chk_preload.blockSignals(False)
        self.lbl_preload.setText(note)
        tip = tr("preload.tip", need=need)
        self.chk_preload.setToolTip(tip)
        self.lbl_preload.setToolTip(tip)

    def _on_preload_toggled(self, on: bool) -> None:
        """Remember the choice; unticking frees the preloaded models at once."""
        pre = getattr(self._win, "preloader", None)
        if pre is not None:
            pre.set_enabled(on)
        else:
            preload.set_enabled(on)
        self.refresh_preload()

    # ------------------------------------------------------------------ speech recognition model
    @staticmethod
    def _default_asr_missing() -> list:
        from workers.pipeline_runner import models_missing

        return [r for r in models_missing() if r in (asr_choice.SMALL, asr_choice.LARGE)]

    def refresh_asr(self) -> None:
        """Select the saved choice and say which model is used and whether it still has to be downloaded."""
        choice = asr_choice.preference()
        self.cmb_asr.blockSignals(True)
        self.cmb_asr.setCurrentIndex(max(0, self.cmb_asr.findData(choice)))
        self.cmb_asr.blockSignals(False)
        try:
            repo, missing = self.asr_resolve(choice), self.asr_missing()
        except Exception:  # noqa: BLE001 - GPU detection / a broken models folder must not break the dialog
            log.warning("could not resolve the speech recognition model", exc_info=True)
            self.lbl_asr_note.setText("")
            return
        name = repo.split("/")[-1]
        self.lbl_asr_note.setText(tr("asrmodel.note_missing", model=name) if missing else tr("asrmodel.note_ready", model=name))

    def _on_asr_changed(self, _index: int = 0) -> None:
        """Remember the choice; a model that is not installed yet is fetched right away by the normal "download all" step
        (visible progress in the main window), never later in the middle of a task."""
        choice = self.cmb_asr.currentData()
        if choice not in asr_choice.CHOICES:
            return
        asr_choice.set_preference(choice)
        try:
            missing = self.asr_missing()
        except Exception:  # noqa: BLE001
            missing = []
        start = getattr(self._win, "start_prefetch", None)
        if missing and start is not None and not self._win.busy:
            start()
        self.refresh_asr()

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
    def _default_pick_save(self, title: str, name: str) -> str:
        """Save-as chooser for the diagnostic report (replaced in tests)."""
        return QFileDialog.getSaveFileName(self, title, str(Path.home() / name), "Zip (*.zip)")[0]

    def save_diagnostics(self) -> Optional[Path]:
        """Settings -> *Save diagnostic report...*: zip the logs, system information and settings to a chosen file."""
        import time

        from infra import diagnostics

        name = time.strftime("voxprint-diagnostics-%Y%m%d-%H%M%S.zip")
        target = (getattr(self, "pick_save", None) or self._default_pick_save)(tr("diag.button"), name)
        if not target:
            return None
        try:
            path = diagnostics.write_report(Path(target))
        except OSError as exc:
            self.notify(tr("diag.button"), tr("diag.failed", err=str(exc)))
            return None
        self.notify(tr("diag.button"), tr("diag.saved", path=str(path)))
        return path

    def _default_pick_folder(self, title: str) -> str:
        """Folder chooser (replaced in tests)."""
        return QFileDialog.getExistingDirectory(self, title)

    def _default_confirm(self, title: str, text: str) -> bool:
        """Yes / no question (replaced in tests)."""
        from ui import std_buttons

        return std_buttons.question(self, title, text)

    def _default_notify(self, title: str, text: str) -> None:
        """Information box (replaced in tests)."""
        from ui import std_buttons

        std_buttons.information(self, title, text)

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
        include_voices = self.chk_voices.isChecked()
        include_models = self.chk_models.isChecked()
        try:
            items = self.collect_items(include_voices, include_models)
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
        self._run(lambda prog, cancel: self.backup_job(Path(target), include_voices, prog, cancel, items=items))
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
        link = self.chk_link.isChecked()
        root = backup.find_backup(Path(src))
        if root is None:
            self._set_status(tr("backup.err_no_manifest", path=str(src)), False)
            self.notify(tr("backup.btn_restore"), tr("backup.err_no_manifest", path=str(src)))
            return False
        try:
            items = [i for i in backup.items_from_manifest(backup.read_manifest(root)) if i.complete
                     and (include or i.kind != backup.KIND_VOICES)
                     and not (link and i.kind == backup.KIND_MODEL)]
        except BackupError as exc:
            self._set_status(exc.user_message, False)
            return False
        size = backup.format_size(sum(i.size for i in items))
        question = tr("backup.confirm_restore", size=size, source=str(root))
        if link:
            question += "\n\n" + tr("backup.link_warn")
        if not self.confirm(tr("backup.btn_restore"), question):
            return False
        self._run(lambda prog, cancel: self.restore_job(root, include, prog, cancel, link=link))
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
        problems = list(getattr(report, "problems", ()) or ())
        if problems:
            text += " " + tr("backup.done_problems", names=", ".join(problems))
        external = getattr(report, "external", "") or ""
        if external:
            text += " " + tr("backup.linked", path=external)
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
        self.cmb_asr.setEnabled(not busy)            # a running task keeps the recogniser it started with
        self.btn_update.setEnabled(not busy and not self._win.updating)
        self.btn_autorepair.setEnabled(self.autorepair_running or (not busy and not self._win.repairing))
        idle = not busy and not self.backing_up
        for b in (self.btn_backup, self.btn_restore, self.btn_repair_restore, self.btn_existing,
                  self.chk_voices, self.chk_models, self.chk_link):
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
