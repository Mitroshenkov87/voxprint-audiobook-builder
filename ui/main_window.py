"""Voxprint main window (PySide6): choose the audio, choose the text, press one button - the rest is automatic.

The main screen keeps only the core workflow (file pickers, "Create voice", optional voice type/description, the
universal-model and dataset buttons, progress, stage chips, status).  Service items (language, updates, model/data
folders, repair, About) live in the Settings dialog (:mod:`ui.settings_dialog`) behind the gear button.  The whole UI
sits in a scroll area so the window stays usable on short screens (e.g. 1366x768 at 150% scaling).

The window never does heavy work itself: tasks run in the QThread workers of :mod:`workers.process_worker`; the
window only reacts to their signals.  Almost every collaborator (runner, updater, health check ...) is injectable
through the constructor, which is how the tests drive it without a GPU, network or real dialogs.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from core import i18n, model_export, voice_info
from core.errors import DatasetMakerError
from core.events import Stage
from core.i18n import tr
from infra import paths, platform_win
from workers.pipeline_runner import (KIND_DATASET, KIND_LORA, KIND_MERGE, TaskRequest, last_adapter, plan_for,
                                     run_task)
from ui.settings_dialog import SettingsDialog
from workers.process_worker import PrefetchWorker, ProcessWorker, RepairWorker, StatusWorker, UpdateWorker

log = logging.getLogger("voxprint.ui")

APP_TITLE = "Voxprint"
#: Hard minimum of the window; the content scrolls below its natural size (see ``_fit_to_screen``).
MIN_WINDOW_W, MIN_WINDOW_H = 480, 320
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".opus", ".mp4"}
TEXT_EXT = {".txt"}
ALL_STAGES: List[Stage] = [Stage.UPDATES, Stage.MODEL, Stage.ALIGN, Stage.SLICE, Stage.TRAIN, Stage.SAVE]

# Dark theme.  Design rules (WCAG AA, i.e. >= 4.5:1 for all readable text, see tests/test_ui.py::test_theme_contrast):
#  * every text colour is OPAQUE - translucent text would take on whatever the blurred desktop behind it looks like;
#  * panels/buttons/inputs are (almost) solid dark fills, so the contrast does not depend on the backdrop;
#  * only the window background itself is translucent in Acrylic mode, with a strong dark tint (~87 % opaque),
#    which keeps a subtle glass feel while the worst case (a white desktop) still ends up dark.
# ``{root_bg}`` / ``{card_bg}`` are the only parts that differ between the Acrylic and the plain fallback look.
TEXT = "#f2f2f5"            # primary text
TEXT_MUTED = "#c4c4d0"      # secondary text (subtitle, file name, status)
TEXT_FAINT = "#a8a8b8"      # tertiary text (hints, footer, inactive stage chips)
TEXT_DISABLED = "#8e8e9e"   # disabled controls (exempt from WCAG, still legible)
TEXT_ON_ACCENT = "#0b1220"  # dark text on the light-blue accent
ACCENT = "#60a5fa"
ACCENT_HOVER = "#7db9ff"
ACCENT_SOFT = "#93c5fd"
ACCENT_STRONG = "#2563eb"   # progress chunk / selection: white text on it passes AA
ROOT_PLAIN = "#17171c"
ROOT_GLASS = "rgba(16,16,22,222)"
CARD_PLAIN = "#202029"
CARD_GLASS = "rgba(34,34,44,238)"
CONTROL_BG = "#2c2c38"      # buttons, combo boxes, line edits
CONTROL_HOVER = "#383846"
CONTROL_BORDER = "#4b4b5c"

STYLE_TEMPLATE = """
* {{{{ font-family: "Segoe UI Variable Text", "Segoe UI", sans-serif; font-size: 14px; color: {text}; }}}}
QWidget#root {{{{ background: {{root_bg}}; }}}}
QLabel#title {{{{ font-size: 26px; font-weight: 600; }}}}
QLabel#subtitle {{{{ color: {muted}; }}}}
QFrame#card {{{{ background: {{card_bg}}; border: 1px solid {border}; border-radius: 12px; }}}}
QLabel#fileLabel {{{{ color: {muted}; }}}}
QPushButton {{{{ background: {control}; border: 1px solid {border}; border-radius: 8px; padding: 8px 16px; }}}}
QPushButton:hover {{{{ background: {hover}; }}}}
QPushButton:pressed {{{{ background: #23232d; }}}}
QPushButton:disabled {{{{ color: {disabled}; background: #1f1f28; border-color: #34343f; }}}}
QPushButton#primary {{{{ background: {accent}; border: 1px solid {soft}; font-size: 17px;
                      font-weight: 600; padding: 14px 20px; color: {on_accent}; }}}}
QPushButton#primary:hover {{{{ background: {accent_hover}; }}}}
QPushButton#primary:disabled {{{{ background: #2a3a55; color: #9db0cc; border-color: #3a4d6e; }}}}
QProgressBar {{{{ background: {control}; border: 1px solid {border}; border-radius: 8px;
               height: 16px; text-align: center; }}}}
QProgressBar::chunk {{{{ background: {strong}; border-radius: 7px; }}}}
QLabel#chip {{{{ color: {faint}; padding: 3px 8px; border-radius: 10px; font-size: 12px; }}}}
QLabel#chip[state="active"] {{{{ color: {on_accent}; background: {soft}; font-weight: 600; }}}}
QLabel#chip[state="done"] {{{{ color: #dcdce4; background: #34343f; }}}}
QLabel#status {{{{ color: {muted}; }}}}
QLabel#ready {{{{ font-size: 22px; font-weight: 600; color: #86efac; }}}}
QLabel#footer {{{{ color: {faint}; font-size: 12px; }}}}
QLabel#hint {{{{ color: {faint}; font-size: 12px; padding-left: 4px; }}}}
QComboBox {{{{ background: {control}; border: 1px solid {border}; border-radius: 8px;
            padding: 6px 12px; min-width: 110px; }}}}
QComboBox:disabled {{{{ color: {disabled}; }}}}
QComboBox QAbstractItemView {{{{ background: #23232b; border: 1px solid {border};
                              selection-background-color: {strong}; }}}}
QLineEdit {{{{ background: {control}; border: 1px solid {border}; border-radius: 8px;
            padding: 6px 12px; selection-background-color: {strong}; }}}}
QPushButton#gear {{{{ padding: 6px 12px; font-size: 18px; }}}}
QScrollArea {{{{ background: transparent; border: none; }}}}
QWidget#content {{{{ background: transparent; }}}}
QScrollBar:vertical {{{{ background: transparent; width: 10px; margin: 2px; }}}}
QScrollBar::handle:vertical {{{{ background: #5a5a6c; border-radius: 4px; min-height: 30px; }}}}
QScrollBar:horizontal {{{{ background: transparent; height: 10px; margin: 2px; }}}}
QScrollBar::handle:horizontal {{{{ background: #5a5a6c; border-radius: 4px; min-width: 30px; }}}}
QScrollBar::add-line, QScrollBar::sub-line {{{{ width: 0; height: 0; }}}}
QDialog#root, QTextBrowser {{{{ color: {text}; }}}}
QDialog#root {{{{ background: {root_plain}; }}}}
""".format(text=TEXT, muted=TEXT_MUTED, faint=TEXT_FAINT, disabled=TEXT_DISABLED, on_accent=TEXT_ON_ACCENT,
           accent=ACCENT, accent_hover=ACCENT_HOVER, soft=ACCENT_SOFT, strong=ACCENT_STRONG,
           control=CONTROL_BG, hover=CONTROL_HOVER, border=CONTROL_BORDER, root_plain=ROOT_PLAIN)

# (foreground, background) pairs that must keep >= 4.5:1; checked by tests/test_ui.py::test_theme_contrast.
CONTRAST_PAIRS = [
    (TEXT, ROOT_PLAIN), (TEXT, CARD_PLAIN), (TEXT, CONTROL_BG), (TEXT, CONTROL_HOVER),
    (TEXT_MUTED, ROOT_PLAIN), (TEXT_MUTED, CARD_PLAIN), (TEXT_FAINT, ROOT_PLAIN), (TEXT_FAINT, CARD_PLAIN),
    (TEXT_ON_ACCENT, ACCENT), (TEXT_ON_ACCENT, ACCENT_HOVER), (TEXT_ON_ACCENT, ACCENT_SOFT),
    (TEXT, ACCENT_STRONG), ("#dcdce4", "#34343f"), ("#86efac", CARD_PLAIN),
]


def privacy_marker() -> Path:
    """State file whose existence means the user has seen the privacy notice."""
    return paths.state_dir() / "privacy_ack"


def privacy_acknowledged() -> bool:
    """True if the first-run privacy notice was already shown."""
    return privacy_marker().exists()


def acknowledge_privacy() -> None:
    """Remember that the privacy notice was shown (failure to write is only logged)."""
    try:
        privacy_marker().write_text("1", encoding="utf-8")
    except OSError:
        log.warning("cannot store privacy ack")



def build_style(glass: bool) -> str:
    """Return the stylesheet: ``glass`` = Acrylic backdrop behind a translucent window, else the solid fallback."""
    if glass:
        return STYLE_TEMPLATE.format(root_bg=ROOT_GLASS, card_bg=CARD_GLASS)
    return STYLE_TEMPLATE.format(root_bg=ROOT_PLAIN, card_bg=CARD_PLAIN)


def open_folder(path: Path) -> None:
    """Open a folder in Explorer (Windows) or the default file manager."""
    p = str(path)
    if sys.platform == "win32":
        try:
            os.startfile(p)  # type: ignore[attr-defined]  # noqa: S606
            return
        except OSError:
            pass
    QDesktopServices.openUrl(QUrl.fromLocalFile(p))


class MainWindow(QWidget):
    """The application window.  Public attributes (buttons, labels, ``audio``/``text``) are what the tests inspect."""
    def __init__(self, runner: Callable[..., Any] = run_task, updater_factory: Optional[Callable[[], Any]] = None,
                 autocheck: bool = True, auto_open_folder: bool = True,
                 prefetch_fn: Optional[Callable[..., Any]] = None, prefetch: bool = False,
                 health_fn: Optional[Callable[[], list]] = None,
                 model_states_fn: Optional[Callable[[], dict]] = None,
                 repair_fn: Optional[Callable[..., Any]] = None) -> None:
        """Build the window.

        ``runner`` executes tasks (default :func:`run_task`); ``updater_factory``, ``prefetch_fn``, ``health_fn``,
        ``model_states_fn`` and ``repair_fn`` replace the real implementations in tests.  ``autocheck`` enables the quiet
        start-up checks, ``prefetch`` the first-run model download, ``auto_open_folder`` opens the result folder when done.
        """
        super().__init__()
        self.runner = runner
        self.updater_factory = updater_factory
        self.auto_open_folder = auto_open_folder
        self.audio: Optional[Path] = None
        self.text: Optional[Path] = None
        self.result_dir: Optional[Path] = None
        self.worker: Optional[ProcessWorker] = None
        self.update_worker: Optional[UpdateWorker] = None
        self.prefetch_worker: Optional[PrefetchWorker] = None
        self.prefetch_fn = prefetch_fn
        self.health_fn = health_fn
        self.model_states_fn = model_states_fn
        self.repair_fn = repair_fn
        self.status_worker: Optional[StatusWorker] = None
        self.repair_worker: Optional[RepairWorker] = None
        self.upgrade_dialog = None
        self._health_reasons: list = []
        self._model_states: dict = {}
        self._last_request: Optional[TaskRequest] = None
        self._chips: dict = {}
        self.backdrop = "plain"
        self.last_error_text = ""
        self._settings: Optional[SettingsDialog] = None

        self.setWindowTitle(APP_TITLE)
        self.setObjectName("root")
        self.setAcceptDrops(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._build()
        self.setStyleSheet(build_style(False))
        self._fit_to_screen()
        self._refresh_buttons()
        self._os_check = platform_win.check_os()
        if prefetch:
            QTimer.singleShot(300, self.start_prefetch)
        if autocheck:
            QTimer.singleShot(1500, self._startup_checks)

    def _fit_to_screen(self) -> None:
        """Pick the initial window size and a small hard minimum.

        The content lives in a scroll area, so the window may be smaller than the content: the default size is the
        content's natural size (at least 820 px wide) limited to ~94% / ~90% of the screen's work area, in logical
        pixels, i.e. already divided by the Windows display scale (125%, 150%...).  On a short screen (1366x768 at
        150% gives ~510 logical px) vertical scrolling takes over instead of the window overflowing the screen.
        """
        hint = self.content.sizeHint()
        w, h = max(820, hint.width()), max(560, hint.height() + 8)
        try:
            avail = (self.screen() or QApplication.primaryScreen()).availableGeometry()
            w, h = min(w, int(avail.width() * 0.94)), min(h, int(avail.height() * 0.90))
        except Exception:  # noqa: BLE001 - no screen (tests): keep the natural size
            pass
        self.setMinimumSize(MIN_WINDOW_W, MIN_WINDOW_H)
        self.resize(max(w, MIN_WINDOW_W), max(h, MIN_WINDOW_H))

    # ------------------------------------------------------------------ interface
    def _card(self) -> QFrame:
        """A rounded panel (``QFrame#card``) used to group related widgets."""
        f = QFrame()
        f.setObjectName("card")
        return f

    def _build(self) -> None:
        """Create all widgets and layouts, then connect the signals.  Texts are filled by :meth:`retranslate`."""
        # The window is a thin shell around a scroll area; all widgets live in ``self.content``.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.content = QWidget()
        self.content.setObjectName("content")
        self.scroll.setWidget(self.content)
        self.scroll.viewport().setAutoFillBackground(False)
        outer.addWidget(self.scroll)
        root = QVBoxLayout(self.content)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)
        head = QHBoxLayout()
        title = QLabel(APP_TITLE)
        title.setObjectName("title")
        head.addWidget(title)
        head.addStretch(1)
        self.btn_settings = QPushButton("\u2699")          # gear: opens the Settings dialog
        self.btn_settings.setObjectName("gear")
        head.addWidget(self.btn_settings)
        root.addLayout(head)
        self.lbl_sub = QLabel()
        self.lbl_sub.setObjectName("subtitle")
        self.lbl_sub.setWordWrap(True)
        root.addWidget(self.lbl_sub)

        card = self._card()
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 14, 16, 14)
        cl.setSpacing(10)
        self.btn_audio = QPushButton()
        self.lbl_audio = QLabel()
        self.btn_text = QPushButton()
        self.lbl_text = QLabel()
        for b, l in ((self.btn_audio, self.lbl_audio), (self.btn_text, self.lbl_text)):
            row = QHBoxLayout()
            b.setMinimumWidth(170)
            l.setObjectName("fileLabel")
            l.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            row.addWidget(b)
            row.addWidget(l, 1)
            cl.addLayout(row)
        root.addWidget(card)

        self.btn_lora = QPushButton()
        self.btn_lora.setObjectName("primary")
        root.addWidget(self.btn_lora)
        self.lbl_hint_lora = QLabel()
        self.lbl_hint_lora.setObjectName("hint")
        self.lbl_hint_lora.setWordWrap(True)
        root.addWidget(self.lbl_hint_lora)

        # Two optional fields; they only end up in voice.json next to the adapter.
        vrow = QHBoxLayout()
        self.cmb_voice_type = QComboBox()
        self.cmb_voice_type.addItem("", "")
        for code in voice_info.VOICE_TYPES:
            self.cmb_voice_type.addItem("", code)
        self.edt_voice_desc = QLineEdit()
        self.edt_voice_desc.setMaxLength(voice_info.MAX_DESCRIPTION_CHARS)
        vrow.addWidget(self.cmb_voice_type)
        vrow.addWidget(self.edt_voice_desc, 1)
        root.addLayout(vrow)

        self.btn_merge = QPushButton()      # universal (merged) model, ~4 GB; only on an explicit click
        root.addWidget(self.btn_merge)
        self.lbl_hint_merge = QLabel()
        self.lbl_hint_merge.setObjectName("hint")
        self.lbl_hint_merge.setWordWrap(True)
        root.addWidget(self.lbl_hint_merge)

        self.btn_dataset = QPushButton()
        root.addWidget(self.btn_dataset)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        chips = QHBoxLayout()
        chips.setSpacing(6)
        for st in ALL_STAGES:
            c = QLabel()
            c.setObjectName("chip")
            c.setProperty("state", "idle")
            self._chips[st] = c
            chips.addWidget(c)
        chips.addStretch(1)
        root.addLayout(chips)

        self.lbl_status = QLabel()
        self.lbl_status.setObjectName("status")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        # «Repair» offer: appears only when the install check fails (stable reason codes, localized)
        self.health_row = QWidget()
        hl = QHBoxLayout(self.health_row)
        hl.setContentsMargins(0, 0, 0, 0)
        self.lbl_health = QLabel()
        self.lbl_health.setObjectName("status")
        self.lbl_health.setWordWrap(True)
        self.btn_repair = QPushButton()
        hl.addWidget(self.lbl_health, 1)
        hl.addWidget(self.btn_repair)
        self.health_row.hide()
        root.addWidget(self.health_row)

        self.lbl_models = QLabel()           # «Models: aligner: ready, base: missing»
        self.lbl_models.setObjectName("hint")
        self.lbl_models.setWordWrap(True)
        self.lbl_models.hide()
        root.addWidget(self.lbl_models)

        self.lbl_ready = QLabel()
        self.lbl_ready.setObjectName("ready")
        self.lbl_ready.hide()
        self.btn_open = QPushButton()
        self.btn_open.hide()
        self.btn_cancel = QPushButton()
        self.btn_cancel.hide()
        root.addWidget(self.lbl_ready)
        root.addWidget(self.btn_open)
        root.addWidget(self.btn_cancel)
        root.addStretch(1)
        self.lbl_privacy = QLabel()
        self.lbl_privacy.setObjectName("footer")
        self.lbl_privacy.setWordWrap(True)
        root.addWidget(self.lbl_privacy)

        self.btn_audio.clicked.connect(self.choose_audio)
        self.btn_text.clicked.connect(self.choose_text)
        self.btn_dataset.clicked.connect(lambda: self.start(KIND_DATASET))
        self.btn_lora.clicked.connect(lambda: self.start(KIND_LORA))
        self.btn_merge.clicked.connect(self.start_merge)
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_open.clicked.connect(self.open_result)
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_repair.clicked.connect(self.start_repair)
        self.retranslate()

    # ------------------------------------------------------------------ language
    def retranslate(self) -> None:
        """Apply the texts of the current language (called on creation and whenever the language changes)."""
        self.lbl_sub.setText(tr("ui.subtitle"))
        self.btn_audio.setText(tr("ui.choose_audio"))
        self.btn_text.setText(tr("ui.choose_text"))
        self.lbl_audio.setText(self.audio.name if self.audio else tr("ui.audio_none"))
        self.lbl_text.setText(self.text.name if self.text else tr("ui.text_none"))
        self.btn_lora.setText(tr("ui.btn_lora"))
        self.btn_lora.setToolTip(tr("ui.tip_lora"))
        self.lbl_hint_lora.setText(tr("ui.hint_lora"))
        self.btn_merge.setText(tr("ui.btn_merge"))
        self.lbl_hint_merge.setText(tr("ui.hint_merge"))
        self.btn_dataset.setText(tr("ui.btn_dataset"))
        self.btn_dataset.setToolTip(tr("ui.tip_dataset"))
        self.btn_settings.setToolTip(tr("ui.settings_tip"))
        self.cmb_voice_type.setToolTip(tr("ui.voice_type_tip"))
        type_labels = {"": tr("ui.voice_type_none"), "male": tr("ui.voice_type_male"),
                       "female": tr("ui.voice_type_female"), "child": tr("ui.voice_type_child"),
                       "other": tr("ui.voice_type_other")}
        for i in range(self.cmb_voice_type.count()):
            self.cmb_voice_type.setItemText(i, type_labels[self.cmb_voice_type.itemData(i)])
        self.edt_voice_desc.setPlaceholderText(tr("ui.voice_desc_placeholder"))
        if self._settings is not None:
            self._settings.retranslate()
            self._settings.sync_language()
        for st, chip in self._chips.items():
            chip.setText(st.label)
        self.lbl_ready.setText(tr("ui.ready"))
        self.btn_open.setText(tr("ui.open_folder"))
        self.btn_cancel.setText(tr("ui.cancel"))
        self.lbl_privacy.setText(tr("ui.footer_privacy"))
        self.btn_repair.setText(tr("ui.btn_repair"))
        self._render_status()
        if not self.busy and not self.lbl_ready.isVisible():
            self.lbl_status.setText(tr("ui.status_idle"))
        self._refresh_buttons()

    def set_language(self, code: str) -> None:
        """Switch the UI language (persisted). Ignored for unknown codes and while a task is running."""
        if self.busy or code not in i18n.LANGS or code == i18n.get_language():
            return
        i18n.set_language(code, persist=True)
        self.retranslate()

    # ------------------------------------------------------------------ settings dialog
    def settings_dialog(self) -> SettingsDialog:
        """Return the (lazily created, reused) Settings dialog."""
        if self._settings is None:
            self._settings = SettingsDialog(self)
        return self._settings

    def open_settings(self) -> None:
        """Show the Settings dialog (modal, except in the offscreen test platform where it must not block)."""
        dlg = self.settings_dialog()
        dlg.setStyleSheet(self.styleSheet())
        dlg.refresh()
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            dlg.show()
            return
        dlg.exec()

    def open_models_folder(self) -> None:
        """Open the folder where Voxprint keeps its downloaded models."""
        open_folder(paths.models_dir())

    def open_data_folder(self) -> None:
        """Open Voxprint's data folder (settings, logs, models)."""
        open_folder(paths.app_home())

    def open_about(self) -> None:
        """Show the About dialog (non-blocking under the offscreen test platform)."""
        from ui.about_dialog import AboutDialog

        dlg = AboutDialog(self)
        self._about = dlg
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":  # do not block in tests
            dlg.show()
            return
        dlg.exec()

    # ------------------------------------------------------------------ window / effects
    def showEvent(self, e) -> None:  # noqa: N802
        """On Windows, enable the Acrylic backdrop once the native window exists; otherwise stay on the plain dark look."""
        super().showEvent(e)
        if sys.platform == "win32" and self.backdrop == "plain":
            self.backdrop = platform_win.apply_backdrop(int(self.winId()))
            self.setStyleSheet(build_style(self.backdrop == "acrylic"))
            if self.backdrop != "acrylic":
                self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

    # ------------------------------------------------------------------ file selection
    def set_audio(self, path: Path) -> None:
        """Remember the chosen recording and update the label and button states."""
        self.audio = Path(path)
        self.lbl_audio.setText(self.audio.name)
        self.lbl_audio.setToolTip(str(self.audio))
        self._refresh_buttons()

    def set_text(self, path: Path) -> None:
        """Remember the chosen text file and update the label and button states."""
        self.text = Path(path)
        self.lbl_text.setText(self.text.name)
        self.lbl_text.setToolTip(str(self.text))
        self._refresh_buttons()

    def choose_audio(self) -> None:
        """File dialog for the recording."""
        exts = " ".join(f"*{e}" for e in sorted(AUDIO_EXT))
        f, _ = QFileDialog.getOpenFileName(self, tr("ui.dlg_audio"), "", tr("ui.filter_audio", exts=exts))
        if f:
            self.set_audio(Path(f))

    def choose_text(self) -> None:
        """File dialog for the text."""
        f, _ = QFileDialog.getOpenFileName(self, tr("ui.dlg_text"), "", tr("ui.filter_text"))
        if f:
            self.set_text(Path(f))

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:  # noqa: N802
        """Accept drags that carry files."""
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent) -> None:  # noqa: N802
        """Dropped ``.txt`` files become the text, known audio extensions become the recording."""
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() in TEXT_EXT:
                self.set_text(p)
            elif p.suffix.lower() in AUDIO_EXT:
                self.set_audio(p)

    # ------------------------------------------------------------------ state
    @property
    def busy(self) -> bool:
        """True while a task or the first-run download is running (the file/voice inputs are then locked)."""
        return bool((self.worker and self.worker.isRunning()) or
                    (self.prefetch_worker and self.prefetch_worker.isRunning()))

    @property
    def updating(self) -> bool:
        """True while an update check is running."""
        return bool(self.update_worker and self.update_worker.isRunning())

    @property
    def repairing(self) -> bool:
        """True while a repair of Voxprint's own environment is running."""
        return bool(self.repair_worker and self.repair_worker.isRunning())

    def _refresh_buttons(self) -> None:
        """Enable/disable controls according to the current state (files chosen, busy, adapter available ...)."""
        busy = self.busy
        ready = bool(self.audio and self.text) and not busy
        self.btn_audio.setEnabled(not busy)
        self.btn_text.setEnabled(not busy)
        self.btn_settings.setEnabled(True)
        self.cmb_voice_type.setEnabled(not busy)
        self.edt_voice_desc.setEnabled(not busy)
        has_adapter = last_adapter() is not None
        self.btn_merge.setEnabled(not busy and has_adapter)
        self.btn_merge.setToolTip(tr("ui.tip_merge") if has_adapter else tr("ui.tip_merge_disabled"))
        self.btn_lora.setEnabled(ready)
        self.btn_dataset.setEnabled(ready)
        if self._settings is not None:
            self._settings.refresh()
        self.btn_cancel.setVisible(bool(self.worker and self.worker.isRunning()))

    def _set_chip(self, stage_label: str) -> None:
        """Highlight the active stage chip and mark the earlier ones as done."""
        active_idx = next((i for i, s in enumerate(ALL_STAGES) if s.label == stage_label), -1)
        for i, s in enumerate(ALL_STAGES):
            chip = self._chips[s]
            state = "active" if i == active_idx else ("done" if 0 <= active_idx and i < active_idx else "idle")
            chip.setProperty("state", state)
            chip.style().unpolish(chip)
            chip.style().polish(chip)

    def _reset_chips(self) -> None:
        """Reset all stage chips to idle."""
        self._set_chip("")

    # ------------------------------------------------------------------ running a task
    def start(self, kind: str, force_cpu: bool = False) -> None:
        """Start a ``dataset`` or ``lora`` task for the chosen files (with the optional voice type/description)."""
        if self.busy or not (self.audio and self.text):
            return
        self._launch(TaskRequest(kind=kind, audio=self.audio, text=self.text, force_cpu=force_cpu,
                                 voice_type=str(self.cmb_voice_type.currentData() or ""),
                                 voice_description=self.edt_voice_desc.text().strip()))

    def start_merge(self) -> bool:
        """"Universal model" button: check free disk space, ask for confirmation, start.  True if started.

        Under the offscreen test platform the confirmation box is skipped.
        """
        adapter = last_adapter()
        if self.busy or adapter is None:
            return False
        try:
            need, _repo = model_export.required_free_gb(adapter)
            free = shutil.disk_usage(adapter).free / 1024 ** 3
        except (DatasetMakerError, OSError) as exc:
            self.on_failed("export", getattr(exc, "user_message", str(exc)), getattr(exc, "details", ""), "")
            return False
        if free < need:
            self.on_failed("export", tr("err.export_disk", need=f"{need:.1f}", free=f"{free:.1f}"), "", "")
            return False
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":  # tests skip the confirmation
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle(tr("ui.merge_confirm_title"))
            box.setText(tr("ui.merge_confirm_text", need=f"{need:.1f}", free=f"{free:.1f}"))
            yes = box.addButton(tr("ui.yes"), QMessageBox.ButtonRole.AcceptRole)
            box.addButton(tr("ui.no"), QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is not yes:
                return False
        self._launch(TaskRequest(kind=KIND_MERGE, adapter_dir=adapter))
        return True

    def _launch(self, req: TaskRequest) -> None:
        """Reset the progress UI and run ``req`` in a :class:`ProcessWorker`."""
        self._last_request = req
        self.lbl_ready.hide()
        self.btn_open.hide()
        self.progress.setValue(0)
        self._reset_chips()
        self.lbl_status.setText(tr("ui.starting"))
        self.worker = ProcessWorker(req, self.runner, self)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.cancelled.connect(self.on_cancelled)
        self.worker.finished.connect(self._refresh_buttons)
        self.worker.start()
        self._refresh_buttons()

    def cancel(self) -> None:
        """Ask the running task to stop."""
        if self.worker:
            self.lbl_status.setText(tr("ui.stopping"))
            self.worker.cancel()

    def on_progress(self, pct: int, stage_label: str, message: str) -> None:
        """Worker signal: update the bar, the active chip and the status line."""
        self.progress.setValue(pct)
        self._set_chip(stage_label)
        self.lbl_status.setText(message)

    def on_done(self, result: Any) -> None:
        """Worker signal: show the result summary (and open the result folder if enabled)."""
        self.progress.setValue(100)
        self._set_chip(Stage.SAVE.label)
        self.result_dir = Path(result.open_dir)
        if getattr(result, "kind", "") == KIND_MERGE:
            self.lbl_status.setText(tr("ui.merge_done", path=result.merged_path) + "\n"
                                    + tr("ui.merge_note", speaker=result.speaker))
        else:
            extra = ""
            if getattr(result, "adapter_path", None):
                extra = "\n" + tr("ui.adapter_saved")
            warn = ("\n" + "\n".join(result.warnings[-3:])) if getattr(result, "warnings", None) else ""
            self.lbl_status.setText(tr("ui.done_segments", n=result.n_segments) + f"{extra}{warn}"
                                    + (f"\n{result.update_summary}" if getattr(result, "update_summary", "") else ""))
        self.lbl_ready.show()
        self.btn_open.show()
        self._refresh_buttons()
        if self.auto_open_folder:
            open_folder(self.result_dir)

    def on_cancelled(self) -> None:
        """Worker signal: the user cancelled the task."""
        self.lbl_status.setText(tr("ui.cancelled"))
        self.progress.setValue(0)
        self._reset_chips()

    def on_failed(self, kind: str, message: str, details: str, url: str) -> None:
        """Worker signal: remember and show the error."""
        self.last_error_text = message
        self.progress.setValue(0)
        self._reset_chips()
        self.lbl_status.setText(message)
        log.error("task failed: kind=%s msg=%s details=%s", kind, message, details)
        self.show_error(kind, message, details, url)

    # ------------------------------------------------------------------ friendly error messages
    def show_error(self, kind: str, message: str, details: str = "", url: str = "") -> None:
        """Friendly message box per error kind; for out-of-memory it offers a retry on the CPU."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(APP_TITLE)
        titles = {"mismatch": tr("ui.err_mismatch"), "oom": tr("ui.err_oom"),
                  "download": tr("ui.err_download"), "text_read": tr("ui.err_text_read"),
                  "audio_read": tr("ui.err_audio_read"), "export": tr("ui.err_export")}
        box.setText(f"<b>{titles.get(kind, tr('ui.err_default'))}</b>")
        body = message
        if kind == "download" and url:
            body += f'<br><br>{tr("ui.model_page")} <a href="{url}">{url}</a>'
        elif details and kind == "other":
            body += f"<br><small>{details}</small>"
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setInformativeText(body.replace("\n", "<br>"))
        retry_cpu = None
        if kind == "oom":
            retry_cpu = box.addButton(tr("ui.retry_cpu"), QMessageBox.ButtonRole.AcceptRole)
            box.addButton(tr("ui.cancel"), QMessageBox.ButtonRole.RejectRole)
        else:
            box.addButton(tr("ui.ok"), QMessageBox.ButtonRole.AcceptRole)
        self._error_box = box
        box.setModal(True)
        box.finished.connect(lambda _=0: None)
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":  # do not block in tests
            self._retry_cpu_button = retry_cpu
            return
        box.exec()
        if retry_cpu is not None and box.clickedButton() is retry_cpu and self._last_request:
            self.start(self._last_request.kind, force_cpu=True)

    # ------------------------------------------------------------------ updates
    def check_updates(self) -> None:
        """Start a manual update check in the background (called from the Settings dialog)."""
        if self.update_worker and self.update_worker.isRunning():
            return
        self.lbl_status.setText(tr("upd.checking"))
        self.progress.setValue(0)
        self.update_worker = UpdateWorker(self.updater_factory, parent=self)
        self.update_worker.progress.connect(lambda p, s, m: (self.progress.setValue(p), self.lbl_status.setText(m)))
        self.update_worker.done.connect(self._on_update_done)
        self.update_worker.offer.connect(lambda offers, w=self.update_worker: self._show_offer(w, offers))
        self.update_worker.failed.connect(lambda k, m, d, u: self.lbl_status.setText(m))
        self.update_worker.finished.connect(self._refresh_buttons)
        self.update_worker.start()
        self._refresh_buttons()

    def _on_update_done(self, summary: str, changed: bool) -> None:
        """Show the result of an update check."""
        self.progress.setValue(100 if changed else 0)
        self.lbl_status.setText(summary or tr("upd.verified_installed") + ".")

    def _show_offer(self, worker: UpdateWorker, offers: list) -> None:
        """Outdated component in the user's environment: ask first (non-blocking dialog), then tell the worker."""
        from ui.upgrade_dialog import UpgradeOfferDialog

        dlg = UpgradeOfferDialog(offers, self)
        self.upgrade_dialog = dlg
        dlg.decided.connect(worker.answer)
        dlg.open()

    # ------------------------------------------------------------------ install health and model state
    def refresh_status(self) -> None:
        """Health of Voxprint's own install + per-model state, computed off the GUI thread."""
        if self.status_worker and self.status_worker.isRunning():
            return
        w = StatusWorker(self.health_fn, self.model_states_fn, parent=self)
        w.done.connect(self._on_status)
        self.status_worker = w
        w.start()

    def _on_status(self, reasons: list, states: dict) -> None:
        """StatusWorker result: store the health reasons and model states and redraw."""
        self._health_reasons = list(reasons)
        self._model_states = dict(states)
        self._render_status()

    def _render_status(self) -> None:
        """Show/hide the repair banner and the one-line model state summary."""
        if self._health_reasons:
            self.lbl_health.setText(tr("ui.repair_offer", reasons="; ".join(self._health_reasons)))
        self.health_row.setVisible(bool(self._health_reasons))
        items = []
        for repo, state in self._model_states.items():
            if state == "ready":
                label = tr("ui.model_state_ready")
            elif state == "partial":
                label = tr("ui.model_state_partial")
            else:
                label = tr("ui.model_state_missing")
            items.append(tr("ui.model_item", short=repo.split("/")[-1], state=label))
        self.lbl_models.setText(tr("ui.models_line", items=", ".join(items)) if items else "")
        self.lbl_models.setVisible(bool(items))

    def start_repair(self) -> None:
        """Rebuild Voxprint's own environment in the background (the banner / Settings button)."""
        if self.busy or (self.repair_worker and self.repair_worker.isRunning()):
            return
        w = RepairWorker(self.repair_fn, parent=self)
        self.repair_worker = w
        self.lbl_status.setText(tr("ui.repair_running"))
        w.progress.connect(lambda p, m: (self.progress.setValue(p), self.lbl_status.setText(m)))
        w.done.connect(self._on_repair_done)
        w.finished.connect(self._refresh_buttons)
        self.btn_repair.setEnabled(False)
        w.start()

    def _on_repair_done(self, rc: int, text: str) -> None:
        """Repair finished: show the text and re-check the install state."""
        self.progress.setValue(0)
        self.lbl_status.setText(text)
        self.btn_repair.setEnabled(True)
        if rc == 0:
            self._health_reasons = []
            self._render_status()
        self.refresh_status()

    def _startup_checks(self) -> None:
        """Privacy notice (once), a soft OS warning, then the quiet weekly update check."""
        self.maybe_show_privacy_notice()
        if not self._os_check.ok and self._os_check.message and \
                os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QMessageBox.information(self, APP_TITLE, self._os_check.message)
        self.refresh_status()
        if self.busy:
            return
        w = UpdateWorker(self.updater_factory, silent=True, only_if_due=True, parent=self)
        w.done.connect(lambda s, changed: changed and self.lbl_status.setText(s))
        w.offer.connect(lambda offers, w=w: self._show_offer(w, offers))
        w.done.connect(lambda *_: self.refresh_status())
        self.update_worker = w
        w.start()

    def maybe_show_privacy_notice(self) -> bool:
        """Show the privacy notice on the very first run.  Under offscreen (tests) nothing is shown or stored."""
        if privacy_acknowledged() or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return False
        QMessageBox.information(self, tr("ui.privacy_title"), tr("ui.privacy_text"))
        acknowledge_privacy()
        return True

    # ------------------------------------------------------------------ first run
    def start_prefetch(self) -> None:
        """Download the required models automatically on the first run (no questions asked)."""
        if self.busy:
            return
        w = PrefetchWorker(self.prefetch_fn, parent=self)
        self.prefetch_worker = w
        self.lbl_status.setText(tr("ui.prefetch_start"))
        w.progress.connect(lambda p, m: (self.progress.setValue(p), self.lbl_status.setText(m)))
        w.done.connect(self._on_prefetch_done)
        w.failed.connect(self._on_prefetch_failed)
        w.finished.connect(self._refresh_buttons)
        w.start()
        self._refresh_buttons()

    def _on_prefetch_done(self, downloaded: list) -> None:
        """First-run download finished."""
        self.progress.setValue(0)
        self.lbl_status.setText(tr("ui.prefetch_done") if downloaded else tr("ui.status_idle"))
        self.refresh_status()

    def _on_prefetch_failed(self, message: str, url: str) -> None:
        """First-run download failed: show the message; the app stays usable and retries next time."""
        self.progress.setValue(0)
        self.lbl_status.setText(message + " " + tr("ui.prefetch_failed"))

    def open_result(self) -> None:
        """Open the folder with the result."""
        if self.result_dir:
            open_folder(self.result_dir)

    def closeEvent(self, e) -> None:  # noqa: N802
        """Decline any pending upgrade offer, cancel a running task and wait for it, then close."""
        if self.upgrade_dialog is not None:      # an unanswered offer = «Not now» (the worker must not wait)
            self.upgrade_dialog.reject()
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        super().closeEvent(e)
