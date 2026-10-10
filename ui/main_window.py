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
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core import adapter_strength, i18n, model_export, train_presets, voice_info
from core import consent as consent_mod
from core.appinfo import APP_DISPLAY_NAME
from core.errors import DatasetMakerError
from core.voice_library import VoiceLibrary
from core.events import Stage
from core.i18n import tr
from infra import paths, platform_win, ui_prefs
from infra.vram_optimizer import detect_gpu
from ui.mini_player import MiniPlayer
from ui.settings_dialog import SettingsDialog
from workers.pipeline_runner import (
    KIND_DATASET,
    KIND_LORA,
    KIND_MERGE,
    KIND_PREVIEW,
    TaskRequest,
    last_adapter,
    run_task,
)
if TYPE_CHECKING:
    from ui.upgrade_dialog import UpgradeOfferDialog

from workers.process_worker import (
    PrefetchWorker,
    ProcessWorker,
    RepairWorker,
    StatusWorker,
    UpdateWorker,
)

log = logging.getLogger("voxprint.ui")

APP_TITLE = APP_DISPLAY_NAME
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
AMBER = "#fcd34d"           # warnings / "personal use only" accents
NOTE_BG = "#2a2418"         # background of warning notes (e.g. the AAC disclaimer)
NOTE_BORDER = "#6b5720"
GREEN = "#86efac"
BADGE_GREEN_TEXT = "#0b1f12"
BADGE_AMBER_TEXT = "#251a02"
# Soft blue / yellow palette: yellow primary buttons, soft blue menus, lists and accents.
PRIMARY = "#f5d76e"          # main action buttons ("Start", "Narrate", the primary Studio card border)
PRIMARY_HOVER = "#f9e291"
PRIMARY_BORDER = "#fbe8a6"
TEXT_ON_PRIMARY = "#1c1604"
PRIMARY_DISABLED_BG = "#36321f"
PRIMARY_DISABLED_TEXT = "#b9b08a"
MENU_BG = "#1c2535"          # drop-down lists and menus: a soft blue tint of the dark background
MENU_SELECTION = "#2f4c78"
# Window transparency (Settings): "more" lets ~25 % more of the blurred desktop through than the default tint.
ROOT_GLASS_MORE = "rgba(16,16,22,167)"
CARD_GLASS_MORE = "rgba(34,34,44,214)"


def recommended_text(text: str) -> str:
    """``text`` with the "recommended" badge (a star and the word) behind it; used for the pre-selected best options."""
    return f"{text}  \u2605 {tr('auto.recommended')}"


def mark_recommended(widget: QWidget, on: bool = True) -> None:
    """Highlight ``widget`` as the recommended choice (bold check box / accent border of a card, see :func:`build_style`)."""
    widget.setProperty("recommended", "true" if on else "false")
    st = widget.style()
    st.unpolish(widget)
    st.polish(widget)


def _check_icon_url() -> str:
    """URL for the tick mark of checked check boxes (an SVG next to the icons); empty if the file is missing."""
    try:
        f = paths.resource_dir() / "assets" / "check.png"
        return f.as_posix() if f.is_file() else ""
    except OSError:
        return ""


STYLE_TEMPLATE = """
* {{{{ font-family: "Segoe UI Variable Text", "Segoe UI", "Ubuntu", "Noto Sans", "Cantarell", "DejaVu Sans", sans-serif; font-size: 14px; color: {text}; }}}}
QWidget#root {{{{ background: {{root_bg}}; }}}}
QLabel#title {{{{ font-size: 22px; font-weight: 600; }}}}
QLabel#subtitle {{{{ color: {muted}; }}}}
QFrame#card {{{{ background: {{card_bg}}; border: 1px solid {border}; border-radius: 12px; }}}}
QLabel#fileLabel {{{{ color: {muted}; }}}}
QPushButton {{{{ background: {control}; border: 1px solid {border}; border-radius: 8px; padding: 6px 14px; }}}}
QPushButton:hover {{{{ background: {hover}; }}}}
QPushButton:pressed {{{{ background: #23232d; }}}}
QPushButton:disabled {{{{ color: {disabled}; background: #1f1f28; border-color: #34343f; }}}}
QPushButton#primary {{{{ background: {primary}; border: 1px solid {primary_border}; font-size: 16px;
                      font-weight: 600; padding: 10px 18px; color: {on_primary}; }}}}
QPushButton#primary:hover {{{{ background: {primary_hover}; }}}}
QPushButton#primary:disabled {{{{ background: {primary_dis_bg}; color: {primary_dis_text}; border-color: #4a4430; }}}}
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
QLabel#warn {{{{ color: #f2c14e; font-size: 12px; padding: 2px 4px; }}}}
QLabel#liveline {{{{ font-weight: 600; }}}}
QLabel#liveline[state="error"] {{{{ color: #f87171; }}}}
QComboBox {{{{ background: {control}; border: 1px solid {border}; border-radius: 8px;
            padding: 6px 12px; min-width: 110px; }}}}
QComboBox:disabled {{{{ color: {disabled}; }}}}
QComboBox QAbstractItemView {{{{ background: {menu_bg}; border: 1px solid {border};
                              selection-background-color: {menu_sel}; selection-color: #ffffff; }}}}
QMenu {{{{ background: {menu_bg}; border: 1px solid {border}; padding: 4px; }}}}
QMenu::item {{{{ padding: 5px 18px; border-radius: 6px; }}}}
QMenu::item:selected {{{{ background: {menu_sel}; color: #ffffff; }}}}
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
QPushButton#back {{{{ padding: 6px 12px; }}}}
QPushButton#bigcard {{{{ background: {{card_bg}}; border: 1px solid {border}; border-radius: 16px; text-align: left; padding: 0; }}}}
QPushButton#bigcard:hover {{{{ background: {hover}; border-color: {accent}; }}}}
QPushButton#bigcard:disabled {{{{ background: #1f1f28; }}}}
QPushButton#bigcard[primary="true"] {{{{ border: 2px solid {primary}; }}}}
QPushButton#bigcard QLabel {{{{ background: transparent; }}}}
QLabel#cardtitle {{{{ font-size: 19px; font-weight: 600; }}}}
QLabel#carddesc {{{{ color: {muted}; }}}}
QLabel#cardnote {{{{ color: {amber}; font-size: 12px; }}}}
QLabel#sectiontitle {{{{ font-size: 15px; font-weight: 600; }}}}
QLabel#badge {{{{ padding: 2px 9px; border-radius: 9px; font-size: 12px; font-weight: 600; }}}}
QLabel#badge[commercial="true"] {{{{ color: {badge_green_text}; background: {green}; }}}}
QLabel#badge[commercial="false"] {{{{ color: {badge_amber_text}; background: {amber}; }}}}
QFrame#note {{{{ background: {note_bg}; border: 1px solid {note_border}; border-radius: 8px; }}}}
QFrame#note QLabel {{{{ color: {amber}; background: transparent; font-size: 12px; }}}}
QFrame#sep {{{{ background: {border}; max-height: 1px; border: none; }}}}
QCheckBox {{{{ spacing: 8px; }}}}
QCheckBox:disabled {{{{ color: {disabled}; }}}}
QCheckBox[recommended="true"] {{{{ font-weight: 600; }}}}
QFrame#card[recommended="true"] {{{{ border: 2px solid {accent}; }}}}
QPushButton[recommended="true"] {{{{ border: 2px solid {accent}; font-weight: 600; }}}}
QCheckBox::indicator {{{{ width: 16px; height: 16px; border: 1px solid {accent}; border-radius: 4px; background: {control}; }}}}
QCheckBox::indicator:checked {{{{ background: {strong}; border-color: {soft}; {check_image} }}}}
QCheckBox::indicator:disabled {{{{ border: 1px solid {disabled}; background: transparent; }}}}
QCheckBox::indicator:checked:disabled {{{{ background: {disabled}; border-color: {disabled}; {check_image} }}}}
QToolButton#expander {{{{ background: transparent; border: none; color: {soft}; font-weight: 600; padding: 4px 2px; }}}}
QPushButton#preset {{{{ padding: 6px 16px; }}}}
QPushButton#preset:checked {{{{ background: {strong}; border-color: {soft}; color: #ffffff; font-weight: 600; }}}}
QSpinBox {{{{ background: {control}; border: 1px solid {border}; border-radius: 8px; padding: 5px 8px; }}}}
QListWidget {{{{ background: {control}; border: 1px solid {border}; border-radius: 8px; padding: 4px; }}}}
QListWidget::item {{{{ padding: 6px; }}}}
QListWidget::item:selected {{{{ background: {strong}; color: #ffffff; }}}}
QDialog#root, QTextBrowser {{{{ color: {text}; }}}}
QTextBrowser {{{{ background: transparent; border: none; }}}}
""".format(text=TEXT, muted=TEXT_MUTED, faint=TEXT_FAINT, disabled=TEXT_DISABLED, on_accent=TEXT_ON_ACCENT,
           accent=ACCENT, soft=ACCENT_SOFT, strong=ACCENT_STRONG,
           control=CONTROL_BG, hover=CONTROL_HOVER, border=CONTROL_BORDER, amber=AMBER,
           note_bg=NOTE_BG, note_border=NOTE_BORDER, green=GREEN, badge_green_text=BADGE_GREEN_TEXT,
           badge_amber_text=BADGE_AMBER_TEXT, primary=PRIMARY, primary_hover=PRIMARY_HOVER, primary_border=PRIMARY_BORDER,
           on_primary=TEXT_ON_PRIMARY, primary_dis_bg=PRIMARY_DISABLED_BG, primary_dis_text=PRIMARY_DISABLED_TEXT,
           menu_bg=MENU_BG, menu_sel=MENU_SELECTION,
           check_image=(f"image: url({_check_icon_url()});" if _check_icon_url() else ""))

# (foreground, background) pairs that must keep >= 4.5:1; checked by tests/test_ui.py::test_theme_contrast.
CONTRAST_PAIRS = [
    (TEXT, ROOT_PLAIN), (TEXT, CARD_PLAIN), (TEXT, CONTROL_BG), (TEXT, CONTROL_HOVER),
    (TEXT_MUTED, ROOT_PLAIN), (TEXT_MUTED, CARD_PLAIN), (TEXT_FAINT, ROOT_PLAIN), (TEXT_FAINT, CARD_PLAIN),
    (TEXT_ON_ACCENT, ACCENT), (TEXT_ON_ACCENT, ACCENT_HOVER), (TEXT_ON_ACCENT, ACCENT_SOFT),
    (TEXT, ACCENT_STRONG), ("#dcdce4", "#34343f"), ("#86efac", CARD_PLAIN),
    (BADGE_GREEN_TEXT, GREEN), (BADGE_AMBER_TEXT, AMBER), (AMBER, CARD_PLAIN), (AMBER, NOTE_BG), (AMBER, ROOT_PLAIN),
    ("#ffffff", ACCENT_STRONG), (ACCENT_SOFT, ROOT_PLAIN), (ACCENT_SOFT, CARD_PLAIN),
    (TEXT_ON_PRIMARY, PRIMARY), (TEXT_ON_PRIMARY, PRIMARY_HOVER), (PRIMARY_DISABLED_TEXT, PRIMARY_DISABLED_BG),
    (TEXT, MENU_BG), ("#ffffff", MENU_SELECTION),
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
    """Return the stylesheet: ``glass`` = Acrylic backdrop behind a translucent window, else the solid fallback.
    The Settings option "Window transparency" picks the tint: default, more (see ``ROOT_GLASS_MORE``) or off (solid)."""
    level = ui_prefs.transparency()
    if glass and level == "more":
        return STYLE_TEMPLATE.format(root_bg=ROOT_GLASS_MORE, card_bg=CARD_GLASS_MORE)
    if glass and level != "off":
        return STYLE_TEMPLATE.format(root_bg=ROOT_GLASS, card_bg=CARD_GLASS)
    return STYLE_TEMPLATE.format(root_bg=ROOT_PLAIN, card_bg=CARD_PLAIN)


def apply_look(win: QWidget) -> None:
    """Backdrop + stylesheet of a top-level window for the current transparency level (Settings).  Called on every show
    and, for the visible windows, when the level changes; hidden windows catch up on their next show.  The Acrylic
    backdrop is asked for once, on a shown window (Windows 11, the native window must exist); "off" only swaps in the
    solid colours (the solid root paints the whole translucent window), so switching back needs no restart."""
    level = ui_prefs.transparency()
    if sys.platform == "win32" and level != "off" and win.isVisible() and not getattr(win, "_backdrop_tried", False):
        win._backdrop_tried = True  # type: ignore[attr-defined]
        if not os.environ.get("VOXPRINT_NO_FADE_IN"):
            # the first frame of a new window is painted grey by Windows before the backdrop and the stylesheet arrive:
            # show it fully transparent and fade it in once the first real frame is ready
            win.setWindowOpacity(0.0)
            QTimer.singleShot(90, lambda w=win: _end_fade_in(w))
        win._backdrop_hwnd = int(win.winId())  # type: ignore[attr-defined]
        win.backdrop = platform_win.apply_backdrop(win._backdrop_hwnd)  # type: ignore[attr-defined]
        if win.backdrop != "acrylic":  # type: ignore[attr-defined]
            win.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
    elif sys.platform == "win32" and level != "off" and win.isVisible() and getattr(win, "backdrop", "") == "acrylic" \
            and getattr(win, "_backdrop_hwnd", 0) != int(win.winId()):
        # Qt re-created the native window (a re-shown dialog, a changed parent): the new HWND has no backdrop yet
        win._backdrop_hwnd = int(win.winId())  # type: ignore[attr-defined]
        platform_win.apply_backdrop(win._backdrop_hwnd)
    style = build_style(getattr(win, "backdrop", "plain") == "acrylic")
    if win.styleSheet() != style:      # re-polishing every widget is not free: only when something changed
        win.setStyleSheet(style)


def _end_fade_in(win: QWidget) -> None:
    """End the fade-in: full opacity, then ask for the Acrylic backdrop again.  ``setWindowOpacity(0)`` made the window
    a layered window while the backdrop was requested, and Windows draws no system backdrop behind a layered window; once
    the layered style is gone the request is repeated, so dialogs (shown modally right away) get it as the windows do."""
    try:
        win.setWindowOpacity(1.0)
        if sys.platform == "win32" and getattr(win, "backdrop", "") == "acrylic" and win.isVisible():
            platform_win.apply_backdrop(int(win.winId()))
            win.update()
    except RuntimeError:               # the window was deleted during the 90 ms
        pass


def audio_seconds(path: Path) -> Optional[float]:
    """Duration of an audio file from its header (no decoding); None when it cannot be read quickly (e.g. some AAC/M4A files)."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:  # noqa: BLE001
        return None


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


class StatusLabel(QLabel):
    """A label that announces every text change (``changed``), so the Studio home screen can mirror the status line."""

    changed = Signal(str)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        """Set the text and emit :attr:`changed`."""
        super().setText(text)
        self.changed.emit(text)


class MainWindow(QWidget):
    """The "Train your voice" window.  Public attributes (buttons, labels, ``audio``/``text``) are what the tests inspect.

    Inside the Studio it is one of several windows: ``enable_studio_nav()`` shows the "Studio" back button, and the
    signals ``go`` (navigation request), ``closing`` and ``language_changed`` let the Studio react.  Standalone (tests)
    it behaves exactly like the old main window.
    """
    go = Signal(str)                 # "studio" - ask the Studio to show another window
    closing = Signal()               # the user closed this window
    language_changed = Signal()      # the UI language was switched here
    voice_type_suggested = Signal(str, float)   # (type, median pitch in Hz) from the background pitch analysis
    noise_scored = Signal(object)               # DNSMOS background score of the chosen recording(s) or None (core/denoise.py)
    denoise_downloaded = Signal(str)            # "" = the clean-up program is ready, else the error text
    def __init__(self, runner: Callable[..., Any] = run_task, updater_factory: Optional[Callable[[], Any]] = None,
                 autocheck: bool = True, auto_open_folder: bool = True,
                 prefetch_fn: Optional[Callable[..., Any]] = None, prefetch: bool = False,
                 health_fn: Optional[Callable[[], list]] = None,
                 model_states_fn: Optional[Callable[[], dict]] = None,
                 repair_fn: Optional[Callable[..., Any]] = None, gpu_fn: Optional[Callable[[], Any]] = None) -> None:
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
        self._gpu = None
        self.library: Optional[VoiceLibrary] = None  # set by the Studio; None -> the default library
        self._last_voice_id = ""
        self.gpu_fn = gpu_fn or detect_gpu
        self.audio_files: List[Path] = []      # no-transcript mode: several files and/or folders
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
        self.upgrade_dialog: Optional[UpgradeOfferDialog] = None
        self._health_reasons: list = []
        self._model_states: dict = {}
        self._last_request: Optional[TaskRequest] = None
        self._chips: dict = {}
        self.backdrop = "plain"
        self.last_error_text = ""
        self._settings: Optional[SettingsDialog] = None
        self._in_studio = False

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
        from ui import screen_fit

        hint = self.content.sizeHint()
        self.setMinimumSize(MIN_WINDOW_W, MIN_WINDOW_H)
        screen_fit.fit(self, max(820, hint.width()), max(560, hint.height() + 8))

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
        root.setContentsMargins(20, 14, 20, 14)
        root.setSpacing(10)
        head = QHBoxLayout()
        self.btn_back = QPushButton()                      # back to the Studio home (hidden outside the Studio)
        self.btn_back.setObjectName("back")
        self.btn_back.hide()
        head.addWidget(self.btn_back)
        self.lbl_title = QLabel()
        self.lbl_title.setObjectName("title")
        head.addWidget(self.lbl_title)
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
        cl.setContentsMargins(14, 10, 14, 10)
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
        # "No transcript" mode: audio only, the app recognises the speech itself (warning + explicit opt-in)
        self.chk_no_text = QCheckBox()
        cl.addWidget(self.chk_no_text)
        self.asr_box = QWidget()
        al = QVBoxLayout(self.asr_box)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(6)
        self.btn_folder = QPushButton()
        al.addWidget(self.btn_folder)
        self.lbl_asr_warning = QLabel()
        self.lbl_asr_warning.setObjectName("warn")
        self.lbl_asr_warning.setWordWrap(True)
        al.addWidget(self.lbl_asr_warning)
        self.chk_asr_ok = QCheckBox()
        al.addWidget(self.chk_asr_ok)
        self.asr_box.hide()
        cl.addWidget(self.asr_box)
        root.addWidget(card)

        # Training presets (Fast / Balanced / Maximum / Manual) with a time estimate and a collapsed advanced panel
        pcard = self._card()
        pl = QVBoxLayout(pcard)
        pl.setContentsMargins(14, 10, 14, 10)
        pl.setSpacing(8)
        prow = QHBoxLayout()
        self.lbl_preset = QLabel()
        self.cmb_preset = QComboBox()
        for code in train_presets.PRESETS:
            self.cmb_preset.addItem("", code)
        self.cmb_preset.setCurrentIndex(train_presets.PRESETS.index(train_presets.DEFAULT_PRESET))
        prow.addWidget(self.lbl_preset)
        prow.addWidget(self.cmb_preset, 1)
        pl.addLayout(prow)
        self.lbl_preset_desc = QLabel()
        self.lbl_preset_desc.setObjectName("hint")
        self.lbl_preset_desc.setWordWrap(True)
        pl.addWidget(self.lbl_preset_desc)
        self.lbl_estimate = QLabel()
        self.lbl_estimate.setWordWrap(True)
        pl.addWidget(self.lbl_estimate)
        self.btn_adv = QToolButton()
        self.btn_adv.setCheckable(True)
        self.btn_adv.setObjectName("expander")
        pl.addWidget(self.btn_adv)
        self.adv_box = QWidget()
        self.adv_box.hide()
        gl = QVBoxLayout(self.adv_box)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(6)
        self.sp_epochs, self.sp_rank, self.sp_alpha, self.sp_accum = QSpinBox(), QSpinBox(), QSpinBox(), QSpinBox()
        for sp, lo, hi in ((self.sp_epochs, 1, 100), (self.sp_rank, 1, 256), (self.sp_alpha, 1, 1024), (self.sp_accum, 1, 32)):
            sp.setRange(lo, hi)
        self.sp_lr = QDoubleSpinBox()
        self.sp_lr.setDecimals(7)
        self.sp_lr.setRange(1e-7, 1e-4)
        self.sp_lr.setSingleStep(5e-7)
        self.adv_rows = []
        for sp in (self.sp_epochs, self.sp_rank, self.sp_alpha, self.sp_lr, self.sp_accum):
            row = QHBoxLayout()
            lab = QLabel()
            row.addWidget(lab, 1)
            row.addWidget(sp)
            gl.addLayout(row)
            self.adv_rows.append(lab)
        self.lbl_adv_hint = QLabel()
        self.lbl_adv_hint.setObjectName("hint")
        self.lbl_adv_hint.setWordWrap(True)
        gl.addWidget(self.lbl_adv_hint)
        pl.addWidget(self.adv_box)
        root.addWidget(pcard)

        # Voice-owner consent: read from the spoken statement at the end of the recording, or chosen by hand
        ccard = self._card()
        cl2 = QVBoxLayout(ccard)
        cl2.setContentsMargins(14, 10, 14, 10)
        cl2.setSpacing(8)
        self.lbl_consent = QLabel()
        self.lbl_consent.setObjectName("sectiontitle")
        cl2.addWidget(self.lbl_consent)
        self.cmb_consent = QComboBox()
        for code in ("auto", "manual", "none"):
            self.cmb_consent.addItem("", code)
        cl2.addWidget(self.cmb_consent)
        self.lbl_consent_hint = QLabel()
        self.lbl_consent_hint.setObjectName("hint")
        self.lbl_consent_hint.setWordWrap(True)
        cl2.addWidget(self.lbl_consent_hint)
        self.manual_consent = QWidget()
        ml = QHBoxLayout(self.manual_consent)
        ml.setContentsMargins(0, 0, 0, 0)
        self.edt_consent_name = QLineEdit()
        self.cmb_consent_scope = QComboBox()
        for sc in consent_mod.SCOPES:
            self.cmb_consent_scope.addItem("", sc)
        self.cmb_consent_scope.setCurrentIndex(consent_mod.SCOPES.index(consent_mod.PRIVATE))
        ml.addWidget(self.edt_consent_name, 1)
        ml.addWidget(self.cmb_consent_scope, 1)
        self.manual_consent.hide()
        cl2.addWidget(self.manual_consent)
        self.chk_consent_clip = QCheckBox()
        self.chk_consent_clip.setChecked(True)
        cl2.addWidget(self.chk_consent_clip)
        self.consent_result = QWidget()           # shown after training: the detected scope for one-click confirmation
        rl = QVBoxLayout(self.consent_result)
        rl.setContentsMargins(0, 0, 0, 0)
        self.lbl_consent_result = QLabel()
        self.lbl_consent_result.setWordWrap(True)
        rl.addWidget(self.lbl_consent_result)
        crow = QHBoxLayout()
        self.cmb_consent_confirm = QComboBox()
        for sc in consent_mod.SCOPES:
            self.cmb_consent_confirm.addItem("", sc)
        self.btn_consent_confirm = QPushButton()
        crow.addWidget(self.cmb_consent_confirm, 1)
        crow.addWidget(self.btn_consent_confirm)
        rl.addLayout(crow)
        self.consent_result.hide()
        cl2.addWidget(self.consent_result)
        root.addWidget(ccard)

        self.btn_lora = QPushButton()
        self.btn_lora.setObjectName("primary")
        root.addWidget(self.btn_lora)
        self.lbl_hint_lora = QLabel()
        self.lbl_hint_lora.setObjectName("hint")
        self.lbl_hint_lora.setWordWrap(True)
        root.addWidget(self.lbl_hint_lora)

        # Quick preview: a short training on a few clips + a ~10 s sample with automatic checks (pitch, WER, does it stop)
        prev = QHBoxLayout()
        self.btn_preview = QPushButton()
        self.chk_compare = QCheckBox()
        self.chk_compare.setChecked(True)               # the best option is pre-selected (see infra.auto_steps)
        self.chk_check = QCheckBox()
        self.chk_check.setChecked(True)
        self.chk_pick = QCheckBox()                     # automatic checkpoint + strength pick (core/checkpoint_pick.py)
        self.chk_pick.setChecked(True)
        mark_recommended(self.chk_compare)
        mark_recommended(self.chk_check)
        mark_recommended(self.chk_pick)
        prev.addWidget(self.btn_preview)
        prev.addWidget(self.chk_compare)
        root.addLayout(prev)
        # the automatic checks and clip options are pre-set well, so they fold away (height-first layout on small screens);
        # their rows keep their own hidden state inside the folded box
        self.btn_checks = QToolButton()
        self.btn_checks.setCheckable(True)
        self.btn_checks.setObjectName("expander")
        root.addWidget(self.btn_checks)
        self.checks_box = QWidget()
        self.checks_box.hide()
        chl = QVBoxLayout(self.checks_box)
        chl.setContentsMargins(0, 0, 0, 0)
        chl.setSpacing(6)
        root.addWidget(self.checks_box)
        self.btn_checks.toggled.connect(self._on_checks_toggled)
        chl.addWidget(self.chk_check)
        chl.addWidget(self.chk_pick)
        # speech-recognition cross-check of the dataset clips (core/clip_check.py): text mode only, CER threshold 10-15 %
        crow = QHBoxLayout()
        self.chk_clipcheck = QCheckBox()
        self.chk_clipcheck.setChecked(True)
        mark_recommended(self.chk_clipcheck)
        self.sp_clipcer = QSpinBox()
        self.sp_clipcer.setRange(10, 15)
        self.sp_clipcer.setValue(12)
        self.sp_clipcer.setSuffix(" %")
        crow.addWidget(self.chk_clipcheck)
        crow.addWidget(self.sp_clipcer)
        crow.addStretch(1)
        self.clipcheck_row = QWidget()
        self.clipcheck_row.setLayout(crow)
        crow.setContentsMargins(0, 0, 0, 0)
        chl.addWidget(self.clipcheck_row)
        self.chk_clipcheck.toggled.connect(lambda on: self.sp_clipcer.setEnabled(on and not self.busy))
        # longest training clip (core/slicer.long_clip_config): 12 s = the classic cut, 15 s default, 20 s on GPUs with >= 10 GB
        lrow = QHBoxLayout()
        lrow.setContentsMargins(0, 0, 0, 0)
        self.lbl_clipmax = QLabel()
        self.sp_clipmax = QSpinBox()
        self.sp_clipmax.setRange(12, 20)
        self.sp_clipmax.setValue(15)
        self.sp_clipmax.setSuffix(" s")
        lrow.addWidget(self.lbl_clipmax)
        lrow.addWidget(self.sp_clipmax)
        lrow.addStretch(1)
        self.clipmax_row = QWidget()
        self.clipmax_row.setLayout(lrow)
        chl.addWidget(self.clipmax_row)
        # optional noise clean-up (core/denoise.py): the row appears ONLY when the recording sounds noisy; never ticked by itself
        drow = QVBoxLayout()
        drow.setContentsMargins(0, 0, 0, 0)
        self.lbl_denoise = QLabel()
        self.lbl_denoise.setWordWrap(True)
        self.lbl_denoise.setObjectName("hint")
        dline = QHBoxLayout()
        self.chk_denoise = QCheckBox()
        self.btn_denoise_dl = QPushButton()
        dline.addWidget(self.chk_denoise)
        dline.addWidget(self.btn_denoise_dl)
        dline.addStretch(1)
        drow.addWidget(self.lbl_denoise)
        drow.addLayout(dline)
        self.denoise_row = QWidget()
        self.denoise_row.setLayout(drow)
        self.denoise_row.setVisible(False)
        root.addWidget(self.denoise_row)
        self.noise_bak = None
        self._noise_token = 0
        self._denoise_downloading = False
        self.noise_scorer = None            # tests: callable(files) -> background score; None = DNSMOS if installed
        self.noise_scored.connect(self._apply_noise_score)
        self.denoise_downloaded.connect(self._on_denoise_downloaded)
        self.btn_denoise_dl.clicked.connect(self.download_denoise)
        self.lbl_preview_estimate = QLabel()
        self.lbl_preview_estimate.setObjectName("hint")
        self.lbl_preview_estimate.setWordWrap(True)
        root.addWidget(self.lbl_preview_estimate)
        self.preview_box = QWidget()
        self.preview_layout = QVBoxLayout(self.preview_box)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_box.hide()
        self.preview_player = MiniPlayer(backend=getattr(self, "_player_backend", None))    # all samples in one seekable player
        self.preview_player.hide()
        self.preview_player.started.connect(self._stop_previewer)
        root.addWidget(self.preview_player)
        root.addWidget(self.preview_box)

        # Optional fields; they only end up in voice.json next to the adapter (the name also names the folders).
        vrow = QHBoxLayout()
        self.edt_voice_name = QLineEdit()           # voice name -> library folder and file names (empty = from the recording)
        self.edt_voice_name.setMaxLength(80)
        vrow.addWidget(self.edt_voice_name, 1)
        self.cmb_voice_type = QComboBox()           # gender ("" / male / female); the name is kept from the older type selector
        for code in ("",) + voice_info.GENDERS:
            self.cmb_voice_type.addItem("", code)
        self.cmb_voice_age = QComboBox()            # age group ("" / child / young / adult / elderly)
        for code in ("",) + voice_info.AGE_GROUPS:
            self.cmb_voice_age.addItem("", code)
        self.edt_voice_desc = QLineEdit()
        self.edt_voice_desc.setMaxLength(voice_info.MAX_DESCRIPTION_CHARS)
        vrow.addWidget(self.edt_voice_desc, 1)
        root.addLayout(vrow)
        grow = QHBoxLayout()                        # gender, age group and the "more" expander share one row
        for combo in (self.cmb_voice_type, self.cmb_voice_age):     # may shrink below the longest item (narrow window)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
        grow.addWidget(self.cmb_voice_type, 1)
        grow.addWidget(self.cmb_voice_age, 1)
        root.addLayout(grow)
        self.lbl_voice_type_hint = QLabel()          # "suggested from the pitch ..." / the file-name rule
        self.lbl_voice_type_hint.setObjectName("hint")
        self.lbl_voice_type_hint.setWordWrap(True)
        root.addWidget(self.lbl_voice_type_hint)
        # Who / where details: folded away by default so the Train window stays short (150 % scaling on small screens)
        self.btn_voice_more = QToolButton()
        self.btn_voice_more.setCheckable(True)
        self.btn_voice_more.setObjectName("expander")
        grow.addWidget(self.btn_voice_more)
        self.voice_more_box = QWidget()
        self.voice_more_box.hide()
        mg = QGridLayout(self.voice_more_box)
        mg.setContentsMargins(0, 0, 0, 0)
        mg.setSpacing(6)
        self.edt_voice_speaker, self.edt_voice_prepared = QLineEdit(), QLineEdit()
        self.edt_voice_org, self.edt_voice_url = QLineEdit(), QLineEdit()
        for i, (w, n) in enumerate(((self.edt_voice_speaker, voice_info.MAX_NAME_CHARS),
                                    (self.edt_voice_prepared, voice_info.MAX_NAME_CHARS),
                                    (self.edt_voice_org, voice_info.MAX_ORGANIZATION_CHARS),
                                    (self.edt_voice_url, voice_info.MAX_URL_CHARS))):
            w.setMaxLength(n)
            mg.addWidget(w, i // 2, i % 2)
        root.addWidget(self.voice_more_box)
        self.btn_voice_more.toggled.connect(self._on_voice_more_toggled)
        self._voice_type_user_set = False             # the user chose a type: the pitch suggestion must not override it
        self._suggest_token = 0
        self.cmb_voice_type.activated.connect(lambda _i: setattr(self, "_voice_type_user_set", True))
        self.voice_type_suggested.connect(self._apply_voice_type_suggestion)

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

        self.lbl_status = StatusLabel()
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
        self.btn_folder.clicked.connect(self.choose_folder)
        self.btn_preview.clicked.connect(lambda: self.start(KIND_PREVIEW))
        self.chk_compare.toggled.connect(lambda _on: self.refresh_estimate())
        self.cmb_preset.currentIndexChanged.connect(self._on_preset)
        self.btn_adv.toggled.connect(self._on_adv_toggled)
        self.cmb_consent.currentIndexChanged.connect(lambda _i: self._on_consent_mode())
        self.btn_consent_confirm.clicked.connect(self.confirm_consent)
        for sp in (self.sp_epochs, self.sp_rank, self.sp_alpha, self.sp_accum, self.sp_lr):
            sp.valueChanged.connect(lambda _v: self.refresh_estimate())
        self.chk_no_text.toggled.connect(self.set_no_transcript)
        self.chk_asr_ok.toggled.connect(lambda _on: self._refresh_buttons())
        self.btn_dataset.clicked.connect(lambda: self.start(KIND_DATASET))
        self.btn_lora.clicked.connect(lambda: self.start(KIND_LORA))
        self.btn_merge.clicked.connect(self.start_merge)
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_back.clicked.connect(lambda: self.go.emit("studio"))
        self.btn_open.clicked.connect(self.open_result)
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_repair.clicked.connect(self.start_repair)
        self.retranslate()

    # ------------------------------------------------------------------ language
    def retranslate(self) -> None:
        """Apply the texts of the current language (called on creation and whenever the language changes)."""
        if hasattr(self, "preview_player"):
            self.preview_player.retranslate()
        self.lbl_title.setText(tr("train.title"))
        self.btn_back.setText(tr("nav.back"))
        if self._in_studio:
            self.setWindowTitle(f"{APP_TITLE} - {tr('train.title')}")
        self.lbl_sub.setText(tr("ui.subtitle"))
        self.btn_audio.setText(tr("ui.choose_audio"))
        self._apply_text_row()
        self.lbl_consent.setText(tr("consent.title"))
        for i, code in enumerate(("auto", "manual", "none")):
            self.cmb_consent.setItemText(i, tr("consent.mode_" + code))
        for combo in (self.cmb_consent_scope, self.cmb_consent_confirm):
            for i, sc in enumerate(consent_mod.SCOPES):
                combo.setItemText(i, tr("consent.scope_" + sc))
        self.edt_consent_name.setPlaceholderText(tr("consent.name_placeholder"))
        self.chk_consent_clip.setText(tr("consent.save_clip"))
        self.btn_consent_confirm.setText(tr("consent.confirm"))
        self._on_consent_mode()
        self.lbl_preset.setText(tr("preset.label"))
        for i, code in enumerate(train_presets.PRESETS):
            self.cmb_preset.setItemText(i, tr("preset." + code))
        self.btn_adv.setText(("\u25be " if self.btn_adv.isChecked() else "\u25b8 ") + tr("preset.advanced"))
        self._on_checks_toggled(self.btn_checks.isChecked())
        for lab, key in zip(self.adv_rows, ("epochs", "rank", "alpha", "lr", "accum")):
            lab.setText(tr("preset.adv_" + key))
        self.lbl_adv_hint.setText(tr("preset.adv_hint"))
        self.refresh_estimate()
        self.chk_no_text.setText(tr("asr.checkbox"))
        self.btn_folder.setText(tr("asr.choose_folder"))
        self.lbl_asr_warning.setText(tr("asr.warning"))
        self.chk_asr_ok.setText(tr("asr.confirm"))
        self._show_audio_label()
        self.btn_preview.setText(tr("preview.button"))
        self.chk_compare.setText(recommended_text(tr("preview.compare")))
        self.chk_check.setText(tr("check.checkbox"))
        self.chk_pick.setText(tr("pick.checkbox"))
        self.chk_pick.setToolTip(tr("pick.tip"))
        self.chk_clipcheck.setText(tr("clipcheck.checkbox"))
        self.chk_clipcheck.setToolTip(tr("clipcheck.tip"))
        self.sp_clipcer.setToolTip(tr("clipcheck.tip"))
        self.lbl_clipmax.setText(tr("clipmax.label"))
        self.chk_denoise.setText(tr("denoise.checkbox"))
        self.chk_denoise.setToolTip(tr("denoise.tip"))
        self.btn_denoise_dl.setText(tr("denoise.download", mb=self._denoise_mb()))
        if self.noise_bak is not None:
            self.lbl_denoise.setText(tr("denoise.hint", bak=f"{self.noise_bak:.1f}"))
        self.lbl_clipmax.setToolTip(tr("clipmax.tip"))
        self.sp_clipmax.setToolTip(tr("clipmax.tip"))
        self.btn_lora.setText(tr("ui.btn_lora"))
        self.btn_lora.setToolTip(tr("ui.tip_lora"))
        self.lbl_hint_lora.setText(tr("ui.hint_lora"))
        self.btn_merge.setText(tr("ui.btn_merge"))
        self.lbl_hint_merge.setText(tr("ui.hint_merge"))
        self.btn_dataset.setText(tr("ui.btn_dataset"))
        self.btn_dataset.setToolTip(tr("ui.tip_dataset"))
        self.btn_settings.setToolTip(tr("ui.settings_tip"))
        from ui.voices_window import age_labels, gender_labels

        self.cmb_voice_type.setToolTip(tr("ui.voice_type_tip"))
        self.cmb_voice_age.setToolTip(tr("ui.voice_age_tip"))
        for combo, labels in ((self.cmb_voice_type, gender_labels()), (self.cmb_voice_age, age_labels())):
            for i in range(combo.count()):
                combo.setItemText(i, labels[combo.itemData(i)])
        self._on_voice_more_toggled(self.btn_voice_more.isChecked())
        self.edt_voice_speaker.setPlaceholderText(tr("ui.voice_speaker_placeholder"))
        self.edt_voice_prepared.setPlaceholderText(tr("ui.voice_prepared_placeholder"))
        self.edt_voice_org.setPlaceholderText(tr("ui.voice_org_placeholder"))
        self.edt_voice_url.setPlaceholderText(tr("ui.voice_url_placeholder"))
        self.edt_voice_desc.setPlaceholderText(tr("ui.voice_desc_placeholder"))
        self.edt_voice_name.setPlaceholderText(tr("ui.voice_name_placeholder"))
        if not self.lbl_voice_type_hint.text():
            self.lbl_voice_type_hint.setText(tr("ui.voice_type_hint"))
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
        self.language_changed.emit()

    def enable_studio_nav(self) -> None:
        """Called by the Studio: show the "back to Studio" button and use the section title as window title."""
        self._in_studio = True
        self.btn_back.show()
        self.setWindowTitle(f"{APP_TITLE} - {tr('train.title')}")

    # ------------------------------------------------------------------ settings dialog
    def settings_dialog(self) -> SettingsDialog:
        """Return the (lazily created, reused) Settings dialog."""
        if self._settings is None:
            self._settings = SettingsDialog(self)
        return self._settings

    def open_settings(self) -> None:
        """Show the Settings dialog (modal, except in the offscreen test platform where it must not block)."""
        dlg = self.settings_dialog()
        dlg.refresh()                       # its look comes from apply_look on show (GlassDialog), not from this window
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
        apply_look(self)

    # ------------------------------------------------------------------ file selection
    def set_audio(self, path: Path) -> None:
        """Remember the chosen recording and update the label and button states."""
        self.audio = Path(path)
        self.preview_scale = None               # a strength picked in a preview belongs to the previous recording
        self.lbl_audio.setText(self.audio.name)
        self.lbl_audio.setToolTip(str(self.audio))
        self._on_preset()
        self._refresh_buttons()
        self._suggest_voice_type(self.audio)
        self._assess_noise()

    def _on_voice_more_toggled(self, on: bool) -> None:
        """Show / hide the optional speaker, prepared-by, organization and project-link fields."""
        self.voice_more_box.setVisible(on)
        self.btn_voice_more.setText(("\u25be " if on else "\u25b8 ") + tr("ui.voice_more"))

    def voice_details(self) -> dict:
        """The optional voice.json fields typed in the Train window, as :class:`TaskRequest` keyword arguments."""
        return dict(voice_type="", gender=str(self.cmb_voice_type.currentData() or ""),
                    age_group=str(self.cmb_voice_age.currentData() or ""),
                    voice_description=self.edt_voice_desc.text().strip(),
                    voice_display_name=self.edt_voice_name.text().strip(),
                    speaker=self.edt_voice_speaker.text().strip(), prepared_by=self.edt_voice_prepared.text().strip(),
                    organization=self.edt_voice_org.text().strip(), project_url=self.edt_voice_url.text().strip())

    def _suggest_voice_type(self, audio: Path) -> None:
        """Estimate male / female from the recording's pitch in a background thread (a hint; the user decides)."""
        import threading

        from core.voice_type import suggest_voice_type

        self._suggest_token += 1
        token = self._suggest_token

        def work() -> None:
            kind, hz = suggest_voice_type(audio)
            if token == self._suggest_token:
                try:
                    self.voice_type_suggested.emit(kind, hz)
                except RuntimeError:                  # the window was closed meanwhile
                    pass

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ optional noise clean-up (core/denoise.py)
    def _noise_inputs(self) -> list:
        if self.no_transcript:
            from core.asr_dataset import expand_inputs

            return expand_inputs(self.audio_files)
        return [self.audio] if self.audio else []

    @staticmethod
    def _denoise_mb() -> int:
        from infra import denoise_tool

        return int(round(denoise_tool.download_size() / 1e6))

    def _assess_noise(self) -> None:
        """Score the background noise of the chosen recording(s) in a background thread (DNSMOS, if installed)."""
        import threading

        from infra import denoise_tool

        self._noise_token += 1
        token = self._noise_token
        self._apply_noise_score(None)
        files = self._noise_inputs()
        if not files or denoise_tool.platform_key() is None:
            return
        scorer = self.noise_scorer
        if scorer is None:
            from core import denoise, mos

            m = mos.default_mos()
            if m is None:                       # no DNSMOS file: no score, no suggestion
                return
            scorer = lambda fs: denoise.background_score(fs, m)  # noqa: E731

        def work() -> None:
            try:
                bak = scorer(files)
            except Exception:  # noqa: BLE001 - a hint must never get in the way
                bak = None
            if token == self._noise_token:
                try:
                    self.noise_scored.emit(bak)
                except RuntimeError:            # the window was closed meanwhile
                    pass

        threading.Thread(target=work, daemon=True).start()

    def _apply_noise_score(self, bak) -> None:
        """Offer the clean-up only for a noisy recording (low DNSMOS background score); it is never ticked by itself."""
        from core import denoise

        self.noise_bak = bak
        show = denoise.should_suggest(bak)
        if bak is not None:
            log.info("recording background score (DNSMOS BAK) %.2f: clean-up %s", bak, "offered" if show else "not offered")
        if not show:
            self.chk_denoise.setChecked(False)
        else:
            self.lbl_denoise.setText(tr("denoise.hint", bak=f"{bak:.1f}"))
        self.denoise_row.setVisible(show)
        self._sync_denoise_controls()

    def _sync_denoise_controls(self) -> None:
        from infra import denoise_tool

        installed = denoise_tool.ready() is not None
        self.chk_denoise.setEnabled(installed and not self.busy)
        if not installed:
            self.chk_denoise.setChecked(False)
        self.btn_denoise_dl.setVisible(not installed)
        self.btn_denoise_dl.setEnabled(not self.busy and not self._denoise_downloading)

    def download_denoise(self, ensure: Optional[Callable[..., Any]] = None) -> None:
        """The separate optional download of the clean-up program, on the user's click only."""
        import threading

        from infra import denoise_tool

        ensure = ensure or denoise_tool.ensure
        self._denoise_downloading = True
        self._sync_denoise_controls()
        self.lbl_status.setText(tr("progress.downloading", short=denoise_tool.LABEL, pct=0))

        def work() -> None:
            try:
                ensure()
                err = ""
            except Exception as exc:  # noqa: BLE001 - shown to the user
                err = str(exc) or type(exc).__name__
            try:
                self.denoise_downloaded.emit(err)
            except RuntimeError:
                pass

        threading.Thread(target=work, daemon=True).start()

    def _on_denoise_downloaded(self, err: str) -> None:
        self._denoise_downloading = False
        self._sync_denoise_controls()
        if err:
            self.lbl_status.setText(tr("denoise.download_failed", err=err))
            return
        self.lbl_status.setText("")
        self.chk_denoise.setChecked(True)       # the user asked for the program just now: use it for this recording

    def _apply_voice_type_suggestion(self, kind: str, hz: float) -> None:
        """Show the suggestion; select it if the user has not chosen a type yet."""
        if not kind:
            self.lbl_voice_type_hint.setText(tr("ui.voice_type_hint"))
            return
        if not self._voice_type_user_set and not self.cmb_voice_type.currentData():
            self.cmb_voice_type.setCurrentIndex(max(0, self.cmb_voice_type.findData(kind)))
        name = tr("ui.voice_type_male") if kind == "male" else tr("ui.voice_type_female")
        self.lbl_voice_type_hint.setText(tr("ui.voice_type_suggest", type=name, hz=int(round(hz))))

    def set_text(self, path: Path) -> None:
        """Remember the chosen text file and update the label and button states."""
        self.text = Path(path)
        self.lbl_text.setText(self.text.name)
        self.lbl_text.setToolTip(str(self.text))
        self._refresh_buttons()

    def reload_auto_steps(self) -> None:
        """Select every recommended automatic option of this window again."""
        self.chk_compare.setChecked(True)
        self.chk_check.setChecked(True)
        self.chk_pick.setChecked(True)
        self.chk_clipcheck.setChecked(True)
        self.sp_clipmax.setValue(15)

    # ------------------------------------------------------------------ quick preview
    def show_previews(self, items: list) -> None:
        """One row per quick variant: settings, automatic checks, Play and "use these settings"."""
        while self.preview_layout.count():
            it = self.preview_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.preview_rows = []
        from workers import preview_runner

        best = preview_runner.recommend(items)
        for item in items:
            row = QFrame()
            row.setObjectName("card")
            if best and item.key == best and len(items) > 1:
                mark_recommended(row)
            rl = QVBoxLayout(row)
            if best and item.key == best and len(items) > 1:
                badge = QLabel(recommended_text(tr("preview.best")))
                badge.setObjectName("sectiontitle")
                rl.addWidget(badge)
            lbl = QLabel()
            lbl.setWordWrap(True)
            rl.addWidget(lbl)
            hint = QLabel()
            hint.setObjectName("hint")
            hint.setWordWrap(True)
            rl.addWidget(hint)
            self._fill_preview_texts(item, lbl, hint)
            btns = QHBoxLayout()
            if item.scales:        # one sample per adapter strength: the selector switches what Play and the checks show
                strength = QComboBox()
                strength.setToolTip(tr("preview.strength_tip"))
                for e in item.scales:
                    text = f"{e['scale']:.2f}"
                    strength.addItem(tr("preview.strength_full", scale=text) if adapter_strength.is_full(e["scale"]) else text,
                                     e["scale"])
                strength.setCurrentIndex(max(0, strength.findData(item.scale)))
                strength.currentIndexChanged.connect(
                    lambda _i, it=item, c=strength, a=lbl, b=hint: (it.use_scale(c.currentData()), self._fill_preview_texts(it, a, b)))
                btns.addWidget(QLabel(tr("preview.strength")))
                btns.addWidget(strength)
            play = QPushButton(tr("preview.play"))
            play.clicked.connect(lambda _c=False, it=item: self.play_preview(it.wav))
            use = QPushButton(tr("preview.use"))
            if best and item.key == best and len(items) > 1:
                mark_recommended(use)
            use.clicked.connect(lambda _c=False, it=item: self.use_preview_settings(it))
            btns.addWidget(play)
            btns.addWidget(use)
            rl.addLayout(btns)
            self.preview_layout.addWidget(row)
            self.preview_rows.append((row, play, use))
        self.preview_box.show()
        wavs = [Path(i.wav) for i in items if i.wav and Path(i.wav).is_file()]
        if wavs:
            self.preview_player.set_files(wavs)
            self.preview_player.show()

    @staticmethod
    def _fill_preview_texts(item, lbl: QLabel, hint: QLabel) -> None:
        """Settings and automatic checks of a preview row (for the currently selected adapter strength)."""
        chk = item.check or {}
        wer = chk.get("wer")
        st = chk.get("semitones")
        lbl.setText(tr("preview.row", v=item.key, epochs=item.epochs, rank=item.lora_r, secs=f"{item.seconds:.0f}",
                       wer="-" if wer is None else f"{round(100 * wer)} %",
                       pitch="-" if st is None else f"{st:+.1f}", verdict=tr("check.verdict_" + chk.get("verdict", "good"))
                       if chk.get("verdict", "good") != "good" else tr("check.verdict_good")))
        if chk.get("sim") is not None or chk.get("mos") is not None:
            lbl.setText(lbl.text() + "\n" + tr("preview.row_extra", sim="-" if chk.get("sim") is None else f"{chk['sim']:.2f}",
                                                mos="-" if chk.get("mos") is None else f"{chk['mos']:.1f}"))
        hint.setText(" ".join(tr("check.sugg_" + c) for c in chk.get("issues", []) if c != "no_pitch"))
        hint.setVisible(bool(hint.text()))

    def _stop_previewer(self) -> None:
        """The mini player started: stop the single-file previewer."""
        if getattr(self, "previewer", None) is not None and hasattr(self.previewer, "stop"):
            self.previewer.stop()

    def play_preview(self, path: Path) -> None:
        """Play a preview sample (``self.previewer`` is replaced in tests)."""
        if getattr(self, "previewer", None) is None:
            from ui.audio_preview import Previewer

            self.previewer = Previewer(self)
        self.preview_player.pause()
        self.previewer.play(Path(path))

    def use_preview_settings(self, item) -> None:
        """The chosen variant's numbers become the Manual settings of the full run (rank, alpha, accumulation, learning rate; the
        epochs go back to the full-run level of the preset, x1.5 for variant B)."""
        full = train_presets.build_plan(self.preset if self.preset != train_presets.MANUAL else train_presets.BALANCED,
                                        self.gpu(), self._expected_clips()[0])
        self.cmb_preset.setCurrentIndex(train_presets.PRESETS.index(train_presets.MANUAL))
        self.sp_rank.setValue(item.lora_r)
        self.sp_alpha.setValue(item.lora_alpha)
        self.sp_accum.setValue(item.grad_accum)
        self.sp_lr.setValue(item.lr)
        self.sp_epochs.setValue(full.epochs if item.key == "A" else int(round(full.epochs * 1.5)))   # B = 50 % more passes
        self.preview_scale = getattr(item, "scale", None)      # goes to the trained voice's voice.json
        if self.preview_scale is None:
            self.lbl_status.setText(tr("preview.chosen", v=item.key))
        else:
            self.lbl_status.setText(tr("preview.chosen_scale", v=item.key, scale=f"{self.preview_scale:.2f}"))

    # ------------------------------------------------------------------ voice-owner consent
    @property
    def consent_mode(self) -> str:
        return str(self.cmb_consent.currentData() or "auto")

    def _on_consent_mode(self) -> None:
        mode = self.consent_mode
        self.manual_consent.setVisible(mode == "manual")
        self.chk_consent_clip.setVisible(mode == "auto")
        self.lbl_consent_hint.setText(tr("consent.hint_" + mode))

    def _task_extras(self) -> dict:
        from infra import projects

        return dict(projects_root=projects.sub(projects.VOICES), compare=self.chk_compare.isChecked(), quality_check=self.chk_check.isChecked(),
                    auto_pick=self.chk_pick.isChecked(), adapter_scale=getattr(self, "preview_scale", None),
                    clip_max_cer=self.sp_clipcer.value() / 100.0 if self.chk_clipcheck.isChecked() else 0.0,
                    max_clip_s=float(self.sp_clipmax.value()),
                    denoise=not self.denoise_row.isHidden() and self.chk_denoise.isChecked())

    def _consent_kwargs(self) -> dict:
        return dict(consent_mode=self.consent_mode, consent_scope=str(self.cmb_consent_scope.currentData()),
                    consent_name=self.edt_consent_name.text().strip(), consent_save_clip=self.chk_consent_clip.isChecked())

    def show_consent_result(self, block: dict, voice_id: str) -> None:
        """After training: show what was detected (name, date, scope, the recognised statement) for confirmation or change."""
        self._last_voice_id = voice_id
        self.cmb_consent_confirm.setCurrentIndex(consent_mod.SCOPES.index(block["scope"]))
        who = block.get("name") or tr("consent.unknown")
        self.lbl_consent_result.setText(tr("consent.detected", name=who, date=block.get("date", ""),
                                           scope=tr("consent.scope_" + block["scope"]),
                                           statement=(block.get("statement") or tr("consent.no_statement"))[:300]))
        self.consent_result.show()

    def confirm_consent(self) -> None:
        """One click: mark the (possibly changed) scope as confirmed in the voice library and in voice.json."""
        if not self._last_voice_id:
            return
        from core.voice_library import VoiceLibrary

        scope = str(self.cmb_consent_confirm.currentData())
        try:
            (self.library or VoiceLibrary()).confirm_consent(self._last_voice_id, scope)
        except DatasetMakerError as exc:
            self.show_error("library", getattr(exc, "user_message", str(exc)))
            return
        self.consent_result.hide()
        self.lbl_status.setText(tr("consent.confirmed", scope=tr("consent.scope_" + scope)))

    # ------------------------------------------------------------------ training presets
    def gpu(self):
        """The detected GPU (cached; ``self.gpu_fn`` is replaced in tests)."""
        if self._gpu is None:
            try:
                self._gpu = self.gpu_fn()
            except Exception:  # noqa: BLE001 - an estimate must never break the window
                from infra.vram_optimizer import GpuInfo

                self._gpu = GpuInfo(False)
        return self._gpu

    @property
    def preset(self) -> str:
        return str(self.cmb_preset.currentData() or train_presets.DEFAULT_PRESET)

    def manual_values(self) -> train_presets.Manual:
        return train_presets.Manual(self.sp_epochs.value(), self.sp_rank.value(), self.sp_alpha.value(),
                                    self.sp_lr.value(), self.sp_accum.value())

    def _expected_clips(self) -> tuple:
        """(number of clips to expect, whether it is a real estimate for the chosen audio or a 10-minute example)."""
        secs = 0.0
        if self.no_transcript:
            from core.asr_dataset import expand_inputs

            for p in expand_inputs(self.audio_files)[:200]:
                secs += audio_seconds(p) or 0.0
        elif self.audio:
            secs = audio_seconds(self.audio) or 0.0
        return (train_presets.clips_from_audio(secs), True) if secs > 0 else (train_presets.clips_from_audio(600), False)

    def _on_adv_toggled(self, on: bool) -> None:
        self.adv_box.setVisible(on)
        self.btn_adv.setText(("\u25be " if on else "\u25b8 ") + tr("preset.advanced"))

    def _on_checks_toggled(self, on: bool) -> None:
        self.checks_box.setVisible(on)
        self.btn_checks.setText(("\u25be " if on else "\u25b8 ") + tr("train.checks_options"))

    def _on_preset(self, _i: int = 0) -> None:
        """Show the values of the chosen preset in the advanced panel (editable only for Manual) and refresh the estimate."""
        n, _real = self._expected_clips()
        plan = train_presets.build_plan(self.preset if self.preset != train_presets.MANUAL else train_presets.BALANCED,
                                        self.gpu(), n)
        if self.preset != train_presets.MANUAL:
            for sp, v in ((self.sp_epochs, plan.epochs), (self.sp_rank, plan.lora_r), (self.sp_alpha, plan.lora_alpha),
                          (self.sp_accum, plan.grad_accum)):
                sp.blockSignals(True); sp.setValue(v); sp.blockSignals(False)
            self.sp_lr.blockSignals(True); self.sp_lr.setValue(plan.lr); self.sp_lr.blockSignals(False)
        for sp in (self.sp_epochs, self.sp_rank, self.sp_alpha, self.sp_accum, self.sp_lr):
            sp.setEnabled(self.preset == train_presets.MANUAL)
        if self.preset == train_presets.MANUAL:
            self.btn_adv.setChecked(True)
        self.refresh_estimate()

    def refresh_estimate(self) -> None:
        """Description + estimated training time of the chosen preset on this GPU for the chosen audio."""
        preset = self.preset
        self.lbl_preset_desc.setText(tr("preset.desc_" + preset))
        n, real = self._expected_clips()
        gpu = self.gpu()
        plan = train_presets.build_plan(preset, gpu, n, manual=self.manual_values() if preset == train_presets.MANUAL else None)
        dur = train_presets.format_duration(train_presets.estimate_seconds(plan, n, gpu), tr("preset.unit_min"),
                                            tr("preset.unit_sec"), tr("preset.unit_hour"))
        where = gpu.name or "GPU"
        from workers import preview_runner

        vs = preview_runner.make_variants(plan, self.chk_compare.isChecked())
        per_variant = preview_runner.SYNTH_CHECK_SEC + preview_runner.EXTRA_SCALE_SEC * (len(adapter_strength.PREVIEW_SCALES) - 1)
        quick = sum(train_presets.estimate_seconds(v.plan, min(preview_runner.MAX_CLIPS, n), gpu) + per_variant for v in vs) + preview_runner.ASR_LOAD_SEC
        self.lbl_preview_estimate.setText(tr("preview.estimate", time=train_presets.format_duration(
            quick, tr("preset.unit_min"), tr("preset.unit_sec"), tr("preset.unit_hour")), n=len(vs)))
        fmt = dict(time=dur, epochs=plan.epochs, rank=plan.lora_r, n=n, gpu=where)
        self.lbl_estimate.setText(tr("preset.estimate", **fmt) if real else tr("preset.estimate_example", **fmt))

    @property
    def no_transcript(self) -> bool:
        """True while the "I don't have a transcript" mode is on."""
        return self.chk_no_text.isChecked()

    def set_no_transcript(self, on: bool) -> None:
        """Switch the audio-only mode: hides the text row, shows the warning and the opt-in; the opt-in resets each time."""
        if self.chk_no_text.isChecked() != on:
            self.chk_no_text.setChecked(on)
            return
        self.chk_asr_ok.setChecked(False)
        self.asr_box.setVisible(on)
        self.clipcheck_row.setVisible(not on)     # the clip cross-check needs a transcript (core/clip_check.py)
        self.clipmax_row.setVisible(not on)       # auto-transcribed datasets cut their own clips (core/asr_dataset.py, up to 20 s)
        self._apply_text_row()      # the text row stays: in this mode it is the OPTIONAL script that was read aloud
        self._show_audio_label()
        self._refresh_buttons()
        self._assess_noise()        # the other mode has other recordings

    def _apply_text_row(self) -> None:
        """Texts of the text row: the transcript (normal mode) or the optional recording script (audio-only mode)."""
        if self.no_transcript:
            self.btn_text.setText(tr("asr.choose_script"))
            self.lbl_text.setText(self.text.name if self.text else tr("asr.script_none"))
        else:
            self.btn_text.setText(tr("ui.choose_text"))
            self.lbl_text.setText(self.text.name if self.text else tr("ui.text_none"))

    def set_audio_files(self, paths: List[Path]) -> None:
        """No-transcript mode: remember several audio files and/or folders."""
        from core.asr_dataset import expand_inputs

        self.audio_files = [Path(p) for p in paths]
        self.preview_scale = None
        self._audio_count = len(expand_inputs(self.audio_files))
        self._show_audio_label()
        self._on_preset()
        self._refresh_buttons()
        self._assess_noise()

    def _show_audio_label(self) -> None:
        if self.no_transcript:
            n = getattr(self, "_audio_count", 0) if self.audio_files else 0
            self.lbl_audio.setText(tr("asr.files_chosen", n=n) if n else tr("asr.files_none"))
            self.lbl_audio.setToolTip("\n".join(str(p) for p in self.audio_files[:20]))
        else:
            self.lbl_audio.setText(self.audio.name if self.audio else tr("ui.audio_none"))
            self.lbl_audio.setToolTip(str(self.audio) if self.audio else "")

    def choose_folder(self) -> None:
        """No-transcript mode: pick a folder with clips (searched recursively)."""
        d = QFileDialog.getExistingDirectory(self, tr("asr.choose_folder"))
        if d:
            self.set_audio_files(self.audio_files + [Path(d)])

    def choose_audio(self) -> None:
        """File dialog for the recording (several files in the no-transcript mode)."""
        exts = " ".join(f"*{e}" for e in sorted(AUDIO_EXT))
        if self.no_transcript:
            fs, _ = QFileDialog.getOpenFileNames(self, tr("ui.dlg_audio"), "", tr("ui.filter_audio", exts=exts))
            if fs:
                self.set_audio_files([Path(f) for f in fs])
            return
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
        if self.no_transcript:   # many files and/or folders at once
            dropped = [Path(u.toLocalFile()) for u in e.mimeData().urls()]
            scripts = [p for p in dropped if p.suffix.lower() in TEXT_EXT]
            if scripts:
                self.set_text(scripts[0])       # a dropped .txt is the optional recording script
            dropped = [p for p in dropped if p.is_dir() or p.suffix.lower() in AUDIO_EXT]
            if dropped:
                self.set_audio_files(self.audio_files + dropped)
            return
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
        if self.no_transcript:   # audio only: needs files and the explicit "I understand" opt-in
            ready = bool(self.audio_files) and self.chk_asr_ok.isChecked() and not busy
        else:
            ready = bool(self.audio and self.text) and not busy
        self.btn_audio.setEnabled(not busy)
        self.btn_text.setEnabled(not busy)
        self.btn_folder.setEnabled(not busy)
        self.chk_no_text.setEnabled(not busy)
        self.chk_asr_ok.setEnabled(not busy)
        self.btn_settings.setEnabled(True)
        for w in (self.cmb_voice_type, self.cmb_voice_age, self.edt_voice_desc, self.edt_voice_name, self.edt_voice_speaker,
                  self.edt_voice_prepared, self.edt_voice_org, self.edt_voice_url):
            w.setEnabled(not busy)
        has_adapter = last_adapter() is not None
        self.btn_merge.setEnabled(not busy and has_adapter)
        self.btn_merge.setToolTip(tr("ui.tip_merge") if has_adapter else tr("ui.tip_merge_disabled"))
        self.btn_lora.setEnabled(ready)
        self.btn_dataset.setEnabled(ready)
        self.btn_preview.setEnabled(ready)
        self.chk_compare.setEnabled(not busy)
        self.chk_check.setEnabled(not busy)
        self.chk_pick.setEnabled(not busy)
        self.chk_clipcheck.setEnabled(not busy)
        self.sp_clipcer.setEnabled(not busy and self.chk_clipcheck.isChecked())
        self.sp_clipmax.setEnabled(not busy)
        self._sync_denoise_controls()
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
        if self.no_transcript:
            if self.busy or not (self.audio_files and self.chk_asr_ok.isChecked()):
                return
            self._launch(TaskRequest(kind=kind, no_transcript=True, audio_files=list(self.audio_files), force_cpu=force_cpu,
                                     text=self.text,      # optional script: the recognised pieces are matched to it (core.script_match)
                                     **self.voice_details(),
                                     preset=self.preset, manual=self.manual_values() if self.preset == train_presets.MANUAL else None,
                                     **self._consent_kwargs(), **self._task_extras()))
            return
        if self.busy or not (self.audio and self.text):
            return
        self._launch(TaskRequest(kind=kind, audio=self.audio, text=self.text, force_cpu=force_cpu,
                                 **self.voice_details(),
                                 preset=self.preset, manual=self.manual_values() if self.preset == train_presets.MANUAL else None,
                                     **self._consent_kwargs(), **self._task_extras()))

    def start_merge(self, adapter: Optional[Path] = None) -> bool:
        """"Universal model" button: check free disk space, ask for confirmation, start.  True if started.

        Under the offscreen test platform the confirmation box is skipped.
        """
        adapter = Path(adapter) if adapter else last_adapter()   # a voice from "My voices" or the last trained one
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
        if getattr(result, "kind", "") == KIND_PREVIEW:
            self.show_previews(result.previews)
            self.lbl_status.setText(tr("preview.ready", n=len(result.previews)))
            self.lbl_ready.show()
            self._refresh_buttons()
            return
        if getattr(result, "kind", "") == KIND_MERGE:
            self.lbl_status.setText(tr("ui.merge_done", path=result.merged_path) + "\n"
                                    + tr("ui.merge_note", speaker=result.speaker))
        else:
            extra = ""
            if getattr(result, "adapter_path", None):
                extra = "\n" + tr("ui.adapter_saved")
                if getattr(result, "voice_id", ""):
                    extra += "\n" + tr("ui.voice_registered")
            warn = ("\n" + "\n".join(result.warnings[-3:])) if getattr(result, "warnings", None) else ""
            self.consent_result.hide()
            cb = getattr(result, "consent", None)
            if cb and not cb.get("confirmed") and getattr(result, "voice_id", ""):
                self.show_consent_result(cb, result.voice_id)
            rep = getattr(result, "asr_report", None)
            if rep is not None:   # no-transcript mode: how much of the audio was usable
                extra += "\n" + tr("asr.report", files=rep.files - rep.files_failed, kept=rep.kept, clips=rep.clips,
                                   kept_min=f"{rep.seconds_kept / 60:.1f}", total_min=f"{rep.seconds_total / 60:.1f}",
                                   pct=int(round(100 * rep.share_kept)))
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
        """Friendly message box per error kind."""
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
        box.addButton(tr("ui.ok"), QMessageBox.ButtonRole.AcceptRole)
        self._error_box = box
        self._retry_cpu_button = None
        box.setModal(True)
        box.finished.connect(lambda _=0: None)
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":  # do not block in tests
            return
        box.exec()

    # ------------------------------------------------------------------ updates
    def check_updates(self) -> None:
        """Start a manual update check in the background (called from the Settings dialog)."""
        if self.update_worker and self.update_worker.isRunning():
            return
        self.lbl_status.setText(tr("upd.checking"))
        self.progress.setValue(0)
        self.update_worker = UpdateWorker(self.updater_factory, parent=self)

        def on_progress(percent, _stage, message) -> None:
            self.progress.setValue(percent)
            self.lbl_status.setText(message)

        self.update_worker.progress.connect(on_progress)
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
        def on_progress(percent, message) -> None:
            self.progress.setValue(percent)
            self.lbl_status.setText(message)

        w.progress.connect(on_progress)
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
            from ui import std_buttons

            std_buttons.information(self, APP_TITLE, self._os_check.message)
        self.refresh_status()
        if self.busy:
            return
        w = UpdateWorker(self.updater_factory, silent=True, only_if_due=True, parent=self)
        def on_done(summary, changed) -> None:
            if changed:
                self.lbl_status.setText(summary)

        w.done.connect(on_done)
        w.offer.connect(lambda offers, w=w: self._show_offer(w, offers))
        w.done.connect(lambda *_: self.refresh_status())
        self.update_worker = w
        w.start()

    def maybe_show_privacy_notice(self) -> bool:
        """Show the privacy notice on the very first run.  Under offscreen (tests) nothing is shown or stored."""
        if privacy_acknowledged() or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return False
        from ui import std_buttons

        std_buttons.information(self, tr("ui.privacy_title"), tr("ui.privacy_text"))
        acknowledge_privacy()
        return True

    # ------------------------------------------------------------------ first run
    def start_prefetch(self, hook: Optional[Callable[[Any], None]] = None) -> Optional[PrefetchWorker]:
        """Download ALL required models automatically on the first run (no questions asked, no "quality" button).

        ``hook(worker)`` runs before the worker starts (the Components window connects its live line there, so no signal is
        lost).  Returns the worker, or None if a task is running."""
        if self.busy:
            return None
        w = PrefetchWorker(self.prefetch_fn, parent=self)
        self.prefetch_worker = w
        self.lbl_status.setText(tr("ui.prefetch_start"))

        def on_progress(percent, message) -> None:
            self.progress.setValue(percent)
            self.lbl_status.setText(message)

        w.progress.connect(on_progress)
        w.done.connect(self._on_prefetch_done)
        w.failed.connect(self._on_prefetch_failed)
        w.finished.connect(self._refresh_buttons)
        if hook is not None:
            hook(w)
        w.start()
        self._refresh_buttons()
        return w

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
        if hasattr(self, "preview_player"):
            self.preview_player.shutdown()
        if self.upgrade_dialog is not None:      # an unanswered offer = «Not now» (the worker must not wait)
            self.upgrade_dialog.reject()
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        super().closeEvent(e)
        self.closing.emit()
