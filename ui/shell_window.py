"""The product window: the shared audiobook screen.

The audio sample (book, voice, pace, player) is this window. Settings, check
and repair, and backup open the existing settings dialog, because the shell
already sends ``view.settings``. Voices, Roles, Re-voice, and Train voice are Sections and Steps in
``docs/ui/FEATURE-MAP.md``. This window does not open that row or wizard yet,
so those windows stay as they are. ``set_voices`` takes one flat id and label
list. This file does not invent sections.

    from ui.shell_window import AudiobookShell
    window = AudiobookShell()
    window.show()
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from voxprint_common.i18n import add_catalogs
from voxprint_common.i18n import tr as shell_tr
from voxprint_common.paths import state_dir
from voxprint_common.suite_settings import set_window_geometry, window_geometry
from voxprint_common.ui.expert import ExpertMode
from voxprint_common.ui.safety import ReviewedOutput
from voxprint_common.ui.simple_image import Phase, SimpleLayout, SimpleWidget
from voxprint_common.ui.theme import build_shell_style
from voxprint_common.ui.tokens import load as load_tokens
from voxprint_common.ui.variants import WINDOW_HEIGHT, WINDOW_WIDTH, state_names, surface_for

from core.voice_library import VoiceLibrary
from infra.mdl_gate import manage_argv, models_ready
from ui.main_window import MainWindow
from ui.narrate_state import (
    SURFACE_ID,
    chips_for_export,
    export_format,
    narrate_presets,
    narrate_settings,
)

# Methods this pin's screen still does not offer. set_voices, input_path, and pause do.
MISSING_SHELL_API = ("open_page",)

# Chip stylesheet padding is 14 px on each side. See QPushButton[chip="true"].
_CHIP_PAD_X = 14
_HISTORY_HEIGHT = 104
_EXPERT_SCROLL_PAD = 18
_DROP_TEXT_PX = 15
_DROP_PAD_X = 16
_EXPORT_SOURCES = ("narr.container", "narr.split_chapters", "narr.file")
_EXPORT_ROWS = ("format", "chapters")

_CATALOGS = Path(__file__).resolve().parent.parent / "locales" / "shell"


def flat_voice_rows(voices: object) -> list[tuple[str, str]]:
    """Return one ``(id, label)`` list for ``set_voices``.

    A pair is one voice. A pair whose second item is a list of pairs is a male
    or female group from the old shape; those voices join the same list.
    """
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(voice_id: object, label: object) -> None:
        ident = str(voice_id).strip()
        text = str(label).strip()
        if not ident or not text or ident in seen:
            return
        seen.add(ident)
        rows.append((ident, text))

    if not isinstance(voices, (list, tuple)):
        return rows
    for item in voices:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        key, value = item
        if isinstance(value, str):
            add(key, value)
            continue
        if isinstance(value, (list, tuple)):
            for nested_id, nested_label in flat_voice_rows(value):
                add(nested_id, nested_label)
    return rows


def library_voice_rows() -> list[tuple[str, str]]:
    """Installed voices as a flat ``set_voices`` list. An empty library stays empty."""
    return flat_voice_rows([(record.id, record.name) for record in VoiceLibrary().list_voices()])


def _needs_restack(form: QWidget) -> bool:
    """True when the expert form still has a side-by-side row."""
    layout = form.layout()
    if layout is None:
        return False
    if not form.property("stackedRows"):
        return True
    for index in range(layout.count()):
        item = layout.itemAt(index)
        sub = None if item is None else item.layout()
        if isinstance(sub, (QHBoxLayout, QFormLayout)):
            return True
    return False


def _take_widgets(layout: QLayout) -> list[QWidget]:
    """Remove every widget from ``layout``, including rows inside a nested box."""
    held: list[QWidget] = []
    while layout.count():
        taken = layout.takeAt(0)
        if taken is None:
            break
        widget = taken.widget()
        if widget is not None:
            held.append(widget)
        nested = taken.layout()
        if nested is not None:
            held.extend(_take_widgets(nested))
    return held


def _layout_holding(layout: QLayout | None, widget: QWidget) -> QLayout | None:
    """Return the nested box that contains ``widget``."""
    if layout is None:
        return None
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item is None:
            continue
        if item.widget() is widget:
            return layout
        found = _layout_holding(item.layout(), widget)
        if found is not None:
            return found
    return None


def _remove_empty_layouts(layout: QLayout) -> None:
    """Drop a box that no longer holds a widget. An empty voice row caps the book column."""
    index = 0
    while index < layout.count():
        item = layout.itemAt(index)
        sub = None if item is None else item.layout()
        if sub is not None and sub.count() == 0:
            layout.takeAt(index)
            sub.deleteLater()
            continue
        index += 1


class _Allow:
    """Let a narration request through. This app has no picture gate."""

    def review_prompt(self, prompt: str) -> str | None:
        """Allow the request.

        Args:
            prompt: The book name the screen is holding.
        """
        return None

    def review_result(self, prompt: str, item_id: str, output: ReviewedOutput) -> str | None:
        """Allow a finished item.

        Args:
            prompt: The book name.
            item_id: The id the caller chose.
            output: The finished file. This app does not review the audio.
        """
        return None


class AudiobookShell:
    """Host for the shared simple screen. It does not draw a second product UI."""

    def __init__(self) -> None:
        """Build the screen at 1200 by 900 the first time, then the saved size."""
        add_catalogs(_CATALOGS)
        self.surface = surface_for(SURFACE_ID)
        self.settings = narrate_settings()
        settings_file, history_file = state_names(self.surface)
        self.layout = SimpleLayout(
            presets=narrate_presets(self.settings),
            settings=self.settings,
            state_dir=state_dir() / SURFACE_ID,
            gate=_Allow(),
            state_name=settings_file,
            history_name=history_file,
        )
        if not models_ready():
            self.layout.unavailable(shell_tr("abb.models_missing"))
        self.screen = SimpleWidget(
            self.layout,
            self.settings,
            on_generate=self._on_generate,
            on_command=self._on_command,
            on_refresh=self._after_paint,
            surface=self.surface,
        )
        self._exporting = False
        self._painting = False
        self._export_key: tuple[object, bool] | None = None
        self._bind_export()
        self._bind_screen()
        self._publish_voices()
        self.screen.refresh()
        self.expert = ExpertMode()
        self._stop_expert = self.expert.subscribe(self.screen.set_expert)
        self._book = ""
        self._settings_host: MainWindow | None = None
        self.widget = QWidget()
        self.widget.setObjectName("shell")
        self.widget.setStyleSheet(build_shell_style(False))
        self.widget.setWindowTitle(shell_tr("splash.app.audiobook-builder"))
        self.widget.setMinimumSize(self.surface.min_width, self.surface.min_height)
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.screen.root)
        self._apply_geometry()
        if self.expert.enabled():
            self.screen.set_expert(True)

    def show(self) -> None:
        """Show the window."""
        self.widget.show()

    def shutdown(self) -> None:
        """Remember the window rectangle and drop the expert listener."""
        self._stop_expert()
        set_window_geometry(self.widget.width(), self.widget.height(), self.widget.x(), self.widget.y())

    def set_book(self, path: str) -> None:
        """Remember a book path the shell cannot return itself.

        Args:
            path: A file the caller already chose.
        """
        self._book = path

    def book_path(self) -> str:
        """Return a book path when the shell, or :meth:`set_book`, has one.

        The drop target keeps the file name only, so a name that is not a
        real file is not a path.
        """
        reader = getattr(self.screen, "input_path", None)
        if callable(reader):
            found = str(reader() or "").strip()
            if found:
                return found
        if self._book:
            return self._book
        name = self.layout.prompt.split(" · ", 1)[0].strip()
        if name and Path(name).is_file():
            return str(Path(name))
        return ""

    def show_progress(self, step: int, total: int, stage: str, eta_seconds: int | None = None) -> None:
        """Paint a narration in progress. The stage text is already translated.

        Args:
            step: Finished steps.
            total: Steps in the job.
            stage: Plain words for the current stage.
            eta_seconds: Seconds still left, or None.
        """
        if not self.layout.prompt.strip():
            self.screen.set_prompt_text("book")
        if self.layout.phase is not Phase.BUSY:
            self.layout.generate()
        self.layout.progress(step, total, stage, eta_seconds=eta_seconds)
        self.screen.refresh()

    def _apply_geometry(self) -> None:
        stored = window_geometry()
        if stored is None:
            self.widget.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
            return
        self.widget.resize(
            max(self.surface.min_width, stored["width"]),
            max(self.surface.min_height, stored["height"]),
        )
        self.widget.move(stored["x"], stored["y"])

    def _after_paint(self) -> None:
        """Size each pill, hide an empty history row, and keep the expert rows from colliding.

        The first paint runs inside the screen constructor, before ``self.screen`` exists.
        """
        if getattr(self, "screen", None) is None or self._painting:
            return
        self._painting = True
        try:
            self._fit_pills()
            self._fit_history()
            self._lift_voice()
            self._fit_source_row()
            self._restack_expert()
            self._activate_voice()
            self._hide_panel_resets()
            QApplication.sendPostedEvents()
            form = self.screen.root.findChild(QWidget, "expertForm")
            if form is not None:
                self._fill_expert_width(form)
        finally:
            self._painting = False

    def _publish_voices(self) -> None:
        """Fill the voice combo from the library. Warm, Clear, and Low stay when it is empty."""
        rows = library_voice_rows()
        if rows:
            self.screen.set_voices(rows)

    def _bind_export(self) -> None:
        """Keep the format chips and the audiobook file on one value."""
        choose = self.layout.choose
        reset_group = self.layout.reset_group
        set_value = self.layout.set_value

        def choose_chip(row_id: str, value: str) -> None:
            choose(row_id, value)
            if row_id in _EXPORT_ROWS:
                self._sync_export("narr.container")

        def reset_chip(group_id: str) -> None:
            reset_group(group_id)
            if group_id in _EXPORT_ROWS:
                self._sync_export("narr.container")

        def edit_value(key: str, value: object) -> None:
            set_value(key, value)
            if self._exporting or key not in _EXPORT_SOURCES:
                return
            self._sync_export(key)

        self.layout.choose = choose_chip
        self.layout.reset_group = reset_chip
        self.layout.set_value = edit_value
        self._adopt_saved_file()

    def _bind_screen(self) -> None:
        """Restack after the expert panel is rebuilt, and repaint linked controls."""
        set_expert = self.screen.set_expert
        edit = self.screen._on_expert_edit

        def show_expert(enabled: bool) -> None:
            form = self.screen.root.findChild(QWidget, "expertForm")
            if form is not None:
                form.setProperty("stackedRows", False)
            set_expert(enabled)
            self._after_paint()

        def on_expert_edit(setting_id: str, value: object) -> None:
            edit(setting_id, value)
            if setting_id in _EXPORT_SOURCES:
                self.screen._paint_expert()

        refresh_resets = self.screen._refresh_resets

        def refresh_and_hide() -> None:
            refresh_resets()
            self._hide_panel_resets()

        self.screen.set_expert = show_expert
        self.screen._on_expert_edit = on_expert_edit
        self.screen._refresh_resets = refresh_and_hide

    def _adopt_saved_file(self) -> None:
        """Point the chips at a saved file when that file is one the chips can name."""
        mapped = chips_for_export(self.layout.get("narr.file"))
        self._exporting = True
        try:
            if mapped is not None:
                container, split = mapped
                if split is None:
                    split = bool(self.layout.get("narr.split_chapters"))
                if self.layout.get("narr.container") != container:
                    self.layout.set_value("narr.container", container)
                if bool(self.layout.get("narr.split_chapters")) != bool(split):
                    self.layout.set_value("narr.split_chapters", bool(split))
            self._export_key = (
                self.layout.get("narr.container"),
                bool(self.layout.get("narr.split_chapters")),
            )
        finally:
            self._exporting = False

    def _sync_export(self, source: str) -> None:
        """Write ``narr.file`` from the chips, or the chips from an expert file edit.

        An expert file the chips cannot name (FLAC, WAV) stays until a format chip changes.
        """
        if self._exporting:
            return
        self._exporting = True
        try:
            container = self.layout.get("narr.container")
            split = bool(self.layout.get("narr.split_chapters"))
            current = self.layout.get("narr.file")
            if source == "narr.file":
                mapped = chips_for_export(current)
                if mapped is None:
                    self._export_key = None
                    return
                new_container, new_split = mapped
                if new_split is None:
                    new_split = split
                if container != new_container:
                    self.layout.set_value("narr.container", new_container)
                if bool(split) != bool(new_split):
                    self.layout.set_value("narr.split_chapters", bool(new_split))
                self._export_key = (new_container, bool(new_split))
                return
            key = (container, split)
            if key == self._export_key:
                return
            wanted = export_format(str(container), split)
            if current != wanted:
                self.layout.set_value("narr.file", wanted)
            self._export_key = key
        finally:
            self._exporting = False

    def _lift_voice(self) -> None:
        """Give the voice dropdown the full window width, above the expert split."""
        root = self.screen.root
        if root.findChild(QWidget, "voiceRow") is not None:
            return
        voice = root.findChild(QComboBox, "voicePicker")
        label = root.findChild(QLabel, "voiceLabel")
        layout = root.layout()
        if voice is None or label is None or layout is None:
            return
        row = QWidget(root)
        row.setObjectName("voiceRow")
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(8)
        label.setParent(row)
        voice.setParent(row)
        voice.setMinimumWidth(0)
        voice.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        box.addWidget(label)
        box.addWidget(voice, 1)
        layout.insertWidget(2, row)

    def _activate_voice(self) -> None:
        """Hold the voice list to the free width of its row, and release that width when the window shrinks."""
        row = self.screen.root.findChild(QWidget, "voiceRow")
        if row is None or row.layout() is None:
            return
        layout = row.layout()
        layout.activate()
        voice = row.findChild(QComboBox, "voicePicker")
        label = row.findChild(QLabel, "voiceLabel")
        if voice is None:
            return
        label_width = 0 if label is None else label.sizeHint().width()
        margins = layout.contentsMargins()
        available = row.width() - label_width - layout.spacing() - margins.left() - margins.right()
        if available < 80:
            return
        voice.setMaximumWidth(16777215)
        voice.setMinimumWidth(available)
        layout.activate()

    def _fit_source_row(self) -> None:
        """Give the book drop one line of text and the spare width beside Create audiobook."""
        root = self.screen.root
        drop = root.findChild(QLabel, "fileDrop")
        if drop is None or root.layout() is None:
            return
        column = _layout_holding(root.layout(), drop)
        if column is not None:
            _remove_empty_layouts(column)
            column.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
            column.invalidate()
        drop.setWordWrap(False)
        font = QFont(drop.font())
        font.setPixelSize(_DROP_TEXT_PX)
        advance = QFontMetrics(font).horizontalAdvance(drop.text())
        density = load_tokens().density
        text_width = advance + _DROP_PAD_X * 2 + density.border_px * 2
        button = root.findChild(QPushButton, "generate")
        button_width = 148
        if button is not None:
            label = QFont(button.font())
            label.setPixelSize(16)
            label.setWeight(QFont.Weight.DemiBold)
            button_width = max(QFontMetrics(label).horizontalAdvance(button.text()) + 40, 148)
            button.setMinimumWidth(button_width)
            button.setMaximumWidth(button_width)
            policy = button.sizePolicy()
            policy.setHorizontalPolicy(QSizePolicy.Policy.Maximum)
            button.setSizePolicy(policy)
        room = root.width() - button_width - 48
        wanted = max(text_width, int(root.width() * 0.40))
        if room >= text_width:
            wanted = min(wanted, room)
        drop.setMaximumWidth(16777215)
        drop.setMinimumWidth(max(wanted, text_width))
        drop.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        root.layout().activate()

    def _hide_panel_resets(self) -> None:
        """Keep the one reset that already sits beside the pace chips."""
        host = self.screen.root.findChild(QWidget, "expertResets")
        if host is None:
            return
        host.hide()
        host.setMaximumHeight(0)

    def _restack_expert(self) -> None:
        """One label, then one control, in each expert row, inside the scroll panel."""
        form = self.screen.root.findChild(QWidget, "expertForm")
        if form is None or not _needs_restack(form):
            return
        layout = form.layout()
        if layout is None:
            return
        index = 0
        while index < layout.count():
            item = layout.itemAt(index)
            sub = None if item is None else item.layout()
            if not isinstance(sub, (QHBoxLayout, QFormLayout)):
                index += 1
                continue
            held = _take_widgets(sub)
            layout.takeAt(index)
            sub.deleteLater()
            for widget in held:
                widget.setMinimumWidth(0)
                widget.setMaximumWidth(16777215)
                if isinstance(widget, QLabel):
                    widget.setWordWrap(True)
                else:
                    widget.setMinimumHeight(max(widget.minimumSizeHint().height(), 1))
                widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
                layout.insertWidget(index, widget)
                index += 1
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, _EXPERT_SCROLL_PAD, 0)
        if form.width() < 80:
            form.resize(240, max(form.height(), 1))
        policy = form.sizePolicy()
        policy.setVerticalPolicy(QSizePolicy.Policy.Minimum)
        policy.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        form.setSizePolicy(policy)
        form.setProperty("stackedRows", True)
        # Size hints stay empty until the new rows are polished.
        QApplication.sendPostedEvents()
        layout.invalidate()
        hint = layout.sizeHint().height()
        if hint <= 0:
            QApplication.processEvents()
            hint = layout.sizeHint().height()
        if hint > 0:
            form.setMinimumHeight(hint)
        self._watch_expert_width(form)
        self._fill_expert_width(form)
        layout.activate()
        form.updateGeometry()

    def _watch_expert_width(self, form: QWidget) -> None:
        """Keep each expert control as wide as the panel after the scroll area resizes it."""
        if getattr(self, "_expert_width_watch", None) is not None:
            return
        host = self

        class _Width(QObject):
            def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
                if event.type() == QEvent.Type.Resize and not host._painting:
                    host._fill_expert_width(watched)
                return False

        watcher = _Width(form)
        form.installEventFilter(watcher)
        self._expert_width_watch = watcher
        self._fill_expert_width(form)

    def _fill_expert_width(self, form: QWidget) -> None:
        layout = form.layout()
        if layout is None or form.width() < 40:
            return
        margins = layout.contentsMargins()
        width = form.width() - margins.left() - margins.right()
        if width < 40:
            return
        for index in range(layout.count()):
            item = layout.itemAt(index)
            widget = None if item is None else item.widget()
            if widget is None:
                continue
            widget.setMaximumWidth(width)
            if not isinstance(widget, QComboBox):
                continue
            widget.setMinimumWidth(width)
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, widget.sizePolicy().verticalPolicy())
        layout.activate()

    def _fit_pills(self) -> None:
        density = load_tokens().density
        pad = _CHIP_PAD_X * 2 + density.border_px * 2
        for button in self.screen.root.findChildren(QPushButton):
            name = str(button.objectName())
            if button.property("chip") != "true" and not name.startswith("reset-") and name != "expertToggle":
                continue
            text = button.text()
            if not text:
                continue
            font = QFont(button.font())
            font.setPixelSize(density.body_px)
            font.setWeight(QFont.Weight.DemiBold)
            advance = QFontMetrics(font).horizontalAdvance(text)
            extra = 8 if name == "expertToggle" else 0
            button.setMinimumWidth(max(advance + pad + extra, 36))

    def _fit_history(self) -> None:
        strip = self.screen.root.findChild(QScrollArea, "historyStrip")
        if strip is None:
            return
        if self.layout.history:
            strip.setFixedHeight(_HISTORY_HEIGHT)
            strip.show()
            return
        strip.hide()
        strip.setFixedHeight(0)

    def _on_generate(self) -> None:
        if self.book_path():
            return
        self.layout.block(shell_tr("abb.need_book"))
        self.screen.refresh()

    def _on_command(self, command_id: str) -> None:
        if command_id == "view.toggle_expert_mode":
            self.expert.toggle()
            return
        if command_id == "view.settings":
            self.open_saved_settings()
            return
        if command_id == "models.manage":
            subprocess.Popen(manage_argv())

    def open_saved_settings(self) -> None:
        """Open the existing settings dialog. It is not a second shell."""
        host = self._settings_host
        if host is None:
            host = MainWindow(autocheck=False, prefetch=False)
            host.hide()
            self._settings_host = host
        host.open_settings()
