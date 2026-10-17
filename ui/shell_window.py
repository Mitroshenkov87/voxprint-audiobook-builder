"""The product window: the shared audiobook screen.

The audio sample (book, voice, pace, player) is this window. Settings, check
and repair, and backup open the existing settings dialog, because the shell
already sends ``view.settings``. Voices, Roles, Re-voice, and Train voice are Sections and Steps in
``docs/ui/FEATURE-MAP.md``. This pin has neither a section row nor a step
wizard, so those windows stay as they are. This file does not invent them.

    from ui.shell_window import AudiobookShell
    window = AudiobookShell()
    window.show()
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import QPushButton, QScrollArea, QVBoxLayout, QWidget

from voxprint_common.i18n import add_catalogs
from voxprint_common.i18n import tr as shell_tr
from voxprint_common.paths import state_dir
from voxprint_common.suite_settings import set_window_geometry, window_geometry
from voxprint_common.ui.expert import ExpertMode
from voxprint_common.ui.simple_image import Phase, SimpleLayout, SimpleWidget
from voxprint_common.ui.theme import build_shell_style
from voxprint_common.ui.tokens import load as load_tokens
from voxprint_common.ui.variants import WINDOW_HEIGHT, WINDOW_WIDTH, state_names, surface_for

from infra.mdl_gate import manage_argv, models_ready
from ui.main_window import MainWindow
from ui.narrate_state import SURFACE_ID, narrate_presets, narrate_settings

# Methods the shell does not offer. Callers look them up and do not invent them.
MISSING_SHELL_API = ("set_voices", "input_path", "open_page", "pause")

# Chip stylesheet padding is 14 px on each side. See QPushButton[chip="true"].
_CHIP_PAD_X = 14
_HISTORY_HEIGHT = 104

_CATALOGS = Path(__file__).resolve().parent.parent / "locales" / "shell"


class _Allow:
    """Let a narration request through. This app has no picture gate."""

    def review_prompt(self, prompt: str) -> str | None:
        """Allow the request.

        Args:
            prompt: The book name the screen is holding.
        """
        return None

    def review_result(self, prompt: str, item_id: str) -> str | None:
        """Allow a finished item.

        Args:
            prompt: The book name.
            item_id: The id the caller chose.
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
        """Size each pill to its bold label, and hide the history row while it is empty.

        The first paint runs inside the screen constructor, before ``self.screen`` exists.
        """
        if getattr(self, "screen", None) is None:
            return
        self._fit_pills()
        self._fit_history()

    def _fit_pills(self) -> None:
        density = load_tokens().density
        pad = _CHIP_PAD_X * 2 + density.border_px * 2 + 8
        for button in self.screen.root.findChildren(QPushButton):
            if button.property("chip") != "true" and not str(button.objectName()).startswith("reset-"):
                continue
            text = button.text()
            if not text:
                continue
            font = QFont(button.font())
            font.setPixelSize(density.body_px)
            font.setWeight(QFont.Weight.DemiBold)
            advance = QFontMetrics(font).horizontalAdvance(text)
            button.setMinimumWidth(max(advance + pad, 36))

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
