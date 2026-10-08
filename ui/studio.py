"""Studio: the first window of Voxprint AI Audiobook Builder - three big cards and a gear button.

* **Narrate a book** (primary) -> :class:`ui.narrate_window.NarrateWindow`
* **Train your voice**         -> the training window (:class:`ui.main_window.MainWindow`, unchanged workflow)
* **My voices**                -> :class:`ui.voices_window.VoicesWindow`
* **Re-voice**                 -> :class:`ui.revoice_window.RevoiceWindow` (speech -> text -> narrated with a chosen voice)
* the gear (top right) opens the Settings dialog: language, updates, model/data folders, repair, About.

The Studio owns the other windows and shows exactly one at a time.  A window asks for navigation with its ``go`` signal
(``"studio" | "train" | "voices" | "narrate" | "revoice"``); background jobs keep running while another window is shown (the Studio
cards show "training in progress" / "narration in progress").  Closing any window ends the application, after cancelling
running jobs (:meth:`StudioWindow.shutdown`).

The Settings dialog was written for the training window; the Studio exposes the same small service API
(``busy``, ``updating``, ``repairing``, ``check_updates`` ...) by delegating to it, and mirrors its status line and progress
bar so that update checks, repairs and the first-run model download are visible on the home screen.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QApplication, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget)

from core import i18n
from core.i18n import tr
from core.voice_library import VoiceLibrary
from infra import hard_exit, preload
from ui.main_window import APP_TITLE, MainWindow, apply_look
from ui.narrate_window import NarrateWindow
from ui.revoice_window import RevoiceWindow
from ui.settings_dialog import SettingsDialog
from ui.voices_window import VoicesWindow
from ui import screen_fit
from ui.window_base import SubWindow, fit_to_screen

PAGES = ("studio", "train", "voices", "narrate", "revoice")


class ActionCard(QPushButton):
    """A big clickable card: title, description and an optional highlighted note (e.g. "train a voice first")."""

    def __init__(self, primary: bool = False) -> None:
        """Build the card; ``primary`` adds the accent border."""
        super().__init__()
        self.setObjectName("bigcard")
        self.setProperty("primary", "true" if primary else "false")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(100)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 12, 18, 12)
        lay.setSpacing(6)
        self.lbl_title = QLabel()
        self.lbl_title.setObjectName("cardtitle")
        self.lbl_desc = QLabel()
        self.lbl_desc.setObjectName("carddesc")
        self.lbl_desc.setWordWrap(True)
        self.lbl_note = QLabel()
        self.lbl_note.setObjectName("cardnote")
        self.lbl_note.setWordWrap(True)
        self.lbl_note.hide()
        for w in (self.lbl_title, self.lbl_desc, self.lbl_note):
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            lay.addWidget(w)
        lay.addStretch(1)

    def set_texts(self, title: str, desc: str, note: str = "") -> None:
        """Set the three texts; an empty ``note`` hides the note line."""
        self.lbl_title.setText(title)
        self.lbl_desc.setText(desc)
        self.lbl_note.setText(note)
        self.lbl_note.setVisible(bool(note))


class StudioWindow(SubWindow):
    """The home window; also the owner of all other windows and of the application's shutdown."""

    def __init__(self, library: Optional[VoiceLibrary] = None, trainer: Optional[MainWindow] = None,
                 voices: Optional[VoicesWindow] = None, narrate: Optional[NarrateWindow] = None,
                 revoice: Optional[RevoiceWindow] = None, **trainer_kwargs: Any) -> None:
        """Create the Studio and its sub-windows.  ``trainer_kwargs`` go to :class:`MainWindow` (autocheck, prefetch ...)."""
        super().__init__(with_back=False, with_gear=True)
        self.library = library or VoiceLibrary()
        self.trainer = trainer or MainWindow(**trainer_kwargs)
        self.trainer.library = self.library      # the Train window confirms the voice-owner consent in the same library
        self.voices_window = voices or VoicesWindow(self.library)
        self.narrate_window = narrate or NarrateWindow(self.library)
        self.revoice_window = revoice or RevoiceWindow()
        self.pages: Dict[str, QWidget] = {"studio": self, "train": self.trainer, "voices": self.voices_window,
                                          "narrate": self.narrate_window, "revoice": self.revoice_window}
        self.current_page = "studio"
        self._settings: Optional[SettingsDialog] = None
        self._about = None
        self._shutting_down = False
        # Optional "Preload models into memory at startup" (Settings, off by default): ticked by the timer below.
        self.preloader = preload.Preloader(self._preload_voice)
        self._build()
        self.trainer.enable_studio_nav()
        self.apply_theme_to_children()
        self._connect()
        self.retranslate()
        fit_to_screen(self, self.content, 1040, 600)   # wide enough for two narrate columns
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

    # ------------------------------------------------------------------ construction
    def _build(self) -> None:
        """Header texts, the three cards and the status strip."""
        self.lbl_tagline = QLabel()
        self.lbl_tagline.setObjectName("subtitle")
        self.lbl_tagline.setWordWrap(True)
        self.body.addWidget(self.lbl_tagline)
        self.card_narrate = ActionCard(primary=True)
        self.card_train = ActionCard()
        self.card_voices = ActionCard()
        self.card_revoice = ActionCard()
        for c in (self.card_narrate, self.card_train, self.card_voices, self.card_revoice):
            self.body.addWidget(c)
        self.lbl_status = QLabel()
        self.lbl_status.setObjectName("status")
        self.lbl_status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.hide()
        self.body.addWidget(self.lbl_status)
        self.body.addWidget(self.progress)
        self.body.addStretch(1)
        self.lbl_privacy = QLabel()
        self.lbl_privacy.setObjectName("footer")
        self.lbl_privacy.setWordWrap(True)
        self.body.addWidget(self.lbl_privacy)

    def _connect(self) -> None:
        """Wire navigation, mirrored status and shutdown."""
        self.card_narrate.clicked.connect(lambda: self.navigate("narrate"))
        self.card_train.clicked.connect(lambda: self.navigate("train"))
        self.card_voices.clicked.connect(lambda: self.navigate("voices"))
        self.card_revoice.clicked.connect(lambda: self.navigate("revoice"))
        self.revoice_window.narrate_file.connect(self.narrate_file)
        assert self.btn_gear is not None
        self.btn_gear.clicked.connect(self.open_settings)
        for name, w in self.pages.items():
            if name == "studio":
                continue
            w.go.connect(self.navigate)           # type: ignore[attr-defined]
            w.closing.connect(self._on_child_closed)   # type: ignore[attr-defined]
        self.voices_window.narrate_with.connect(self.narrate_with_voice)
        self.voices_window.library_changed.connect(self.refresh)
        self.voices_window.full_model_requested.connect(self._full_model)
        self.trainer.language_changed.connect(self.retranslate_all)
        self.trainer.lbl_status.changed.connect(self._mirror_status)
        self.trainer.progress.valueChanged.connect(self._mirror_progress)

    def apply_theme_to_children(self) -> None:
        """Sub-windows share the Studio's look (plain vs. Acrylic is applied per window when it is shown)."""
        for name, w in self.pages.items():
            if name not in ("studio", "train"):
                w.apply_style_from(self)       # type: ignore[attr-defined]

    def apply_look_all(self) -> None:
        """Window transparency changed in Settings: restyle the visible pages now (hidden ones on their next show)."""
        for w in set(self.pages.values()):
            if w.isVisible():
                apply_look(w)

    # ------------------------------------------------------------------ texts
    def window_title(self) -> str:
        """The Studio is titled with the product name."""
        return APP_TITLE

    def retranslate(self) -> None:
        """Apply the current language to the Studio's own widgets."""
        super().retranslate()
        self.lbl_tagline.setText(tr("studio.tagline"))
        assert self.btn_gear is not None
        self.btn_gear.setToolTip(tr("ui.settings_tip"))
        self.lbl_privacy.setText(tr("ui.footer_privacy"))
        self.refresh()

    def retranslate_all(self) -> None:
        """Language changed: retranslate every window."""
        for name, w in self.pages.items():
            if name != "train":              # the training window retranslates itself before announcing the change
                w.retranslate()              # type: ignore[attr-defined]
        if self._settings is not None:
            self._settings.retranslate()
            self._settings.sync_language()

    # ------------------------------------------------------------------ state
    def refresh(self) -> None:
        """Update the cards (voice count, running jobs) and the status strip."""
        n = len(self.library.list_voices())
        narrating, training = self.narrate_window.busy, self.trainer.busy
        self.card_narrate.set_texts(tr("studio.narrate_title"), tr("studio.narrate_desc"),
                                    tr("studio.narrate_running") if narrating else
                                    (tr("studio.narrate_no_voice") if n == 0 else ""))
        self.card_train.set_texts(tr("studio.train_title"), tr("studio.train_desc"),
                                  tr("studio.train_running") if training else "")
        self.card_voices.set_texts(tr("studio.voices_title"), tr("studio.voices_desc"),
                                   tr("studio.voices_count", n=n) if n else "")
        self.card_revoice.set_texts(tr("studio.revoice_title"), tr("studio.revoice_desc"))
        self.progress.setVisible(self.progress.value() > 0 and (training or self.trainer.updating
                                                                 or self.trainer.repairing))
        self.preloader.tick(busy=self.busy or self.trainer.updating or self.trainer.repairing)

    def _preload_voice(self) -> Any:
        """The voice selected in the narrator (its base model is what Narrate will load), else the first installed voice."""
        vid = self.narrate_window.selected_voice_id()
        voice = self.library.get(vid) if vid else None
        if voice is None:
            voices = self.library.list_voices()
            voice = voices[0] if voices else None
        return voice

    @property
    def busy(self) -> bool:
        """True while a training run / download or a narration job runs."""
        return bool(self.trainer.busy or self.narrate_window.busy or self.revoice_window.busy)

    def _mirror_status(self, text: str) -> None:
        """Show the training window's status line (updates, repair, first-run download) on the home screen."""
        self.lbl_status.setText(text)

    def _mirror_progress(self, value: int) -> None:
        """Mirror the training window's progress bar."""
        self.progress.setValue(value)
        self.refresh()

    # ------------------------------------------------------------------ navigation
    def navigate(self, page: str) -> None:
        """Show ``page`` and hide the current window."""
        if page not in self.pages or page == self.current_page and self.pages[page].isVisible():
            return
        old = self.pages[self.current_page]
        new = self.pages[page]
        if page == "voices":
            self.voices_window.refresh()
        elif page == "narrate":
            self.narrate_window.refresh_voices()
        if old is not new:
            new.resize(old.size())
            new.move(old.pos())
            want = getattr(new, "wide_width", lambda: 0)()
            if want > new.width():                 # e.g. the two Narrate columns: wider, capped to the screen, centred
                screen_fit.fit(new, want, new.height())
        self.current_page = page
        new.show()
        new.raise_()
        new.activateWindow()
        if old is not new:
            old.hide()
        self.refresh()

    def _full_model(self, adapter: str) -> None:
        """"Export -> full model" in My voices: the training window builds the standalone model of that voice."""
        self.navigate("train")
        self.trainer.start_merge(Path(adapter))

    def narrate_with_voice(self, voice_id: str) -> None:
        """"Narrate with this voice" on a voice card: open the narrator with that voice selected."""
        self.navigate("narrate")
        self.narrate_window.select_voice(voice_id)

    def narrate_file(self, path: str) -> None:
        """Re-voice handed over its edited text: open the narrator with it loaded."""
        self.navigate("narrate")
        self.narrate_window.load_book_file(Path(path))

    def show_studio(self) -> None:
        """Show the home window (application start)."""
        self.current_page = "studio"
        self.show()

    # ------------------------------------------------------------------ Settings dialog service API (see SettingsDialog)
    @property
    def updating(self) -> bool:
        """True while an update check runs."""
        return self.trainer.updating

    @property
    def repairing(self) -> bool:
        """True while a repair runs."""
        return self.trainer.repairing

    def set_language(self, code: str) -> None:
        """Switch the language of the whole application (refused while a job runs)."""
        if self.busy or code not in i18n.LANGS or code == i18n.get_language():
            return
        i18n.set_language(code, persist=True)
        self.trainer.retranslate()
        self.retranslate_all()

    def start_prefetch(self, hook=None):
        """The first-run download of all models (runs in the Train window, which owns the status line); see
        :meth:`ui.main_window.MainWindow.start_prefetch`."""
        return self.trainer.start_prefetch(hook)

    def reload_auto_steps(self) -> None:
        """Select the recommended options in the training and narration windows again."""
        self.trainer.reload_auto_steps()
        self.narrate_window.reload_auto_steps()

    def check_updates(self) -> None:
        """Start an update check (progress shows on the home screen)."""
        self.trainer.check_updates()

    def start_repair(self) -> None:
        """Repair Voxprint's environment."""
        self.trainer.start_repair()

    def open_models_folder(self) -> None:
        """Open the models folder."""
        self.trainer.open_models_folder()

    def open_data_folder(self) -> None:
        """Open the data folder."""
        self.trainer.open_data_folder()

    def open_about(self) -> None:
        """Show the About dialog."""
        from ui.about_dialog import AboutDialog

        dlg = AboutDialog(self)
        self._about = dlg
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            dlg.show()
            return
        dlg.exec()

    def settings_dialog(self) -> SettingsDialog:
        """The (lazily created) Settings dialog."""
        if self._settings is None:
            self._settings = SettingsDialog(self)
        return self._settings

    def open_settings(self) -> None:
        """Show the Settings dialog (non-blocking under the offscreen test platform)."""
        dlg = self.settings_dialog()
        dlg.setStyleSheet(self.styleSheet())
        dlg.refresh()
        dlg.refresh_preload()                  # RAM and installed models may have changed since the dialog was built
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            dlg.show()
            return
        dlg.exec()

    # ------------------------------------------------------------------ shutdown
    def shutdown(self) -> None:
        """Cancel running jobs and wait for them (idempotent)."""
        if self._shutting_down:
            return
        self._shutting_down = True
        self._timer.stop()
        self.preloader.free("exit")
        self.narrate_window.shutdown()
        self.voices_window.shutdown()
        self.revoice_window.shutdown()
        self.trainer.close()
        for w in (self.voices_window, self.narrate_window, self.revoice_window):
            w.hide()

    def _on_child_closed(self) -> None:
        """A sub-window was closed by the user: end the application."""
        hard_exit.fire()                          # armed (real run): kill every helper process and leave at once
        if self._shutting_down:
            return
        self.shutdown()
        app = QApplication.instance()
        if app is not None:
            QTimer.singleShot(0, app.quit)

    def closeEvent(self, e) -> None:  # noqa: N802
        """Closing the Studio ends the application - hard: all child processes are killed, nothing lingers."""
        hard_exit.fire()
        self.shutdown()
        super().closeEvent(e)
