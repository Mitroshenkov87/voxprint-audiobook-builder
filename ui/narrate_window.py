""""Narrate a book": choose a book (TXT / FB2 / EPUB), a voice and an output format, press Start.

The window only collects choices and shows progress; the work happens in :class:`workers.narrate_worker.NarrateWorker`
(chunk-by-chunk synthesis with the voice's LoRA adapter, resumable, see :mod:`core.narration`).  Pause/Resume and Cancel
are available while it runs; finished chunks are cached on disk, so Start after a cancel or a crash continues where it
stopped.

Output formats (see :mod:`core.audiobook_export`): the default is one Ogg Opus file with chapter markers; per-chapter MP3
is the most compatible alternative; M4B (AAC, for Apple Books) is a separate, clearly marked opt-in with a legal note and
can be hidden completely (:mod:`infra.features`); more formats and the bitrates sit under "Other formats and quality".

Extension points that are deliberately *not* implemented yet are only mentioned in one line of text ("coming later"):
text clean-up, translation and multi-voice role markup.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QProgressBar,
                               QPushButton, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from core import audiobook_export as ex
from core import narration as nr
from core.book_parsers import SUPPORTED_EXTENSIONS, Book, load_book
from core.errors import DatasetMakerError
from core.i18n import tr
from core.voice_library import VoiceLibrary
from infra import features
from ui.main_window import open_folder
from ui.voices_window import make_badge
from ui.window_base import SubWindow, card_frame, fit_to_screen, hint_label
from workers.narrate_worker import NarrateWorker
from workers.narration_runner import NarrationJob, default_output_dir, format_eta, run_narration

log = logging.getLogger("voxprint.ui.narrate")

#: Formats shown directly (in this order); the M4B entry is separated and carries the legal note.
MAIN_FORMATS: Tuple[str, ...] = (ex.FORMAT_OPUS_SINGLE, ex.FORMAT_MP3_CHAPTERS)
OTHER_FORMATS: Tuple[str, ...] = (ex.FORMAT_M4B_OPUS, ex.FORMAT_OPUS_CHAPTERS, ex.FORMAT_MP3_SINGLE,
                                  ex.FORMAT_FLAC_CHAPTERS, ex.FORMAT_WAV_CHAPTERS)
CHARS_PER_SECOND = 14.0         # rough speaking rate used for the "about N hours" estimate


def format_texts() -> Dict[str, Tuple[str, str]]:
    """``{format: (name, where it plays)}`` in the current language."""
    return {
        ex.FORMAT_OPUS_SINGLE: (tr("fmt.opus_single"), tr("fmt.opus_single_where")),
        ex.FORMAT_MP3_CHAPTERS: (tr("fmt.mp3_chapters"), tr("fmt.mp3_chapters_where")),
        ex.FORMAT_M4B: (tr("fmt.m4b"), tr("fmt.m4b_where")),
        ex.FORMAT_M4B_OPUS: (tr("fmt.m4b_opus"), tr("fmt.m4b_opus_where")),
        ex.FORMAT_OPUS_CHAPTERS: (tr("fmt.opus_chapters"), tr("fmt.opus_chapters_where")),
        ex.FORMAT_MP3_SINGLE: (tr("fmt.mp3_single"), tr("fmt.mp3_single_where")),
        ex.FORMAT_FLAC_CHAPTERS: (tr("fmt.flac_chapters"), tr("fmt.flac_chapters_where")),
        ex.FORMAT_WAV_CHAPTERS: (tr("fmt.wav_chapters"), tr("fmt.wav_chapters_where")),
    }


def estimate_hours(book: Book) -> float:
    """Rough length of the narration in hours (characters / speaking rate)."""
    return book.total_chars / CHARS_PER_SECOND / 3600.0


class NarrateWindow(SubWindow):
    """The "Narrate a book" window."""

    def __init__(self, library: Optional[VoiceLibrary] = None, runner: Callable[..., Any] = run_narration,
                 pick_book: Optional[Callable[[], str]] = None, pick_folder: Optional[Callable[[], str]] = None,
                 out_dir: Optional[Path] = None, aac_allowed: Optional[bool] = None,
                 auto_open_folder: bool = True) -> None:
        """Build the window.  ``runner`` and the file pickers are injectable (tests); ``aac_allowed`` overrides the feature flag."""
        super().__init__(with_back=True)
        self.library = library or VoiceLibrary()
        self.runner = runner
        self._pick_book, self._pick_folder = pick_book, pick_folder
        self.out_dir: Path = Path(out_dir) if out_dir else default_output_dir()
        self.aac_allowed = features.aac_enabled() if aac_allowed is None else aac_allowed
        self.auto_open_folder = auto_open_folder
        self.book: Optional[Book] = None
        self.book_path: Optional[Path] = None
        self.worker: Optional[NarrateWorker] = None
        self.result: Optional[nr.NarrationResult] = None
        self.format_checks: Dict[str, QCheckBox] = {}
        self.format_desc: Dict[str, QLabel] = {}
        self.format_names: Dict[str, QLabel] = {}
        self._build()
        self.retranslate()
        self.refresh_voices()
        fit_to_screen(self, self.content, 820, 600)

    # ------------------------------------------------------------------ construction
    def _format_row(self, fmt: str, parent_layout: QVBoxLayout, checked: bool = False) -> QCheckBox:
        """A format check box with its one-line description below."""
        chk = QCheckBox()
        chk.setChecked(checked)
        desc = hint_label()
        desc.setContentsMargins(26, 0, 0, 0)
        parent_layout.addWidget(chk)
        parent_layout.addWidget(desc)
        self.format_checks[fmt], self.format_desc[fmt] = chk, desc
        chk.toggled.connect(lambda _c: self._refresh_buttons())
        return chk

    def _build(self) -> None:
        """Create all widgets (texts come from :meth:`retranslate`)."""
        self.lbl_intro = hint_label()
        self.body.addWidget(self.lbl_intro)

        # --- book ---
        c = card_frame()
        v = QVBoxLayout(c)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(8)
        self.lbl_book_title = QLabel()
        self.lbl_book_title.setObjectName("sectiontitle")
        row = QHBoxLayout()
        self.btn_book = QPushButton()
        self.lbl_book = QLabel()
        self.lbl_book.setObjectName("fileLabel")
        row.addWidget(self.btn_book)
        row.addWidget(self.lbl_book, 1)
        self.lbl_book_info = hint_label()
        self.lbl_book_error = QLabel()
        self.lbl_book_error.setObjectName("cardnote")
        self.lbl_book_error.setWordWrap(True)
        for w in (self.lbl_book_title,):
            v.addWidget(w)
        v.addLayout(row)
        v.addWidget(self.lbl_book_info)
        v.addWidget(self.lbl_book_error)
        self.body.addWidget(c)

        # --- voice ---
        c = card_frame()
        v = QVBoxLayout(c)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(8)
        self.lbl_voice_title = QLabel()
        self.lbl_voice_title.setObjectName("sectiontitle")
        v.addWidget(self.lbl_voice_title)
        vrow = QHBoxLayout()
        self.cmb_voice = QComboBox()
        self.cmb_voice.setMinimumWidth(260)
        self.badge_box = QHBoxLayout()
        vrow.addWidget(self.cmb_voice)
        vrow.addLayout(self.badge_box)
        vrow.addStretch(1)
        v.addLayout(vrow)
        self.lbl_voice_info = hint_label()
        v.addWidget(self.lbl_voice_info)
        self.no_voice = QWidget()
        nv = QVBoxLayout(self.no_voice)
        nv.setContentsMargins(0, 0, 0, 0)
        self.lbl_no_voice = QLabel()
        self.lbl_no_voice.setWordWrap(True)
        nrow = QHBoxLayout()
        self.btn_to_train = QPushButton()
        self.btn_to_voices = QPushButton()
        nrow.addWidget(self.btn_to_train)
        nrow.addWidget(self.btn_to_voices)
        nrow.addStretch(1)
        nv.addWidget(self.lbl_no_voice)
        nv.addLayout(nrow)
        v.addWidget(self.no_voice)
        self.body.addWidget(c)

        # --- output ---
        c = card_frame()
        v = QVBoxLayout(c)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(6)
        self.lbl_format_title = QLabel()
        self.lbl_format_title.setObjectName("sectiontitle")
        v.addWidget(self.lbl_format_title)
        self._format_row(ex.FORMAT_OPUS_SINGLE, v, checked=True)
        self.lbl_players = hint_label()
        self.lbl_players.setContentsMargins(26, 0, 0, 0)
        v.addWidget(self.lbl_players)
        self._format_row(ex.FORMAT_MP3_CHAPTERS, v)
        # M4B (AAC): separated, with a legal note
        self.aac_box = QWidget()
        av = QVBoxLayout(self.aac_box)
        av.setContentsMargins(0, 6, 0, 0)
        sep = QFrame()
        sep.setObjectName("sep")
        av.addWidget(sep)
        self._format_row(ex.FORMAT_M4B, av)
        self.note = QFrame()
        self.note.setObjectName("note")
        nl = QVBoxLayout(self.note)
        nl.setContentsMargins(10, 8, 10, 8)
        self.lbl_aac_note = QLabel()
        self.lbl_aac_note.setWordWrap(True)
        self.lbl_aac_disclaimer = QLabel()
        self.lbl_aac_disclaimer.setWordWrap(True)
        nl.addWidget(self.lbl_aac_note)
        nl.addWidget(self.lbl_aac_disclaimer)
        av.addWidget(self.note)
        v.addWidget(self.aac_box)
        self.aac_box.setVisible(self.aac_allowed)
        self.lbl_aac_disclaimer.setVisible(False)
        self.format_checks[ex.FORMAT_M4B].toggled.connect(self._on_aac_toggled)
        # other formats + bitrates
        self.btn_other = QToolButton()
        self.btn_other.setObjectName("expander")
        self.btn_other.setCheckable(True)
        v.addWidget(self.btn_other)
        self.other_box = QWidget()
        ov = QVBoxLayout(self.other_box)
        ov.setContentsMargins(0, 0, 0, 0)
        ov.setSpacing(6)
        for fmt in OTHER_FORMATS:
            self._format_row(fmt, ov)
        brow = QHBoxLayout()
        self.spn_opus, self.spn_mp3, self.spn_aac = QSpinBox(), QSpinBox(), QSpinBox()
        for spn, lo, hi, val in ((self.spn_opus, 12, 128, 32), (self.spn_mp3, 32, 320, 96), (self.spn_aac, 24, 256, 64)):
            spn.setRange(lo, hi)
            spn.setValue(val)
            spn.setSuffix(" kbit/s")
        self.lbl_rate_opus, self.lbl_rate_mp3, self.lbl_rate_aac = QLabel(), QLabel(), QLabel()
        for lbl, spn in ((self.lbl_rate_opus, self.spn_opus), (self.lbl_rate_mp3, self.spn_mp3),
                         (self.lbl_rate_aac, self.spn_aac)):
            brow.addWidget(lbl)
            brow.addWidget(spn)
        brow.addStretch(1)
        self.rates_row = brow
        ov.addLayout(brow)
        self.lbl_rates_hint = hint_label()
        ov.addWidget(self.lbl_rates_hint)
        self.other_box.setVisible(False)
        v.addWidget(self.other_box)
        self.btn_other.toggled.connect(self._on_other_toggled)
        # output folder + options
        frow = QHBoxLayout()
        self.btn_out = QPushButton()
        self.lbl_out = QLabel()
        self.lbl_out.setObjectName("fileLabel")
        frow.addWidget(self.btn_out)
        frow.addWidget(self.lbl_out, 1)
        v.addLayout(frow)
        self.chk_titles = QCheckBox()
        self.chk_titles.setChecked(True)
        v.addWidget(self.chk_titles)
        self.body.addWidget(c)

        self.lbl_later = hint_label()
        self.body.addWidget(self.lbl_later)

        # --- run ---
        self.btn_start = QPushButton()
        self.btn_start.setObjectName("primary")
        self.body.addWidget(self.btn_start)
        ctl = QHBoxLayout()
        self.btn_pause = QPushButton()
        self.btn_cancel = QPushButton()
        ctl.addWidget(self.btn_pause)
        ctl.addWidget(self.btn_cancel)
        ctl.addStretch(1)
        self.body.addLayout(ctl)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.body.addWidget(self.progress)
        self.lbl_status = QLabel()
        self.lbl_status.setObjectName("status")
        self.lbl_status.setWordWrap(True)
        self.body.addWidget(self.lbl_status)
        self.lbl_ready = QLabel()
        self.lbl_ready.setObjectName("ready")
        self.btn_open = QPushButton()
        self.body.addWidget(self.lbl_ready)
        self.body.addWidget(self.btn_open)
        self.body.addStretch(1)
        self.lbl_footer = QLabel()
        self.lbl_footer.setObjectName("footer")
        self.lbl_footer.setWordWrap(True)
        self.body.addWidget(self.lbl_footer)
        self.lbl_ready.hide()
        self.btn_open.hide()

        self.btn_book.clicked.connect(self.choose_book)
        self.btn_out.clicked.connect(self.choose_folder)
        self.btn_start.clicked.connect(self.start)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_open.clicked.connect(self.open_result)
        self.btn_to_train.clicked.connect(lambda: self.go.emit("train"))
        self.btn_to_voices.clicked.connect(lambda: self.go.emit("voices"))
        self.cmb_voice.currentIndexChanged.connect(self._on_voice_changed)

    # ------------------------------------------------------------------ texts
    def window_title(self) -> str:
        """Localized window title."""
        return tr("narr.title")

    def retranslate(self) -> None:
        """Apply the current language."""
        super().retranslate()
        self.lbl_intro.setText(tr("narr.intro"))
        self.lbl_book_title.setText(tr("narr.book"))
        self.btn_book.setText(tr("narr.choose_book"))
        self.lbl_book.setText(self.book_path.name if self.book_path else tr("narr.no_book"))
        self._render_book_info()
        self.lbl_voice_title.setText(tr("narr.voice"))
        self.lbl_no_voice.setText(tr("narr.no_voice"))
        self.btn_to_train.setText(tr("studio.train_title"))
        self.btn_to_voices.setText(tr("studio.voices_title"))
        self.lbl_format_title.setText(tr("narr.format"))
        names = format_texts()
        for fmt, chk in self.format_checks.items():
            chk.setText(names[fmt][0])
            self.format_desc[fmt].setText(names[fmt][1])
        self.lbl_players.setText(tr("narr.opus_players"))
        self.lbl_aac_note.setText(tr("narr.aac_note"))
        self.lbl_aac_disclaimer.setText(tr("narr.aac_disclaimer"))
        self._on_other_toggled(self.btn_other.isChecked())
        self.lbl_rate_opus.setText(tr("narr.rate_opus"))
        self.lbl_rate_mp3.setText(tr("narr.rate_mp3"))
        self.lbl_rate_aac.setText(tr("narr.rate_aac"))
        self.lbl_rates_hint.setText(tr("narr.rates_hint"))
        self.btn_out.setText(tr("narr.choose_folder"))
        self.lbl_out.setText(str(self.out_dir))
        self.chk_titles.setText(tr("narr.speak_titles"))
        self.lbl_later.setText(tr("narr.coming_later"))
        self.btn_start.setText(tr("narr.start"))
        self.btn_pause.setText(tr("narr.resume") if self._paused() else tr("narr.pause"))
        self.btn_cancel.setText(tr("ui.cancel"))
        self.lbl_ready.setText(tr("ui.ready"))
        self.btn_open.setText(tr("ui.open_folder"))
        self.lbl_footer.setText(tr("narr.footer"))
        if not self.busy and not self.lbl_status.text():
            self.lbl_status.setText(tr("narr.idle"))
        self._render_voice_info()
        self._refresh_buttons()

    def _on_other_toggled(self, open_: bool) -> None:
        """Expand / collapse "Other formats and quality"."""
        self.other_box.setVisible(open_)
        self.btn_other.setText(("\u25be " if open_ else "\u25b8 ") + tr("narr.other_formats"))

    def _on_aac_toggled(self, checked: bool) -> None:
        """The full AAC disclaimer is shown while M4B is selected."""
        self.lbl_aac_disclaimer.setVisible(checked)

    # ------------------------------------------------------------------ book
    def choose_book(self) -> None:
        """Pick a book file and load it."""
        if self._pick_book:
            path = self._pick_book()
        else:
            exts = " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS)
            path, _ = QFileDialog.getOpenFileName(self, tr("narr.choose_book"), "", f"{tr('narr.book_files')} ({exts})")
        if path:
            self.load_book_file(Path(path))

    def load_book_file(self, path: Path) -> bool:
        """Parse ``path``; on failure the problem is shown under the book (nothing else changes)."""
        try:
            book = load_book(path)
        except DatasetMakerError as exc:
            self.lbl_book_error.setText(exc.user_message)
            log.warning("book load failed: %s (%s)", exc.user_message, exc.details)
            return False
        self.book, self.book_path = book, Path(path)
        self.lbl_book.setText(self.book_path.name)
        self.lbl_book_error.setText("")
        self.result = None
        self.lbl_ready.hide()
        self.btn_open.hide()
        self._render_book_info()
        self._refresh_buttons()
        return True

    def _render_book_info(self) -> None:
        """Title, author, number of chapters and a rough length."""
        if self.book is None:
            self.lbl_book_info.setText("")
            return
        b = self.book
        hours = estimate_hours(b)
        length = tr("narr.hours", h=f"{hours:.1f}") if hours >= 1 else tr("narr.minutes", m=max(1, round(hours * 60)))
        head = f"{b.title}" + (f" \u2014 {b.author}" if b.author else "")
        self.lbl_book_info.setText(f"{head}\n" + tr("narr.book_info", chapters=len(b.chapters), length=length))

    # ------------------------------------------------------------------ voices
    def refresh_voices(self, select: str = "") -> None:
        """Reload the voice list (keeps the selection if possible)."""
        current = select or str(self.cmb_voice.currentData() or "")
        self.cmb_voice.blockSignals(True)
        self.cmb_voice.clear()
        for rec in self.library.list_voices():
            self.cmb_voice.addItem(rec.name, rec.id)
        idx = self.cmb_voice.findData(current)
        self.cmb_voice.setCurrentIndex(idx if idx >= 0 else 0)
        self.cmb_voice.blockSignals(False)
        has = self.cmb_voice.count() > 0
        self.cmb_voice.setVisible(has)
        self.no_voice.setVisible(not has)
        self._render_voice_info()
        self._refresh_buttons()

    def select_voice(self, voice_id: str) -> None:
        """Select a voice by id (used by "Narrate with this voice")."""
        self.refresh_voices(select=voice_id)

    def selected_voice_id(self) -> str:
        """Id of the selected voice ("" if none)."""
        return str(self.cmb_voice.currentData() or "")

    def _on_voice_changed(self, _i: int = 0) -> None:
        """A different voice was selected."""
        self._render_voice_info()
        self._refresh_buttons()

    def _render_voice_info(self) -> None:
        """Licence badge and one-line info for the selected voice."""
        while self.badge_box.count():
            it = self.badge_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        rec = self.library.get(self.selected_voice_id()) if self.selected_voice_id() else None
        if rec is None:
            self.lbl_voice_info.setText("")
            return
        self.badge_box.addWidget(make_badge(rec.license, rec.commercial_use, str(rec.info.get("license_url", ""))))
        lang = rec.language.capitalize() if rec.language else ""
        self.lbl_voice_info.setText(" \u00b7 ".join(p for p in (lang, rec.info.get("author", "")) if p)
                                    + ("\n" + tr("narr.voice_personal_note") if not rec.commercial_use else ""))

    # ------------------------------------------------------------------ output
    def choose_folder(self) -> None:
        """Pick the parent folder of the audiobook."""
        path = self._pick_folder() if self._pick_folder else QFileDialog.getExistingDirectory(
            self, tr("narr.choose_folder"), str(self.out_dir))
        if path:
            self.out_dir = Path(path)
            self.lbl_out.setText(str(self.out_dir))

    def selected_formats(self) -> Set[str]:
        """Formats checked in the UI (M4B only counts while AAC is allowed)."""
        out = {f for f, chk in self.format_checks.items() if chk.isChecked()}
        if not self.aac_allowed:
            out.discard(ex.FORMAT_M4B)
        return out

    def options(self) -> nr.NarrationOptions:
        """Build the narration options from the controls."""
        return nr.NarrationOptions(
            formats=self.selected_formats(),
            bitrates=ex.Bitrates(self.spn_aac.value(), self.spn_mp3.value(), self.spn_opus.value()),
            speak_titles=self.chk_titles.isChecked(), allow_aac=self.aac_allowed)

    # ------------------------------------------------------------------ state
    @property
    def busy(self) -> bool:
        """True while a narration job runs (also while paused)."""
        return bool(self.worker and self.worker.isRunning())

    def _paused(self) -> bool:
        """True if the running job is paused."""
        return bool(self.worker and self.worker.pause_token.paused)

    def _refresh_buttons(self) -> None:
        """Enable/disable controls from the current state."""
        busy = self.busy
        ready = bool(self.book and self.selected_voice_id() and self.selected_formats()) and not busy
        self.btn_start.setEnabled(ready)
        self.btn_book.setEnabled(not busy)
        self.btn_out.setEnabled(not busy)
        self.cmb_voice.setEnabled(not busy)
        self.btn_pause.setVisible(busy)
        self.btn_cancel.setVisible(busy)

    # ------------------------------------------------------------------ run
    def start(self) -> bool:
        """Start (or resume) the narration of the loaded book; returns False if something is missing."""
        rec = self.library.get(self.selected_voice_id()) if self.selected_voice_id() else None
        if not (self.book and rec and self.selected_formats()) or self.busy:
            return False
        job = NarrationJob(self.book, rec, self.out_dir, self.options())
        self.result = None
        self.lbl_ready.hide()
        self.btn_open.hide()
        self.progress.setValue(0)
        self.lbl_status.setText(tr("narr.starting"))
        w = NarrateWorker(job, self.runner, parent=self)
        w.progress.connect(self.on_progress)
        w.done.connect(self.on_done)
        w.failed.connect(self.on_failed)
        w.cancelled.connect(self.on_cancelled)
        w.finished.connect(self._refresh_buttons)
        self.worker = w
        w.start()
        self._refresh_buttons()
        return True

    def toggle_pause(self) -> None:
        """Pause / resume the running job."""
        if not self.worker:
            return
        if self._paused():
            self.worker.resume()
            self.lbl_status.setText(tr("narr.resumed"))
        else:
            self.worker.pause()
            self.lbl_status.setText(tr("narr.paused"))
        self.btn_pause.setText(tr("narr.resume") if self._paused() else tr("narr.pause"))

    def cancel(self) -> None:
        """Stop the job (finished parts stay cached)."""
        if self.worker:
            self.lbl_status.setText(tr("ui.stopping"))
            self.worker.cancel()

    def on_progress(self, p: nr.NarrationProgress) -> None:
        """Worker signal: bar, chunk counter and ETA."""
        self.progress.setValue(int(p.fraction * 100))
        eta = format_eta(p.eta)
        text = p.message
        if p.phase == "synth" and eta:
            text = f"{p.message}  \u00b7  {tr('narr.eta', eta=eta)}"
        self.lbl_status.setText(text)

    def on_done(self, result: nr.NarrationResult) -> None:
        """Worker signal: show the result."""
        self.result = result
        self.progress.setValue(100)
        self.lbl_status.setText(tr("narr.done_summary", chapters=result.chapters, files=len(result.files)))
        self.lbl_ready.show()
        self.btn_open.show()
        self._refresh_buttons()
        if self.auto_open_folder:
            open_folder(result.out_dir)

    def on_cancelled(self) -> None:
        """Worker signal: cancelled."""
        self.lbl_status.setText(tr("narr.cancelled"))

    def on_failed(self, kind: str, message: str, details: str) -> None:
        """Worker signal: failed - the message is shown; finished chunks are still cached."""
        self.progress.setValue(0)
        self.lbl_status.setText(message + "  " + tr("narr.failed_resume"))
        log.error("narration failed: kind=%s %s | %s", kind, message, details)

    def open_result(self) -> None:
        """Open the folder with the audiobook."""
        if self.result:
            open_folder(self.result.out_dir)

    def shutdown(self) -> None:
        """Cancel a running job and wait for the thread (application exit)."""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(8000)

    def closeEvent(self, e) -> None:  # noqa: N802
        """Stop the worker before closing."""
        self.shutdown()
        super().closeEvent(e)
