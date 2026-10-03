""""Narrate a book": choose a book (TXT / FB2 / EPUB), a voice and an output format, press Start.

The window only collects choices and shows progress; the work happens in :class:`workers.narrate_worker.NarrateWorker`
(chunk-by-chunk synthesis with the voice's LoRA adapter, resumable, see :mod:`core.narration`).  Pause/Resume and Cancel
are available while it runs; finished chunks are cached on disk, so Start after a cancel or a crash continues where it
stopped.

Output formats (see :mod:`core.audiobook_export`): the default is one Ogg Opus file with chapter markers; per-chapter MP3
is the most compatible alternative; M4B (AAC, for Apple Books) is a separate, clearly marked opt-in with a legal note and
can be hidden completely (:mod:`infra.features`).  The quality is chosen with three preset buttons (Compact / Standard /
High, see :data:`core.audiobook_export.QUALITY_PRESETS`); exact bitrates, the output folder, the chapter-title option and a
sample of the prepared text sit in the collapsed "Advanced" section, the rarely needed formats in "Other formats".

"Prepare the text" is fully automatic (no editor, no review): the rule-based steps of :mod:`core.text_prep` are check
boxes (all on by default), the optional AI clean-up of :mod:`core.text_cleanup` needs a one-time model download (button in
the row, :class:`workers.narrate_worker.TextModelDownloadWorker`) and works for Russian books only.  Steps that do not exist
yet (punctuation model, stress marks, translation, speaker roles) are shown greyed out in a collapsed "coming later" list.
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
from core import text_prep
from core.book_parsers import SUPPORTED_EXTENSIONS, Book, load_book
from core.errors import DatasetMakerError
from core import voice_info
from core.i18n import tr
from core.voice_library import VoiceLibrary
from infra import features, text_models
from infra import voice_catalog as catalog
from infra import voice_repository as repo
from ui.main_window import open_folder
from ui.mini_player import MiniPlayer
from ui.voices_window import make_badge, make_scope_badge
from ui.window_base import SubWindow, card_frame, fit_to_screen, hint_label
from workers.narrate_worker import NarrateWorker, RepoDownloadWorker, RepoIndexWorker, TextModelDownloadWorker
from workers.narration_runner import NarrationJob, default_output_dir, format_eta, run_narration

log = logging.getLogger("voxprint.ui.narrate")

#: Formats shown directly (in this order); the M4B entry is separated and carries the legal note.
MAIN_FORMATS: Tuple[str, ...] = (ex.FORMAT_OPUS_SINGLE, ex.FORMAT_MP3_CHAPTERS)
OTHER_FORMATS: Tuple[str, ...] = (ex.FORMAT_M4B_OPUS, ex.FORMAT_OPUS_CHAPTERS, ex.FORMAT_MP3_SINGLE,
                                  ex.FORMAT_FLAC_CHAPTERS, ex.FORMAT_WAV_CHAPTERS)
CHARS_PER_SECOND = 14.0         # rough speaking rate used for the "about N hours" estimate
#: Rule-based preparation steps shown as check boxes (all on by default), in display order.
RULE_STEPS: Tuple[str, ...] = (text_prep.STEP_LAYOUT, text_prep.STEP_NOISE, text_prep.STEP_QUOTES, text_prep.STEP_LINKS,
                               text_prep.STEP_HEADINGS, text_prep.STEP_NUMBERS, text_prep.STEP_ABBREV)
#: Steps that are only announced (no code yet): greyed out in the collapsed "coming later" list.
LATER_STEPS: Tuple[str, ...] = (text_models.STEP_PUNCT, text_models.STEP_STRESS, text_models.STEP_TRANSLATE,
                                text_models.STEP_ROLES)
SPELLFIX_MODEL = "sage-ru"      # registry key of the model behind the "fix typos" step
SAMPLE_CHARS = 420              # length of the prepared-text sample


def prep_texts() -> Dict[str, Tuple[str, str]]:
    """``{step: (name, one-line description)}`` of the preparation steps in the current language."""
    return {
        text_prep.STEP_LAYOUT: (tr("prep.layout"), tr("prep.layout_d")),
        text_prep.STEP_NOISE: (tr("prep.noise"), tr("prep.noise_d")),
        text_prep.STEP_QUOTES: (tr("prep.quotes"), tr("prep.quotes_d")),
        text_prep.STEP_LINKS: (tr("prep.links"), tr("prep.links_d")),
        text_prep.STEP_HEADINGS: (tr("prep.headings"), tr("prep.headings_d")),
        text_prep.STEP_NUMBERS: (tr("prep.numbers"), tr("prep.numbers_d")),
        text_prep.STEP_ABBREV: (tr("prep.abbrev"), tr("prep.abbrev_d")),
        text_models.STEP_SPELLFIX: (tr("prep.spellfix"), tr("prep.spellfix_d")),
    }


def later_texts() -> Dict[str, str]:
    """``{step: name}`` of the announced (not yet available) steps."""
    return {
        text_models.STEP_PUNCT: tr("prep.later_punct"),
        text_models.STEP_STRESS: tr("prep.later_stress"),
        text_models.STEP_TRANSLATE: tr("prep.later_translate"),
        text_models.STEP_ROLES: tr("prep.later_roles"),
    }


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
                 auto_open_folder: bool = True, model_state: Callable[[Any], str] = text_models.state,
                 model_ensure: Callable[..., Any] = text_models.ensure,
                 plan_builder: Callable[..., Any] = text_models.build_plan,
                 fetch: Callable[..., Any] = repo.fetch_index, download: Callable[..., Any] = repo.download_voice,
                 auto_refresh: bool = True, player_backend: Any = None) -> None:
        """Build the window.  ``runner``, the file pickers and the text-model hooks (``model_state(model)``,
        ``model_ensure(model, progress)``, ``plan_builder(rule_steps, neural_steps)``) are injectable (tests);
        ``aac_allowed`` overrides the feature flag."""
        super().__init__(with_back=True)
        self.library = library or VoiceLibrary()
        self._fetch, self._download = fetch, download
        self._player_backend = player_backend
        self.entries: list = repo.load_cache()          # online voices not installed yet (cached index; refreshed in the background)
        self._index_worker = None
        self._dl_worker = None
        self._auto_refresh = auto_refresh
        self.runner = runner
        self._pick_book, self._pick_folder = pick_book, pick_folder
        self.out_dir: Path = Path(out_dir) if out_dir else default_output_dir()
        self.aac_allowed = features.aac_enabled() if aac_allowed is None else aac_allowed
        self.auto_open_folder = auto_open_folder
        self.model_state, self.model_ensure, self.plan_builder = model_state, model_ensure, plan_builder
        self.book: Optional[Book] = None
        self.book_path: Optional[Path] = None
        self.worker: Optional[NarrateWorker] = None
        self.result: Optional[nr.NarrationResult] = None
        self.format_checks: Dict[str, QCheckBox] = {}
        self.format_desc: Dict[str, QLabel] = {}
        self.format_names: Dict[str, QLabel] = {}
        self.prep_checks: Dict[str, QCheckBox] = {}
        self.prep_desc: Dict[str, QLabel] = {}
        self.later_checks: Dict[str, QCheckBox] = {}
        self.preset_buttons: Dict[str, QPushButton] = {}
        self.model_worker: Optional[TextModelDownloadWorker] = None
        self._model_msg = ""                    # last download message ("" = show the state text)
        self._spell_wanted = True               # the user's wish for the AI step (default on once it is available)
        self._build()
        self.retranslate()
        self._sync_preset()
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

    def _prep_row(self, key: str, parent_layout: QVBoxLayout, checked: bool) -> QCheckBox:
        """A preparation check box with its one-line description below."""
        chk = QCheckBox()
        chk.setChecked(checked)
        desc = hint_label()
        desc.setContentsMargins(26, 0, 0, 0)
        parent_layout.addWidget(chk)
        parent_layout.addWidget(desc)
        self.prep_checks[key], self.prep_desc[key] = chk, desc
        chk.toggled.connect(lambda c, k=key: self._on_prep_toggled(k, c))
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

        # --- prepare the text (automatic: rules, then the optional AI clean-up) ---
        c = card_frame()
        v = QVBoxLayout(c)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(6)
        self.lbl_prep_title = QLabel()
        self.lbl_prep_title.setObjectName("sectiontitle")
        v.addWidget(self.lbl_prep_title)
        self.lbl_prep_hint = hint_label()
        v.addWidget(self.lbl_prep_hint)
        for key in RULE_STEPS:
            self._prep_row(key, v, checked=True)
        self._prep_row(text_models.STEP_SPELLFIX, v, checked=False)
        mrow = QHBoxLayout()
        mrow.setContentsMargins(26, 0, 0, 0)
        self.lbl_model_state = QLabel()
        self.lbl_model_state.setObjectName("cardnote")
        self.btn_model_download = QPushButton()
        mrow.addWidget(self.lbl_model_state, 1)
        mrow.addWidget(self.btn_model_download)
        v.addLayout(mrow)
        self.btn_more_prep = QToolButton()
        self.btn_more_prep.setObjectName("expander")
        self.btn_more_prep.setCheckable(True)
        v.addWidget(self.btn_more_prep)
        self.more_prep_box = QWidget()
        mv = QVBoxLayout(self.more_prep_box)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(4)
        for key in LATER_STEPS:
            chk = QCheckBox()
            chk.setEnabled(False)
            mv.addWidget(chk)
            self.later_checks[key] = chk
        self.lbl_later_note = hint_label()
        mv.addWidget(self.lbl_later_note)
        self.more_prep_box.setVisible(False)
        v.addWidget(self.more_prep_box)
        self.btn_more_prep.toggled.connect(self._on_more_prep_toggled)
        self.body.addWidget(c)

        # --- output format and quality ---
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
        # quality presets
        self.lbl_quality = QLabel()
        self.lbl_quality.setObjectName("sectiontitle")
        v.addWidget(self.lbl_quality)
        prow = QHBoxLayout()
        for name in ex.QUALITY_PRESETS:
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setObjectName("preset")
            btn.clicked.connect(lambda _c=False, n=name: self.apply_preset(n))
            prow.addWidget(btn)
            self.preset_buttons[name] = btn
        prow.addStretch(1)
        v.addLayout(prow)
        self.lbl_preset_info = hint_label()
        v.addWidget(self.lbl_preset_info)
        # other formats (collapsed)
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
        self.other_box.setVisible(False)
        v.addWidget(self.other_box)
        self.btn_other.toggled.connect(self._on_other_toggled)
        # advanced (collapsed): exact bitrates, output folder, chapter titles, text sample
        self.btn_advanced = QToolButton()
        self.btn_advanced.setObjectName("expander")
        self.btn_advanced.setCheckable(True)
        v.addWidget(self.btn_advanced)
        self.advanced_box = QWidget()
        dv = QVBoxLayout(self.advanced_box)
        dv.setContentsMargins(0, 0, 0, 0)
        dv.setSpacing(6)
        brow = QHBoxLayout()
        self.spn_opus, self.spn_mp3, self.spn_aac = QSpinBox(), QSpinBox(), QSpinBox()
        std = ex.QUALITY_PRESETS[ex.DEFAULT_PRESET]
        for spn, lo, hi, val in ((self.spn_opus, 12, 128, std.opus_kbps), (self.spn_mp3, 32, 320, std.mp3_kbps),
                                 (self.spn_aac, 24, 256, std.aac_kbps)):
            spn.setRange(lo, hi)
            spn.setValue(val)
            spn.setSuffix(" kbit/s")
            spn.valueChanged.connect(self._on_rate_changed)
        self.lbl_rate_opus, self.lbl_rate_mp3, self.lbl_rate_aac = QLabel(), QLabel(), QLabel()
        for lbl, spn in ((self.lbl_rate_opus, self.spn_opus), (self.lbl_rate_mp3, self.spn_mp3),
                         (self.lbl_rate_aac, self.spn_aac)):
            brow.addWidget(lbl)
            brow.addWidget(spn)
        brow.addStretch(1)
        self.rates_row = brow
        dv.addLayout(brow)
        self.lbl_rates_hint = hint_label()
        dv.addWidget(self.lbl_rates_hint)
        frow = QHBoxLayout()
        self.btn_out = QPushButton()
        self.lbl_out = QLabel()
        self.lbl_out.setObjectName("fileLabel")
        frow.addWidget(self.btn_out)
        frow.addWidget(self.lbl_out, 1)
        dv.addLayout(frow)
        self.chk_titles = QCheckBox()
        self.chk_titles.setChecked(True)
        dv.addWidget(self.chk_titles)
        self.lbl_sample_title = QLabel()
        self.lbl_sample_title.setObjectName("sectiontitle")
        self.lbl_sample = hint_label()
        self.lbl_sample.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        dv.addWidget(self.lbl_sample_title)
        dv.addWidget(self.lbl_sample)
        self.advanced_box.setVisible(False)
        v.addWidget(self.advanced_box)
        self.btn_advanced.toggled.connect(self._on_advanced_toggled)
        self.body.addWidget(c)

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
        self.player = MiniPlayer(backend=self._player_backend)          # live listening of the finished parts
        self.player.hide()
        self.body.addWidget(self.player)
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
        self.btn_model_download.clicked.connect(self.download_model)
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
        if hasattr(self, "player"):
            self.player.retranslate()
            self.player.setToolTip(tr("narr.player_hint"))
        self.lbl_intro.setText(tr("narr.intro"))
        self.lbl_book_title.setText(tr("narr.book"))
        self.btn_book.setText(tr("narr.choose_book"))
        self.lbl_book.setText(self.book_path.name if self.book_path else tr("narr.no_book"))
        self._render_book_info()
        self.lbl_voice_title.setText(tr("narr.voice"))
        self.lbl_no_voice.setText(tr("narr.no_voice"))
        self.btn_to_train.setText(tr("studio.train_title"))
        self.btn_to_voices.setText(tr("studio.voices_title"))
        self.lbl_prep_title.setText(tr("narr.prep_title"))
        self.lbl_prep_hint.setText(tr("narr.prep_hint"))
        for key, (name, desc) in prep_texts().items():
            self.prep_checks[key].setText(name)
            self.prep_desc[key].setText(desc)
        self.btn_model_download.setText(tr("prep.model_download"))
        later = later_texts()
        for key, chk in self.later_checks.items():
            chk.setText(f"{later[key]}  \u00b7  {tr('prep.later_tag')}")
        self.lbl_later_note.setText(tr("prep.later_note"))
        self._on_more_prep_toggled(self.btn_more_prep.isChecked())
        self._refresh_model_row()
        self.lbl_format_title.setText(tr("narr.format"))
        self.lbl_quality.setText(tr("narr.quality"))
        self.preset_buttons["compact"].setText(tr("narr.preset_compact"))
        self.preset_buttons["standard"].setText(tr("narr.preset_standard"))
        self.preset_buttons["high"].setText(tr("narr.preset_high"))
        self._sync_preset()
        self._on_advanced_toggled(self.btn_advanced.isChecked())
        self.lbl_sample_title.setText(tr("narr.sample"))
        self._update_sample()
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
        """Expand / collapse "Other formats"."""
        self.other_box.setVisible(open_)
        self.btn_other.setText(("\u25be " if open_ else "\u25b8 ") + tr("narr.other_formats"))

    def _on_advanced_toggled(self, open_: bool) -> None:
        """Expand / collapse "Advanced" (bitrates, output folder, chapter titles, text sample)."""
        self.advanced_box.setVisible(open_)
        self.btn_advanced.setText(("\u25be " if open_ else "\u25b8 ") + tr("narr.advanced"))
        if open_:
            self._update_sample()

    def _on_more_prep_toggled(self, open_: bool) -> None:
        """Expand / collapse the greyed-out "coming later" steps."""
        self.more_prep_box.setVisible(open_)
        self.btn_more_prep.setText(("\u25be " if open_ else "\u25b8 ") + tr("prep.more"))

    # ------------------------------------------------------------------ quality presets
    def apply_preset(self, name: str) -> None:
        """Set the three bitrates from a preset (Compact / Standard / High)."""
        b = ex.QUALITY_PRESETS[name]
        for spn, val in ((self.spn_opus, b.opus_kbps), (self.spn_mp3, b.mp3_kbps), (self.spn_aac, b.aac_kbps)):
            spn.blockSignals(True)
            spn.setValue(val)
            spn.blockSignals(False)
        self._sync_preset()

    def current_preset(self) -> str:
        """Name of the preset equal to the spin boxes, ``""`` for custom values."""
        return ex.preset_for(self.bitrates())

    def bitrates(self) -> ex.Bitrates:
        """Bitrates of the spin boxes in the Advanced section."""
        return ex.Bitrates(self.spn_aac.value(), self.spn_mp3.value(), self.spn_opus.value())

    def _on_rate_changed(self, _v: int = 0) -> None:
        """A bitrate was edited by hand: the matching preset (if any) is highlighted, otherwise none."""
        self._sync_preset()

    def _sync_preset(self) -> None:
        """Highlight the matching preset button, update the one-line description and the size estimate."""
        name = self.current_preset()
        for key, btn in self.preset_buttons.items():
            btn.setChecked(key == name)
        text = {"compact": tr("narr.preset_compact_d"), "standard": tr("narr.preset_standard_d"),
                "high": tr("narr.preset_high_d"), "": tr("narr.preset_custom_d")}[name]
        b = self.bitrates()
        chosen = self.selected_formats()
        parts = []
        if chosen & {ex.FORMAT_OPUS_SINGLE, ex.FORMAT_OPUS_CHAPTERS, ex.FORMAT_M4B_OPUS} or not chosen:
            parts.append("Opus " + tr("narr.mb", n=round(ex.megabytes_per_hour(b.opus_kbps))))
        if chosen & {ex.FORMAT_MP3_CHAPTERS, ex.FORMAT_MP3_SINGLE}:
            parts.append("MP3 " + tr("narr.mb", n=round(ex.megabytes_per_hour(b.mp3_kbps))))
        if ex.FORMAT_M4B in chosen:
            parts.append("AAC " + tr("narr.mb", n=round(ex.megabytes_per_hour(b.aac_kbps))))
        self.lbl_preset_info.setText(text + " " + tr("narr.size_hint", sizes=" \u00b7 ".join(parts)))

    # ------------------------------------------------------------------ text preparation
    def selected_rule_steps(self) -> Set[str]:
        """Rule-based preparation steps that are checked."""
        return {k for k in RULE_STEPS if self.prep_checks[k].isChecked()}

    def selected_neural_steps(self) -> Set[str]:
        """Neural steps that are checked and usable (the clean-up model must be downloaded)."""
        chk = self.prep_checks[text_models.STEP_SPELLFIX]
        return {text_models.STEP_SPELLFIX} if chk.isChecked() and chk.isEnabled() else set()

    def book_language(self) -> str:
        """Language code of the loaded book (``""`` = unknown / no book)."""
        return text_prep.resolve_language(self.book) if self.book else ""

    def _on_prep_toggled(self, key: str, checked: bool) -> None:
        """A preparation check box was clicked."""
        if key == text_models.STEP_SPELLFIX and self.prep_checks[key].isEnabled():
            self._spell_wanted = checked
        self._update_sample()
        self._refresh_buttons()

    def _refresh_model_row(self) -> None:
        """State of the AI clean-up row: ready / needs a download / downloading / wrong language."""
        model = text_models.get(SPELLFIX_MODEL)
        lang = self.book_language()
        lang_ok = not lang or lang in model.languages
        ready = self.model_state(model) == text_models.STATE_READY
        downloading = bool(self.model_worker and self.model_worker.isRunning())
        chk = self.prep_checks[text_models.STEP_SPELLFIX]
        usable = ready and lang_ok and not self.busy
        chk.blockSignals(True)
        chk.setEnabled(usable)
        chk.setChecked(usable and self._spell_wanted)
        chk.blockSignals(False)
        if not lang_ok:
            msg = tr("prep.model_lang")
        elif self._model_msg:
            msg = self._model_msg
        elif ready:
            msg = tr("prep.model_ready")
        else:
            msg = tr("prep.model_needs", size=model.size_mb)
        self.lbl_model_state.setText(msg)
        self.btn_model_download.setVisible(not ready and lang_ok)
        self.btn_model_download.setEnabled(not downloading and not self.busy)

    def download_model(self) -> bool:
        """Download the clean-up model in the background; returns False if nothing was started."""
        if self.model_worker and self.model_worker.isRunning():
            return False
        model = text_models.get(SPELLFIX_MODEL)
        w = TextModelDownloadWorker(model, self.model_ensure, parent=self)
        w.progress.connect(self._on_model_progress)
        w.done.connect(self._on_model_done)
        w.failed.connect(self._on_model_failed)
        self.model_worker = w
        self._model_msg = tr("prep.model_downloading", pct=0)
        self._refresh_model_row()
        w.start()
        self._refresh_model_row()
        return True

    def _on_model_progress(self, fraction: float) -> None:
        """Download progress of the clean-up model."""
        self._model_msg = tr("prep.model_downloading", pct=int(fraction * 100))
        self.lbl_model_state.setText(self._model_msg)

    def _on_model_done(self, _key: str) -> None:
        """The model is on disk: the step becomes available (and is switched on by default)."""
        self._model_msg = ""
        self._spell_wanted = True
        self._refresh_model_row()
        self._refresh_buttons()

    def _on_model_failed(self, message: str) -> None:
        """The download failed: the message is shown, the button stays for a retry."""
        self._model_msg = tr("prep.model_failed", error=message)
        self._refresh_model_row()

    def sample_text(self) -> str:
        """The prepared version of the first paragraph the rules change (else of the first paragraph)."""
        if not self.book:
            return ""
        lang = self.book_language()
        opts = text_prep.PrepOptions(frozenset(self.selected_rule_steps()))
        paras = [p.strip() for ch in self.book.chapters[:3] for p in ch.text.split("\n\n") if len(p.strip()) >= 30][:60]
        if not paras:
            return ""
        first = ""
        for para in paras:
            out = text_prep.prepare_text_block(para, lang, opts) if opts.steps else para
            if out != para:
                first = out
                break
        if not first:
            first = (text_prep.prepare_text_block(paras[0], lang, opts) if opts.steps else paras[0]) + "  " + tr("narr.sample_same")
        return first if len(first) <= SAMPLE_CHARS else first[:SAMPLE_CHARS].rsplit(" ", 1)[0] + "\u2026"

    def _update_sample(self) -> None:
        """Refresh the read-only sample in the Advanced section (only while it is visible, it is cheap but pointless otherwise)."""
        if not self.book:
            self.lbl_sample.setText(tr("narr.sample_none"))
        elif self.advanced_box.isVisible() or self.btn_advanced.isChecked():
            self.lbl_sample.setText(self.sample_text())

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
        self._refresh_model_row()
        self._update_sample()
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
        for it in catalog.build(self.library, self.entries):
            if it.installed:
                self.cmb_voice.addItem(it.name, it.key)
            else:
                self.cmb_voice.addItem(tr("narr.voice_remote_item", name=it.name, size=catalog.size_text(it.size_bytes)), it.key)
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
        key = self.selected_voice_id()
        if catalog.is_remote_key(key):
            self._render_remote_info(key)
            return
        rec = self.library.get(key) if key else None
        if rec is None:
            self.lbl_voice_info.setText("")
            return
        self.badge_box.addWidget(make_badge(rec.license, rec.commercial_use, str(rec.info.get("license_url", ""))))
        self.badge_box.addWidget(make_scope_badge(rec.scope))
        lang = rec.language.capitalize() if rec.language else ""
        self.lbl_voice_info.setText(" \u00b7 ".join(p for p in (lang, rec.info.get("author", "")) if p)
                                    + self._scope_note(rec))

    def _render_remote_info(self, key: str) -> None:
        """Badges and note of an online voice that is downloaded on first use."""
        e = catalog.find_entry(self.entries, key)
        if e is None:
            self.lbl_voice_info.setText("")
            return
        self.badge_box.addWidget(make_badge(e.license, e.commercial_use, e.license_url))
        self.badge_box.addWidget(make_scope_badge(catalog.scope_for_license(e.license)))
        lang = e.language.capitalize() if e.language else ""
        note = "\n" + tr("narr.voice_test_only_note") if e.license == voice_info.LICENSE_TEST_ONLY else ""
        self.lbl_voice_info.setText(" \u00b7 ".join(p for p in (lang, e.author) if p) + "\n" + tr("narr.voice_remote_note") + note)

    @staticmethod
    def _scope_note(rec) -> str:
        """Reminder under the voice: what the voice owner allowed (nothing for a commercial scope)."""
        if rec.test_only:
            return "\n" + tr("narr.voice_test_only_note")
        if rec.scope == "private_only":
            return "\n" + tr("narr.voice_personal_note")
        if rec.scope == "public_noncommercial":
            return "\n" + tr("narr.voice_noncommercial_note")
        return ""

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
            formats=self.selected_formats(), bitrates=self.bitrates(),
            speak_titles=self.chk_titles.isChecked(), allow_aac=self.aac_allowed,
            prep=self.plan_builder(self.selected_rule_steps(), self.selected_neural_steps()))

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
        for chk in self.prep_checks.values():
            if chk is not self.prep_checks[text_models.STEP_SPELLFIX]:
                chk.setEnabled(not busy)
        self._refresh_model_row()
        self._sync_preset()
        self.cmb_voice.setEnabled(not busy)
        self.btn_pause.setVisible(busy)
        self.btn_cancel.setVisible(busy)

    # ------------------------------------------------------------------ run
    def start(self) -> bool:
        """Start (or resume) the narration of the loaded book; returns False if something is missing."""
        key = self.selected_voice_id()
        if catalog.is_remote_key(key):
            return self._download_then_start(key)
        rec = self.library.get(key) if key else None
        if not (self.book and rec and self.selected_formats()) or self.busy:
            return False
        job = NarrationJob(self.book, rec, self.out_dir, self.options())
        self.result = None
        self.lbl_ready.hide()
        self.btn_open.hide()
        self.progress.setValue(0)
        self.lbl_status.setText(tr("narr.starting"))
        w = NarrateWorker(job, self.runner, parent=self)
        self.player.stop()
        self.player.hide()
        w.plan.connect(self.on_plan)
        w.progress.connect(self.on_progress)
        w.done.connect(self.on_done)
        w.failed.connect(self.on_failed)
        w.cancelled.connect(self.on_cancelled)
        w.finished.connect(self._refresh_buttons)
        self.worker = w
        w.start()
        self._refresh_buttons()
        return True

    # ------------------------------------------------------------------ online voices
    def refresh_remote(self) -> None:
        """Re-read the online index in the background (no-op while the repository is not configured)."""
        if not repo.is_configured() or (self._index_worker is not None and self._index_worker.isRunning()):
            return
        w = RepoIndexWorker(fetch=self._fetch, parent=self)
        w.done.connect(self._on_index)
        self._index_worker = w
        w.start()

    def _on_index(self, res) -> None:
        """New index: update the list, keeping the selection."""
        if res.voices:
            self.entries = list(res.voices)
            self.refresh_voices()

    def showEvent(self, e) -> None:  # noqa: N802
        """Refresh the online index each time the window opens."""
        super().showEvent(e)
        if self._auto_refresh:
            self.refresh_remote()

    def _download_then_start(self, key: str) -> bool:
        """The selected voice is online only: download it (SHA-256 checked), select it and start the narration."""
        entry = catalog.find_entry(self.entries, key)
        if entry is None or not (self.book and self.selected_formats()) or self.busy or (
                self._dl_worker is not None and self._dl_worker.isRunning()):
            return False
        self.lbl_status.setText(tr("narr.voice_downloading", name=entry.display_name, p=0))
        self.btn_start.setEnabled(False)
        w = RepoDownloadWorker([entry], self.library, download=self._download, parent=self)
        w.progress.connect(lambda f, _n: self.lbl_status.setText(tr("narr.voice_downloading", name=entry.display_name, p=int(f * 100))))
        w.failed.connect(lambda m: self.lbl_status.setText(m))
        w.finished_all.connect(self._on_voice_downloaded)
        self._dl_worker = w
        w.start()
        return True

    def _on_voice_downloaded(self, ids: list) -> None:
        """Download finished: select the new voice and start (or just refresh on failure)."""
        if not ids:
            self.refresh_voices()
            return
        self.refresh_voices(select=ids[0])
        self.lbl_status.setText("")
        self.start()

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

    def on_plan(self, paths: list) -> None:
        """The job announced its chunk files: show the player; it follows the parts as they are made."""
        self.player.set_plan([Path(p) for p in paths], live=True)
        self.player.show()

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
        rec = self.library.get(self.selected_voice_id()) if self.selected_voice_id() else None
        reminder = ""      # the voice owner's scope is repeated when the files are ready: private results must stay local
        if rec is not None and rec.scope == "private_only":
            reminder = "\n" + (tr("narr.done_test_only_reminder") if rec.test_only else tr("narr.done_private_reminder"))
        elif rec is not None and rec.scope == "public_noncommercial":
            reminder = "\n" + tr("narr.done_noncommercial_reminder")
        self.lbl_status.setText(tr("narr.done_summary", chapters=result.chapters, files=len(result.files)) + reminder)
        self.lbl_ready.show()
        self.btn_open.show()
        if self.player.isVisibleTo(self) or self.player.queue.planned:
            self.player.set_final(result.files[0] if result.files else None)
            self.player.show()
        self._refresh_buttons()
        if self.auto_open_folder:
            open_folder(result.out_dir)

    def on_cancelled(self) -> None:
        """Worker signal: cancelled."""
        self.lbl_status.setText(tr("narr.cancelled"))
        self.player.set_final(None)

    def on_failed(self, kind: str, message: str, details: str) -> None:
        """Worker signal: failed - the message is shown; finished chunks are still cached."""
        self.progress.setValue(0)
        self.player.set_final(None)
        self.lbl_status.setText(message + "  " + tr("narr.failed_resume"))
        log.error("narration failed: kind=%s %s | %s", kind, message, details)

    def open_result(self) -> None:
        """Open the folder with the audiobook."""
        if self.result:
            open_folder(self.result.out_dir)

    def shutdown(self) -> None:
        """Cancel a running job and wait for the thread (application exit)."""
        self.player.shutdown()
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(8000)
        if self.model_worker and self.model_worker.isRunning():
            self.model_worker.wait(8000)
        for w in (self._dl_worker, self._index_worker):
            if w is not None and w.isRunning():
                w.wait(3000)

    def closeEvent(self, e) -> None:  # noqa: N802
        """Stop the worker before closing."""
        self.shutdown()
        super().closeEvent(e)
