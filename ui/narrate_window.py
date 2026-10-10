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

"Prepare the text" is one switch, on by default: the rule-based steps of :mod:`core.text_prep`, plus the Russian typo
model of :mod:`core.text_cleanup` when that model is already downloaded.  Translation stays its own card and is off
until the user turns it on.  The AI text model (Gemma) is a separate card: literary translation, a narration rewrite,
and speaker marks.  Those boxes are clickable when the model is downloaded and look disabled when it is not.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QApplication, QSlider, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from core import audiobook_export as ex
from core import narration as nr
from core import pauses as pz
from core import ordinals
from core import pace as pc
from core import ai_disclosure
from core import text_prep
from core import workspace as ws
from core.book_parsers import SUPPORTED_EXTENSIONS, Book, load_book
from core.errors import DatasetMakerError
from core import num_words as nw
from core import translate as tl
from core import soundscape as soundscape_mod
from core import speakers as spk
from core import voice_info
from core.i18n import tr
from core.languages import language_name
from core.voice_library import VoiceLibrary
from infra import bundled_voices, features, llm_tool, soundscape_model, text_models
from infra import voice_catalog as catalog
from infra import voice_repository as repo
from ui.main_window import mark_recommended, open_folder, recommended_text
from ui import job_dialogs
from ui.mini_player import MiniPlayer
from ui.speaker_dialog import SpeakerDialog
from ui.voices_window import make_badge, make_scope_badge
from ui.suite_icons import apply_button
from ui.window_base import ColumnFlow, SubWindow, card_frame, fit_to_screen, hint_label
from workers.narrate_worker import NarrateWorker, RepoDownloadWorker, RepoIndexWorker, TextModelDownloadWorker, TextModelsDownloadWorker
from workers.narration_runner import NarrationJob, default_output_dir, format_eta, run_narration

log = logging.getLogger("voxprint.ui.narrate")

#: Formats shown directly (in this order); the M4B entry is separated and carries the legal note.
MAIN_FORMATS: Tuple[str, ...] = (ex.FORMAT_OPUS_SINGLE, ex.FORMAT_MP3_CHAPTERS)
OTHER_FORMATS: Tuple[str, ...] = (ex.FORMAT_M4B_OPUS, ex.FORMAT_OPUS_CHAPTERS, ex.FORMAT_MP3_SINGLE,
                                  ex.FORMAT_FLAC_CHAPTERS, ex.FORMAT_WAV_CHAPTERS)
CHARS_PER_SECOND = 14.0         # rough speaking rate used for the "about N hours" estimate
#: Rule-based preparation steps behind the one "Prepare text" switch (on by default).
RULE_STEPS: Tuple[str, ...] = (text_prep.STEP_LAYOUT, text_prep.STEP_NOISE, text_prep.STEP_QUOTES, text_prep.STEP_LINKS,
                               text_prep.STEP_HEADINGS, text_prep.STEP_NUMBERS, text_prep.STEP_ABBREV, text_prep.STEP_YO)
SPELLFIX_MODEL = "sage-ru"      # registry key of the Russian typo model included in "Prepare text" when it is downloaded
SAMPLE_CHARS = 420              # length of the prepared-text sample


def language_names() -> Dict[str, str]:
    """``{code: name}`` of the translation languages in the UI language."""
    return {"en": tr("narr.translate_lang_en"), "ru": tr("narr.translate_lang_ru"), "de": tr("narr.translate_lang_de")}


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
                 translate_plan_builder: Callable[..., Any] = text_models.build_translate_plan,
                 fetch: Callable[..., Any] = repo.fetch_index, download: Callable[..., Any] = repo.download_voice,
                 auto_refresh: bool = True, player_backend: Any = None,
                 ask_place: Optional[Callable[[Path, Path], str]] = None,
                 ask_cleanup: Optional[Callable[[ws.JobFiles], Optional[dict]]] = None,
                 llm_status: Callable[[], str] = llm_tool.status, llm_ensure: Callable[..., Any] = llm_tool.ensure,
                 llm_plan: Callable[[], Any] = llm_tool.make_plan) -> None:
        """Build the window.  ``runner``, the file pickers and the text-model hooks (``model_state(model)``,
        ``model_ensure(model, progress)``, ``plan_builder(rule_steps, neural_steps)``) are injectable (tests);
        ``aac_allowed`` overrides the feature flag; ``ask_place(book, job_dir)`` / ``ask_cleanup(files)`` replace the
        working-folder dialogs (:mod:`ui.job_dialogs`)."""
        super().__init__(with_back=True)
        self.library = library or VoiceLibrary()
        self._fetch, self._download = fetch, download
        self._player_backend = player_backend
        self.entries: list = repo.load_cache()          # online voices not installed yet (cached index; refreshed in the background)
        self._index_worker: Optional[RepoIndexWorker] = None
        self._dl_worker: Optional[RepoDownloadWorker] = None
        self._auto_refresh = auto_refresh
        self.runner = runner
        self._pick_book, self._pick_folder = pick_book, pick_folder
        # the working folder: every job lives in <folder>/<book title>/ (core/workspace.py); remembered between starts
        self.out_dir: Path = Path(out_dir) if out_dir else ws.load_folder(default_output_dir())
        self._ask_place = ask_place or (lambda book, job: job_dialogs.ask_book_place(self, book, job))
        self._ask_cleanup = ask_cleanup or (lambda files: job_dialogs.ask_cleanup(self, files))
        self._place_asked: Set[Path] = set()     # books the user already answered for (no second question on a resume)
        self.aac_allowed = features.aac_enabled() if aac_allowed is None else aac_allowed
        self.auto_open_folder = auto_open_folder
        self.model_state, self.model_ensure, self.plan_builder = model_state, model_ensure, plan_builder
        self.translate_plan_builder = translate_plan_builder
        # the optional AI text model (literary translation / prepare text for narration), see infra/llm_tool.py
        self.llm_status, self.llm_ensure, self.llm_plan = llm_status, llm_ensure, llm_plan
        self.llm_worker: Optional[TextModelsDownloadWorker] = None
        self._llm_msg = ""
        self.tr_worker: Optional[TextModelsDownloadWorker] = None
        self._tr_msg = ""
        self._tr_src_cache: Tuple[int, str] = (0, "")
        self.book: Optional[Book] = None
        self.book_path: Optional[Path] = None
        self.worker: Optional[NarrateWorker] = None
        self.result: Optional[nr.NarrationResult] = None
        self.format_checks: Dict[str, QCheckBox] = {}
        self.format_desc: Dict[str, QLabel] = {}
        self.format_names: Dict[str, QLabel] = {}
        self.preset_buttons: Dict[str, QPushButton] = {}
        self.model_worker: Optional[TextModelDownloadWorker] = None
        self._model_msg = ""                    # last download message ("" = show the state text)
        self.speaker_lines: Optional[List[spk.SpeakerLine]] = None
        self._speaker_preview: Optional[SpeakerDialog] = None
        self._spk_touched = {"male": False, "male2": False, "female": False}
        self._choice_status = True              # the idle / ready line, until a job writes its own status
        self._button_retry = False
        self._build()
        self.retranslate()
        self._sync_preset()
        self.refresh_voices()
        fit_to_screen(self, self.content, 1040, 600)

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

    def wide_width(self) -> int:
        """Window width for two card columns (Studio.navigate widens to it when the screen allows)."""
        return self.flow.two_column_width() + 80        # + body margins, scroll bar and frame

    def _build(self) -> None:
        """Create all widgets (texts come from :meth:`retranslate`)."""
        self.lbl_intro = hint_label()
        self.body.addWidget(self.lbl_intro)
        self.flow = ColumnFlow()                    # the six cards: two columns on wide windows (window_base)
        self.body.addWidget(self.flow)

        # --- book ---
        c = card_frame()
        v = QVBoxLayout(c)
        v.setContentsMargins(14, 10, 14, 10)
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
        self.flow.add(c)

        # --- voice ---
        c = card_frame()
        v = QVBoxLayout(c)
        v.setContentsMargins(14, 10, 14, 10)
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
        self.chk_disclosure = QCheckBox()        # spoken AI disclosure at the start (core/ai_disclosure.py), opt-in
        self.chk_disclosure.setChecked(ai_disclosure.load_enabled())
        self.chk_disclosure.toggled.connect(ai_disclosure.save_enabled)
        v.addWidget(self.chk_disclosure)
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
        self.flow.add(c)

        # --- prepare the text: one switch (rules, and the Russian typo model when it is already downloaded) ---
        c = card_frame()
        v = QVBoxLayout(c)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(6)
        self.lbl_prep_title = QLabel()
        self.lbl_prep_title.setObjectName("sectiontitle")
        v.addWidget(self.lbl_prep_title)
        self.lbl_prep_hint = hint_label()
        v.addWidget(self.lbl_prep_hint)
        self.chk_prepare = QCheckBox()
        self.chk_prepare.setChecked(True)
        mark_recommended(self.chk_prepare)
        v.addWidget(self.chk_prepare)
        self.lbl_prep_desc = hint_label()
        self.lbl_prep_desc.setContentsMargins(26, 0, 0, 0)
        v.addWidget(self.lbl_prep_desc)
        mrow = QHBoxLayout()
        mrow.setContentsMargins(26, 0, 0, 0)
        self.lbl_model_state = QLabel()
        self.lbl_model_state.setObjectName("cardnote")
        self.lbl_model_state.setWordWrap(True)
        self.btn_model_download = QPushButton()
        apply_button(self.btn_model_download, "download")
        mrow.addWidget(self.lbl_model_state, 1)
        mrow.addWidget(self.btn_model_download)
        v.addLayout(mrow)
        self.chk_prepare.toggled.connect(self._on_prepare_toggled)
        self.flow.add(c)

        # --- translate the book (optional) ---
        c = card_frame()
        v = QVBoxLayout(c)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(6)
        self.lbl_tr_title = QLabel()
        self.lbl_tr_title.setObjectName("sectiontitle")
        v.addWidget(self.lbl_tr_title)
        trow = QHBoxLayout()
        self.chk_translate = QCheckBox()
        self.cmb_translate = QComboBox()
        for code in tl.LANGUAGES:
            self.cmb_translate.addItem(code, code)
        trow.addWidget(self.chk_translate)
        trow.addWidget(self.cmb_translate)
        trow.addStretch(1)
        v.addLayout(trow)
        self.tr_box = QWidget()
        tv = QVBoxLayout(self.tr_box)
        tv.setContentsMargins(26, 0, 0, 0)
        tv.setSpacing(4)
        srow = QHBoxLayout()
        self.lbl_tr_state = QLabel()
        self.lbl_tr_state.setObjectName("cardnote")
        self.lbl_tr_state.setWordWrap(True)
        self.btn_tr_download = QPushButton()
        apply_button(self.btn_tr_download, "download")
        srow.addWidget(self.lbl_tr_state, 1)
        srow.addWidget(self.btn_tr_download)
        tv.addLayout(srow)
        self.lbl_tr_voice = hint_label()
        tv.addWidget(self.lbl_tr_voice)
        v.addWidget(self.tr_box)
        self.lbl_tr_note = hint_label()
        v.addWidget(self.lbl_tr_note)
        self.chk_translate.toggled.connect(lambda _c: self._refresh_buttons())
        self.cmb_translate.currentIndexChanged.connect(lambda _i: self._refresh_buttons())
        self.btn_tr_download.clicked.connect(self.download_translate_models)
        self.flow.add(c)

        # --- the optional AI text model: literary translation, prepare text for narration (off by default) ---
        c = card_frame()
        v = QVBoxLayout(c)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(6)
        self.lbl_llm_title = QLabel()
        self.lbl_llm_title.setObjectName("sectiontitle")
        v.addWidget(self.lbl_llm_title)
        lrow = QHBoxLayout()
        self.lbl_llm_state = QLabel()
        self.lbl_llm_state.setObjectName("cardnote")
        self.lbl_llm_state.setWordWrap(True)
        self.btn_llm_download = QPushButton()
        apply_button(self.btn_llm_download, "download")
        lrow.addWidget(self.lbl_llm_state, 1)
        lrow.addWidget(self.btn_llm_download)
        v.addLayout(lrow)
        self.chk_literary = QCheckBox()
        self.chk_llm_prepare = QCheckBox()
        self.chk_speakers = QCheckBox()
        v.addWidget(self.chk_literary)
        v.addWidget(self.chk_llm_prepare)
        v.addWidget(self.chk_speakers)
        self.spk_box = QWidget()
        sv = QVBoxLayout(self.spk_box)
        sv.setContentsMargins(26, 0, 0, 0)
        sv.setSpacing(4)
        self.lbl_spk_narrator = hint_label()
        sv.addWidget(self.lbl_spk_narrator)
        for attr, combo_name in (("lbl_spk_male", "cmb_spk_male"), ("lbl_spk_male2", "cmb_spk_male2"),
                                 ("lbl_spk_female", "cmb_spk_female")):
            row = QHBoxLayout()
            label, combo = QLabel(), QComboBox()
            setattr(self, attr, label)
            setattr(self, combo_name, combo)
            row.addWidget(label)
            row.addWidget(combo, 1)
            sv.addLayout(row)
        self.btn_spk_preview = QPushButton()
        sv.addWidget(self.btn_spk_preview)
        self.spk_box.setVisible(False)
        v.addWidget(self.spk_box)
        prow = QHBoxLayout()
        self.lbl_llm_note = hint_label()
        self.btn_llm_prompts = QPushButton()
        self.btn_llm_prompts.setObjectName("link")
        prow.addWidget(self.lbl_llm_note, 1)
        prow.addWidget(self.btn_llm_prompts)
        v.addLayout(prow)
        self.btn_llm_download.clicked.connect(self.download_llm)
        self.btn_llm_prompts.clicked.connect(self.edit_prompts)
        self.chk_literary.toggled.connect(lambda _c: self._refresh_buttons())
        self.chk_llm_prepare.toggled.connect(lambda _c: self._refresh_buttons())
        self.chk_speakers.toggled.connect(lambda _c: self._refresh_buttons())
        self.cmb_spk_male.currentIndexChanged.connect(lambda _i: self._on_spk_combo("male"))
        self.cmb_spk_male2.currentIndexChanged.connect(lambda _i: self._on_spk_combo("male2"))
        self.cmb_spk_female.currentIndexChanged.connect(lambda _i: self._on_spk_combo("female"))
        self.btn_spk_preview.clicked.connect(self.preview_speakers)
        self.flow.add(c)

        # --- output format and quality ---
        c = card_frame()
        v = QVBoxLayout(c)
        v.setContentsMargins(14, 10, 14, 10)
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
        # per-chunk speech-recognition check + regeneration (core/chunk_check.py): on with the High preset, off otherwise
        self.chk_check_chunks = QCheckBox()
        self.chk_check_chunks.setChecked(ex.DEFAULT_PRESET == "high")
        v.addWidget(self.chk_check_chunks)
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
        # advanced (collapsed): exact bitrates, working folder, chapter titles, text sample
        # pause strength: explicit silence between commas, sentences, paragraphs ... (core/pauses.py)
        self.chk_pauses = QCheckBox()            # opt-in: explicit pauses can make the model swallow short words
        self.chk_pauses.setChecked(pz.load_enabled())
        v.addWidget(self.chk_pauses)
        self.lbl_pauses = QLabel()
        self.sld_pauses = QSlider(Qt.Orientation.Horizontal)
        self.sld_pauses.setRange(0, len(pz.LEVELS) - 1)
        self.sld_pauses.setPageStep(1)
        self.sld_pauses.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sld_pauses.setTickInterval(1)
        self.sld_pauses.setFixedWidth(260)
        self.sld_pauses.setValue(pz.load_level())
        self.lbl_pauses_value = QLabel()
        prow2 = QHBoxLayout()
        prow2.addWidget(self.lbl_pauses)
        prow2.addWidget(self.sld_pauses)
        prow2.addWidget(self.lbl_pauses_value)
        prow2.addStretch(1)
        v.addLayout(prow2)
        self.lbl_pauses_hint = hint_label()
        v.addWidget(self.lbl_pauses_hint)
        self.sld_pauses.valueChanged.connect(self._on_pauses_changed)
        self.chk_pauses.toggled.connect(self._on_pauses_toggled)
        self._on_pauses_toggled(self.chk_pauses.isChecked(), save=False)
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
        frow = QHBoxLayout()                      # working folder: every file of a job goes to <folder>/<book title>/
        self.btn_out = QPushButton()
        self.lbl_out = QLabel()
        self.lbl_out.setObjectName("fileLabel")
        frow.addWidget(self.btn_out)
        frow.addWidget(self.lbl_out, 1)
        dv.addLayout(frow)
        self.lbl_out_hint = hint_label()
        dv.addWidget(self.lbl_out_hint)
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
        self.flow.add(c)

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
        self.chk_prepare.setText(recommended_text(tr("prep.one")))
        self.lbl_prep_desc.setText(tr("prep.one_d"))
        self.btn_model_download.setText(tr("prep.model_download"))
        self._refresh_model_row()
        self.lbl_tr_title.setText(tr("narr.translate_title"))
        self.chk_translate.setText(tr("narr.translate_check"))
        for i, code in enumerate(tl.LANGUAGES):
            self.cmb_translate.setItemText(i, language_names()[code])
        self.btn_tr_download.setText(tr("prep.model_download"))
        self.lbl_tr_note.setText(tr("narr.translate_note"))
        self.lbl_llm_title.setText(tr("llm.title"))
        self.btn_llm_download.setText(tr("prep.model_download"))
        self.chk_literary.setText(tr("llm.literary"))
        self.chk_llm_prepare.setText(tr("llm.prepare"))
        self.chk_speakers.setText(tr("spk.check"))
        self.lbl_spk_male.setText(tr("spk.male"))
        self.lbl_spk_male2.setText(tr("spk.male2"))
        self.cmb_spk_male2.setToolTip(tr("spk.male2_tip"))
        self.lbl_spk_female.setText(tr("spk.female"))
        self.btn_spk_preview.setText(tr("spk.preview"))
        self.lbl_llm_note.setText(tr("llm.note"))
        self.btn_llm_prompts.setText(tr("llm.prompts"))
        self.btn_llm_prompts.setToolTip(tr("llm.prompts_tip"))
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
        self.lbl_out_hint.setText(tr("work.folder_hint"))
        self.chk_titles.setText(tr("narr.speak_titles"))
        self.chk_check_chunks.setText(tr("narr.check_chunks"))
        self.chk_check_chunks.setToolTip(tr("narr.check_chunks_tip"))
        self.chk_pauses.setText(tr("narr.pauses_enable"))
        self.chk_disclosure.setText(tr("narr.ai_disclosure"))
        self.chk_disclosure.setToolTip(tr("narr.ai_disclosure_tip"))
        self.lbl_pauses.setText(tr("narr.pauses"))
        self.lbl_pauses_hint.setText(tr("narr.pauses_hint"))
        self._on_pauses_changed(self.sld_pauses.value(), save=False)
        self.btn_start.setText(tr("narr.start"))
        self.btn_pause.setText(tr("narr.resume") if self._paused() else tr("narr.pause"))
        self.btn_cancel.setText(tr("ui.cancel"))
        self.lbl_ready.setText(tr("ui.ready"))
        self.btn_open.setText(tr("ui.open_folder"))
        self.lbl_footer.setText(tr("narr.footer"))
        self._sync_choice_status()
        self._render_voice_info()
        self._refresh_buttons()

    def _on_pauses_changed(self, level: int, save: bool = True) -> None:
        """Show the name of the chosen pause strength (and the sentence / paragraph lengths); remember the choice."""
        prof = pz.PauseProfile(level)
        names = (tr("narr.pauses_1"), tr("narr.pauses_2"), tr("narr.pauses_3"), tr("narr.pauses_4"), tr("narr.pauses_5"))
        self.lbl_pauses_value.setText(tr("narr.pauses_value", name=names[prof.level], sentence=prof.ms(pz.SENTENCE),
                                         paragraph=prof.ms(pz.PARAGRAPH)))
        if save:
            pz.save_level(prof.level)

    def _on_pauses_toggled(self, on: bool, save: bool = True) -> None:
        """Explicit pauses on/off: the slider only matters when they are on."""
        for w in (self.lbl_pauses, self.sld_pauses, self.lbl_pauses_value, self.lbl_pauses_hint):
            w.setEnabled(on and not getattr(self, "busy", False))
        if save:
            pz.save_enabled(on)

    def pause_profile(self) -> "pz.PauseProfile | None":
        """The pause lengths chosen with the slider, or None (packed chunks, model's own phrasing) when switched off."""
        if not self.chk_pauses.isChecked():
            return None
        return pz.PauseProfile(self.sld_pauses.value(), lengths=pz.load_lengths())

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

    # ------------------------------------------------------------------ quality presets
    def apply_preset(self, name: str) -> None:
        """Set the three bitrates from a preset (Compact / Standard / High)."""
        b = ex.QUALITY_PRESETS[name]
        for spn, val in ((self.spn_opus, b.opus_kbps), (self.spn_mp3, b.mp3_kbps), (self.spn_aac, b.aac_kbps)):
            spn.blockSignals(True)
            spn.setValue(val)
            spn.blockSignals(False)
        self.chk_check_chunks.setChecked(name == "high")   # the "quality" mode checks every chunk; the user may still change it
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
        """Rule-based preparation steps. The one switch turns them all on or all off."""
        steps = set(RULE_STEPS) if self.chk_prepare.isChecked() else set()
        if self.book is not None and self.book.explicit_yo:
            steps.discard(text_prep.STEP_YO)
        return steps

    def _spellfix_applies(self) -> bool:
        """True when Prepare text is on and the Russian typo model is downloaded for this book."""
        if not self.chk_prepare.isChecked():
            return False
        model = text_models.get(SPELLFIX_MODEL)
        lang = self.book_language()
        lang_ok = not lang or lang in model.languages
        return lang_ok and self.model_state(model) == text_models.STATE_READY

    def selected_neural_steps(self) -> Set[str]:
        """The Russian typo step, when Prepare text is on and that model is already downloaded."""
        return {text_models.STEP_SPELLFIX} if self._spellfix_applies() else set()

    def reload_auto_steps(self) -> None:
        """Turn Prepare text on again (the recommended choice)."""
        self.chk_prepare.setChecked(True)
        self._refresh_model_row()
        self._refresh_buttons()

    def book_language(self) -> str:
        """Language code of the loaded book (``""`` = unknown / no book)."""
        return text_prep.resolve_language(self.book) if self.book else ""

    def _on_prepare_toggled(self, _checked: bool) -> None:
        """The Prepare text switch was clicked."""
        self._update_sample()
        self._refresh_buttons()

    def _refresh_model_row(self) -> None:
        """State of the Russian typo model: ready / needs a download / downloading / wrong language."""
        model = text_models.get(SPELLFIX_MODEL)
        lang = self.book_language()
        lang_ok = not lang or lang in model.languages
        ready = self.model_state(model) == text_models.STATE_READY
        downloading = bool(self.model_worker and self.model_worker.isRunning())
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
        self._refresh_model_row()
        self._refresh_buttons()

    def _on_model_failed(self, message: str) -> None:
        """The download failed: the message is shown, the button stays for a retry."""
        self._model_msg = tr("prep.model_failed", error=message)
        self._refresh_model_row()

    # ------------------------------------------------------------------ translation
    def translate_target(self) -> str:
        """Language code chosen in the "Translate to" list."""
        return str(self.cmb_translate.currentData() or "")

    def book_source_language(self) -> str:
        """Detected language of the loaded book (``""`` = unknown / not supported); cached per book."""
        if self.book is None:
            return ""
        if self._tr_src_cache[0] != id(self.book):
            self._tr_src_cache = (id(self.book), tl.detect_book_language(self.book))
        return self._tr_src_cache[1]

    def translate_models_needed(self) -> list:
        """Registry models of the translation hops (``[]`` if not applicable or the pair is unsupported)."""
        src, dst = self.book_source_language(), self.translate_target()
        if not src or not dst or src == dst:
            return []
        try:
            return text_models.translate_models(src, dst)
        except ValueError:
            return []

    def _translate_status(self) -> str:
        """``off`` | ``same`` | ``unsupported`` | ``needs_model`` | ``ready`` for the current selection."""
        if not self.chk_translate.isChecked() or self.book is None:
            return "off"
        src, dst = self.book_source_language(), self.translate_target()
        if src == dst:
            return "same"
        if not self.translate_models_needed():
            return "unsupported"
        if any(self.model_state(m) != text_models.STATE_READY for m in self.translate_models_needed()):
            return "needs_model"
        return "ready"

    def _translate_ok(self) -> bool:
        """False while translation is wanted but cannot run (unsupported source, model not downloaded yet)."""
        return self._translate_status() in ("off", "same", "ready")

    def translate_plan(self):
        """The :class:`core.translate.TranslatePlan` for the job, or ``None`` (off, same language, not possible)."""
        if self._translate_status() != "ready":
            return None
        plan = self.translate_plan_builder(self.translate_target(), self.book_source_language())
        if plan is not None and self.chk_literary.isChecked() and self.chk_literary.isEnabled():
            plan.llm = self.llm_plan()
        return plan

    # ------------------------------------------------------------------ the AI text model
    def _refresh_llm_row(self) -> None:
        """State text, Download button and the text-model options. Each box is clickable when the model is ready."""
        st = self.llm_status()
        downloading = bool(self.llm_worker and self.llm_worker.isRunning())
        msgs = {"ready": tr("llm.ready"), "needs_download": tr("llm.needs", size=round(llm_tool.download_mb() / 1000, 1)),
                "low_vram": tr("llm.low_vram", need=int(llm_tool.MIN_VRAM_GB + 0.5)), "unsupported": tr("llm.unsupported")}
        self.lbl_llm_state.setText(self._llm_msg or msgs.get(st, ""))
        self.btn_llm_download.setVisible(st == "needs_download")
        self.btn_llm_download.setEnabled(not downloading and not self.busy)
        usable = st == "ready" and not self.busy
        for chk in (self.chk_literary, self.chk_llm_prepare, self.chk_speakers):
            chk.setEnabled(usable)
            chk.setToolTip("" if usable else tr("llm.need_model"))
        if usable:
            self.chk_literary.setToolTip(tr("llm.literary_when"))
        self.spk_box.setVisible(self.chk_speakers.isChecked())
        self.spk_box.setEnabled(usable)
        self._fill_speaker_combos()

    def llm_prepare_plan(self):
        """The ``LLMPlan`` for "Prepare text for narration", or ``None`` when off / not usable."""
        if self.chk_llm_prepare.isChecked() and self.chk_llm_prepare.isEnabled():
            return self.llm_plan()
        return None

    def _on_spk_combo(self, which: str) -> None:
        """The user picked a male or female voice (including None). Later refreshes keep that pick."""
        self._spk_touched[which] = True

    def _fill_speaker_combos(self) -> None:
        """Library voices in the male and female lists (retired voices such as Boaz are left out). A still-valid choice is
        kept; otherwise the shipped cast (Natan, Shimon, Miriam) when installed, else the first voice of that gender that is
        not the narrator, and for the second male voice the next such male voice (None when there is no third male voice)."""
        voices = bundled_voices.offered_for_roles(self.library.list_voices())
        narrator = self.selected_voice_id()
        defaults = spk.default_role_picks(voices, narrator, bundled_voices.preferred_ids(voices))   # Natan / Shimon / Miriam
        for which, combo in (("male", self.cmb_spk_male), ("male2", self.cmb_spk_male2), ("female", self.cmb_spk_female)):
            current = str(combo.currentData() or "")
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(tr("spk.none"), "")
            for rec in voices:
                combo.addItem(rec.name, rec.id)
            idx = combo.findData(current) if current else 0
            if self._spk_touched[which]:
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                d = combo.findData(defaults[which]) if defaults[which] else 0
                combo.setCurrentIndex(d if d >= 0 else 0)
            combo.blockSignals(False)
        rec = self.library.get(narrator) if narrator else None
        self.lbl_spk_narrator.setText(tr("spk.narrator", name=rec.name if rec is not None else tr("spk.none")))

    def speaker_cast(self) -> Optional[spk.SpeakerCast]:
        """Marks and extra voices when Mark speakers is on and at least one extra voice differs from the narrator."""
        if not (self.chk_speakers.isChecked() and self.chk_speakers.isEnabled()):
            return None
        male, female = str(self.cmb_spk_male.currentData() or ""), str(self.cmb_spk_female.currentData() or "")
        male2 = str(self.cmb_spk_male2.currentData() or "") if male else ""
        narr = self.selected_voice_id()
        characters = dict(self.book.voice_cast) if self.book is not None else {}
        file_marks = tuple(getattr(self.book, "speaker_marks", ()) or ()) if self.book is not None else ()
        lines = self.speaker_lines
        if file_marks and lines is None:
            lines = [spk.SpeakerLine(role, name) for role, name in file_marks]
        cast = spk.SpeakerCast(lines=lines, male_id=male, female_id=female, male2_id=male2,
                               tagger=None if (lines is not None or file_marks) else self.llm_plan(), narrator_id=narr,
                               characters=characters)
        return cast if cast.uses_several(narr) else None

    def preview_speakers(self) -> bool:
        """Tag the loaded book (unless the user already edited the marks) and open the preview. No modal on offscreen."""
        if self.book is None or not self.chk_speakers.isEnabled():
            return False
        paras = [text for _ci, text in spk.paragraphs(self.book)]
        lines = self.speaker_lines
        file_marks = tuple(getattr(self.book, "speaker_marks", ()) or ())
        if file_marks and (lines is None or len(lines) != len(paras)):
            lines = [spk.SpeakerLine(role, name) for role, name in file_marks]
        elif lines is None or len(lines) != len(paras):
            plan = self.llm_plan()
            if plan is None:
                return False
            lines = list(spk.tag_paragraphs(paras, self.book_language() or "en", plan))
        dlg = SpeakerDialog(lines, paras, self)
        self._speaker_preview = dlg
        dlg.accepted.connect(lambda: setattr(self, "speaker_lines", dlg.lines()))
        if QApplication.platformName() == "offscreen":
            dlg.show()
            return True
        dlg.exec()
        return True

    def _sync_choice_status(self) -> None:
        """Idle until a book and a voice are chosen, then a line that says they are. A running or finished job keeps its own text."""
        if self.busy or self.result is not None or not self._choice_status:
            return
        if self.book is not None and self.selected_voice_id():
            self.lbl_status.setText(tr("narr.ready"))
        else:
            self.lbl_status.setText(tr("narr.idle"))

    def download_llm(self) -> bool:
        """Download llama.cpp + the model (one optional download, ~7 GB); False if nothing was started."""
        if (self.llm_worker and self.llm_worker.isRunning()) or self.llm_status() != "needs_download":
            return False
        from types import SimpleNamespace

        w = TextModelsDownloadWorker([SimpleNamespace(key=llm_tool.KEY)],
                                     lambda _m, progress: self.llm_ensure(lambda f, msg="": progress(None, f, msg)), parent=self)
        def on_progress(fraction) -> None:
            setattr(self, "_llm_msg", tr("prep.model_downloading", pct=int(fraction * 100)))
            self.lbl_llm_state.setText(self._llm_msg)

        def on_done(_key) -> None:
            setattr(self, "_llm_msg", "")
            self._refresh_buttons()

        def on_failed(message) -> None:
            setattr(self, "_llm_msg", tr("prep.model_failed", error=message))
            self._refresh_llm_row()

        w.progress.connect(on_progress)
        w.done.connect(on_done)
        w.failed.connect(on_failed)
        self.llm_worker = w
        self._llm_msg = tr("prep.model_downloading", pct=0)
        w.start()
        self._refresh_llm_row()
        return True

    def edit_prompts(self) -> Path:
        """Copy our prompt files to ``<data folder>/prompts`` (only the missing ones) and open that folder: edited copies
        there are used instead of ours (core/llm_text.load_prompt); delete a file to get ours back."""
        import shutil

        from infra import paths

        dest = paths.app_home() / "prompts"
        dest.mkdir(parents=True, exist_ok=True)
        for f in sorted((paths.resource_dir() / "prompts").glob("*.txt")):
            if not (dest / f.name).exists():
                shutil.copyfile(f, dest / f.name)
        open_folder(dest)
        return dest

    def _refresh_translate_row(self) -> None:
        """Texts and the Download button of the translation card."""
        on = self.chk_translate.isChecked()
        self.tr_box.setVisible(on)
        status = self._translate_status()
        src, dst = self.book_source_language(), self.translate_target()
        name = lambda code: language_names().get(code, code or "?")   # noqa: E731
        models = self.translate_models_needed()
        downloading = bool(self.tr_worker and self.tr_worker.isRunning())
        if status == "same":
            msg = tr("narr.translate_same")
        elif status == "unsupported":
            msg = tr("narr.translate_unsupported", source=name(src))
        elif self._tr_msg:
            msg = self._tr_msg
        elif status == "needs_model":
            missing = [m for m in models if self.model_state(m) != text_models.STATE_READY]
            msg = tr("narr.translate_needs", size=sum(m.size_mb for m in missing))
        elif status == "ready":
            msg = tr("narr.translate_ready")
        else:
            msg = ""
        if status in ("ready", "needs_model") and len(models) > 1:
            msg += "  " + tr("narr.translate_pivot", source=name(src), target=name(dst))
        if status in ("ready", "needs_model") and src:
            msg = tr("narr.translate_detected", lang=name(src)) + "  " + msg
        self.lbl_tr_state.setText(msg)
        self.btn_tr_download.setVisible(status == "needs_model")
        self.btn_tr_download.setEnabled(not downloading and not self.busy)
        rec = self.library.get(self.selected_voice_id()) if self.selected_voice_id() else None
        vlang = nw.lang_code(rec.language) if rec is not None and rec.language else ""
        note = ""
        if on and status in ("ready", "needs_model") and rec is not None and vlang and vlang != dst:
            note = tr("narr.translate_voice", voice=name(vlang), target=name(dst))
        self.lbl_tr_voice.setText(note)
        self.lbl_tr_voice.setVisible(bool(note))

    def download_translate_models(self) -> bool:
        """Download the missing translation models one after another; returns False if nothing was started."""
        if self.tr_worker and self.tr_worker.isRunning():
            return False
        missing = [m for m in self.translate_models_needed() if self.model_state(m) != text_models.STATE_READY]
        if not missing:
            return False
        w = TextModelsDownloadWorker(missing, self.model_ensure, parent=self)
        w.progress.connect(self._on_tr_progress)
        w.done.connect(self._on_tr_done)
        w.failed.connect(self._on_tr_failed)
        self.tr_worker = w
        self._tr_msg = tr("prep.model_downloading", pct=0)
        w.start()
        self._refresh_translate_row()
        return True

    def _on_tr_progress(self, fraction: float) -> None:
        """Download progress of the translation models."""
        self._tr_msg = tr("prep.model_downloading", pct=int(fraction * 100))
        self.lbl_tr_state.setText(self._tr_msg)

    def _on_tr_done(self, _keys: str) -> None:
        """The models are on disk: the translation can be used."""
        self._tr_msg = ""
        self._refresh_buttons()

    def _on_tr_failed(self, message: str) -> None:
        """The download failed: the message is shown, the button stays for a retry."""
        self._tr_msg = tr("prep.model_failed", error=message)
        self._refresh_translate_row()

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
        self.speaker_lines = None
        if book.speaker_marks:
            self.speaker_lines = [spk.SpeakerLine(role, name) for role, name in book.speaker_marks]
            self.chk_speakers.setChecked(True)
        self._choice_status = True
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
        items = catalog.build(self.library, self.entries)
        if not current:   # nothing chosen yet: the shipped narrator (Levi) when it is installed, else the first voice
            current = bundled_voices.preferred_ids([it.record for it in items if it.record is not None]).get("narrator", "")
        self.cmb_voice.blockSignals(True)
        self.cmb_voice.clear()
        for it in items:
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
        lang = language_name(rec.language)
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
        lang = language_name(e.language)
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
            ws.save_folder(self.out_dir)

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
            speak_titles=self.chk_titles.isChecked(), allow_aac=self.aac_allowed, pauses=self.pause_profile(),
            pause_lengths=pz.load_lengths(), pace=pc.load(),      # Settings: pause lengths, reading speed and style
            prep=self.plan_builder(self.selected_rule_steps(), self.selected_neural_steps()),
            translate=self.translate_plan(), ai_disclosure=self.chk_disclosure.isChecked(),
            check_chunks=self.chk_check_chunks.isChecked(), llm_prepare=self.llm_prepare_plan(),
            speakers=self.speaker_cast(),
            ordinals=ordinals.load_enabled(),                    # Settings: ordinal numbers by context
            yo=self.chk_prepare.isChecked() and not (self.book is not None and self.book.explicit_yo),
            sound=(soundscape_mod.SoundRequest()
                   if self.book is not None and soundscape_model.enabled() and soundscape_mod.requested(self.book)
                   else None))

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
        ready = ready and self._translate_ok()
        self.btn_start.setEnabled(ready)
        self.btn_book.setEnabled(not busy)
        self.btn_out.setEnabled(not busy)
        self.chk_prepare.setEnabled(not busy)
        self._refresh_model_row()
        self._refresh_translate_row()
        self._refresh_llm_row()
        self._sync_preset()
        self.chk_translate.setEnabled(not busy)
        self.cmb_translate.setEnabled(not busy)
        self.cmb_voice.setEnabled(not busy)
        self.chk_disclosure.setEnabled(not busy)
        self.chk_check_chunks.setEnabled(not busy)
        self.chk_pauses.setEnabled(not busy)
        self.sld_pauses.setEnabled(not busy and self.chk_pauses.isChecked())
        self.btn_pause.setVisible(busy)
        self.btn_cancel.setVisible(busy)
        self._sync_choice_status()

    # ------------------------------------------------------------------ run
    def start(self) -> bool:
        """Start (or resume) the narration of the loaded book; returns False if something is missing."""
        key = self.selected_voice_id()
        if catalog.is_remote_key(key):
            return self._download_then_start(key)
        rec = self.library.get(key) if key else None
        if not (self.book and rec and self.selected_formats()) or self.busy:
            return False
        options = self.options()
        self._offer_book_copy(nr.job_dir_for(self.book, self.out_dir, options))
        extra = {}
        if options.speakers is not None:
            for vid in options.speakers.extra_ids(rec.id):
                other = self.library.get(vid)
                if other is not None:
                    extra[vid] = other
        job = NarrationJob(self.book, rec, self.out_dir, options, extra_voices=extra)
        self.result = None
        self._choice_status = False
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
        w.finished.connect(self._after_worker)
        self.worker = w
        w.start()
        self._refresh_buttons()
        return True

    def _offer_book_copy(self, job_dir: Path) -> None:
        """Ask once per book whether its file may be copied/moved into the job folder (never done without asking)."""
        src = self.book_path
        if src is None or not src.is_file() or ws.is_inside(src, job_dir) or src in self._place_asked:
            return
        mode = self._ask_place(src, job_dir)
        self._place_asked.add(src)
        if mode == ws.PLACE_LEAVE:
            return
        try:
            self.book_path = ws.place_book(src, job_dir, mode)
        except OSError as exc:                           # e.g. the file is open elsewhere: narrate it where it is
            log.warning("cannot %s the book into %s: %s", mode, job_dir, exc)
            return
        self._place_asked.add(self.book_path)
        self.lbl_book.setText(self.book_path.name)

    def _cleanup_job(self, result: nr.NarrationResult) -> bool:
        """Finished job: let the user choose what stays in the job folder; returns False if the result audio was removed."""
        files = ws.scan_job(result.out_dir, result.files, self.book_path)
        choice = self._ask_cleanup(files)
        if choice is None:                               # "later": nothing is deleted
            return True
        freed = ws.clean_job(files, **choice)
        if freed:
            log.info("job folder %s cleaned, %.1f MB freed", result.out_dir, freed / 1024 ** 2)
        return bool(choice.get("keep_results", True))

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
        self._choice_status = False          # keep the download line; do not replace it with "book and voice are chosen"
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
        cc = result.chunk_check or {}
        checked = tr("narr.check_summary", n=cc.get("regenerated", 0), total=cc.get("checked", 0),
                     bad=cc.get("still_bad", 0)) if cc.get("checked") else ""
        self.lbl_status.setText(tr("narr.done_summary", chapters=result.chapters, files=len(result.files))
                                + (" " + checked if checked else "") + reminder)
        self.lbl_ready.show()
        self.btn_open.show()
        kept = self._cleanup_job(result)
        if self.player.isVisibleTo(self) or self.player.queue.planned:
            self.player.set_final(result.files[0] if (result.files and kept) else None)
            self.player.show()
        self._refresh_buttons()
        if self.auto_open_folder:
            open_folder(result.out_dir)

    def on_cancelled(self) -> None:
        """Worker signal: cancelled."""
        self.lbl_status.setText(tr("narr.cancelled"))
        self.player.set_final(None)

    def _after_worker(self) -> None:
        """Enable Start once the worker has really stopped.

        ``QThread.finished`` can be delivered while ``isRunning()`` is still true. Refreshing in that slot leaves
        Start disabled after the thread has stopped, and a later event-loop turn then sees the failure line with the
        button off. Retry on the next turn until the thread is idle, then refresh.
        """
        if self.busy:
            if not self._button_retry:
                self._button_retry = True
                QTimer.singleShot(0, self._retry_buttons)
            return
        self._button_retry = False
        self._refresh_buttons()

    def _retry_buttons(self) -> None:
        self._button_retry = False
        self._after_worker()

    def on_failed(self, kind: str, message: str, details: str) -> None:
        """Worker signal: failed - the message is shown; finished chunks are still cached."""
        self.progress.setValue(0)
        self.player.set_final(None)
        self.lbl_status.setText(message + "  " + tr("narr.failed_resume"))
        log.error("narration failed: kind=%s %s | %s", kind, message, details)
        self._after_worker()

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
        if self.tr_worker and self.tr_worker.isRunning():
            self.tr_worker.wait(8000)
        for w in (self._dl_worker, self._index_worker):
            if w is not None and w.isRunning():
                w.wait(3000)

    def closeEvent(self, e) -> None:  # noqa: N802
        """Stop the worker before closing."""
        self.shutdown()
        super().closeEvent(e)
