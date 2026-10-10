""""My voices": the voice library as a list of cards, plus import and the online repository.

Each card shows the name, language/length/epochs, author, description, a **licence badge** ("commercial use OK" /
"personal use only", derived from the licence in ``voice.json``), and buttons: preview sample, narrate with this voice,
edit details, delete.  Header buttons import a voice from a folder or ``.zip`` and open the repository dialog
(:class:`RepoDialog`) that lists downloadable voices with their licences.

All file-system work is done by :class:`core.voice_library.VoiceLibrary`; dialogs and the audio previewer are injectable
so the tests drive the window without real dialogs or sound devices.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, cast

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core import adapter_strength, voice_info
from core.errors import DatasetMakerError
from core.i18n import tr
from core.languages import language_name
from core.voice_library import VoiceLibrary, VoiceRecord
from infra import voice_catalog as catalog
from infra import voice_repository as repo
from ui.audio_preview import Previewer
from ui.glass import GlassDialog
from ui.window_base import SubWindow, card_frame, fit_to_screen, hint_label
from workers.narrate_worker import RepoDownloadWorker, RepoIndexWorker

log = logging.getLogger("voxprint.ui.voices")


def voice_type_label(code: str) -> str:
    """Localized name of a ``voice_type`` value ("" = not set)."""
    labels = {"": tr("ui.voice_type_none"), "male": tr("ui.voice_type_male"), "female": tr("ui.voice_type_female"),
              "child": tr("ui.voice_type_child"), "other": tr("ui.voice_type_other")}
    return labels.get(code, labels[""])


def gender_labels() -> dict:
    """Localized names of the ``gender`` values ("" = not set)."""
    return {"": tr("ui.voice_gender_none"), "male": tr("ui.voice_gender_male"), "female": tr("ui.voice_gender_female")}


def age_labels() -> dict:
    """Localized names of the ``age_group`` values ("" = not set)."""
    return {"": tr("ui.voice_age_none"), "child": tr("ui.voice_age_child"), "young": tr("ui.voice_age_young"),
            "adult": tr("ui.voice_age_adult"), "elderly": tr("ui.voice_age_elderly")}


def fill_combo(combo: QComboBox, labels: dict, current: str = "") -> None:
    """(Re)fill ``combo`` with ``labels`` (code -> text), keeping the code ``current`` selected."""
    combo.clear()
    for code, text in labels.items():
        combo.addItem(text, code)
    combo.setCurrentIndex(max(0, combo.findData(current)))


def badge_text(license_id: str, commercial: bool) -> str:
    """Text of the licence badge: what the licence allows, plus the licence id."""
    if commercial:
        return tr("voices.badge_commercial", license=license_id)
    return tr("voices.badge_personal", license=license_id)


def duration_text(seconds: float) -> str:
    """``"25 min"`` / ``"1 h 05 min"`` for the amount of training speech."""
    m = int(round(seconds / 60))
    return tr("voices.minutes", n=m) if m < 60 else tr("voices.hours_minutes", h=m // 60, m=f"{m % 60:02d}")


def meta_line(rec: VoiceRecord) -> str:
    """One line: language, training length, epochs, voice type and author."""
    parts = [language_name(rec.language), duration_text(float(rec.info.get("duration", 0))),
             tr("voices.epochs", n=int(rec.info.get("epochs", 0)))]
    parts.append(voice_type_label(str(rec.info.get("voice_type") or "")))      # older voices: "not specified"
    age = str(rec.info.get("age_group") or "")
    if age and age != "child":                                                 # "child" is the type label already
        parts.append(age_labels().get(age, ""))
    if rec.info.get("author"):
        parts.append(tr("voices.by_author", author=rec.info["author"]))
    return " \u00b7 ".join(p for p in parts if p)


def make_badge(license_id: str, commercial: bool, url: str = "") -> QLabel:
    """The coloured licence badge (``QLabel#badge`` with a ``commercial`` property for the style sheet)."""
    b = QLabel(badge_text(license_id, commercial))
    b.setObjectName("badge")
    b.setProperty("commercial", "true" if commercial else "false")
    if url:
        b.setToolTip(url)
    return b


def make_scope_badge(scope: str) -> QLabel:
    """Badge of the voice-owner's usage scope (commercial / public non-commercial / private only)."""
    b = QLabel(tr("consent.badge_" + scope))
    b.setObjectName("badge")
    b.setProperty("commercial", "true" if scope == "commercial" else "false")
    b.setToolTip(tr("consent.tip_" + scope))
    return b


class RemoteCard(QFrame):
    """A voice that is in the online index but not on this computer yet: badges, size and a *Download* button."""

    download = Signal(str)

    def __init__(self, item: "catalog.CatalogItem", busy: bool = False, progress: float = -1.0) -> None:
        """``progress`` >= 0 shows the running download."""
        super().__init__()
        self.setObjectName("card")
        self.key = item.key
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(6)
        top = QHBoxLayout()
        self.lbl_name = QLabel(item.name)
        self.lbl_name.setObjectName("sectiontitle")
        top.addWidget(self.lbl_name)
        self.badge = make_badge(item.license, item.commercial, item.license_url)
        top.addWidget(self.badge)
        self.scope_badge = make_scope_badge(item.scope)
        top.addWidget(self.scope_badge)
        top.addStretch(1)
        lay.addLayout(top)
        size = catalog.size_text(item.size_bytes)
        meta = [p for p in (language_name(item.language), item.author and tr("voices.by_author", author=item.author),
                            size) if p]
        self.lbl_meta = QLabel(" \u00b7 ".join(meta))
        self.lbl_meta.setObjectName("carddesc")
        lay.addWidget(self.lbl_meta)
        self.lbl_desc = QLabel(item.description)
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setVisible(bool(item.description))
        lay.addWidget(self.lbl_desc)
        if item.license == voice_info.LICENSE_TEST_ONLY:
            self.lbl_note = QLabel(tr("voices.test_only_note"))
            self.lbl_note.setObjectName("hint")
            self.lbl_note.setWordWrap(True)
            lay.addWidget(self.lbl_note)
        elif item.entry is not None and item.entry.bundled:      # comes with Voxprint: licence stated before the download
            self.lbl_note = QLabel(tr("voices.bundled_note", license=item.license))
            self.lbl_note.setObjectName("hint")
            self.lbl_note.setWordWrap(True)
            lay.addWidget(self.lbl_note)
        row = QHBoxLayout()
        self.lbl_state = QLabel(tr("voices.remote_state"))
        self.lbl_state.setObjectName("carddesc")
        self.btn_download = QPushButton(tr("voices.remote_download") if progress < 0 else tr("voices.remote_downloading", p=int(progress * 100)))
        self.btn_download.setEnabled(not busy and progress < 0)
        self.btn_download.clicked.connect(lambda: self.download.emit(self.key))
        row.addWidget(self.lbl_state)
        row.addStretch(1)
        row.addWidget(self.btn_download)
        lay.addLayout(row)


class VoiceCard(QFrame):
    """One voice of the library."""

    preview = Signal(str)
    narrate = Signal(str)
    edit = Signal(str)
    delete = Signal(str)
    export = Signal(str)
    open_folder = Signal(str)

    def __init__(self, rec: VoiceRecord, playing: bool = False) -> None:
        """Build the card for ``rec``; ``playing`` shows the stop symbol on the preview button."""
        super().__init__()
        self.setObjectName("card")
        self.voice_id = rec.id
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(6)
        top = QHBoxLayout()
        self.lbl_name = QLabel(rec.name)
        self.lbl_name.setObjectName("sectiontitle")
        top.addWidget(self.lbl_name)
        self.badge = make_badge(rec.license, rec.commercial_use, str(rec.info.get("license_url", "")))
        top.addWidget(self.badge)
        self.scope_badge = make_scope_badge(rec.scope)
        top.addWidget(self.scope_badge)
        top.addStretch(1)
        lay.addLayout(top)
        self.lbl_meta = QLabel(meta_line(rec))
        self.lbl_meta.setObjectName("carddesc")
        self.lbl_meta.setWordWrap(True)
        lay.addWidget(self.lbl_meta)
        self.lbl_desc = QLabel(rec.description)
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setVisible(bool(rec.description))
        lay.addWidget(self.lbl_desc)
        self.lbl_note = None
        if rec.test_only:      # test-only voice: always remind of its terms
            self.lbl_note = QLabel(tr("voices.test_only_note"))
            self.lbl_note.setObjectName("hint")
            self.lbl_note.setWordWrap(True)
            lay.addWidget(self.lbl_note)
        elif rec.bundled:      # an open voice that comes with Voxprint: read-only, licence stated
            self.lbl_note = QLabel(tr("voices.bundled_note", license=rec.license))
            self.lbl_note.setObjectName("hint")
            self.lbl_note.setWordWrap(True)
            lay.addWidget(self.lbl_note)
        row = QHBoxLayout()
        self.btn_preview = QPushButton(("\u25a0 " + tr("voices.stop")) if playing else ("\u25b6 " + tr("voices.preview")))
        self.btn_preview.setEnabled(rec.preview_path is not None)
        self.btn_preview.setToolTip(tr("voices.preview_tip") if rec.preview_path else tr("voices.no_preview"))
        self.btn_narrate = QPushButton(tr("voices.narrate"))
        self.btn_edit = QPushButton(tr("voices.edit"))
        self.btn_delete = QPushButton(tr("voices.delete"))
        self.btn_export = QPushButton(tr("voices.export"))
        self.btn_folder = QPushButton(tr("voices.open_folder"))
        if rec.bundled:
            for b in (self.btn_edit, self.btn_delete):
                b.setEnabled(False)
                b.setToolTip(tr("voices.bundled_tip"))
        for b in (self.btn_preview, self.btn_narrate, self.btn_edit, self.btn_export, self.btn_folder, self.btn_delete):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.btn_preview.clicked.connect(lambda: self.preview.emit(self.voice_id))
        self.btn_narrate.clicked.connect(lambda: self.narrate.emit(self.voice_id))
        self.btn_edit.clicked.connect(lambda: self.edit.emit(self.voice_id))
        self.btn_delete.clicked.connect(lambda: self.delete.emit(self.voice_id))
        self.btn_export.clicked.connect(lambda: self.export.emit(self.voice_id))
        self.btn_folder.clicked.connect(lambda: self.open_folder.emit(self.voice_id))


class VoiceEditDialog(GlassDialog):
    """Edit name, author, speaker, prepared by, organization, project link, licence, gender / age and description.

    A form layout (label left, field right) keeps the dialog short enough for 150 % display scaling."""

    def __init__(self, rec: VoiceRecord, parent: Optional[QWidget] = None) -> None:
        """Fill the fields from ``rec``; :meth:`values` returns what the user entered."""
        super().__init__(parent)
        self.setObjectName("root")
        self.setMinimumWidth(560)
        if parent is not None:
            self.setStyleSheet(parent.styleSheet())
        self.setWindowTitle(tr("voices.edit_title"))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(10)
        self.edt_name = QLineEdit(str(rec.info.get("name") or rec.name))   # the stored default, not the localized display name
        self.edt_name.setMaxLength(voice_info.MAX_NAME_CHARS)
        self.edt_author = QLineEdit(str(rec.info.get("author", "")))
        self.edt_author.setPlaceholderText(tr("voices.author_placeholder"))
        self.cmb_license = QComboBox()
        for lic in voice_info.LICENSES:
            self.cmb_license.addItem(lic, lic)
        if self.cmb_license.findData(rec.license) < 0:
            self.cmb_license.addItem(rec.license, rec.license)
        self.cmb_license.setCurrentIndex(self.cmb_license.findData(rec.license))
        self.lbl_license_note = hint_label()
        self.cmb_gender, self.cmb_age = QComboBox(), QComboBox()
        fill_combo(self.cmb_gender, gender_labels(), str(rec.info.get("gender", "")))
        fill_combo(self.cmb_age, age_labels(), str(rec.info.get("age_group", "")))
        ga = QHBoxLayout()
        ga.setContentsMargins(0, 0, 0, 0)
        ga.addWidget(self.cmb_gender, 1)
        ga.addWidget(self.cmb_age, 1)
        self.edt_speaker = QLineEdit(str(rec.info.get("speaker", "")))
        self.edt_prepared = QLineEdit(str(rec.info.get("prepared_by", "")))
        self.edt_org = QLineEdit(str(rec.info.get("organization", "")))
        self.edt_url = QLineEdit(str(rec.info.get("project_url", "")))
        self.edt_url.setPlaceholderText("https://")
        for w, n in ((self.edt_speaker, voice_info.MAX_NAME_CHARS), (self.edt_prepared, voice_info.MAX_NAME_CHARS),
                     (self.edt_org, voice_info.MAX_ORGANIZATION_CHARS), (self.edt_url, voice_info.MAX_URL_CHARS)):
            w.setMaxLength(n)
        for w in (self.edt_speaker, self.edt_prepared, self.edt_org):
            w.setPlaceholderText(tr("voices.author_placeholder"))
        self.edt_desc = QLineEdit(str(rec.info.get("description", "")))
        self.edt_desc.setMaxLength(voice_info.MAX_DESCRIPTION_CHARS)
        self.sp_strength = QDoubleSpinBox()          # LoRA strength at narration (core/adapter_strength.py)
        self.sp_strength.setRange(adapter_strength.MIN_SCALE, adapter_strength.MAX_SCALE)
        self.sp_strength.setSingleStep(0.05)
        self.sp_strength.setDecimals(2)
        self.sp_strength.setValue(rec.adapter_scale)
        self.sp_strength.setToolTip(tr("voices.strength_note"))
        form = QFormLayout()
        form.setSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for label, w in ((tr("voices.field_name"), self.edt_name), (tr("voices.field_speaker"), self.edt_speaker),
                         (tr("voices.field_gender_age"), ga), (tr("voices.field_author"), self.edt_author),
                         (tr("voices.field_prepared_by"), self.edt_prepared),
                         (tr("voices.field_organization"), self.edt_org), (tr("voices.field_project_url"), self.edt_url),
                         (tr("voices.field_description"), self.edt_desc), (tr("voices.field_license"), self.cmb_license),
                         (tr("voices.field_strength"), self.sp_strength)):
            form.addRow(label, w)
        lay.addLayout(form)
        lay.addWidget(self.lbl_license_note)      # outside the form: a wrapped label in a form row gets cut off
        self.lbl_strength_note = hint_label()
        self.lbl_strength_note.setText(tr("voices.strength_note"))
        lay.addWidget(self.lbl_strength_note)
        self.lbl_rights = hint_label()
        self.lbl_rights.setText(tr("voices.rights_note"))
        lay.addWidget(self.lbl_rights)
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_cancel = QPushButton(tr("voices.cancel"))
        self.btn_save = QPushButton(tr("voices.save"))
        self.btn_save.setObjectName("primary")
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_save)
        lay.addLayout(row)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self.accept)
        self.cmb_license.currentIndexChanged.connect(self._update_note)
        self._update_note()

    def _update_note(self, _i: int = 0) -> None:
        """Explain what the selected licence allows."""
        lic = str(self.cmb_license.currentData())
        if lic == voice_info.LICENSE_TEST_ONLY:
            self.lbl_license_note.setText(tr("voices.test_only_note"))
        else:
            self.lbl_license_note.setText(tr("voices.license_commercial") if voice_info.license_allows_commercial(lic)
                                          else tr("voices.license_personal"))

    def values(self) -> dict:
        """The edited fields, ready for :meth:`VoiceLibrary.update`."""
        return {"name": self.edt_name.text(), "author": self.edt_author.text(), "speaker": self.edt_speaker.text(),
                "prepared_by": self.edt_prepared.text(), "organization": self.edt_org.text(),
                "project_url": self.edt_url.text(), "license": str(self.cmb_license.currentData()),
                "gender": str(self.cmb_gender.currentData() or ""), "age_group": str(self.cmb_age.currentData() or ""),
                "description": self.edt_desc.text(), "adapter_scale": round(self.sp_strength.value(), 2)}


class RepoDialog(GlassDialog):
    """Browse the online voice index and download selected voices (SHA-256 verified)."""

    voices_added = Signal(list)

    def __init__(self, library: VoiceLibrary, parent: Optional[QWidget] = None,
                 fetch: Callable[..., Any] = repo.fetch_index,
                 download: Callable[..., Any] = repo.download_voice) -> None:
        """``fetch``/``download`` are injectable (tests)."""
        super().__init__(parent)
        self.library, self._fetch, self._download = library, fetch, download
        self.entries: List[repo.RepoVoice] = []
        self._index_worker: Optional[RepoIndexWorker] = None
        self._dl_worker: Optional[RepoDownloadWorker] = None
        self.setObjectName("root")
        self.setMinimumSize(560, 460)
        if parent is not None:
            self.setStyleSheet(parent.styleSheet())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(10)
        self.lbl_title = QLabel()
        self.lbl_title.setObjectName("title")
        lay.addWidget(self.lbl_title)
        urow = QHBoxLayout()
        self.edt_url = QLineEdit(repo.index_url())
        self.btn_refresh = QPushButton()
        urow.addWidget(self.edt_url, 1)
        urow.addWidget(self.btn_refresh)
        lay.addLayout(urow)
        self.lbl_state = hint_label()
        lay.addWidget(self.lbl_state)
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        lay.addWidget(self.list, 1)
        self.lbl_detail = hint_label()
        lay.addWidget(self.lbl_detail)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.hide()
        lay.addWidget(self.progress)
        brow = QHBoxLayout()
        brow.addStretch(1)
        self.btn_close = QPushButton()
        self.btn_download = QPushButton()
        self.btn_download.setObjectName("primary")
        brow.addWidget(self.btn_close)
        brow.addWidget(self.btn_download)
        lay.addLayout(brow)
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_close.clicked.connect(self.reject)
        self.btn_download.clicked.connect(self.download_selected)
        self.list.itemSelectionChanged.connect(self._on_selection)
        self.retranslate()
        self._on_selection()

    def retranslate(self) -> None:
        """Texts in the current language."""
        self.setWindowTitle(tr("voices.repo_title"))
        self.lbl_title.setText(tr("voices.repo_title"))
        self.btn_refresh.setText(tr("voices.repo_refresh"))
        self.btn_close.setText(tr("about.btn_close"))
        self.btn_download.setText(tr("voices.repo_download"))
        self.edt_url.setToolTip(tr("voices.repo_url_tip"))

    # ------------------------------------------------------------------ index
    def refresh(self) -> None:
        """Save the URL, then load the index in the background."""
        url = self.edt_url.text().strip()
        if url != repo.index_url():
            repo.set_index_url("" if url == repo.DEFAULT_INDEX_URL else url)
        self.lbl_state.setText(tr("voices.repo_loading"))
        self.btn_refresh.setEnabled(False)
        w = RepoIndexWorker(url or None, self._fetch, parent=self)
        w.done.connect(self._on_index)
        self._index_worker = w
        w.start()

    def _on_index(self, res: repo.IndexResult) -> None:
        """Show the index entries or the reason there are none."""
        self.btn_refresh.setEnabled(True)
        self.list.clear()
        self.entries = list(res.voices)
        if res.error == "not_configured":
            self.lbl_state.setText(tr("voices.repo_not_configured"))
        elif res.error == "unreachable":
            self.lbl_state.setText(tr("voices.repo_unreachable"))
        elif res.error == "invalid":
            self.lbl_state.setText(tr("voices.repo_invalid"))
        elif not self.entries:
            self.lbl_state.setText(tr("voices.repo_empty"))
        else:
            self.lbl_state.setText(tr("voices.repo_found", n=len(self.entries)))
        for e in self.entries:
            it = QListWidgetItem(f"{e.display_name}  \u00b7  {language_name(e.language)}  \u00b7  {badge_text(e.license, e.commercial_use)}")
            it.setData(Qt.ItemDataRole.UserRole, e.id)
            self.list.addItem(it)
        self._on_selection()

    def _selected(self) -> List[repo.RepoVoice]:
        """Entries selected in the list."""
        ids = {i.data(Qt.ItemDataRole.UserRole) for i in self.list.selectedItems()}
        return [e for e in self.entries if e.id in ids]

    def _on_selection(self) -> None:
        """Enable the download button and show the details of the selected voice."""
        sel = self._selected()
        self.btn_download.setEnabled(bool(sel) and not self._downloading)
        if len(sel) == 1:
            e = sel[0]
            self.lbl_detail.setText(f"{e.description}\n{tr('voices.by_author', author=e.author) if e.author else ''}"
                                    f"  {e.license_url}".strip())
        else:
            self.lbl_detail.setText("")

    @property
    def _downloading(self) -> bool:
        """True while a download runs."""
        return bool(self._dl_worker and self._dl_worker.isRunning())

    # ------------------------------------------------------------------ download
    def download_selected(self) -> None:
        """Download and import the selected voices in the background."""
        sel = self._selected()
        if not sel or self._downloading:
            return
        self.progress.setValue(0)
        self.progress.show()
        self.btn_download.setEnabled(False)
        w = RepoDownloadWorker(sel, self.library, self._download, parent=self)
        w.progress.connect(lambda f, n: (self.progress.setValue(int(f * 100)), self.lbl_state.setText(n)))
        w.failed.connect(lambda m: self.lbl_state.setText(m))
        w.finished_all.connect(self._on_downloaded)
        self._dl_worker = w
        w.start()

    def _on_downloaded(self, ids: list) -> None:
        """Downloads finished (possibly partially)."""
        self.progress.hide()
        if ids:
            self.lbl_state.setText(tr("voices.repo_downloaded", n=len(ids)))
            self.voices_added.emit(ids)
        self._on_selection()

    def done(self, r: int) -> None:  # noqa: D401
        """Stop a running download when the dialog is closed."""
        if self._dl_worker and self._dl_worker.isRunning():
            self._dl_worker.cancel()
            self._dl_worker.wait(3000)
        super().done(r)


class VoicesWindow(SubWindow):
    """The "My voices" window."""

    narrate_with = Signal(str)       # voice id: open the narrator with this voice selected
    library_changed = Signal()
    full_model_requested = Signal(str)   # adapter folder: build the standalone ~4 GB model (training window's merge)

    def __init__(self, library: Optional[VoiceLibrary] = None, previewer: Optional[Previewer] = None,
                 confirm: Optional[Callable[[str, str], bool]] = None,
                 pick_folder: Optional[Callable[[], str]] = None, pick_zip: Optional[Callable[[], str]] = None,
                 repo_dialog_factory: Optional[Callable[..., RepoDialog]] = None,
                 fetch: Callable[..., Any] = repo.fetch_index, download: Callable[..., Any] = repo.download_voice,
                 auto_refresh: bool = True) -> None:
        """Build the window; every dialog/IO collaborator can be replaced for tests.  ``fetch`` / ``download`` are the
        repository hooks; the index is refreshed in the background when the window opens (``auto_refresh``)."""
        super().__init__(with_back=True)
        self._fetch, self._download = fetch, download
        self.entries: List[repo.RepoVoice] = repo.load_cache()      # the cached index shows at once, also offline
        self._index_worker: Optional[RepoIndexWorker] = None
        self._dl_worker: Optional[RepoDownloadWorker] = None
        self._dl_key, self._dl_progress = "", -1.0
        self.library = library or VoiceLibrary()
        self.previewer = previewer or Previewer(self)
        self.previewer.stopped.connect(self.refresh)
        self._confirm = confirm
        self._pick_folder, self._pick_zip = pick_folder, pick_zip
        self._repo_dialog_factory = repo_dialog_factory
        self._playing_id = ""
        self.repo_dialog: Optional[RepoDialog] = None
        self.lbl_intro = hint_label()
        self.body.addWidget(self.lbl_intro)
        bar = QHBoxLayout()
        self.btn_import = QPushButton()
        menu = QMenu(self.btn_import)
        self.act_folder = menu.addAction("")
        self.act_zip = menu.addAction("")
        self.btn_import.setMenu(menu)
        self.btn_repo = QPushButton()
        bar.addWidget(self.btn_import)
        bar.addWidget(self.btn_repo)
        self.btn_reload = QPushButton()
        bar.addWidget(self.btn_reload)
        bar.addStretch(1)
        self.body.addLayout(bar)
        self.lbl_status = QLabel()
        self.lbl_status.setObjectName("status")
        self.lbl_status.setWordWrap(True)
        self.body.addWidget(self.lbl_status)
        self.cards_box = QVBoxLayout()
        self.cards_box.setSpacing(10)
        self.body.addLayout(self.cards_box)
        # empty state
        self.empty = card_frame()
        el = QVBoxLayout(self.empty)
        el.setContentsMargins(16, 12, 16, 12)
        self.lbl_empty = QLabel()
        self.lbl_empty.setWordWrap(True)
        self.btn_train = QPushButton()
        self.btn_train.setObjectName("primary")
        el.addWidget(self.lbl_empty)
        el.addWidget(self.btn_train)
        self.body.addWidget(self.empty)
        self.body.addStretch(1)
        self.lbl_rights = QLabel()
        self.lbl_rights.setObjectName("footer")
        self.lbl_rights.setWordWrap(True)
        self.body.addWidget(self.lbl_rights)

        self.act_folder.triggered.connect(self.import_folder_dialog)
        self.act_zip.triggered.connect(self.import_zip_dialog)
        self.btn_repo.clicked.connect(self.open_repository)
        self.btn_reload.clicked.connect(self.refresh_remote)
        self.btn_train.clicked.connect(lambda: self.go.emit("train"))
        self.retranslate()
        self.refresh()
        fit_to_screen(self, self.content, 760, 520)
        if auto_refresh:
            self.refresh_remote()

    def window_title(self) -> str:
        """Localized window title."""
        return tr("voices.title")

    def retranslate(self) -> None:
        """Apply the current language."""
        super().retranslate()
        self.lbl_intro.setText(tr("voices.intro"))
        self.btn_import.setText(tr("voices.import"))
        self.act_folder.setText(tr("voices.import_folder"))
        self.act_zip.setText(tr("voices.import_zip"))
        self.btn_repo.setText(tr("voices.repo_button"))
        self.btn_reload.setText(tr("voices.remote_refresh"))
        self.lbl_empty.setText(tr("voices.empty"))
        self.btn_train.setText(tr("voices.empty_train"))
        self.lbl_rights.setText(tr("voices.rights_note"))
        if hasattr(self, "cards_box"):
            self.refresh()

    # ------------------------------------------------------------------ list
    def refresh(self) -> None:
        """Rebuild the cards from the library."""
        while self.cards_box.count():
            it = self.cards_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        items = catalog.build(self.library, self.entries)
        voices = [i for i in items if i.installed]
        for it in voices:
            rec = cast(VoiceRecord, it.record)
            c = VoiceCard(rec, playing=(self.previewer.current is not None and rec.preview_path == self.previewer.current))
            c.preview.connect(self.toggle_preview)
            c.narrate.connect(self.narrate_with.emit)
            c.edit.connect(self.edit_voice)
            c.delete.connect(self.delete_voice)
            c.export.connect(self.export_voice)
            c.open_folder.connect(self.open_voice_folder)
            self.cards_box.addWidget(c)
        busy = self._dl_worker is not None and self._dl_worker.isRunning()
        for it in (i for i in items if not i.installed):
            rc = RemoteCard(it, busy=busy, progress=self._dl_progress if it.key == self._dl_key and busy else -1.0)
            rc.download.connect(self.download_remote)
            self.cards_box.addWidget(rc)
        self.empty.setVisible(not items)

    # ------------------------------------------------------------------ online voices
    def shutdown(self) -> None:
        """Application exit: wait for the index / download threads (a QThread destroyed while running aborts the process)."""
        for w in (self._dl_worker, self._index_worker):
            if w is not None and w.isRunning():
                if hasattr(w, "cancel"):
                    w.cancel()
                w.wait(3000)

    def refresh_remote(self) -> None:
        """Fetch the index in the background (does nothing while the repository is not configured)."""
        if not repo.is_configured() or (self._index_worker is not None and self._index_worker.isRunning()):
            return
        self.lbl_status.setText(tr("voices.repo_loading"))
        w = RepoIndexWorker(fetch=self._fetch, parent=self)
        w.done.connect(self._on_index)
        self._index_worker = w
        w.start()

    def _on_index(self, res: "repo.IndexResult") -> None:
        """New index (or the cached one when offline): rebuild the list."""
        if res.voices:
            self.entries = list(res.voices)
        if res.offline:
            self.lbl_status.setText(tr("voices.repo_offline"))
        elif res.error and res.error != "not_configured":
            self.lbl_status.setText(tr("voices.repo_unreachable"))
        else:
            self.lbl_status.setText("")
        self.refresh()

    def download_remote(self, key: str) -> None:
        """Download one index voice (SHA-256 checked, resumable) and add it to the library."""
        entry = catalog.find_entry(self.entries, key)
        if entry is None or (self._dl_worker is not None and self._dl_worker.isRunning()):
            return
        self._dl_key, self._dl_progress = key, 0.0
        w = RepoDownloadWorker([entry], self.library, download=self._download, parent=self)
        w.progress.connect(self._on_dl_progress)
        w.failed.connect(lambda m: self.lbl_status.setText(m))
        w.finished_all.connect(self._on_downloaded)
        self._dl_worker = w
        w.start()
        self.refresh()

    def _on_dl_progress(self, frac: float, _name: str) -> None:
        self._dl_progress = max(0.0, frac)
        self.lbl_status.setText(tr("voices.remote_downloading", p=int(self._dl_progress * 100)))

    def _on_downloaded(self, ids: list) -> None:
        self._dl_progress, self._dl_key = -1.0, ""
        if ids:
            self.lbl_status.setText(tr("voices.repo_downloaded", n=len(ids)))
            self.library_changed.emit()
        self.refresh()

    def card_ids(self) -> List[str]:
        """Ids of the cards currently shown (tests)."""
        return [self.cards_box.itemAt(i).widget().voice_id for i in range(self.cards_box.count())
                if self.cards_box.itemAt(i).widget() is not None]

    def _error(self, exc: DatasetMakerError) -> None:
        """Show a library error in the status line."""
        self.lbl_status.setText(exc.user_message)
        log.warning("voice library: %s (%s)", exc.user_message, exc.details)

    # ------------------------------------------------------------------ actions
    def toggle_preview(self, voice_id: str) -> None:
        """Play the voice's sample, or stop it if it is playing."""
        rec = self.library.get(voice_id)
        if rec is None or rec.preview_path is None:
            return
        if self.previewer.current == rec.preview_path:
            self.previewer.stop()
            self.refresh()
            return
        self.previewer.play(rec.preview_path)
        self.refresh()

    def import_folder_dialog(self) -> None:
        """Ask for an adapter folder and import it."""
        path = self._pick_folder() if self._pick_folder else QFileDialog.getExistingDirectory(
            self, tr("voices.import_folder"))
        if path:
            self.import_path(Path(path))

    def import_zip_dialog(self) -> None:
        """Ask for a ``.zip`` and import it."""
        if self._pick_zip:
            path = self._pick_zip()
        else:
            path, _ = QFileDialog.getOpenFileName(self, tr("voices.import_zip"), "", "ZIP (*.zip)")
        if path:
            self.import_path(Path(path))

    def import_path(self, path: Path) -> Optional[VoiceRecord]:
        """Import a folder or zip into the library; the result (or error) is shown in the status line."""
        try:
            rec = self.library.import_zip(path) if path.is_file() else self.library.import_folder(path)
        except DatasetMakerError as exc:
            self._error(exc)
            return None
        self.lbl_status.setText(tr("voices.imported", name=rec.name))
        self.refresh()
        self.library_changed.emit()
        return rec

    def edit_voice(self, voice_id: str) -> None:
        """Open the edit dialog and save the changes."""
        rec = self.library.get(voice_id)
        if rec is None or rec.bundled:
            return
        dlg = VoiceEditDialog(rec, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.apply_edit(voice_id, dlg.values())

    def apply_edit(self, voice_id: str, values: dict) -> None:
        """Save edited details (separate from the dialog so it can be tested without ``exec``)."""
        try:
            self.library.update(voice_id, **values)
        except DatasetMakerError as exc:
            self._error(exc)
            return
        self.refresh()
        self.library_changed.emit()

    def export_voice(self, voice_id: str) -> None:
        """Ask what to export: the small voice file (default) or a full standalone model (~4 GB, built separately)."""
        rec = self.library.get(voice_id)
        if rec is None:
            return
        mb = max(1, round(self.library.size_bytes(voice_id) / 1024 ** 2))
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(tr("voices.export_title"))
        box.setText(tr("voices.export_text", name=rec.name, mb=mb))
        small = box.addButton(tr("voices.export_small", mb=mb), QMessageBox.ButtonRole.AcceptRole)
        full = box.addButton(tr("voices.export_full"), QMessageBox.ButtonRole.ActionRole)
        box.addButton(tr("voices.cancel"), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(small)
        box.exec()
        if box.clickedButton() is full:
            self.full_model_requested.emit(str(rec.path))
            return
        if box.clickedButton() is not small:
            return
        default = str(self.library.root / f"{rec.id}.zip")      # the voices folder, not Downloads
        path, _ = QFileDialog.getSaveFileName(self, tr("voices.export_title"), default, "Voxprint voice (*.zip)")
        if path:
            self.do_export(voice_id, Path(path))

    def do_export(self, voice_id: str, dest: Path) -> Optional[Path]:
        """Write the export (separate from the dialogs so it can be tested)."""
        try:
            return self.library.export_zip(voice_id, dest)
        except (DatasetMakerError, OSError) as exc:
            self._error(exc) if isinstance(exc, DatasetMakerError) else log.warning("export failed: %s", exc)
            return None

    def open_voice_folder(self, voice_id: str) -> None:
        """Show the voice's folder in the file manager."""
        rec = self.library.get(voice_id)
        if rec is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(rec.path)))

    def delete_voice(self, voice_id: str) -> None:
        """Delete a voice after asking for confirmation."""
        rec = self.library.get(voice_id)
        if rec is None or rec.bundled:
            return
        if not self._ask(tr("voices.delete_title"), tr("voices.delete_confirm", name=rec.name)):
            return
        try:
            if self.previewer.current and rec.path in self.previewer.current.parents:
                self.previewer.stop()
            self.library.delete(voice_id)
        except DatasetMakerError as exc:
            self._error(exc)
            return
        self.lbl_status.setText(tr("voices.deleted", name=rec.name))
        self.refresh()
        self.library_changed.emit()

    def _ask(self, title: str, text: str) -> bool:
        """Yes/No question; tests inject ``confirm``; under offscreen the default answer is *no* (never blocks)."""
        if self._confirm is not None:
            return bool(self._confirm(title, text))
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return False
        from ui import std_buttons

        return std_buttons.question(self, title, text)

    def open_repository(self) -> None:
        """Show the repository dialog (non-blocking offscreen)."""
        if self._repo_dialog_factory:
            dlg = self._repo_dialog_factory(self.library, self)
        else:
            dlg = RepoDialog(self.library, self)
        def on_added(_ids) -> None:
            self.refresh()
            self.library_changed.emit()

        dlg.voices_added.connect(on_added)
        self.repo_dialog = dlg
        dlg.refresh()
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            dlg.show()
        else:
            dlg.exec()
