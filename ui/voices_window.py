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
from typing import Any, Callable, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMenu, QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget)

from core import voice_info
from core.errors import DatasetMakerError
from core.i18n import tr
from core.voice_library import VoiceLibrary, VoiceRecord
from infra import voice_repository as repo
from ui.audio_preview import Previewer
from ui.window_base import SubWindow, card_frame, fit_to_screen, hint_label
from workers.narrate_worker import RepoDownloadWorker, RepoIndexWorker

log = logging.getLogger("voxprint.ui.voices")


def voice_type_label(code: str) -> str:
    """Localized name of a ``voice_type`` value ("" = not set)."""
    labels = {"": tr("ui.voice_type_none"), "male": tr("ui.voice_type_male"), "female": tr("ui.voice_type_female"),
              "child": tr("ui.voice_type_child"), "other": tr("ui.voice_type_other")}
    return labels.get(code, labels[""])


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
    parts = [rec.language.capitalize() if rec.language else "", duration_text(float(rec.info.get("duration", 0))),
             tr("voices.epochs", n=int(rec.info.get("epochs", 0)))]
    if rec.info.get("voice_type"):
        parts.append(voice_type_label(str(rec.info["voice_type"])))
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


class VoiceCard(QFrame):
    """One voice of the library."""

    preview = Signal(str)
    narrate = Signal(str)
    edit = Signal(str)
    delete = Signal(str)

    def __init__(self, rec: VoiceRecord, playing: bool = False) -> None:
        """Build the card for ``rec``; ``playing`` shows the stop symbol on the preview button."""
        super().__init__()
        self.setObjectName("card")
        self.voice_id = rec.id
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
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
        self.lbl_desc = QLabel(str(rec.info.get("description", "")))
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setVisible(bool(rec.info.get("description")))
        lay.addWidget(self.lbl_desc)
        row = QHBoxLayout()
        self.btn_preview = QPushButton(("\u25a0 " + tr("voices.stop")) if playing else ("\u25b6 " + tr("voices.preview")))
        self.btn_preview.setEnabled(rec.preview_path is not None)
        self.btn_preview.setToolTip(tr("voices.preview_tip") if rec.preview_path else tr("voices.no_preview"))
        self.btn_narrate = QPushButton(tr("voices.narrate"))
        self.btn_edit = QPushButton(tr("voices.edit"))
        self.btn_delete = QPushButton(tr("voices.delete"))
        for b in (self.btn_preview, self.btn_narrate, self.btn_edit, self.btn_delete):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.btn_preview.clicked.connect(lambda: self.preview.emit(self.voice_id))
        self.btn_narrate.clicked.connect(lambda: self.narrate.emit(self.voice_id))
        self.btn_edit.clicked.connect(lambda: self.edit.emit(self.voice_id))
        self.btn_delete.clicked.connect(lambda: self.delete.emit(self.voice_id))


class VoiceEditDialog(QDialog):
    """Edit name, author, licence, voice type and description of a voice."""

    def __init__(self, rec: VoiceRecord, parent: Optional[QWidget] = None) -> None:
        """Fill the fields from ``rec``; :meth:`values` returns what the user entered."""
        super().__init__(parent)
        self.setObjectName("root")
        self.setMinimumWidth(480)
        if parent is not None:
            self.setStyleSheet(parent.styleSheet())
        self.setWindowTitle(tr("voices.edit_title"))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(10)
        self.edt_name = QLineEdit(rec.name)
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
        self.cmb_type = QComboBox()
        for code in ("",) + voice_info.VOICE_TYPES:
            self.cmb_type.addItem(voice_type_label(code), code)
        self.cmb_type.setCurrentIndex(max(0, self.cmb_type.findData(str(rec.info.get("voice_type", "")))))
        self.edt_desc = QLineEdit(str(rec.info.get("description", "")))
        self.edt_desc.setMaxLength(voice_info.MAX_DESCRIPTION_CHARS)
        for label, w in ((tr("voices.field_name"), self.edt_name), (tr("voices.field_author"), self.edt_author),
                         (tr("voices.field_license"), self.cmb_license), (None, self.lbl_license_note),
                         (tr("voices.field_type"), self.cmb_type), (tr("voices.field_description"), self.edt_desc)):
            if label:
                lay.addWidget(QLabel(label))
            lay.addWidget(w)
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
        self.lbl_license_note.setText(tr("voices.license_commercial") if voice_info.license_allows_commercial(lic)
                                      else tr("voices.license_personal"))

    def values(self) -> dict:
        """The edited fields, ready for :meth:`VoiceLibrary.update`."""
        return {"name": self.edt_name.text(), "author": self.edt_author.text(),
                "license": str(self.cmb_license.currentData()), "voice_type": str(self.cmb_type.currentData()),
                "description": self.edt_desc.text()}


class RepoDialog(QDialog):
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
        lay.setContentsMargins(24, 20, 24, 20)
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
            it = QListWidgetItem(f"{e.name}  \u00b7  {e.language}  \u00b7  {badge_text(e.license, e.commercial_use)}")
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

    def __init__(self, library: Optional[VoiceLibrary] = None, previewer: Optional[Previewer] = None,
                 confirm: Optional[Callable[[str, str], bool]] = None,
                 pick_folder: Optional[Callable[[], str]] = None, pick_zip: Optional[Callable[[], str]] = None,
                 repo_dialog_factory: Optional[Callable[..., RepoDialog]] = None) -> None:
        """Build the window; every dialog/IO collaborator can be replaced for tests."""
        super().__init__(with_back=True)
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
        bar.addStretch(1)
        self.body.addLayout(bar)
        self.lbl_status = QLabel()
        self.lbl_status.setObjectName("status")
        self.lbl_status.setWordWrap(True)
        self.body.addWidget(self.lbl_status)
        self.cards_box = QVBoxLayout()
        self.cards_box.setSpacing(12)
        self.body.addLayout(self.cards_box)
        # empty state
        self.empty = card_frame()
        el = QVBoxLayout(self.empty)
        el.setContentsMargins(20, 18, 20, 18)
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
        self.btn_train.clicked.connect(lambda: self.go.emit("train"))
        self.retranslate()
        self.refresh()
        fit_to_screen(self, self.content, 760, 520)

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
        voices = self.library.list_voices()
        for rec in voices:
            c = VoiceCard(rec, playing=(self.previewer.current is not None and rec.preview_path == self.previewer.current))
            c.preview.connect(self.toggle_preview)
            c.narrate.connect(self.narrate_with.emit)
            c.edit.connect(self.edit_voice)
            c.delete.connect(self.delete_voice)
            self.cards_box.addWidget(c)
        self.empty.setVisible(not voices)

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
        if rec is None:
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

    def delete_voice(self, voice_id: str) -> None:
        """Delete a voice after asking for confirmation."""
        rec = self.library.get(voice_id)
        if rec is None:
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
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def open_repository(self) -> None:
        """Show the repository dialog (non-blocking offscreen)."""
        if self._repo_dialog_factory:
            dlg = self._repo_dialog_factory(self.library, self)
        else:
            dlg = RepoDialog(self.library, self)
        dlg.voices_added.connect(lambda ids: (self.refresh(), self.library_changed.emit()))
        self.repo_dialog = dlg
        dlg.refresh()
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            dlg.show()
        else:
            dlg.exec()
