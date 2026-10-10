"""The "Re-voice" window: one recording or audio file, a voice from the library, then one of two actions.

* **Convert to text** - the installed speech recogniser, an editable text, then *Voice this text* (the narrator, that voice).
* **Re-voice recording** - direct conversion into the chosen voice (timing and intonation stay). The OpenVoice V2 model
  (~130 MB) is part of the standard first-run download; when it is still missing the button is NOT grey: a click fetches it
  first and then converts (*Download model* does only the download).

Recording uses Qt Multimedia (record / stop / play). Recognition and conversion run on worker threads. Nothing here
downloads a model without a click.
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core import revoice, voice_convert
from core.errors import CancelledByUser
from core.events import CancelToken
from core.i18n import tr
from core.voice_library import VoiceLibrary
from infra import vc_model
from ui.audio_preview import Previewer
from ui.suite_icons import apply_button
from ui.window_base import SubWindow, card_frame, fit_to_screen, hint_label

log = logging.getLogger("voxprint.revoice")


def default_asr():
    """The installed recogniser (0.6B or 1.7B, whichever is there), or None - this window never downloads it."""
    from core.asr import make_default_asr
    from infra import asr_choice

    found = asr_choice.ready()
    return make_default_asr(str(found[1]), "auto") if found is not None else None


def revoice_dir() -> Path:
    """``<projects folder>/Re-voice``: recordings, converted audio and the texts handed to the narrator."""
    from infra import projects

    return projects.sub(projects.REVOICE)


def record_format():
    """The dictaphone's format: Ogg Opus when Qt can encode it, else FLAC, WAV or M4A (the first the backend offers).

    Windows' media backend usually cannot encode Opus, so the recording is FLAC and :func:`core.revoice.to_opus` turns it
    into Opus. The models only ever see PCM decoded in memory. Previews play the Opus file.
    """
    from PySide6.QtMultimedia import QMediaFormat

    enc = QMediaFormat.ConversionMode.Encode
    probe = QMediaFormat()
    supported = probe.supportedFileFormats(enc)
    if QMediaFormat.FileFormat.Ogg in supported:
        fmt = QMediaFormat(QMediaFormat.FileFormat.Ogg)
        if QMediaFormat.AudioCodec.Opus in fmt.supportedAudioCodecs(enc):
            fmt.setAudioCodec(QMediaFormat.AudioCodec.Opus)
            return fmt
    for ff in (QMediaFormat.FileFormat.FLAC, QMediaFormat.FileFormat.Wave, QMediaFormat.FileFormat.Mpeg4Audio):
        if ff in supported:
            return QMediaFormat(ff)
    return probe


class TranscribeWorker(QThread):
    """Runs :func:`core.revoice.transcribe_files`.  Signals: ``progress(fraction, file title)``, ``done(text)``, ``failed(msg)``."""

    progress = Signal(float, str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, files: List[Path], asr_factory: Callable[[], Any], parent=None) -> None:
        super().__init__(parent)
        self.files, self.asr_factory, self.cancel = list(files), asr_factory, CancelToken()

    def run(self) -> None:  # noqa: D401
        asr = None
        try:
            asr = self.asr_factory()
            if asr is None:
                self.failed.emit(tr("revoice.no_asr"))
                return
            chapters = revoice.transcribe_files(self.files, asr, None, lambda f, t: self.progress.emit(f, t), self.cancel)
            self.done.emit(revoice.to_text(chapters))
        except CancelledByUser:
            self.failed.emit(tr("revoice.cancelled"))
        except Exception as exc:  # noqa: BLE001 - unreadable file, recogniser error: shown, logged with traceback
            log.exception("re-voice transcription failed")
            self.failed.emit(str(exc))
        finally:
            if asr is not None:
                asr.unload()


class ConvertWorker(QThread):
    """Direct conversion of one recording.  Signals: ``progress(fraction)``, ``done(opus path)``, ``failed(msg)``."""

    progress = Signal(float)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, source: Path, reference: Path, out_dir: Path, factory: Optional[Callable[[], Any]], parent=None) -> None:
        super().__init__(parent)
        self.source, self.reference, self.out_dir = source, reference, out_dir
        self.factory, self.cancel = factory, CancelToken()

    def run(self) -> None:  # noqa: D401
        converter = None
        try:
            converter = voice_convert.make_converter(self.factory)
            path = voice_convert.convert_file(
                self.source, self.reference, self.out_dir, converter, cancel=self.cancel,
                progress=lambda f: self.progress.emit(f))
            self.done.emit(str(path))
        except CancelledByUser:
            self.failed.emit(tr("revoice.cancelled"))
        except Exception as exc:  # noqa: BLE001
            log.exception("direct re-voice failed")
            self.failed.emit(getattr(exc, "user_message", None) or str(exc))
        finally:
            if converter is not None:
                converter.unload()


class DownloadWorker(QThread):
    """Fetches the optional converter. ``done("")`` on success, ``done(error)`` otherwise."""

    progress = Signal(float, str)
    done = Signal(str)

    def __init__(self, ensure: Callable[..., Any], parent=None) -> None:
        super().__init__(parent)
        self.ensure = ensure

    def run(self) -> None:  # noqa: D401
        try:
            self.ensure(progress=lambda f, m="": self.progress.emit(f, m))
            self.done.emit("")
        except Exception as exc:  # noqa: BLE001 - download errors are shown on the status line
            log.exception("voice-conversion model download failed")
            self.done.emit(str(exc))


class RevoiceWindow(SubWindow):
    """Record or choose a file, pick a voice, then convert to text or re-voice the recording directly."""

    narrate_file = Signal(str)

    def __init__(self, asr_factory: Callable[[], Any] = default_asr, pick_files: Optional[Callable[[], List[str]]] = None,
                 out_dir: Optional[Path] = None, previewer: Optional[Previewer] = None,
                 library: Optional[VoiceLibrary] = None, vc_factory: Optional[Callable[[], Any]] = None,
                 vc_status: Optional[Callable[[], str]] = None, vc_ensure: Optional[Callable[..., Any]] = None) -> None:
        super().__init__(with_back=True)
        self.asr_factory, self._pick_files = asr_factory, pick_files
        self.out_dir = Path(out_dir) if out_dir else revoice_dir()
        self.previewer = previewer or Previewer(self)
        self.library = library
        self.vc_factory = vc_factory
        self.vc_status = vc_status or (lambda: "ready" if vc_model.ready() else "needs_download")
        self.vc_ensure = vc_ensure or vc_model.ensure
        self.files: List[Path] = []
        self.output: Optional[Path] = None          # the Opus file of the last direct conversion (what Play previews)
        self.worker: Optional[TranscribeWorker] = None
        self.converter: Optional[ConvertWorker] = None
        self.downloader: Optional[DownloadWorker] = None
        self._rec = None
        self._session = None
        self._input = None
        self._build()
        self.retranslate()
        fit_to_screen(self, self.content, 860, 620)

    # ------------------------------------------------------------------ construction
    def _build(self) -> None:
        c = card_frame()
        v = QVBoxLayout(c)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(6)
        self.lbl_audio = QLabel()
        self.lbl_audio.setObjectName("sectiontitle")
        v.addWidget(self.lbl_audio)
        self.lbl_audio_hint = hint_label()
        v.addWidget(self.lbl_audio_hint)
        row = QHBoxLayout()
        self.btn_record = QPushButton()
        self.btn_play = QPushButton()
        self.btn_add = QPushButton()
        self.btn_remove = QPushButton()
        for b in (self.btn_record, self.btn_play, self.btn_add, self.btn_remove):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.lst_files = QListWidget()
        self.lst_files.setMaximumHeight(72)
        self.lst_files.setVisible(False)
        v.addWidget(self.lst_files)
        self.lbl_rec = hint_label()
        v.addWidget(self.lbl_rec)
        self.body.addWidget(c)

        c = card_frame()
        v = QVBoxLayout(c)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(6)
        self.lbl_voice = QLabel()
        self.lbl_voice.setObjectName("sectiontitle")
        v.addWidget(self.lbl_voice)
        self.cmb_voice = QComboBox()
        self.cmb_voice.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_voice.setMinimumContentsLength(12)
        v.addWidget(self.cmb_voice)
        self.lbl_no_voice = hint_label()
        v.addWidget(self.lbl_no_voice)
        actions = QHBoxLayout()
        actions.setSpacing(12)
        left, right = QWidget(), QWidget()
        lv, rv = QVBoxLayout(left), QVBoxLayout(right)
        for box in (lv, rv):
            box.setContentsMargins(0, 4, 0, 0)
            box.setSpacing(2)
        self.btn_to_text = QPushButton()
        self.lbl_to_text = hint_label()
        lv.addWidget(self.btn_to_text)
        lv.addWidget(self.lbl_to_text)
        self.btn_direct = QPushButton()
        self.lbl_direct = hint_label()
        self.btn_vc_download = QPushButton()
        apply_button(self.btn_vc_download, "download")
        for b in (self.btn_to_text, self.btn_direct, self.btn_vc_download):
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        rv.addWidget(self.btn_direct)
        rv.addWidget(self.lbl_direct)
        rv.addWidget(self.btn_vc_download)
        actions.addWidget(left, 1)
        actions.addWidget(right, 1)
        v.addLayout(actions)
        self.body.addWidget(c)

        c = card_frame()
        v = QVBoxLayout(c)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(6)
        self.lbl_text = QLabel()
        self.lbl_text.setObjectName("sectiontitle")
        v.addWidget(self.lbl_text)
        row = QHBoxLayout()
        self.btn_cancel = QPushButton()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.progress, 1)
        v.addLayout(row)
        self.lbl_state = hint_label()
        v.addWidget(self.lbl_state)
        self.ed_text = QPlainTextEdit()
        self.ed_text.setMinimumHeight(110)
        v.addWidget(self.ed_text)
        self.btn_narrate = QPushButton()
        self.btn_narrate.setObjectName("primary")
        v.addWidget(self.btn_narrate)
        self.btn_advanced = QToolButton()
        self.btn_advanced.setObjectName("expander")
        self.btn_advanced.setCheckable(True)
        v.addWidget(self.btn_advanced)
        self.advanced_box = QWidget()
        dv = QVBoxLayout(self.advanced_box)
        dv.setContentsMargins(0, 0, 0, 0)
        dv.setSpacing(4)
        self.lbl_advanced_hint = hint_label()
        dv.addWidget(self.lbl_advanced_hint)
        nrow = QHBoxLayout()
        self.lbl_name = QLabel()
        self.ed_name = QLineEdit()
        nrow.addWidget(self.lbl_name)
        nrow.addWidget(self.ed_name, 1)
        dv.addLayout(nrow)
        self.advanced_box.setVisible(False)
        v.addWidget(self.advanced_box)
        self.body.addWidget(c)
        self.body.addStretch(1)

        self.btn_record.clicked.connect(self.toggle_record)
        self.btn_play.clicked.connect(self.play_selected)
        self.btn_add.clicked.connect(self.add_files)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_to_text.clicked.connect(self.transcribe)
        self.btn_direct.clicked.connect(self.convert_direct)
        self.btn_vc_download.clicked.connect(self.download_model)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_narrate.clicked.connect(self.narrate)
        self.btn_advanced.toggled.connect(self._on_advanced)
        self.ed_text.textChanged.connect(self._refresh)
        self.lst_files.currentRowChanged.connect(lambda _r: self._refresh())
        self.cmb_voice.currentIndexChanged.connect(lambda _i: self._refresh())

    def window_title(self) -> str:
        return tr("revoice.title")

    def retranslate(self) -> None:
        super().retranslate()
        self.lbl_audio.setText(tr("revoice.audio"))
        self.lbl_audio_hint.setText(tr("revoice.audio_hint"))
        self.btn_play.setText(tr("revoice.play"))
        self.btn_add.setText(tr("revoice.add"))
        self.btn_remove.setText(tr("revoice.remove"))
        self.lbl_voice.setText(tr("revoice.voice"))
        self.lbl_no_voice.setText(tr("narr.no_voice"))
        self.btn_to_text.setText(tr("revoice.to_text"))
        self.lbl_to_text.setText(tr("revoice.to_text_hint"))
        self.btn_direct.setText(tr("revoice.direct"))
        self.lbl_text.setText(tr("revoice.text"))
        self.btn_cancel.setText(tr("ui.cancel"))
        self.lbl_name.setText(tr("revoice.name"))
        self.lbl_advanced_hint.setText(tr("revoice.advanced_hint"))
        self.btn_narrate.setText(tr("revoice.narrate"))
        self.ed_text.setPlaceholderText(tr("revoice.text_hint"))
        self._on_advanced(self.btn_advanced.isChecked())
        self.refresh_voices()

    def _on_advanced(self, open_: bool) -> None:
        """Expand or collapse the title and the note about several files."""
        self.advanced_box.setVisible(open_)
        self.btn_advanced.setText(("\u25be " if open_ else "\u25b8 ") + tr("narr.advanced"))

    # ------------------------------------------------------------------ state
    @property
    def busy(self) -> bool:
        running = lambda w: bool(w and w.isRunning())
        return running(self.worker) or running(self.converter) or running(self.downloader)

    def _vc_ready(self) -> bool:
        return self.vc_status() == "ready"

    def selected_voice_id(self) -> str:
        """Id of the voice chosen for both actions ("" when the library is empty)."""
        return str(self.cmb_voice.currentData() or "")

    def _voice_record(self):
        vid = self.selected_voice_id()
        if self.library is None or not vid:
            return None
        return self.library.get(vid)

    def _has_ref(self) -> bool:
        rec = self._voice_record()
        return bool(rec and rec.preview_path)

    def _current_file(self) -> Optional[Path]:
        """The recording the direct converter uses: the selected row, else the only file."""
        row = self.lst_files.currentRow()
        if 0 <= row < len(self.files):
            return self.files[row]
        return self.files[0] if len(self.files) == 1 else None

    def refresh_voices(self, select: str = "") -> None:
        """Reload the library into the combo (keeps the selection when it still exists)."""
        current = select or self.selected_voice_id()
        self.cmb_voice.blockSignals(True)
        self.cmb_voice.clear()
        if self.library is not None:
            for rec in self.library.list_voices():
                self.cmb_voice.addItem(rec.name, rec.id)
        idx = self.cmb_voice.findData(current)
        self.cmb_voice.setCurrentIndex(idx if idx >= 0 else 0)
        self.cmb_voice.blockSignals(False)
        self._refresh()

    def _refresh(self) -> None:
        busy, recording = self.busy, self._rec is not None
        ready, has_voice = self._vc_ready(), bool(self.selected_voice_id())
        self.btn_record.setText(tr("revoice.stop") if recording else tr("revoice.record"))
        self.btn_record.setEnabled(not busy)
        can_play = (self.output is not None or self._current_file() is not None) and not recording
        self.btn_play.setEnabled(can_play)
        self.btn_add.setEnabled(not busy and not recording)
        self.btn_remove.setEnabled(not busy and self.lst_files.currentRow() >= 0)
        self.btn_to_text.setEnabled(not busy and not recording and bool(self.files))
        self.btn_direct.setEnabled(not busy and not recording and has_voice and self._has_ref()
                                   and self._current_file() is not None)
        self.btn_vc_download.setVisible(not ready)
        self.btn_vc_download.setEnabled(not busy)
        try:
            mb = vc_model.download_mb()
        except Exception:  # noqa: BLE001 - a broken manifest must not blank the window
            mb = 0
        self.btn_vc_download.setText(tr("revoice.vc_download", mb=mb))
        if has_voice and not self._has_ref():
            self.lbl_direct.setText(tr("revoice.no_ref"))
        else:
            self.lbl_direct.setText(tr("revoice.direct_hint"))
        self.btn_cancel.setVisible(busy)
        self.progress.setVisible(busy)
        has = self.cmb_voice.count() > 0
        self.cmb_voice.setVisible(has)
        self.lbl_no_voice.setVisible(not has)
        self.btn_narrate.setEnabled(not busy and has_voice and bool(self.ed_text.toPlainText().strip()))

    def set_files(self, files: List[Path]) -> None:
        self.files = [Path(f) for f in files]
        self.output = None
        self.lst_files.clear()
        self.lst_files.addItems([f.name for f in self.files])
        self.lst_files.setVisible(bool(self.files))
        if self.files:
            self.lst_files.setCurrentRow(0)
        if not self.ed_name.text().strip() and self.files:
            self.ed_name.setText(revoice.chapter_title(self.files[0]))
        self._refresh()

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel.cancel()
        if self.converter is not None:
            self.converter.cancel.cancel()

    # ------------------------------------------------------------------ audio
    def add_files(self) -> None:
        if self._pick_files is not None:
            picked = self._pick_files()
        else:
            pattern = " ".join("*" + e for e in revoice.AUDIO_EXTENSIONS)
            picked, _ = QFileDialog.getOpenFileNames(self, tr("revoice.add"), "", f"{tr('revoice.audio_files')} ({pattern})")
        if picked:
            self.set_files(self.files + sorted(Path(p) for p in picked))

    def remove_selected(self) -> None:
        r = self.lst_files.currentRow()
        if r >= 0:
            self.set_files(self.files[:r] + self.files[r + 1:])

    def _play_path(self) -> Optional[Path]:
        """Play previews the last direct conversion when there is one, otherwise the selected recording."""
        if self.output is not None and self.output.is_file():
            return self.output
        return self._current_file()

    def play_selected(self) -> None:
        path = self._play_path()
        if path is not None:
            self.previewer.play(path)

    def toggle_record(self) -> None:
        """Start recording from the default microphone, or stop and add the recording to the list."""
        from PySide6.QtMultimedia import QAudioInput, QMediaCaptureSession, QMediaRecorder

        if self._rec is not None:
            self._rec.stop()
            return
        self.previewer.stop()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        session, audio_in, rec = QMediaCaptureSession(), QAudioInput(), QMediaRecorder()
        self._session, self._input = session, audio_in
        # PySide6 stubs type these constructors as None.
        if session is None or audio_in is None or rec is None:
            return
        session.setAudioInput(audio_in)
        session.setRecorder(rec)
        rec.setMediaFormat(record_format())
        rec.setQuality(QMediaRecorder.Quality.HighQuality)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        rec.setOutputLocation(QUrl.fromLocalFile(str(self.out_dir / f"recording {stamp}")))
        rec.recorderStateChanged.connect(self._on_rec_state)
        rec.errorOccurred.connect(lambda _e, msg: self.lbl_rec.setText(tr("revoice.rec_failed", error=msg)))
        self._rec = rec
        rec.record()
        self.lbl_rec.setText(tr("revoice.recording"))
        self._refresh()

    def _on_rec_state(self, state) -> None:
        from PySide6.QtMultimedia import QMediaRecorder

        if state != QMediaRecorder.RecorderState.StoppedState or self._rec is None:
            return
        path = Path(self._rec.actualLocation().toLocalFile())
        self._rec = self._session = self._input = None
        if path.is_file() and path.stat().st_size > 0:
            from core import audio_utils

            path = revoice.to_opus(path, audio_utils.ensure_ffmpeg())
            self.set_files(self.files + [path])
            self.lst_files.setCurrentRow(len(self.files) - 1)
            self.lbl_rec.setText(tr("revoice.recorded", name=path.name))
        self._refresh()

    # ------------------------------------------------------------------ recognition, direct conversion, hand-over
    def transcribe(self) -> bool:
        if self.busy or not self.files:
            return False
        w = TranscribeWorker(self.files, self.asr_factory, self)
        w.progress.connect(lambda f, t: (self.progress.setValue(int(f * 100)),
                                         self.lbl_state.setText(tr("revoice.transcribing", name=t))))
        w.done.connect(self._on_done)
        def on_failed(message) -> None:
            self.lbl_state.setText(tr("revoice.failed", error=message))
            self._refresh()

        w.failed.connect(on_failed)
        w.finished.connect(self._refresh)
        self.worker = w
        self.progress.setValue(0)
        w.start()
        self._refresh()
        return True

    def _on_done(self, text: str) -> None:
        self.ed_text.setPlainText(text)
        self.lbl_state.setText(tr("revoice.done"))
        self._refresh()

    def convert_direct(self) -> bool:
        """Convert the selected recording into the chosen voice. No text step."""
        src = self._current_file()
        rec = self._voice_record()
        if self.busy or src is None or rec is None or not rec.preview_path:
            return False
        if not self._vc_ready():          # the small model is standard; if it is still missing, fetch it, then convert
            return self.download_model(then_convert=True)
        w = ConvertWorker(src, rec.preview_path, self.out_dir, self.vc_factory, self)
        w.progress.connect(lambda f: self.progress.setValue(int(f * 100)))
        w.done.connect(self._on_converted)
        def on_failed(message) -> None:
            self.lbl_state.setText(tr("revoice.vc_failed", error=message))
            self._refresh()

        w.failed.connect(on_failed)
        w.finished.connect(self._refresh)
        self.converter = w
        self.progress.setValue(0)
        self.lbl_state.setText(tr("revoice.converting", name=rec.name))
        w.start()
        self._refresh()
        return True

    def _on_converted(self, path: str) -> None:
        self.output = Path(path)
        self.lbl_state.setText(tr("revoice.converted", name=self.output.name))
        self.lbl_rec.setText(tr("revoice.converted", name=self.output.name))
        self._refresh()

    def download_model(self, then_convert: bool = False) -> bool:
        """Fetch the converter (``then_convert``: and convert right after). Nothing while a job runs or the model is there."""
        if self.busy or self._vc_ready():
            return False
        self._convert_after = then_convert
        w = DownloadWorker(self.vc_ensure, self)
        w.progress.connect(lambda f, m: (self.progress.setValue(int(f * 100)), self.lbl_state.setText(m or self.lbl_state.text())))
        w.done.connect(self._on_downloaded)
        w.finished.connect(self._refresh)
        self.downloader = w
        self.progress.setValue(0)
        w.start()
        self._refresh()
        return True

    def _on_downloaded(self, error: str) -> None:
        if error:
            self.lbl_state.setText(tr("revoice.download_failed", error=error))
        else:
            self.lbl_state.setText(tr("revoice.download_ok"))
        self._refresh()
        if not error and getattr(self, "_convert_after", False):
            self._convert_after = False
            from PySide6.QtCore import QTimer

            QTimer.singleShot(0, self.convert_direct)   # after the download worker has finished (busy is False then)

    def narrate(self) -> Optional[Path]:
        """Save the edited text and ask the Studio to open it in the narrator with the chosen voice."""
        text = self.ed_text.toPlainText().strip()
        if not text or not self.selected_voice_id():
            return None
        p = revoice.save_text(text + "\n", self.out_dir, self.ed_name.text().strip() or "Re-voice")
        self.narrate_file.emit(str(p))
        return p

    def shutdown(self) -> None:
        if self._rec is not None:
            self._rec.stop()
        for w in (self.worker, self.converter):
            if w is not None and w.isRunning():
                w.cancel.cancel()
                w.wait(30000)
        if self.downloader is not None and self.downloader.isRunning():
            self.downloader.wait(30000)
