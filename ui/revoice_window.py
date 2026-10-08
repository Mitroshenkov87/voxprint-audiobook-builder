"""The "Re-voice" window: record from the microphone or import audio files -> speech recognition -> editable text ->
the normal narrator with a voice of your choice (see :mod:`core.revoice`).  Recording uses Qt Multimedia (record / stop /
play); recognition runs on a worker thread with the installed Qwen3-ASR model (nothing is downloaded here)."""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from core import revoice
from core.errors import CancelledByUser
from core.events import CancelToken
from core.i18n import tr
from ui.audio_preview import Previewer
from ui.window_base import SubWindow, card_frame, fit_to_screen, hint_label

log = logging.getLogger("voxprint.revoice")


def default_asr():
    """The installed recogniser (0.6B or 1.7B, whichever is there), or None - this window never downloads."""
    from core.asr import make_default_asr
    from infra import asr_choice

    found = asr_choice.ready()
    return make_default_asr(str(found[1]), "auto") if found is not None else None


def revoice_dir() -> Path:
    """``<projects folder>/Re-voice``: recordings and the texts handed to the narrator (:mod:`infra.projects`)."""
    from infra import projects

    return projects.sub(projects.REVOICE)


def record_format():
    """The dictaphone's format: Ogg Opus (small; speech needs no lossless copy, the models get PCM decoded in memory),
    else FLAC, WAV or M4A - whatever the platform's Qt Multimedia backend can encode first."""
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
        if ff in supported:                         # ffmpeg / soundfile read any of them
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
                asr.unload()                        # free the GPU before the narrator loads the voice


class RevoiceWindow(SubWindow):
    """Record / import -> recognise -> edit -> ``narrate_file(path)`` (the Studio opens the narrator with that text)."""

    narrate_file = Signal(str)

    def __init__(self, asr_factory: Callable[[], Any] = default_asr, pick_files: Optional[Callable[[], List[str]]] = None,
                 out_dir: Optional[Path] = None, previewer: Optional[Previewer] = None) -> None:
        super().__init__(with_back=True)
        self.asr_factory, self._pick_files = asr_factory, pick_files
        self.out_dir = Path(out_dir) if out_dir else revoice_dir()
        self.previewer = previewer or Previewer(self)
        self.files: List[Path] = []
        self.worker: Optional[TranscribeWorker] = None
        self._rec = None                            # QMediaRecorder while recording
        self._session = None
        self._input = None
        self._build()
        self.retranslate()
        fit_to_screen(self, self.content, 820, 560)

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
        self.lst_files.setMaximumHeight(140)
        v.addWidget(self.lst_files)
        self.lbl_rec = hint_label()
        v.addWidget(self.lbl_rec)
        self.body.addWidget(c)

        c = card_frame()
        v = QVBoxLayout(c)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(6)
        self.lbl_text = QLabel()
        self.lbl_text.setObjectName("sectiontitle")
        v.addWidget(self.lbl_text)
        row = QHBoxLayout()
        self.btn_transcribe = QPushButton()
        self.btn_cancel = QPushButton()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        row.addWidget(self.btn_transcribe)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.progress, 1)
        v.addLayout(row)
        self.lbl_state = hint_label()
        v.addWidget(self.lbl_state)
        self.ed_text = QPlainTextEdit()
        self.ed_text.setMinimumHeight(220)
        v.addWidget(self.ed_text)
        row = QHBoxLayout()
        self.lbl_name = QLabel()
        self.ed_name = QLineEdit()
        self.btn_narrate = QPushButton()
        self.btn_narrate.setObjectName("primary")
        row.addWidget(self.lbl_name)
        row.addWidget(self.ed_name, 1)
        row.addWidget(self.btn_narrate)
        v.addLayout(row)
        self.body.addWidget(c)
        self.body.addStretch(1)

        self.btn_record.clicked.connect(self.toggle_record)
        self.btn_play.clicked.connect(self.play_selected)
        self.btn_add.clicked.connect(self.add_files)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_transcribe.clicked.connect(self.transcribe)
        self.btn_cancel.clicked.connect(lambda: self.worker.cancel.cancel() if self.worker else None)
        self.btn_narrate.clicked.connect(self.narrate)
        self.ed_text.textChanged.connect(self._refresh)
        self.lst_files.currentRowChanged.connect(lambda _r: self._refresh())

    def window_title(self) -> str:
        return tr("revoice.title")

    def retranslate(self) -> None:
        super().retranslate()
        self.lbl_audio.setText(tr("revoice.audio"))
        self.lbl_audio_hint.setText(tr("revoice.audio_hint"))
        self.btn_play.setText(tr("revoice.play"))
        self.btn_add.setText(tr("revoice.add"))
        self.btn_remove.setText(tr("revoice.remove"))
        self.lbl_text.setText(tr("revoice.text"))
        self.btn_transcribe.setText(tr("revoice.transcribe"))
        self.btn_cancel.setText(tr("ui.cancel"))
        self.lbl_name.setText(tr("revoice.name"))
        self.btn_narrate.setText(tr("revoice.narrate"))
        self.ed_text.setPlaceholderText(tr("revoice.text_hint"))
        self._refresh()

    # ------------------------------------------------------------------ state
    @property
    def busy(self) -> bool:
        return bool(self.worker and self.worker.isRunning())

    def _refresh(self) -> None:
        busy, recording = self.busy, self._rec is not None
        self.btn_record.setText(tr("revoice.stop") if recording else tr("revoice.record"))
        self.btn_record.setEnabled(not busy)
        self.btn_play.setEnabled(self.lst_files.currentRow() >= 0 and not recording)
        self.btn_add.setEnabled(not busy and not recording)
        self.btn_remove.setEnabled(not busy and self.lst_files.currentRow() >= 0)
        self.btn_transcribe.setEnabled(not busy and not recording and bool(self.files))
        self.btn_cancel.setVisible(busy)
        self.progress.setVisible(busy)
        self.btn_narrate.setEnabled(not busy and bool(self.ed_text.toPlainText().strip()))

    def set_files(self, files: List[Path]) -> None:
        self.files = [Path(f) for f in files]
        self.lst_files.clear()
        self.lst_files.addItems([f.name for f in self.files])
        if not self.ed_name.text().strip() and self.files:
            self.ed_name.setText(revoice.chapter_title(self.files[0]))
        self._refresh()

    # ------------------------------------------------------------------ audio
    def add_files(self) -> None:
        if self._pick_files is not None:
            picked = self._pick_files()
        else:
            pattern = " ".join("*" + e for e in revoice.AUDIO_EXTENSIONS)
            picked, _ = QFileDialog.getOpenFileNames(self, tr("revoice.add"), "", f"{tr('revoice.audio_files')} ({pattern})")
        if picked:
            self.set_files(self.files + sorted(Path(p) for p in picked))       # several files = chapters, in name order

    def remove_selected(self) -> None:
        r = self.lst_files.currentRow()
        if r >= 0:
            self.set_files(self.files[:r] + self.files[r + 1:])

    def play_selected(self) -> None:
        r = self.lst_files.currentRow()
        if r >= 0:
            self.previewer.play(self.files[r])

    def toggle_record(self) -> None:
        """Start recording from the default microphone, or stop and add the recording to the list."""
        from PySide6.QtMultimedia import QAudioInput, QMediaCaptureSession, QMediaRecorder

        if self._rec is not None:
            self._rec.stop()
            return
        self.previewer.stop()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._session, self._input, rec = QMediaCaptureSession(), QAudioInput(), QMediaRecorder()
        self._session.setAudioInput(self._input)
        self._session.setRecorder(rec)
        rec.setMediaFormat(record_format())
        rec.setQuality(QMediaRecorder.Quality.HighQuality)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        rec.setOutputLocation(QUrl.fromLocalFile(str(self.out_dir / f"recording {stamp}")))    # the backend adds the suffix
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

            path = revoice.to_opus(path, audio_utils.ensure_ffmpeg())     # stored as Opus; decoded to PCM in memory for the models
            self.set_files(self.files + [path])
            self.lst_files.setCurrentRow(len(self.files) - 1)
            self.lbl_rec.setText(tr("revoice.recorded", name=path.name))
        self._refresh()

    # ------------------------------------------------------------------ recognition and hand-over
    def transcribe(self) -> bool:
        if self.busy or not self.files:
            return False
        w = TranscribeWorker(self.files, self.asr_factory, self)
        w.progress.connect(lambda f, t: (self.progress.setValue(int(f * 100)),
                                         self.lbl_state.setText(tr("revoice.transcribing", name=t))))
        w.done.connect(self._on_done)
        w.failed.connect(lambda m: (self.lbl_state.setText(tr("revoice.failed", error=m)), self._refresh()))
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

    def narrate(self) -> Optional[Path]:
        """Save the edited text and ask the Studio to open it in the narrator."""
        text = self.ed_text.toPlainText().strip()
        if not text:
            return None
        p = revoice.save_text(text + "\n", self.out_dir, self.ed_name.text().strip() or "Re-voice")
        self.narrate_file.emit(str(p))
        return p

    def shutdown(self) -> None:
        if self._rec is not None:
            self._rec.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel.cancel()
            self.worker.wait(30000)
