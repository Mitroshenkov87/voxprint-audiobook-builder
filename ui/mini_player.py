"""Mini audio player: play / pause / seek over a list of files, optionally *live* while they are still being produced.

Used in the Narrate window (listen to the finished chunks while the narration runs; the player follows new chunks) and in the
Train window (the quick-preview samples).  The playlist logic is :class:`core.play_queue.PlayQueue`; the audio output is a small
backend object (``QtBackend`` by default, a fake in tests).  The player only polls a few file paths on a slow timer and decodes
on the audio thread of Qt Multimedia, so it does not slow down synthesis (which runs in another thread / on the GPU).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout

from core.i18n import tr
from core.play_queue import PlayQueue, format_clock

log = logging.getLogger("voxprint.miniplayer")

POLL_MS = 1500


class QtBackend(QObject):
    """One audio file at a time through QMediaPlayer (created on first use).  Signals: ``position(ms)``, ``ended()``, ``failed()``."""

    position = Signal(int)
    ended = Signal()
    failed = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        """Lazy: nothing audio-related is created until :meth:`load`."""
        super().__init__(parent)
        self._player = None
        self._audio = None

    def _ensure(self) -> bool:
        if self._player is not None:
            return True
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

            self._audio = QAudioOutput()
            self._player = QMediaPlayer()
            self._player.setAudioOutput(self._audio)
            self._player.positionChanged.connect(self.position.emit)
            self._player.mediaStatusChanged.connect(self._status)
            self._player.errorOccurred.connect(lambda *_a: self.failed.emit())
            return True
        except Exception:  # noqa: BLE001 - no backend / no audio device
            log.info("Qt Multimedia unavailable", exc_info=True)
            return False

    def _status(self, st) -> None:
        from PySide6.QtMultimedia import QMediaPlayer

        if st == QMediaPlayer.MediaStatus.EndOfMedia:
            self.ended.emit()

    def load(self, path: Path, start_ms: int = 0, play: bool = True) -> bool:
        """Open ``path`` at ``start_ms`` and play it; False if audio is unavailable."""
        if not self._ensure():
            self.failed.emit()
            return False
        self._player.setSource(QUrl.fromLocalFile(str(path)))
        if start_ms:
            self._player.setPosition(int(start_ms))
        if play:
            self._player.play()
        return True

    def play(self) -> None:
        """Resume."""
        if self._player is not None:
            self._player.play()

    def pause(self) -> None:
        """Pause."""
        if self._player is not None:
            self._player.pause()

    def stop(self) -> None:
        """Stop and release the file."""
        if self._player is not None:
            self._player.stop()
            self._player.setSource(QUrl())

    def seek(self, ms: int) -> None:
        """Move inside the current file."""
        if self._player is not None:
            self._player.setPosition(int(ms))


class MiniPlayer(QFrame):
    """Play/pause button, seek slider, clock and a status line.

    ``set_plan(paths, live=True)`` starts following a growing list of files; ``set_files(paths)`` is a fixed playlist."""

    started = Signal()          # playback was started by the user (other players should stop)

    def __init__(self, backend: Any = None, poll_ms: int = POLL_MS, queue: Optional[PlayQueue] = None) -> None:
        """``backend`` / ``queue`` are injectable (tests)."""
        super().__init__()
        self.setObjectName("card")
        self.queue = queue or PlayQueue()
        self.backend = backend or QtBackend(self)
        self.backend.position.connect(self._on_position)
        self.backend.ended.connect(self._on_ended)
        self.backend.failed.connect(self._on_failed)
        self._index = -1                 # chunk loaded in the backend
        self._want_play = False          # the user pressed play (so keep going, also across gaps)
        self._waiting = False            # at the end of the ready chunks, waiting for the next one
        self._live = False
        self._final: Optional[Path] = None
        self._dragging = False
        self._pos_s = 0.0                # timeline position
        self._error = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(4)
        self.lbl_title = QLabel()
        self.lbl_title.setObjectName("carddesc")
        lay.addWidget(self.lbl_title)
        row = QHBoxLayout()
        self.btn_play = QPushButton()
        self.btn_play.setFixedWidth(88)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.lbl_time = QLabel("0:00 / 0:00")
        self.lbl_time.setMinimumWidth(96)
        row.addWidget(self.btn_play)
        row.addWidget(self.slider, 1)
        row.addWidget(self.lbl_time)
        lay.addLayout(row)
        self.lbl_status = QLabel()
        self.lbl_status.setObjectName("hint")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)
        self.timer = QTimer(self)
        self.timer.setInterval(poll_ms)
        self.timer.timeout.connect(self.poll)
        self.btn_play.clicked.connect(self.toggle)
        self.slider.sliderPressed.connect(lambda: setattr(self, "_dragging", True))
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_value)
        self.retranslate()

    # ------------------------------------------------------------------ text
    def retranslate(self) -> None:
        """Apply the current language."""
        self.lbl_title.setText(tr("player.title_live") if self._live else tr("player.title"))
        self._update_ui()

    def _update_ui(self) -> None:
        playing = self._want_play and not self._waiting
        self.btn_play.setText(tr("player.pause") if self._want_play else tr("player.play"))
        self.btn_play.setEnabled(self.queue.ready > 0 or self._final is not None)
        self.lbl_time.setText(f"{format_clock(self._pos_s)} / {format_clock(self.queue.total)}")
        if self._error:
            text = tr("player.unavailable")
        elif self._live and not self.queue.complete:
            text = tr("player.live_status", ready=self.queue.ready, total=self.queue.planned)
            if self._waiting and self._want_play:
                text = tr("player.waiting") + "  \u00b7  " + text
        elif self._final is not None and self._live:
            text = tr("player.finished")
        else:
            text = ""
        self.lbl_status.setText(text)
        self.lbl_status.setVisible(bool(text))
        self.btn_play.setProperty("playing", playing)

    # ------------------------------------------------------------------ playlist
    def set_plan(self, paths: Sequence[Path], live: bool = True) -> None:
        """Follow a (growing) ordered list of files; polling starts if some are still missing."""
        self._live = live
        self._final = None
        self.queue.set_plan(paths)
        self.poll()
        self._sync_timer()
        self.retranslate()

    def set_files(self, paths: Sequence[Path]) -> None:
        """A fixed playlist that already exists (quick-preview samples)."""
        self.stop()
        self.queue.paths, self.queue.durations = [], []
        self.set_plan(paths, live=False)

    def set_final(self, path: Optional[Path]) -> None:
        """The job is over: nothing more will appear.  If idle, the finished file becomes the playlist."""
        self.timer.stop()
        self.queue.poll()
        if path is not None and not (self._want_play and not self._waiting):
            self.stop()
            self.queue.paths, self.queue.durations = [], []
            self.queue.set_plan([path])
            self.queue.poll()
            self._live = False
        self._final = Path(path) if path else None
        self.retranslate()
        if self._waiting and self._want_play:      # was waiting for a chunk that will never come
            self._want_play = False
            self._waiting = False
            self._update_ui()

    def poll(self) -> None:
        """Look for new chunks (slow timer); continues playback that was waiting for one."""
        added = self.queue.poll()
        if added:
            self.slider.blockSignals(True)
            self.slider.setRange(0, int(self.queue.total * 1000))
            self.slider.blockSignals(False)
            if self._waiting and self._want_play:
                self._waiting = False
                self._load(self._index + 1, 0.0)
        self._sync_timer()
        self._update_ui()

    def _sync_timer(self) -> None:
        if self._live and not self.queue.complete and self._final is None:
            if not self.timer.isActive():
                self.timer.start()
        else:
            self.timer.stop()

    # ------------------------------------------------------------------ transport
    def toggle(self) -> None:
        """Play / pause."""
        if self._want_play:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        """Play from the current position (from the start if nothing was played)."""
        if self.queue.ready == 0:
            return
        self._want_play = True
        self._error = False
        self.started.emit()
        if self._index < 0 or self._index >= self.queue.ready:
            self._load(0, 0.0)
        elif self._waiting:
            self.poll()
        else:
            self.backend.play()
        self._update_ui()

    def pause(self) -> None:
        """Pause (the position is kept)."""
        self._want_play = False
        self.backend.pause()
        self._update_ui()

    def stop(self) -> None:
        """Stop and rewind."""
        self._want_play = False
        self._waiting = False
        self._index = -1
        self._pos_s = 0.0
        self.backend.stop()
        self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._update_ui()

    def play_index(self, i: int) -> None:
        """Start the chunk ``i`` (e.g. the sample of a variant)."""
        if 0 <= i < self.queue.ready:
            self._want_play = True
            self._error = False
            self.started.emit()
            self._load(i, 0.0)
            self._update_ui()

    def seek(self, seconds: float) -> None:
        """Jump to a timeline position (works while paused too)."""
        loc = self.queue.locate(seconds)
        if loc is None:
            return
        i, off = loc
        self._pos_s = self.queue.start_of(i) + off
        if i == self._index:
            self.backend.seek(int(off * 1000))
        else:
            self._load(i, off, play=self._want_play)
        self._update_ui()

    def _load(self, i: int, offset_s: float, play: bool = True) -> None:
        path = self.queue.path(i)
        if path is None:
            return
        self._index = i
        self._waiting = False
        self._pos_s = self.queue.start_of(i) + offset_s
        self.backend.load(path, int(offset_s * 1000), play=play and self._want_play)

    # ------------------------------------------------------------------ backend events
    def _on_position(self, ms: int) -> None:
        if self._index < 0:
            return
        self._pos_s = self.queue.start_of(self._index) + ms / 1000.0
        if not self._dragging:
            self.slider.blockSignals(True)
            self.slider.setValue(int(self._pos_s * 1000))
            self.slider.blockSignals(False)
        self._update_ui()

    def _on_ended(self) -> None:
        """A chunk finished: next one, or wait for the synthesis to produce it."""
        if not self._want_play:
            return
        if self.queue.has_next(self._index):
            self._load(self._index + 1, 0.0)
        elif self._live and not self.queue.complete and self._final is None:
            self._waiting = True
        else:
            self._want_play = False
            self._index = -1
            self._pos_s = 0.0
            self.slider.setValue(0)
        self._update_ui()

    def _on_failed(self) -> None:
        """No audio device / unreadable file."""
        self._error = True
        self._want_play = False
        self._update_ui()

    def _on_slider_value(self, v: int) -> None:
        if not self._dragging and not self.slider.signalsBlocked():       # keyboard / click on the groove
            self.seek(v / 1000.0)

    def _on_slider_released(self) -> None:
        self._dragging = False
        self.seek(self.slider.value() / 1000.0)

    def shutdown(self) -> None:
        """Stop everything (window closing)."""
        self.timer.stop()
        self.backend.stop()
