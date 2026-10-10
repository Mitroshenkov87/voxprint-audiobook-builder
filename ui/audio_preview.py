"""Plays a short voice sample.  Uses Qt Multimedia when available; otherwise hands the file to the system player.

The class is tiny on purpose: the voices window takes a ``previewer`` argument, so tests inject a fake and never touch
audio devices.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtGui import QDesktopServices

log = logging.getLogger("voxprint.preview")


class Previewer(QObject):
    """Play/stop one audio file at a time.  ``stopped`` is emitted when playback ends or is stopped."""

    stopped = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        """Create lazily-initialised player state."""
        super().__init__(parent)
        self._player = None
        self._audio = None
        self.current: Optional[Path] = None

    def _ensure_player(self) -> bool:
        """Create the QMediaPlayer on first use; False if Qt Multimedia cannot be used."""
        if self._player is not None:
            return True
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

            audio = QAudioOutput()
            player = QMediaPlayer()
            self._audio = audio
            self._player = player
            # PySide6 stubs type these constructors as None.
            if player is None or audio is None:
                return False
            player.setAudioOutput(audio)
            player.playbackStateChanged.connect(self._on_state)
            return True
        except Exception:  # noqa: BLE001 - missing backend / no audio device
            log.info("Qt Multimedia unavailable; falling back to the system player", exc_info=True)
            return False

    def _on_state(self, state) -> None:
        """Playback state changed: report the end."""
        try:
            from PySide6.QtMultimedia import QMediaPlayer

            if state == QMediaPlayer.PlaybackState.StoppedState:
                self.current = None
                self.stopped.emit()
        except Exception:  # noqa: BLE001
            pass

    def play(self, path: Path) -> None:
        """Start playing ``path`` (stops anything that is playing)."""
        self.stop()
        self.current = Path(path)
        if self._ensure_player():
            player = self._player
            if player is None:
                return
            player.setSource(QUrl.fromLocalFile(str(path)))
            player.play()
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            self.current = None
            self.stopped.emit()

    def stop(self) -> None:
        """Stop playback."""
        if self._player is not None:
            self._player.stop()
        self.current = None
