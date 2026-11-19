"""Ordered list of audio files that appear one after another (finished narration chunks), as one seekable timeline.

Qt-free.  The narration job tells the UI the full ordered list of chunk files up front (:func:`core.narration.narrate_book`
``on_plan``); the files are written atomically by the synthesis thread, so a file that exists is complete.  The player polls
:meth:`PlayQueue.poll` (a few ``stat`` calls, nothing is read from the synthesis thread) and plays the ready prefix; new chunks
extend the timeline, so listening follows the synthesis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import soundfile as sf


def file_seconds(path: Path) -> float:
    """Duration of an audio file from its header (cheap); raises ``OSError``/``RuntimeError`` if unreadable."""
    return float(sf.info(str(path)).duration)


class PlayQueue:
    """Chunks in playing order; only the contiguous *ready* prefix is playable."""

    def __init__(self, duration_of: Callable[[Path], float] = file_seconds) -> None:
        """``duration_of`` is injectable (tests)."""
        self._duration_of = duration_of
        self.paths: List[Path] = []
        self.durations: List[float] = []        # of the ready prefix

    # ------------------------------------------------------------------ plan
    def set_plan(self, paths: Sequence[Path]) -> None:
        """Set the full ordered list.  An unchanged list keeps what is already known; a different one starts over."""
        new = [Path(p) for p in paths]
        if new == self.paths:
            return
        keep = 0
        while keep < min(len(new), len(self.paths), len(self.durations)) and new[keep] == self.paths[keep]:
            keep += 1
        self.paths = new
        self.durations = self.durations[:keep]

    def poll(self) -> int:
        """Extend the ready prefix with the files that exist now; returns how many chunks became ready."""
        added = 0
        while len(self.durations) < len(self.paths):
            p = self.paths[len(self.durations)]
            try:
                if not p.is_file() or p.stat().st_size <= 0:
                    break
                self.durations.append(max(0.0, self._duration_of(p)))
            except (OSError, RuntimeError):
                break
            added += 1
        return added

    # ------------------------------------------------------------------ timeline
    @property
    def ready(self) -> int:
        """Number of playable chunks."""
        return len(self.durations)

    @property
    def planned(self) -> int:
        """Number of chunks the job will produce."""
        return len(self.paths)

    @property
    def complete(self) -> bool:
        """True when every planned chunk is ready."""
        return bool(self.paths) and self.ready == self.planned

    @property
    def total(self) -> float:
        """Seconds of the ready prefix."""
        return float(sum(self.durations))

    def start_of(self, index: int) -> float:
        """Timeline position (s) where chunk ``index`` starts."""
        return float(sum(self.durations[:max(0, index)]))

    def locate(self, seconds: float) -> Optional[Tuple[int, float]]:
        """``(chunk index, offset in the chunk in s)`` of a timeline position (clamped); ``None`` if nothing is ready."""
        if not self.durations:
            return None
        t = max(0.0, seconds)
        acc = 0.0
        for i, d in enumerate(self.durations):
            if t < acc + d or i == len(self.durations) - 1:
                return i, min(max(0.0, t - acc), d)
            acc += d
        return None  # pragma: no cover

    def path(self, index: int) -> Optional[Path]:
        """File of a ready chunk."""
        return self.paths[index] if 0 <= index < self.ready else None

    def has_next(self, index: int) -> bool:
        """True if chunk ``index + 1`` is ready."""
        return index + 1 < self.ready


def format_clock(seconds: float) -> str:
    """``m:ss`` or ``h:mm:ss``."""
    s = max(0, int(seconds))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"
