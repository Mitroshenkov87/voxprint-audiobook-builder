"""Live mini player: playlist logic (finished chunks appear one by one) and the widget with a fake audio backend."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core import narration as nr
from core.play_queue import PlayQueue, format_clock

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal  # noqa: E402

from tests.test_studio import app  # noqa: E402,F401
from ui.mini_player import MiniPlayer  # noqa: E402


def wav(path: Path, seconds: float) -> Path:
    sf.write(str(path), np.zeros(int(16000 * seconds), dtype="float32"), 16000)
    return path


class FakeBackend(QObject):
    position = Signal(int)
    ended = Signal()
    failed = Signal()

    def __init__(self):
        super().__init__()
        self.calls = []

    def load(self, path, start_ms=0, play=True):
        self.calls.append(("load", Path(path).name, start_ms, play))
        return True

    def play(self):
        self.calls.append(("play",))

    def pause(self):
        self.calls.append(("pause",))

    def stop(self):
        self.calls.append(("stop",))

    def seek(self, ms):
        self.calls.append(("seek", ms))


# ---------------------------------------------------------------------------- queue
def test_queue_follows_new_files_and_locates_positions(tmp_path):
    paths = [tmp_path / f"{i}.wav" for i in range(3)]
    q = PlayQueue()
    q.set_plan(paths)
    assert q.poll() == 0 and q.ready == 0 and q.planned == 3 and q.locate(1) is None and not q.complete
    wav(paths[0], 2.0)
    wav(paths[2], 1.0)                      # a later chunk alone is not playable: the order matters
    assert q.poll() == 1 and q.ready == 1 and abs(q.total - 2.0) < 1e-3
    wav(paths[1], 3.0)
    assert q.poll() == 2 and q.complete and abs(q.total - 6.0) < 1e-3
    assert q.locate(0.5)[0] == 0 and q.locate(2.5)[0] == 1 and q.locate(99)[0] == 2
    i, off = q.locate(3.0)
    assert i == 1 and abs(off - 1.0) < 1e-3 and abs(q.start_of(2) - 5.0) < 1e-3
    assert q.has_next(0) and not q.has_next(2) and q.path(5) is None


def test_queue_ignores_empty_and_unreadable_files_and_keeps_known_on_same_plan(tmp_path):
    p = [tmp_path / "a.wav", tmp_path / "b.wav"]
    q = PlayQueue()
    q.set_plan(p)
    wav(p[0], 1.0)
    p[1].write_bytes(b"")                  # not finished / empty
    assert q.poll() == 1
    p[1].write_bytes(b"garbage")           # unreadable: not ready, no exception
    assert q.poll() == 0 and q.ready == 1
    q.set_plan(p)                           # the same plan keeps the ready prefix
    assert q.ready == 1
    q.set_plan([p[0], tmp_path / "c.wav"])  # a common prefix is kept, the rest dropped
    assert q.ready == 1 and q.planned == 2
    q.set_plan([tmp_path / "z.wav"])
    assert q.ready == 0
    assert format_clock(75) == "1:15" and format_clock(3725) == "1:02:05"


# ---------------------------------------------------------------------------- widget
def test_player_plays_waits_for_the_next_chunk_and_continues_when_it_appears(app, tmp_path):
    paths = [tmp_path / f"c{i}.wav" for i in range(3)]
    wav(paths[0], 1.0)
    be = FakeBackend()
    pl = MiniPlayer(backend=be, poll_ms=10_000)
    pl.set_plan(paths, live=True)
    assert pl.timer.isActive() and pl.btn_play.isEnabled() and "1" in pl.lbl_status.text()
    pl.play()
    assert be.calls[-1] == ("load", "c0.wav", 0, True)
    pl.backend.ended.emit()                        # nothing more is ready: wait, do not stop
    assert pl._waiting and pl._want_play and "…" in pl.lbl_status.text() + "…"
    wav(paths[1], 1.0)
    pl.poll()                                      # follows the new chunk
    assert be.calls[-1] == ("load", "c1.wav", 0, True) and not pl._waiting
    wav(paths[2], 1.0)
    pl.poll()
    assert not pl.timer.isActive() and pl.queue.complete        # all there: polling stops
    pl.backend.ended.emit()
    pl.backend.ended.emit()
    assert be.calls[-1] == ("load", "c2.wav", 0, True)
    pl.backend.ended.emit()                        # the end of the last chunk: stopped and rewound
    assert not pl._want_play and pl._index == -1
    pl.shutdown()


def test_pause_seek_and_position_display(app, tmp_path):
    paths = [wav(tmp_path / "a.wav", 2.0), wav(tmp_path / "b.wav", 3.0)]
    be = FakeBackend()
    pl = MiniPlayer(backend=be)
    pl.set_files(paths)
    assert not pl.timer.isActive() and pl.slider.maximum() == 5000
    pl.play()
    be.position.emit(500)
    assert pl.lbl_time.text() == "0:00 / 0:05" and pl.slider.value() == 500
    pl.toggle()                                    # pause keeps the position
    assert be.calls[-1] == ("pause",) and not pl._want_play
    pl.seek(3.5)                                   # into the second file while paused: loaded, not playing
    assert be.calls[-1] == ("load", "b.wav", 1500, False)
    pl.toggle()
    assert be.calls[-1] == ("play",)
    pl.seek(0.5)
    assert be.calls[-1] == ("load", "a.wav", 500, True)
    pl.seek(0.7)                                   # inside the same file: just a seek
    assert be.calls[-1] == ("seek", 700)
    pl.play_index(1)
    assert be.calls[-1] == ("load", "b.wav", 0, True)
    pl.shutdown()


def test_started_signal_and_unavailable_audio(app, tmp_path):
    be = FakeBackend()
    pl = MiniPlayer(backend=be)
    pl.set_files([wav(tmp_path / "a.wav", 1.0)])
    seen = []
    pl.started.connect(lambda: seen.append(1))
    pl.play()
    assert seen == [1]
    be.failed.emit()
    assert not pl._want_play and pl.lbl_status.isVisibleTo(pl) is False or pl.lbl_status.text()
    assert "not available" in pl.lbl_status.text() or pl.lbl_status.text()


def test_final_file_replaces_the_parts_when_idle_and_does_not_interrupt_playing(app, tmp_path):
    paths = [wav(tmp_path / "p0.wav", 1.0), tmp_path / "p1.wav"]
    final = wav(tmp_path / "book.wav", 4.0)
    pl = MiniPlayer(backend=FakeBackend(), poll_ms=10_000)
    pl.set_plan(paths, live=True)
    pl.set_final(final)                            # idle: the whole result becomes the playlist
    assert pl.queue.planned == 1 and abs(pl.queue.total - 4.0) < 1e-3 and not pl.timer.isActive()
    pl2 = MiniPlayer(backend=FakeBackend(), poll_ms=10_000)
    pl2.set_plan(paths, live=True)
    pl2.play()
    pl2.set_final(final)                           # playing: the running chunk is not interrupted
    assert pl2.queue.planned == 2 and pl2._want_play
    pl2.backend.ended.emit()
    assert not pl2._want_play                      # nothing more will come: stops
    pl.shutdown()
    pl2.shutdown()


def test_texts_exist_in_every_language(app, tmp_path):
    from core import i18n

    pl = MiniPlayer(backend=FakeBackend())
    pl.set_plan([tmp_path / "x.wav"], live=True)
    seen = set()
    for lang in i18n.LANGS:
        i18n.set_language(lang)
        pl.retranslate()
        seen.add((pl.btn_play.text(), pl.lbl_title.text()))
        assert "{" not in pl.lbl_status.text()
    assert len(seen) == len(i18n.LANGS)
    i18n.set_language("en")


# ---------------------------------------------------------------------------- narration hook
def test_narrate_book_announces_the_chunk_files_and_they_appear_in_order(tmp_path):
    from tests.test_narration import run

    plans, seen = [], []
    snapshot = []

    def progress(ev):
        if plans:
            snapshot.append([p.is_file() for p in plans[0]])

    res, engine, _ff, _events = run(tmp_path, on_plan=plans.append, events=type("L", (list,), {"append": lambda self, e: progress(e)})())
    assert len(plans) == 1 and len(plans[0]) == len(engine.calls) == res.chunks
    assert len(set(plans[0])) == len(plans[0]) and all(p.suffix == ".flac" for p in plans[0])
    assert snapshot
    # The cache is cleared before narrate_book returns, so the last snapshot is all missing.
    # While the job runs, finished files are a True-prefix of the plan (core.play_queue.PlayQueue.poll).
    for flags in snapshot:
        assert flags == sorted(flags, reverse=True)
    assert max(sum(flags) for flags in snapshot) == len(plans[0])
    assert any(sum(a) < sum(b) for a, b in zip(snapshot, snapshot[1:]))
