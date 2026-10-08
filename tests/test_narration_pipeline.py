"""CPU work overlapped with synthesis: the CPU budget, chapter assembly / per-chapter encodes while the GPU still works,
parallel export with a fixed result order, byte-identical output, bounded memory and the early model load."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core import audiobook_export as ex
from core import cpu_budget
from core import narration as nr
from core.book_parsers import Book, Chapter
from tests.test_narration import FakeEngine, FakeFfmpeg


@pytest.fixture(autouse=True)
def _english():
    from core import i18n
    i18n.set_language("en")


def book(n=4):
    return Book("Pipe Book", "Ann Lee", "en", [Chapter(f"Part {i + 1}", f"Sentence {i} one. Another sentence {i} here.\n\n"
                                                             f"Paragraph two of {i}.") for i in range(n)])


@pytest.mark.parametrize("cores, cuda, workers, torch_threads", [
    (1, True, 1, 1), (2, True, 1, 2), (4, True, 2, 2), (8, True, 5, 2), (24, True, 8, 4), (64, True, 8, 4),
    (8, False, 1, 6), (4, False, 1, 3), (1, False, 1, 1),
])
def test_cpu_budget_leaves_cores_for_the_gpu_feeder_and_the_ui(cores, cuda, workers, torch_threads, monkeypatch):
    monkeypatch.delenv(cpu_budget.ENV_WORKERS, raising=False)
    b = cpu_budget.plan(cuda, cores)
    assert (b.workers, b.torch_threads) == (workers, torch_threads)
    if cuda and cores >= 4:
        assert b.workers + 1 + (2 if cores >= 6 else 1) <= cores       # pool + GPU feeder + UI/OS reserve fit


def test_cpu_budget_override(monkeypatch):
    monkeypatch.setenv(cpu_budget.ENV_WORKERS, "3")
    assert cpu_budget.plan(True, 16).workers == 3
    monkeypatch.setenv(cpu_budget.ENV_WORKERS, "nonsense")
    assert cpu_budget.plan(True, 16).workers == 8


class SlowFfmpeg(FakeFfmpeg):
    """Fake ffmpeg whose jobs take time (the earlier-submitted ones longest), recording start times."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.starts = {}

    def __call__(self, cmd):
        if "-encoders" not in cmd:
            self.starts[cmd[-1]] = time.monotonic()
            time.sleep(0.05)
        return super().__call__(cmd)


def chapters(tmp_path, n=5):
    import numpy as np
    import soundfile as sf
    out = []
    for i in range(n):
        wav = tmp_path / f"c{i}.wav"
        sf.write(str(wav), np.zeros(240, dtype=np.float32), 24000, subtype="PCM_16")
        out.append(ex.ChapterAudio(i, f"Ch {i}", wav, 0.01))
    return out


def test_parallel_export_returns_the_serial_order(tmp_path):
    fmts = set(ex.ALL_FORMATS)
    meta = ex.BookMeta("T", "A", "N")
    serial = ex.export_formats("ffmpeg", fmts, chapters(tmp_path), meta, tmp_path / "s", run=FakeFfmpeg())
    with ThreadPoolExecutor(6) as pool:
        par = ex.export_formats("ffmpeg", fmts, chapters(tmp_path), meta, tmp_path / "p", run=SlowFfmpeg(), submit=pool.submit)
    rel = lambda r, root: [f.relative_to(root) for f in r.files]   # noqa: E731
    assert rel(serial, tmp_path / "s") == rel(par, tmp_path / "p") and len(par.files) > 20


def test_export_error_of_a_parallel_job_surfaces(tmp_path):
    with ThreadPoolExecutor(4) as pool, pytest.raises(nr.NarrationError):
        ex.export_formats("ffmpeg", {ex.FORMAT_MP3_CHAPTERS, ex.FORMAT_OPUS_SINGLE}, chapters(tmp_path),
                          ex.BookMeta("T"), tmp_path / "o", run=FakeFfmpeg(fail_on="libmp3lame"), submit=pool.submit)


def test_chapters_are_assembled_and_encoded_while_synthesis_goes_on(tmp_path):
    engine = FakeEngine(delay=0.03)
    stamps = []
    orig = engine.synthesize

    def synth(text):
        stamps.append(time.monotonic())
        return orig(text)
    engine.synthesize = synth
    ff = SlowFfmpeg()
    res = nr.narrate_book(book(4), lambda: engine, engine.tag, tmp_path / "out", language="english", ffmpeg="ffmpeg",
                          run=ff, options=nr.NarrationOptions(formats={ex.FORMAT_MP3_CHAPTERS}))
    first_chapter = next(f for f in res.files if f.name.startswith("1 - ") or f.name.startswith("01 - "))
    assert ff.starts[str(first_chapter)] < stamps[-1]          # chapter 1 was encoded before the last chunk was spoken
    assert [f.name for f in res.files][:4] == [f"0{i} - Part {i}.mp3" for i in range(1, 5)]


def test_output_is_byte_identical_with_one_or_many_workers(tmp_path, monkeypatch):
    def produce(sub, workers):
        monkeypatch.setenv(cpu_budget.ENV_WORKERS, str(workers))
        res = nr.narrate_book(book(5), lambda: FakeEngine(), FakeEngine.tag, tmp_path / sub, language="english",
                              ffmpeg=None, options=nr.NarrationOptions(formats={ex.FORMAT_WAV_CHAPTERS}))
        return [(f.relative_to(tmp_path / sub), f.read_bytes()) for f in res.files]
    assert produce("one", 1) == produce("many", 6)


def test_pending_writes_are_bounded(tmp_path, monkeypatch):
    engine = FakeEngine()
    saved, peak = [], []
    orig = nr.ChunkCache.save

    def slow_save(self, key, audio, sr):
        time.sleep(0.03)
        orig(self, key, audio, sr)
        saved.append(key)
    orig_synth = engine.synthesize

    def synth(text):
        peak.append(len(engine.calls) - len(saved))            # chunks made but not on disk yet
        return orig_synth(text)
    monkeypatch.setattr(nr.ChunkCache, "save", slow_save)
    engine.synthesize = synth
    nr.narrate_book(book(6), lambda: engine, engine.tag, tmp_path / "out", language="english", ffmpeg=None,
                    options=nr.NarrationOptions(formats={ex.FORMAT_WAV_CHAPTERS}))
    assert max(peak) <= 4 + 1                                  # max(4, 2 x batch 1) waiting + the one being made


def test_fresh_job_loads_the_model_while_the_text_is_prepared(tmp_path):
    seen = {}

    def factory():
        seen["thread"] = threading.current_thread().name
        return FakeEngine()
    nr.narrate_book(book(2), factory, "t", tmp_path / "out", language="english", ffmpeg=None,
                    options=nr.NarrationOptions(formats={ex.FORMAT_WAV_CHAPTERS}))
    assert seen["thread"].startswith("engine-load")


def test_early_loaded_engine_is_closed_when_the_job_fails_before_synthesis(tmp_path):
    engine = FakeEngine()
    loaded = threading.Event()

    def factory():
        loaded.set()
        return engine

    def bad_plan(files):
        loaded.wait(2)
        raise RuntimeError("live player broke")
    with pytest.raises(RuntimeError):
        nr.narrate_book(book(2), factory, "t", tmp_path / "out", language="english", ffmpeg=None, on_plan=bad_plan,
                        options=nr.NarrationOptions(formats={ex.FORMAT_WAV_CHAPTERS}))
    deadline = time.monotonic() + 2
    while not engine.closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert engine.closed


def test_pause_also_holds_background_assembly(tmp_path):
    pause = nr.PauseToken()
    done_at = {}
    orig = nr.assemble_chapter

    def spy(*a, **k):
        r = orig(*a, **k)
        done_at.setdefault("first", time.monotonic())
        return r
    opts = nr.NarrationOptions(formats={ex.FORMAT_WAV_CHAPTERS}, keep_cache=True)
    nr.narrate_book(book(2), lambda: FakeEngine(), "t", tmp_path / "out", language="english", ffmpeg=None, options=opts)
    pause.pause()                    # all chunks cached now: only the background assembly is left, and it must wait
    t0 = time.monotonic()
    threading.Timer(0.3, pause.resume).start()
    import unittest.mock as um
    with um.patch.object(nr, "assemble_chapter", spy):
        nr.narrate_book(book(2), lambda: FakeEngine(), "t", tmp_path / "out", language="english", ffmpeg=None,
                        pause=pause, options=opts)
    assert done_at["first"] - t0 >= 0.25


def test_low_priority_run_returns_code_and_output():
    import sys
    rc, out = ex.default_run([sys.executable, "-c", "import sys; print('hi'); sys.stderr.write('err'); sys.exit(3)"])
    assert rc == 3 and "hi" in out and "err" in out
