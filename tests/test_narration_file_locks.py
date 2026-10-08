"""Build 667 log: narration stopped with WinError 32 while renaming a chunk ``<key>.part.flac`` -> ``<key>.flac``.

Cause: two chunks with the same text (a repeated verse) share a cache key, and two writer threads wrote the same temporary
file at once.  Now every write has its own temporary file, a locked rename is retried with backoff, a twin chunk already
on disk is kept, and a file that stays locked stops the job with a clear message.  Fakes only (no model, no ffmpeg)."""
import os
import threading
from pathlib import Path

import numpy as np
import pytest

from core import audiobook_export as ex
from core import narration as nr
from core.errors import NarrationError
from tests.test_narration import SR, FakeEngine, FakeFfmpeg, run
from core.book_parsers import Book, Chapter


def _locked(winerror=32):
    exc = PermissionError(13, "The process cannot access the file because it is being used by another process")
    exc.winerror = winerror
    return exc


def test_each_write_uses_its_own_temporary_file(tmp_path, monkeypatch):
    cache = nr.ChunkCache(tmp_path / "c")
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(Path(src).name)
        real_replace(src, dst)

    monkeypatch.setattr(nr.os, "replace", spy)
    key = cache.key("t", "And God saw that it was good.")
    for _ in range(2):
        cache.save(key, np.full(100, 0.1, np.float32), SR)
    assert len(seen) == 2 and seen[0] != seen[1] and all(n.startswith(key) and n.endswith(".part.flac") for n in seen)
    assert cache.has(key) and not list(cache.dir.glob("*.part.flac"))


def test_parallel_writes_of_one_key_do_not_collide(tmp_path):
    cache = nr.ChunkCache(tmp_path / "c")
    key = cache.key("t", "twin")
    errors, start = [], threading.Barrier(8)

    def write(i):
        try:
            start.wait(5)
            cache.save(key, np.full(2000, 0.01 * (i + 1), np.float32), SR)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert errors == [] and cache.load(key) is not None and not list(cache.dir.glob("*.part.flac"))


def test_a_locked_rename_is_retried_with_backoff(tmp_path, monkeypatch):
    src, dst = tmp_path / "a.part.flac", tmp_path / "a.flac"
    src.write_bytes(b"x")
    calls, pauses = [], []
    real_replace = os.replace

    def flaky(s, d):
        calls.append(1)
        if len(calls) < 3:
            raise _locked()
        real_replace(s, d)

    monkeypatch.setattr(nr.os, "replace", flaky)
    nr.replace_with_retry(src, dst, sleep=pauses.append)
    assert dst.read_bytes() == b"x" and len(calls) == 3 and pauses == [0.2, 0.4]


def test_other_permission_errors_are_not_retried(tmp_path, monkeypatch):
    def denied(s, d):
        raise _locked(winerror=1314)               # a privilege error is not "in use"

    monkeypatch.setattr(nr.os, "replace", denied)
    pauses = []
    with pytest.raises(PermissionError):
        nr.replace_with_retry(tmp_path / "a", tmp_path / "b", sleep=pauses.append)
    assert pauses == []


def test_a_file_that_stays_locked_gives_a_clear_error(tmp_path, monkeypatch):
    from core import i18n

    i18n.set_language("en")
    cache = nr.ChunkCache(tmp_path / "c")
    monkeypatch.setattr(nr, "LOCKED_FIRST_PAUSE", 0.0)
    monkeypatch.setattr(nr.os, "replace", lambda s, d: (_ for _ in ()).throw(_locked()))
    key = cache.key("t", "x")
    with pytest.raises(NarrationError) as err:
        cache.save(key, np.full(100, 0.1, np.float32), SR)
    assert f"{key}.flac" in str(err.value) and "another program" in str(err.value)
    assert not list(cache.dir.glob("*.part.flac"))            # the temporary file does not stay behind


def test_a_twin_already_on_disk_is_kept_when_the_rename_stays_locked(tmp_path, monkeypatch):
    cache = nr.ChunkCache(tmp_path / "c")
    key = cache.key("t", "twin")
    cache.save(key, np.full(100, 0.1, np.float32), SR)
    monkeypatch.setattr(nr, "LOCKED_FIRST_PAUSE", 0.0)
    monkeypatch.setattr(nr.os, "replace", lambda s, d: (_ for _ in ()).throw(_locked()))
    cache.save(key, np.full(100, 0.2, np.float32), SR)        # the reader holds the finished twin: no error
    assert cache.load(key) is not None and not list(cache.dir.glob("*.part.flac"))


def test_a_book_with_repeated_sentences_narrates_with_parallel_writers(tmp_path):
    verse = "And there was evening, and there was morning."
    book = Book("Repeats", "A", "en", [Chapter("One", " ".join([verse] * 12)), Chapter("Two", f"{verse} {verse}")])
    res, engine, ff, _ = run(tmp_path, book=book, options=nr.NarrationOptions(keep_cache=True))
    assert res.chunks >= 2 and res.files
    assert not list((res.out_dir / ".cache").glob("*.part.flac"))


def test_old_partial_files_are_swept_but_fresh_ones_kept(tmp_path):
    cache = nr.ChunkCache(tmp_path / "c")
    cache.dir.mkdir(parents=True)
    old, fresh = cache.dir / "abc.part.flac", cache.dir / "def.1-2-3.part.flac"
    old.write_bytes(b"x")
    fresh.write_bytes(b"y")
    os.utime(old, (1, 1))
    assert cache.sweep_partial() == 1 and not old.exists() and fresh.exists()
    assert nr.ChunkCache(tmp_path / "missing").sweep_partial() == 0


def test_a_leftover_partial_file_does_not_count_as_cached(tmp_path, monkeypatch):
    """The engine is pre-loaded only for a fresh job; a stray .part.flac must not make the job look resumed."""
    from tests.test_narration import book3

    loads = []
    engine = FakeEngine()
    out = tmp_path / "out"
    job = nr.job_dir_for(book3(), out, nr.NarrationOptions())
    (job / ".cache").mkdir(parents=True)
    (job / ".cache" / "stale.9-9-9.part.flac").write_bytes(b"x")     # fresh, so not swept
    res = nr.narrate_book(book3(), lambda: loads.append(1) or engine, engine.tag, out, language="english",
                          ffmpeg="ffmpeg", run=FakeFfmpeg())
    assert res.files and loads == [1]


# ----------------------------------------------------------------------------- the final export (ffmpeg output)
def test_export_waits_for_an_output_file_held_by_a_player(tmp_path, monkeypatch):
    out = tmp_path / "book.mp3"
    out.write_bytes(b"old")
    state = {"locked": 2}

    def in_use(p):
        if state["locked"]:
            state["locked"] -= 1
            return True
        return False

    monkeypatch.setattr(ex, "file_in_use", in_use)
    ff, pauses = FakeFfmpeg(), []
    ex._run_checked(ff, ["ffmpeg", "-i", "x", str(out)], out, sleep=pauses.append)
    assert out.read_bytes() == b"fake" and pauses == [0.2, 0.4]


def test_export_reports_a_file_that_stays_open(tmp_path, monkeypatch):
    from core import i18n

    i18n.set_language("en")
    out = tmp_path / "book.mp3"
    out.write_bytes(b"old")
    monkeypatch.setattr(ex, "file_in_use", lambda p: True)
    with pytest.raises(NarrationError) as err:
        ex._run_checked(lambda cmd: (1, "Permission denied"), ["ffmpeg", str(out)], out, sleep=lambda s: None)
    assert "book.mp3" in str(err.value) and "another program" in str(err.value)


def test_a_plain_ffmpeg_failure_is_not_retried(tmp_path):
    from core import i18n

    i18n.set_language("en")
    out = tmp_path / "book.mp3"
    calls = []
    with pytest.raises(NarrationError) as err:
        ex._run_checked(lambda cmd: calls.append(1) or (1, "boom"), ["ffmpeg", str(out)], out, sleep=lambda s: None)
    assert calls == [1] and "Could not create" in str(err.value)


def test_file_in_use_is_false_for_a_normal_or_missing_file(tmp_path):
    p = tmp_path / "f.mp3"
    assert not ex.file_in_use(p)
    p.write_bytes(b"x")
    assert not ex.file_in_use(p)
