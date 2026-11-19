"""Book working folder (core/workspace.py): remembered folder, book copy only with permission, clean-up after a job."""
from pathlib import Path

import pytest

from core import narration as nr
from core import workspace as ws
from core.book_parsers import Book, Chapter


def job_folder(tmp_path):
    job = tmp_path / "work" / "My Book"
    for d in (".cache", ".work", ".debug", ".translation"):
        (job / d).mkdir(parents=True)
        (job / d / "x.bin").write_bytes(b"0" * 100)
    (job / "My Book.m4b").write_bytes(b"a" * 1000)
    (job / "My Book - MP3").mkdir()
    (job / "My Book - MP3" / "01.mp3").write_bytes(b"b" * 500)
    (job / "translation_de.txt").write_text("Hallo", encoding="utf-8")
    (job / "book.fb2").write_text("<fb2/>", encoding="utf-8")
    return job, [job / "My Book.m4b", job / "My Book - MP3" / "01.mp3"]


def test_folder_is_remembered_and_falls_back_to_the_default(tmp_path):
    default = tmp_path / "default"
    assert ws.load_folder(default) == default
    ws.save_folder(tmp_path / "mine")
    assert ws.load_folder(default) == tmp_path / "mine"           # absent leaf of an existing parent: created later
    ws.save_folder(tmp_path / "gone" / "deeper")
    assert ws.load_folder(default) == default                     # the drive/parent is gone


@pytest.mark.parametrize("mode", [ws.PLACE_COPY, ws.PLACE_MOVE])
def test_place_book_copies_or_moves_and_never_overwrites(tmp_path, mode):
    src = tmp_path / "in" / "book.txt"
    src.parent.mkdir()
    src.write_text("text", encoding="utf-8")
    job = tmp_path / "job"
    (job).mkdir()
    (job / "book.txt").write_text("other", encoding="utf-8")
    got = ws.place_book(src, job, mode)
    assert got == job / "book (2).txt" and got.read_text(encoding="utf-8") == "text"
    assert (job / "book.txt").read_text(encoding="utf-8") == "other"
    assert src.exists() == (mode == ws.PLACE_COPY)
    assert ws.place_book(got, job, mode) == got                   # already inside: nothing happens


def test_place_book_leave_and_identical_copy(tmp_path):
    src = tmp_path / "book.txt"
    src.write_text("same", encoding="utf-8")
    job = tmp_path / "job"
    assert ws.place_book(src, job, ws.PLACE_LEAVE) == src and not job.exists()
    first = ws.place_book(src, job, ws.PLACE_COPY)
    assert ws.place_book(src, job, ws.PLACE_COPY) == first and len(list(job.iterdir())) == 1


def test_scan_sorts_the_groups(tmp_path):
    job, results = job_folder(tmp_path)
    f = ws.scan_job(job, results, job / "book.fb2")
    assert f.results == results and f.original == [job / "book.fb2"]
    assert set(f.prepared) == {job / ".debug", job / ".translation", job / "translation_de.txt"}
    assert set(f.temp) == {job / ".cache", job / ".work"}
    assert ws.scan_job(job, results, tmp_path / "elsewhere.fb2").original == []   # a book outside the folder is never touched


def test_clean_keeps_the_chosen_groups_and_always_drops_temp(tmp_path):
    job, results = job_folder(tmp_path)
    freed = ws.clean_job(ws.scan_job(job, results, job / "book.fb2"), keep_results=True, keep_original=False,
                         keep_prepared=False)
    assert freed == 4 * 100 + len("<fb2/>") + len("Hallo")
    assert sorted(p.name for p in job.iterdir()) == ["My Book - MP3", "My Book.m4b"]


def test_clean_without_results_removes_empty_chapter_folders(tmp_path):
    job, results = job_folder(tmp_path)
    ws.clean_job(ws.scan_job(job, results, job / "book.fb2"), keep_results=False)
    assert sorted(p.name for p in job.iterdir()) == [".debug", ".translation", "book.fb2", "translation_de.txt"]


def test_job_dir_for_matches_narrate_book_naming(tmp_path):
    book = Book("A: Title", "X", "en", [Chapter("One", "Text here.")])
    assert nr.job_dir_for(book, tmp_path) == tmp_path / "A Title"           # no ":" in a Windows folder name
