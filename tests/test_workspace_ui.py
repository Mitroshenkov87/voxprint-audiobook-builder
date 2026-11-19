"""Narrate window and the book working folder: copy only with permission, clean-up dialog, remembered folder."""
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from core import narration as nr  # noqa: E402
from core import workspace as ws  # noqa: E402
from tests.test_studio import app, lib  # noqa: E402,F401


def test_narrate_window_asks_before_copying_and_offers_cleanup(app, lib, tmp_path):
    from core import i18n
    from tests.test_studio import add_voice, wait_for, write_book
    from tests.test_voice_catalog_ui import narrate
    i18n.set_language("en")
    add_voice(lib, tmp_path, "Anna")
    asked, cleaned = [], []

    def runner(job, progress, cancel, pause):
        jd = nr.job_dir_for(job.book, job.out_dir, job.options)
        (jd / ".cache").mkdir(parents=True, exist_ok=True)
        (jd / "B.m4b").write_bytes(b"x")
        return nr.NarrationResult(jd, [jd / "B.m4b"], chapters=1, chunks=1)

    def ask_cleanup(files):
        cleaned.append(files)
        return {"keep_results": True, "keep_original": False, "keep_prepared": True}

    n = narrate(lib, tmp_path, [], runner=runner, ask_place=lambda b, j: asked.append((b, j)) or ws.PLACE_COPY,
                ask_cleanup=ask_cleanup)
    assert n.btn_out.text() == "Working folder…" and n.advanced_box.isAncestorOf(n.lbl_out_hint)
    book = write_book(tmp_path)
    n.load_book_file(book)
    assert n.start() and wait_for(lambda: n.result is not None)
    job = nr.job_dir_for(n.book, n.out_dir, n.options())
    assert asked == [(book, job)] and book.exists() and n.book_path.parent == job
    assert cleaned and cleaned[0].original == [job / "book.txt"]
    assert not (job / ".cache").exists() and not (job / "book.txt").exists() and (job / "B.m4b").exists()
    n.result = None
    assert n.start() and wait_for(lambda: n.result is not None)
    assert len(asked) == 1                                          # asked once per book
    n.shutdown()


def test_choose_folder_is_remembered(app, lib, tmp_path):
    from tests.test_voice_catalog_ui import narrate
    n = narrate(lib, tmp_path, [], pick_folder=lambda: str(tmp_path / "chosen"))
    n.choose_folder()
    assert ws.load_folder(Path("x")) == tmp_path / "chosen"
    n.shutdown()
