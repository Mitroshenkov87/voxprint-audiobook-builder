"""Russian letter yo before synthesis: the safe dictionary, the Prepare step, and the narrate flag.

No model is downloaded. The base TTS model is not loaded: a fake engine records the text it would have spoken.
"""
import hashlib
from pathlib import Path

import pytest

from core import text_prep as tp
from core import yo
from core.book_parsers import Book, Chapter
from core.book_prep import PrepPlan

ROOT = Path(__file__).resolve().parents[1]
DICT = ROOT / "core" / "data" / "yo_safe.txt"
DICT_SHA = "11fc6d9c3cc6d0fff6a21cd642fa6aecbf3f9eb784f84ccd78141f32e51e0693"


def test_shipped_dictionary_is_the_pinned_mit_file():
    raw = DICT.read_bytes()
    assert len(raw) == 864903 and hashlib.sha256(raw).hexdigest() == DICT_SHA
    assert b"\r" not in raw
    licence = (ROOT / "core" / "data" / "yo_safe.LICENSE").read_text(encoding="utf-8")
    assert "MIT" in licence and "Denis Seleznev" in licence and "e2yo/eyo-kernel" in licence
    assert "core\\data;core\\data" in (ROOT / "build.bat").read_text(encoding="utf-8")
    assert "core\\data;core\\data" in (ROOT / "build_thin.bat").read_text(encoding="utf-8")
    from tools.make_linux_package import app_files

    shipped = set(app_files(ROOT))
    assert "core/data/yo_safe.txt" in shipped and "core/data/yo_safe.LICENSE" in shipped


def test_endings_underscore_and_capitals_follow_the_dictionary_rules():
    table = yo.load("Ёлкин(|а|е)\n_киёв# город Киев\nещё\n")
    assert yo.restore("Елкин, Елкина и ЕЛКИН.", table) == "Ёлкин, Ёлкина и ЕЛКИН."
    assert yo.restore("киев и Киев", table) == "киёв и Киев"
    assert yo.restore("еще Еще ЕЩЕ", table) == "ещё Ещё ЕЩЕ"


def test_real_dictionary_restores_only_sure_words_and_keeps_abbreviations():
    text = "В тексте еще все берег. мед. училище."
    out, n = yo.restore_counted(text)
    assert n >= 1 and out == "В тексте ещё все берег. мед. училище."
    assert "тёкст" not in out and "\u0301" not in out and "+" not in out
    assert yo.restore(out) == out
    assert yo.restore("Hello, world.") == "Hello, world."
    assert yo.restore("") == ""


def test_prepare_step_is_russian_only_and_on_by_default():
    src = "В тексте еще все. мед. училище."
    assert tp.prepare_text_block(src, "ru") == "В тексте ещё все. мед. училище."
    assert tp.prepare_text_block(src, "en") == src
    assert tp.prepare_text_block(src, "de", tp.PrepOptions(frozenset({tp.STEP_YO}))) == src
    book = Book("T", "", "en", [Chapter("Still.", src)])
    prepared, rep = tp.prepare_book(book, tp.PrepOptions(frozenset({tp.STEP_YO})))
    assert rep.skipped == [tp.STEP_YO] and prepared.chapters[0].text == src and prepared.chapters[0].title == "Still."
    ru = Book("T", "", "ru", [Chapter("Еще.", src)])
    prepared_ru, rep_ru = tp.prepare_book(ru, tp.PrepOptions(frozenset({tp.STEP_YO})))
    assert rep_ru.skipped == [] and prepared_ru.chapters[0].title == "Ещё."
    assert prepared_ru.chapters[0].text == "В тексте ещё все. мед. училище."
    assert tp.STEP_YO in tp.STEP_KEYS and tp.STEP_YO in tp.RU_ONLY_STEPS


def test_chunk_step_restores_yo_before_the_number_reader_and_can_be_disabled():
    from core import narration as nr

    on = nr.text_steps("Russian", "ru", nr.NarrationOptions(ordinals=False))
    said = nr.prepare_text("В тексте еще мед. училище.", nr.NarrationOptions(), on)
    assert "ещё" in said and "текст" in said and "тёкст" not in said and "мёд" not in said
    off = nr.text_steps("Russian", "ru", nr.NarrationOptions(ordinals=False, yo=False))
    plain = nr.prepare_text("В тексте еще мед. училище.", nr.NarrationOptions(), off)
    assert "еще" in plain and "ещё" not in plain
    assert nr.text_steps("English", "en", nr.NarrationOptions(ordinals=False, yo=True)) is None


def test_fake_engine_receives_the_restored_text(tmp_path):
    from core import narration as nr
    from tests.test_narration import FakeEngine, run

    book = Book("T", "A", "ru", [Chapter("", "В тексте еще все. мед. училище.")])
    options = nr.NarrationOptions(
        speak_titles=False, ordinals=False, yo=True,
        prep=PrepPlan(rules=tp.PrepOptions(frozenset({tp.STEP_YO, tp.STEP_NUMBERS}))),
    )
    _res, engine, _ff, _ev = run(tmp_path, engine=FakeEngine(), book=book, language="Russian", options=options)
    said = " ".join(engine.calls)
    assert "В тексте ещё все. мед. училище." in said
    _res, engine2, _ff, _ev = run(
        tmp_path / "off", engine=FakeEngine(), book=book, language="Russian",
        options=nr.NarrationOptions(speak_titles=False, ordinals=False, yo=False))
    assert "еще" in " ".join(engine2.calls) and "ещё" not in " ".join(engine2.calls)


def test_cli_restores_yo_by_default(tmp_path):
    import cli as user_cli
    from tests.test_user_cli import _make_voice

    lib = _make_voice(tmp_path)
    book = tmp_path / "b.txt"
    book.write_text("Глава 1\n\nВ тексте еще все.\n", encoding="utf-8")
    seen = []

    def fake_run(job, progress, cancel, pause):
        seen.append((job.options.yo, None if job.options.prep is None else set(job.options.prep.rules.steps)))
        return type("R", (), {"out_dir": job.out_dir, "files": []})()

    base = ["narrate", str(book), "--voice", "Model Voice", "--out", str(tmp_path / "o")]
    assert user_cli.main(base, run_narration_fn=fake_run, library=lib) == 0
    assert user_cli.main(base + ["--no-yo"], run_narration_fn=fake_run, library=lib) == 0
    assert user_cli.main(base + ["--yo"], run_narration_fn=fake_run, library=lib) == 0
    assert seen == [(True, {tp.STEP_YO}), (False, None), (True, {tp.STEP_YO})]
