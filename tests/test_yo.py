"""Russian letter yo before synthesis: the safe dictionary, the Prepare step, and the narrate flag.

No model is downloaded. The base TTS model is not loaded: a fake engine records the text it would have spoken.
"""
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from core import text_prep as tp
from core import yo
from core.book_parsers import Book, Chapter
from core.book_prep import PrepPlan

ROOT = Path(__file__).resolve().parents[1]
DICT = ROOT / "core" / "data" / "yo_safe.txt"
DICT_SHA = "11fc6d9c3cc6d0fff6a21cd642fa6aecbf3f9eb784f84ccd78141f32e51e0693"


def test_dictionary_sources_check_out_as_lf():
    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    for name in (
        "core/data/yo_safe.txt",
        "core/data/yo_not_safe.txt",
        "core/data/yo_additions.json",
        "core/data/yo_safe.LICENSE",
        "core/data/YO_DATASET.md",
    ):
        assert f"{name} text eol=lf" in attrs
    assert "core/data/*.gz binary" in attrs
    assert "* text=auto" not in attrs


def test_shipped_dictionary_is_the_pinned_mit_file():
    raw = DICT.read_bytes()
    assert len(raw) == 864903 and hashlib.sha256(raw).hexdigest() == DICT_SHA
    assert b"\r" not in raw
    licence = (ROOT / "core" / "data" / "yo_safe.LICENSE").read_text(encoding="utf-8")
    assert "MIT" in licence and "Denis Seleznev" in licence and "e2yo/eyo-kernel" in licence
    runtime = ROOT / "core" / "data" / "yo_runtime.tsv.gz"
    dataset = ROOT / "core" / "data" / "yo_dataset.jsonl.gz"
    assert runtime.stat().st_size == 1323641 and dataset.stat().st_size == 1257057
    assert "core\\data;core\\data" in (ROOT / "build.bat").read_text(encoding="utf-8")
    assert "core\\data;core\\data" in (ROOT / "build_thin.bat").read_text(encoding="utf-8")
    from tools.make_linux_package import app_files

    shipped = set(app_files(ROOT))
    assert "core/data/yo_safe.txt" in shipped and "core/data/yo_runtime.tsv.gz" in shipped
    assert "core/data/yo_dataset.jsonl.gz" in shipped and "core/data/YO_DATASET.md" in shipped


def test_endings_underscore_and_capitals_follow_the_dictionary_rules():
    table = yo.load("Ёлкин(|а|е)\n_киёв# город Киев\nещё\n")
    assert yo.restore("Елкин, Елкина и ЕЛКИН.", table) == "Ёлкин, Ёлкина и ЕЛКИН."
    assert yo.restore("киев и Киев", table) == "киёв и Киев"
    assert yo.restore("еще Еще ЕЩЕ", table) == "ещё Ещё ЕЩЕ"


def test_dataset_records_plain_text_and_rebuilds_the_same_bytes():
    from tools.build_yo_dataset import build

    before = (ROOT / "core" / "data" / "yo_runtime.tsv.gz").read_bytes()
    dataset = (ROOT / "core" / "data" / "yo_dataset.jsonl.gz").read_bytes()
    rows = [json.loads(line) for line in gzip.decompress(dataset).decode("utf-8").splitlines()]
    plain = next(row for row in rows if row["word"] == "текст")
    assert plain["yo_form"] == "текст" and plain["type"] == "unambiguous" and plain["stress"] == 0
    assert plain["source"] == "project"
    homo = next(row for row in rows if row["word"] == "все")
    assert homo["type"] == "homograph" and homo["yo_form"] == "всё" and homo["variants"]
    keys = {line.split("\t", 1)[0] for line in gzip.decompress(before).decode("utf-8").splitlines()}
    assert "текст" not in keys and "все" not in keys and "берег" not in keys
    stats = build()
    assert stats["runtime_bytes"] == 1323641 and stats["dataset_rows"] == 139996
    rebuilt = (ROOT / "core" / "data" / "yo_runtime.tsv.gz").read_bytes()
    rebuilt_dataset = (ROOT / "core" / "data" / "yo_dataset.jsonl.gz").read_bytes()
    # Python 3.12 gzip.compress(mtime=0) writes OS byte 3; GzipFile writes 255.
    # The Windows runner is 3.11, so the shipped header is the GzipFile one.
    assert before[:10] == bytes.fromhex("1f8b08000000000002ff")
    assert rebuilt == before and rebuilt_dataset == dataset


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
    assert tp.prepare_text_block(src, "ru") == "В тексте ещё всё. мед. училище."
    assert tp.prepare_text_block(src, "en") == src
    assert tp.prepare_text_block(src, "de", tp.PrepOptions(frozenset({tp.STEP_YO}))) == src
    book = Book("T", "", "en", [Chapter("Still.", src)])
    prepared, rep = tp.prepare_book(book, tp.PrepOptions(frozenset({tp.STEP_YO})))
    assert rep.skipped == [tp.STEP_YO] and prepared.chapters[0].text == src and prepared.chapters[0].title == "Still."
    ru = Book("T", "", "ru", [Chapter("Еще.", src)])
    prepared_ru, rep_ru = tp.prepare_book(ru, tp.PrepOptions(frozenset({tp.STEP_YO})))
    assert rep_ru.skipped == [] and prepared_ru.chapters[0].title == "Ещё."
    assert prepared_ru.chapters[0].text == "В тексте ещё всё. мед. училище."
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
    assert "В тексте ещё всё. мед. училище." in said
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


FIXTURES = ROOT / "tests" / "fixtures" / "yo"

VSE_CASES = [
    # kept as "vse": a plural word, pronoun, subject or verb next to it, or no rule fires
    ("Все люди пришли.", "Все люди пришли."),
    ("Пришли все.", "Пришли все."),
    ("Мы все.", "Мы все."),
    ("Все они знали.", "Все они знали."),
    ("Все ее книги на полке.", "Все её книги на полке."),
    ("Все пятеро молчали.", "Все пятеро молчали."),
    ("Все громко засмеялись.", "Все громко засмеялись."),
    ("Все, кто пришел, сели.", "Все, кто пришёл, сели."),
    ("Все это знают.", "Все это знают."),
    ("Все так делают.", "Все так делают."),
    ("Все будут рады.", "Все будут рады."),
    ("Все остальные ушли.", "Все остальные ушли."),
    ("Все хорошо знают это.", "Все хорошо знают это."),
    ("Он съел все яблоки.", "Он съел все яблоки."),
    ("Все семь дней.", "Все семь дней."),
    ("Все мы люди.", "Все мы люди."),
    ("ВСЕ будет хорошо.", "ВСЕ будет хорошо."),
    # written as "vsyo": a singular verb or neuter adjective, a clause end, ", chto", fixed phrases, comparatives
    ("Он все понял.", "Он всё понял."),
    ("Все равно я пойду.", "Всё равно я пойду."),
    ("Он все-таки пришел.", "Он всё-таки пришёл."),
    ("Все было тихо.", "Всё было тихо."),
    ("Вот и все.", "Вот и всё."),
    ("«Вот и все», — сказал он.", "«Вот и всё», — сказал он."),
    ("Это все, что у меня есть.", "Это всё, что у меня есть."),
    ("Все хорошо.", "Всё хорошо."),
    ("Становилось все темнее.", "Становилось всё темнее."),
    ("Она все говорила и говорила.", "Она всё говорила и говорила."),
    ("Я все знаю.", "Я всё знаю."),
    ("Все еще идет дождь.", "Всё ещё идёт дождь."),
    ("Все дело в том, что он устал.", "Всё дело в том, что он устал."),
    ("Все будет хорошо.", "Всё будет хорошо."),
    ("Все новое пугает.", "Всё новое пугает."),
    ("Там все: письма, книги.", "Там всё: письма, книги."),
]


@pytest.mark.parametrize("src,want", VSE_CASES)
def test_vse_is_decided_by_context(src, want):
    assert yo.restore(src) == want
    assert yo.restore(want) == want


def test_vse_context_can_be_turned_off_and_is_counted():
    assert yo.restore("Вот и все.", context=False) == "Вот и все."
    out, n = yo.restore_counted("Вот и все. Все люди здесь.")
    assert out == "Вот и всё. Все люди здесь." and n == 1
    assert yo.context_path().name == "yo_context.json" and yo.context_path().is_file()


def test_dialog_fixture_restores_every_yo_and_adds_none():
    """The 700 test dialogue: 22 yo in the reference (6 of them "vsyo"), none wrong."""
    import re

    src = (FIXTURES / "dialog-ai-torah-01.txt").read_text(encoding="utf-8")
    ref = (FIXTURES / "dialog-ai-torah-01.yo-reference.txt").read_text(encoding="utf-8")
    out = yo.restore(src)
    words = re.compile(r"\w+")
    a, b, c = words.findall(src), words.findall(ref), words.findall(out)
    assert len(a) == len(b) == len(c)
    yo_lo, yo_up = chr(0x0451), chr(0x0401)
    needed = [i for i, w in enumerate(b) if yo_lo in w or yo_up in w]
    assert len(needed) == 22
    assert [i for i in needed if c[i] != b[i]] == []
    assert [i for i, (r, o) in enumerate(zip(b, c)) if r != o] == []
    assert out == ref
