"""Ordinal numbers by context before synthesis (core/ordinals.py + core/ordinal_rules.py): text rules, the narration
pipeline wiring, the Settings checkbox, the CLI flag and the chunk check.  Fakes only."""
from types import SimpleNamespace

import pytest

from core import narration as nr
from core import num_words as nw
from core import ordinals as od
from core.ordinal_rules import RULES


# ----------------------------------------------------------------------------- Russian
@pytest.mark.parametrize("text, spoken", [
    ("день 1", "день первый"),
    ("Глава 1. В начале", "Глава первая. В начале"),
    ("И был вечер, и было утро: День 3.", "И был вечер, и было утро: День третий."),
    ("в главе 3 сказано", "в главе третьей сказано"),
    ("до дня 7", "до дня седьмого"),
    ("Глава IV", "Глава четвёртая"),
    ("часть 2", "часть вторая"),
    ("стих 16", "стих шестнадцатый"),
    ("псалом 22", "псалом двадцать второй"),
    ("том 3", "том третий"),
    ("явление 2", "явление второе"),
    ("страница 125", "страница сто двадцать пятая"),
    ("1-й день, 2-я глава", "первый день, вторая глава"),
    ("3-го января", "третьего января"),
    ("в 5-м томе", "в пятом томе"),
    ("с 5-м томом", "с пятым томом"),
    ("на 3-м этаже", "на третьем этаже"),
    ("1-й главы", "первой главы"),
    ("90-х годов", "девяностых годов"),
    ("в 90-е", "в девяностые"),
    ("1990-е годы", "тысяча девятьсот девяностые годы"),
    ("2026-го года", "две тысячи двадцать шестого года"),
])
def test_russian_ordinals_follow_the_noun(text, spoken):
    assert od.apply_ordinals(text, "Russian") == spoken


@pytest.mark.parametrize("text", [
    "Он дал 5 рублей",             # no trigger word: the normalizer reads the cardinal
    "главы 1-3",                   # range
    "главы 1 – 3",
    "глава 1:5",                   # reference
    "стих 3.16",
    "главы 1 и 2",                 # list
    "главы 1, 2",
    "день 1 сентября",             # date
    "глава 12а",                   # not a plain number
    "в 3 дня",                     # number before the noun = a count
    "Глава iv",                    # lower-case "roman" is a word, not a numeral
])
def test_russian_numbers_that_are_not_ordinals_stay(text):
    assert od.apply_ordinals(text, "ru") == text


def test_stressed_ordinals_keep_their_ending():
    """num_words declined "второй" to "вторый" in the nominative (also used by dates and the AI disclosure)."""
    assert nw.ru_ordinal(2) == "второй" and nw.ru_ordinal(6) == "шестой" and nw.ru_ordinal(40) == "сороковой"
    assert nw.ru_ordinal(22, "acc") == "двадцать второй" and nw.ru_ordinal(2, "nom", "f") == "вторая"
    assert nw.ru_ordinal(2, "gen") == "второго" and nw.ru_ordinal(1, "nom") == "первый"


# ----------------------------------------------------------------------------- English and German
def test_english_written_ordinals_and_heading_roman_numerals():
    text = "On the 1st and 22nd and 103rd day.\nChapter IV\nIn this part I explain. Part II: The End. Chapter 4"
    assert od.apply_ordinals(text, "English") == (
        "On the first and twenty second and one hundred third day.\nChapter four\n"
        "In this part I explain. Part two: The End. Chapter 4")


def test_german_dot_ordinals_take_the_article_ending():
    text = "Das 3. Kapitel. Am 3. Oktober. Der 2. Teil. 1. Kapitel. Im 21. Jahrhundert. Ein 2. Mal. Kapitel 3"
    assert od.apply_ordinals(text, "de") == ("Das dritte Kapitel. Am dritten Oktober. Der zweite Teil. erstes Kapitel. "
                                             "Im einundzwanzigsten Jahrhundert. Ein zweites Mal. Kapitel 3")
    assert od.de_cardinal(1999) == "eintausendneunhundertneunundneunzig" and od.de_ordinal_stem(103) == "einhundertdritt"


def test_unsupported_language_and_empty_text_pass_through():
    assert od.apply_ordinals("chapitre 1", "French") == "chapitre 1"
    assert od.apply_ordinals("", "ru") == "" and od.ordinal_step("Auto") is None
    assert od.language_key("ru-RU") == "ru" and od.language_key("Deutsch") == "de"
    assert set(od.supported_languages()) == {"ru", "en", "de"} <= set(RULES)


def test_roman_values():
    assert [od.roman_value(t) for t in ("I", "IV", "XIV", "MCMXC", "IIII", "", "IC")] == [1, 4, 14, 1990, None, None, None]


def test_every_russian_trigger_form_has_a_known_gender_and_case():
    for lemma, (gender, forms) in RULES["ru"]["after"].items():
        assert gender in ("m", "f", "n", "p") and lemma in forms
        assert set(forms.values()) <= set(nw.CASES)
        for form in forms:
            assert od.apply_ordinals(f"{form} 1", "ru") != f"{form} 1", form


# ----------------------------------------------------------------------------- the narration pipeline
def test_ordinals_run_before_the_russian_normalizer():
    steps = nr.text_steps("Russian", "ru", nr.NarrationOptions())
    spoken = nr.prepare_text("Глава 2. У меня 3 яблока.", nr.NarrationOptions(), steps)
    assert spoken.startswith("Глава вторая") and "3" not in spoken
    off = nr.text_steps("Russian", "ru", nr.NarrationOptions(ordinals=False))
    # ru-normalizr itself already reads "глава 2" as an ordinal, but not "день" / "стих" / "псалом"
    assert nr.prepare_text("День 1, стих 16.", nr.NarrationOptions(), steps) == "День первый, стих шестнадцатый."
    assert "первый" not in nr.prepare_text("День 1.", nr.NarrationOptions(), off)


def test_auto_language_falls_back_to_the_book_language_and_english_has_no_base_normalizer():
    assert nr.text_steps("", "ru", nr.NarrationOptions())("день 1") == "день первый"
    en = nr.text_steps("English", "en", nr.NarrationOptions())
    assert en("the 2nd day") == "the second day" and nr.text_steps("English", "en", nr.NarrationOptions(ordinals=False)) is None
    assert nr.chain_steps(None, None) is None


def test_narration_sends_the_ordinal_text_to_the_engine(tmp_path):
    from core.book_parsers import Book, Chapter
    from tests.test_narration import FakeEngine, run

    book = Book("Days", "A", "en", [Chapter("Chapter IV", "It was the 1st day."), Chapter("Part 2", "On the 3rd.")])
    _, engine, _, _ = run(tmp_path, book=book, options=nr.NarrationOptions(speak_titles=True))
    said = " ".join(engine.calls)
    assert "first day" in said and "third" in said and "Chapter four" in said
    _, engine2, _, _ = run(tmp_path / "off", engine=FakeEngine(), book=book,
                           options=nr.NarrationOptions(speak_titles=True, ordinals=False))
    assert "1st" in " ".join(engine2.calls)


def test_chunk_check_reads_the_recognised_text_the_same_way():
    from core import chunk_check as cc

    ch = cc.make_default_checker("Russian", ready=lambda repo: f"/m/{repo}")
    assert ch.normalize("день 2") == "день второй"
    off = cc.make_default_checker("Russian", cc.ChunkCheckOptions(ordinals=False), ready=lambda repo: f"/m/{repo}")
    assert "второй" not in off.normalize("день 2")


def test_runner_passes_the_setting_to_the_chunk_check(monkeypatch):
    from core import chunk_check as cc
    from workers import narration_runner as runner
    from tests.test_narration import book3

    made = []
    monkeypatch.setattr(cc, "make_default_checker", lambda lang, opts: made.append(opts.ordinals) or "C")
    monkeypatch.setattr(runner.nr, "narrate_book", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(runner.tts_engine, "make_engine_factory", lambda v, lang: None)
    monkeypatch.setattr(runner.tts_engine, "engine_tag", lambda v, lang: "t")
    voice = SimpleNamespace(language="Russian", name="V")
    runner.run_narration(runner.NarrationJob(book3(), voice, "out", nr.NarrationOptions(check_chunks=True, ordinals=False)),
                         lambda p: None, None, None)
    assert made == [False]


# ----------------------------------------------------------------------------- the setting, Settings dialog and CLI
def test_setting_defaults_on_and_is_remembered():
    assert od.load_enabled() is True
    od.save_enabled(False)
    assert od.load_enabled() is False
    od._file().write_text("not json", encoding="utf-8")
    assert od.load_enabled() is True


def test_cli_flags(tmp_path):
    import cli as user_cli
    from tests.test_user_cli import _make_voice

    lib = _make_voice(tmp_path)
    book = tmp_path / "b.txt"
    book.write_text("Глава 1\n\nДень 1. Текст главы.\n", encoding="utf-8")
    seen = []

    def fake_run(job, progress, cancel, pause):
        seen.append(job.options.ordinals)
        return SimpleNamespace(out_dir=job.out_dir, files=[])

    base = ["narrate", str(book), "--voice", "Model Voice", "--out", str(tmp_path / "o")]
    assert user_cli.main(base, run_narration_fn=fake_run, library=lib) == 0
    assert user_cli.main(base + ["--no-ordinals"], run_narration_fn=fake_run, library=lib) == 0
    od.save_enabled(False)
    assert user_cli.main(base, run_narration_fn=fake_run, library=lib) == 0
    assert user_cli.main(base + ["--ordinals"], run_narration_fn=fake_run, library=lib) == 0
    assert seen == [True, False, False, True]


def test_settings_dialog_checkbox(tmp_path):
    pytest.importorskip("PySide6")
    from core import i18n
    from core.voice_library import VoiceLibrary
    from tests.test_studio import make_studio
    from PySide6.QtWidgets import QApplication
    from ui.settings_dialog import SettingsDialog

    _ = QApplication.instance() or QApplication([])
    i18n.set_language("ru")
    d = SettingsDialog(make_studio(VoiceLibrary(tmp_path / "voices")))
    assert d.chk_ordinals.isChecked() and d.chk_ordinals.text() == i18n.tr("narrset.ordinals")
    assert d.chk_soundscape.isChecked() is False
    assert d.chk_soundscape.text() == i18n.tr("narrset.soundscape")
    d.chk_ordinals.setChecked(False)
    assert od.load_enabled() is False
    d.reset_narration()
    assert d.chk_ordinals.isChecked() and od.load_enabled() is True
    assert d.chk_soundscape.isChecked() is False
    i18n.set_language("en")


def test_translations_exist_in_every_ui_language():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "locales"
    for lang in ("en", "ru", "de", "uk", "lv"):
        cat = json.loads((root / f"{lang}.json").read_text(encoding="utf-8"))
        for key in ("narrset.ordinals", "narrset.ordinals_tip", "err.narration_file_in_use"):
            assert cat.get(key), (lang, key)
