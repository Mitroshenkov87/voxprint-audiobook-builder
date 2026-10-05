"""Spoken AI disclosure (core/ai_disclosure.py): phrase per language (month + year only), first chunk, opt-in UI switch."""
import datetime as dt

import pytest

from core import ai_disclosure as ad
from core import narration as nr
from core.book_parsers import Book, Chapter
from tests.test_narration import FakeEngine, run

OCT = dt.date(2026, 10, 5)


def test_phrases_say_month_and_year_in_words_without_the_day():
    assert ad.phrase("russian", "Анна", OCT) == ("Эта книга озвучена с помощью искусственного интеллекта, голос Анна, "
                                                "октябрь две тысячи двадцать шестого года.")
    assert ad.phrase("en", "Anna", OCT).endswith("voice Anna, October twenty twenty six.")
    assert ad.phrase("de", "Anna", OCT) == ("Dieses Buch wurde mit Hilfe künstlicher Intelligenz vertont, Stimme Anna, "
                                           "Oktober zweitausendsechsundzwanzig.")
    assert ad.phrase("french", "Anna", OCT).startswith("This book")                # unsupported language -> English
    assert "5" not in ad.phrase("ru", "Анна", OCT) and "пят" not in ad.phrase("ru", "Анна", OCT)
    assert ad.phrase("en", "", OCT) == "This book was narrated with the help of artificial intelligence, October twenty twenty six."


@pytest.mark.parametrize("year,words", [(2000, "zweitausend"), (2021, "zweitausendeinundzwanzig"),
                                        (2100, "zweitausendeinhundert"), (1999, "neunzehnhundertneunundneunzig")])
def test_german_years(year, words):
    assert ad.de_year(year) == words


def test_disclosure_is_the_first_chunk_in_the_narrated_language(tmp_path):
    book = Book("B", "A", "en", [Chapter("One", "First sentence is here.")])
    _res, eng, *_ = run(tmp_path, engine=FakeEngine(), book=book, options=nr.NarrationOptions(
        speak_titles=True, ai_disclosure=True, disclosure_date=OCT))
    assert eng.calls[0] == ad.phrase("english", "Anna", OCT) and eng.calls[1:] == ["One.", "First sentence is here."]
    _res, eng2, *_ = run(tmp_path / "off", engine=FakeEngine(), book=book, options=nr.NarrationOptions(speak_titles=False))
    assert eng2.calls == ["First sentence is here."]                                   # default OFF


def test_prepend_renumbers():
    from core.chunker import Chunk
    out = ad.prepend([Chunk(0, 2, "a", 100), Chunk(1, 2, "b", 100)], "AI.")
    assert [(c.index, c.chapter, c.text) for c in out] == [(0, 2, "AI."), (1, 2, "a"), (2, 2, "b")]
    assert ad.prepend([], "AI.") == []


def test_switch_is_remembered_and_off_by_default():
    assert ad.load_enabled() is False
    ad.save_enabled(True)
    assert ad.load_enabled() is True


def test_narrate_window_checkbox_tooltip_and_option(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from core import i18n
    from core.voice_library import VoiceLibrary
    from tests.test_voice_catalog_ui import narrate
    QApplication.instance() or QApplication([])
    i18n.set_language("en")
    n = narrate(VoiceLibrary(tmp_path / "voices"), tmp_path, [], runner=lambda *a: None)
    assert not n.chk_disclosure.isChecked() and not n.options().ai_disclosure
    assert "EU AI Act" in n.chk_disclosure.toolTip()
    n.chk_disclosure.setChecked(True)
    assert n.options().ai_disclosure and ad.load_enabled()
    i18n.set_language("ru")
    n.retranslate()
    assert "ИИ" in n.chk_disclosure.text() and "AI Act" in n.chk_disclosure.toolTip()
    n.shutdown()
