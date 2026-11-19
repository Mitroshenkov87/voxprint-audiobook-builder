"""Rule-based book preparation (Russian and English): numbers, dates, abbreviations, layout, headings, links, noise."""
import json
import re
from collections import Counter

import pytest

from core import num_words as nw
from core import text_prep as tp
from core.book_parsers import Book, Chapter


def prep(text, lang, steps=None):
    opts = tp.PrepOptions(frozenset(steps)) if steps is not None else tp.PrepOptions()
    return tp.prepare_text_block(text, lang, opts)


# ----------------------------------------------------------------------------- number words

def test_russian_cardinals_gender_and_thousand():
    assert nw.ru_cardinal(1) == "один" and nw.ru_cardinal(1, "f") == "одна" and nw.ru_cardinal(1, "n") == "одно"
    assert nw.ru_cardinal(2, "f") == "две" and nw.ru_cardinal(21, "f") == "двадцать одна"
    assert nw.ru_cardinal(1000) == "тысяча" and nw.ru_cardinal(2000) == "две тысячи"
    assert nw.ru_cardinal(1984) == "тысяча девятьсот восемьдесят четыре"
    assert nw.ru_cardinal(5000000) == "пять миллионов" and nw.ru_cardinal(0) == "ноль"
    assert nw.ru_cardinal(12345) == "двенадцать тысяч триста сорок пять"


def test_russian_ordinals_cases_and_genders():
    assert nw.ru_ordinal(5) == "пятый" and nw.ru_ordinal(5, "gen", "f") == "пятой"
    assert nw.ru_ordinal(3, "gen") == "третьего" and nw.ru_ordinal(3, "nom", "f") == "третья"
    assert nw.ru_ordinal(2, "prep") == "втором" and nw.ru_ordinal(40, "dat") == "сороковому"
    assert nw.ru_ordinal(1999, "gen") == "тысяча девятьсот девяносто девятого"
    assert nw.ru_ordinal(2000, "gen") == "двухтысячного" and nw.ru_ordinal(12, "nom", "n") == "двенадцатое"
    assert nw.ru_ordinal(21, "ins") == "двадцать первым" and nw.ru_ordinal(90, "nom", "p") == "девяностые"


def test_russian_decimals_and_plurals():
    assert nw.ru_decimal("2", "5") == "две целых пять десятых"
    assert nw.ru_decimal("1", "05") == "одна целая пять сотых"
    assert nw.ru_decimal("21", "1") == "двадцать одна целая одна десятая"
    assert nw.ru_plural(21, ("рубль", "рубля", "рублей")) == "рубль" and nw.ru_plural(12, ("а", "б", "в")) == "в"


def test_english_numbers():
    assert nw.en_cardinal(1234) == "one thousand two hundred thirty four" and nw.en_cardinal(0) == "zero"
    assert nw.en_ordinal(21) == "twenty first" and nw.en_ordinal(12) == "twelfth" and nw.en_ordinal(40) == "fortieth"
    assert nw.en_year(1999) == "nineteen ninety nine" and nw.en_year(1905) == "nineteen oh five"
    assert nw.en_year(2010) == "twenty ten" and nw.en_year(2000) == "two thousand" and nw.en_year(1900) == "nineteen hundred"
    assert nw.en_decimal("2", "50") == "two point five zero"


def test_language_codes():
    assert [nw.lang_code(x) for x in ("Russian", "ru", "English", "de", "French", "")] == ["ru", "ru", "en", "de", "", ""]


# ----------------------------------------------------------------------------- numbers in text (ru)

@pytest.mark.parametrize("src,expected", [
    ("в 1999 г. вышла", "в тысяча девятьсот девяносто девятом году вышла"),
    ("с 1999 года", "с тысяча девятьсот девяносто девятого года"),
    ("к 2000 году", "к двухтысячному году"),
    ("12 мая 1985 года", "двенадцатого мая тысяча девятьсот восемьдесят пятого года"),
    ("1 января", "первого января"),
    ("03.11.2020", "третьего ноября две тысячи двадцатого года"),
    ("2020-05-12", "двенадцатого мая две тысячи двадцатого года"),
    ("цена 5 руб.", "цена пять рублей"), ("21 руб", "двадцать один рубль"), ("100 ₽", "сто рублей"),
    ("5,50 ₽", "пять рублей пятьдесят копеек"), ("$5 млн", "пять миллионов долларов"), ("3 € ", "три евро "),
    ("скидка 15%", "скидка пятнадцать процентов"), ("1% и 2,5%", "один процент и две целых пять десятых процента"),
    ("60 км/ч", "шестьдесят километров в час"), ("21 минута", "двадцать одна минута"), ("2,5 кг", "две целых пять десятых килограмма"),
    ("5 тыс. человек", "пять тысяч человек"), ("в XIX веке", "в девятнадцатом веке"), ("XX век", "двадцатый век"),
    ("в 5-м классе", "в пятом классе"), ("5-му дому", "пятому дому"), ("1-го числа", "первого числа"), ("2-я мировая", "вторая мировая"),
    ("90-е годы", "девяностые годы"), ("№7", "номер семь"), ("2 женщины и 2 стола", "две женщины и два стола"),
    ("1 000 рублей", "тысяча рублей"), ("1 женщина", "одна женщина"), ("1 окно", "одно окно"),
    ("в 1941—1945 гг.", "в тысяча девятьсот сорок первом — тысяча девятьсот сорок пятом годах"),
])
def test_russian_numbers(src, expected):
    assert prep(src, "ru", [tp.STEP_NUMBERS]).strip() == expected.strip()


def test_russian_numbers_leave_no_digits_in_a_typical_paragraph():
    text = ("В 1812 г. армия насчитывала 600 000 человек; 12.06.1812 началась война. Цена: 3 руб. 50 коп., вес 2,5 кг, "
            "температура 20°C, скорость 5 км/ч, 7-й полк, глава 3, 10% из 200.")
    out = prep(text, "ru", [tp.STEP_NUMBERS])
    assert not re.search(r"\d", out), out


# ----------------------------------------------------------------------------- numbers in text (en)

@pytest.mark.parametrize("src,expected", [
    ("In 1999 he left", "In nineteen ninety nine he left"), ("since 2010", "since twenty ten"),
    ("the 1990s", "the nineteen nineties"), ("on May 12, 2020", "on May twelfth, twenty twenty"),
    ("12 May 2020", "the twelfth of May twenty twenty"), ("the 21st century", "the twenty first century"),
    ("$5.50", "five dollars fifty cents"), ("$1", "one dollar"), ("£3", "three pounds"), ("$5 million", "five million dollars"),
    ("25%", "twenty five percent"), ("2.5%", "two point five percent"), ("10 km", "ten kilometers"), ("1 kg", "one kilogram"),
    ("1,000 men", "one thousand men"), ("3.14", "three point one four"), ("room 42", "room forty two"), ("No. 5", "number five"),
])
def test_english_numbers_in_text(src, expected):
    assert prep(src, "en", [tp.STEP_NUMBERS]) == expected


def test_unsupported_language_keeps_digits_but_runs_neutral_steps():
    out = prep("Sie kamen 1999 an.[1] Mehr bei www.example.org.", "de")
    assert "1999" in out and "[1]" not in out and "Link" in out


# ----------------------------------------------------------------------------- abbreviations

@pytest.mark.parametrize("src,expected,lang", [
    ("яблоки, груши и т. д.", "яблоки, груши и так далее.", "ru"),
    ("и т.д. Затем", "и так далее. Затем", "ru"),
    ("то есть т.е. так", "то есть то есть так", "ru"),
    ("им. Пушкина", "имени Пушкина", "ru"), ("г. Москва", "город Москва", "ru"), ("проф. Иванов", "профессор Иванов", "ru"),
    ("Mr. Smith and Dr. Jones", "Mister Smith and Doctor Jones", "en"),
    ("fruit, nuts, etc. Then", "fruit, nuts, et cetera. Then", "en"), ("St. Petersburg", "Saint Petersburg", "en"),
    ("this vs. that", "this versus that", "en"),
])
def test_abbreviations(src, expected, lang):
    assert prep(src, lang, [tp.STEP_ABBREV]) == expected


def test_year_followed_by_capitalized_word_is_not_read_as_city():
    assert prep("в 1999 г. Москва", "ru", [tp.STEP_NUMBERS, tp.STEP_ABBREV]) == "в тысяча девятьсот девяносто девятом году Москва"


# ----------------------------------------------------------------------------- layout

def test_layout_invisible_characters_hyphenation_and_wrapped_lines():
    src = "Это длин\u00adное сло\u200bво. Пере-\nнос слова и кое-\nчто ещё, что про-\nдолжается\nна следующей строке.\n\nНовый абзац."
    out = prep(src, "ru", [tp.STEP_LAYOUT])
    assert out == "Это длинное слово. Перенос слова и кое-что ещё, что продолжается на следующей строке.\n\nНовый абзац."


def test_layout_keeps_poems_and_makes_dialogue_lines_paragraphs():
    poem = "Мороз и солнце;\nдень чудесный!\nЕщё ты дремлешь,\nдруг прелестный"
    assert prep(poem, "ru", [tp.STEP_LAYOUT]) == poem
    dialogue = "— Привет, как дела у тебя сегодня?\n— Хорошо, спасибо тебе большое за вопрос."
    assert prep(dialogue, "ru", [tp.STEP_LAYOUT]) == dialogue.replace("\n", "\n\n")


def test_layout_line_break_after_a_full_stop_is_a_new_paragraph_and_ligatures_are_expanded():
    out = prep("First long sentence ends here.\nSecond one starts here and \ufb01nishes.", "en", [tp.STEP_LAYOUT])
    assert out == "First long sentence ends here.\n\nSecond one starts here and finishes."


def test_layout_spacing():
    assert prep("A  lot   of spaces , and ( odd ) ones .", "en", [tp.STEP_LAYOUT]) == "A lot of spaces, and (odd) ones."


# ----------------------------------------------------------------------------- quotes and dashes

def test_quotes_and_dashes():
    out = prep("\u201cHello,\u201d she said \u2014 it\u2019s ok -- really... Wait!!!", "en", [tp.STEP_QUOTES])
    assert out == '"Hello," she said \u2014 it\'s ok \u2014 really\u2026 Wait!'
    assert prep("- Привет!\n-- Пока.", "ru", [tp.STEP_QUOTES]) == "\u2014 Привет!\n\u2014 Пока."
    assert prep("слово - слово \u2013 слово", "ru", [tp.STEP_QUOTES]) == "слово \u2014 слово \u2014 слово"
    assert prep("1914\u20131918", "ru", [tp.STEP_QUOTES]) == "1914 \u2014 1918"
    assert prep("кто-то и 3-й", "ru", [tp.STEP_QUOTES]) == "кто-то и 3-й"          # hyphens inside words stay


# ----------------------------------------------------------------------------- footnotes, page numbers, links

def test_footnotes_and_page_numbers_are_removed():
    src = "Текст со сноской[1] и звёздочкой* и надстрочной\u00b2 цифрой.\n\n42\n\nСтр. 43\n\n- 44 -\n\nСледующий абзац."
    out = prep(src, "ru", [tp.STEP_NOISE])
    assert out == "Текст со сноской и звёздочкой и надстрочной цифрой.\n\nСледующий абзац."


def test_running_headers_repeated_on_many_pages_are_removed_but_dialogue_stays():
    paras = []
    for i in range(6):
        paras += ["ВОЙНА И МИР. ТОМ ПЕРВЫЙ", f"Текст страницы номер {i}, достаточно длинный, чтобы быть абзацем.", "— Да."]
    out = prep("\n\n".join(paras), "ru", [tp.STEP_NOISE])
    assert "ТОМ ПЕРВЫЙ" not in out and out.count("— Да.") == 6


def test_links_and_emails():
    assert prep("Смотри https://example.com/a?b=1. Пиши a.b@mail.ru!", "ru", [tp.STEP_LINKS]) == \
        "Смотри ссылка. Пиши адрес электронной почты!"
    assert prep("See www.example.org, or write to me@x.io.", "en", [tp.STEP_LINKS]) == "See link, or write to email address."


# ----------------------------------------------------------------------------- headings

def test_roman_numerals_and_headings():
    assert [tp.roman_to_int(x) for x in ("IV", "XIX", "MCMXCIX", "IIII", "", "ABC")] == [4, 19, 1999, 0, 0, 0]
    ctx = tp.PrepOptions()
    assert tp.prepare_title("ГЛАВА XII.", "ru", ctx) == "Глава двенадцатая"
    assert tp.prepare_title("Часть II", "ru", ctx) == "Часть вторая"
    assert tp.prepare_title("Глава 5", "ru", ctx) == "Глава пятая"
    assert tp.prepare_title("ТОМ ПЕРВЫЙ", "ru", ctx) == "Том первый"
    assert tp.prepare_title("Chapter IV. The Light", "en", ctx) == "Chapter four. The Light"
    assert tp.prepare_title("CHAPTER 7", "en", ctx) == "Chapter seven"
    assert tp.prepare_title("XIV", "ru", ctx) == "четырнадцать"
    assert tp.prepare_title("Введение.", "ru", ctx) == "Введение"
    assert tp.prepare_title("NASA", "en", ctx) == "NASA"                                # short acronym is not "fixed"


def test_body_headings():
    out = prep("Текст.\n\nГЛАВА III\n\nXIV\n\nСОВСЕМ ДРУГАЯ ЧАСТЬ\n\nПродолжение.", "ru", [tp.STEP_HEADINGS])
    assert out == "Текст.\n\nГлава третья\n\nчетырнадцать\n\nСовсем другая часть\n\nПродолжение."


# ----------------------------------------------------------------------------- whole book

def test_prepare_book_is_pure_deterministic_and_reports(tmp_path):
    book = Book("T", "A", "ru", [Chapter("ГЛАВА I", "В 1999 г. было 5 яблок[1].\n\nИ т. д."), Chapter("", "Пусто.")])
    a, rep = tp.prepare_book(book)
    b, _ = tp.prepare_book(book)
    assert [c.text for c in a.chapters] == [c.text for c in b.chapters]
    assert a.chapters[0].title == "Глава первая" and a.chapters[1].title == ""
    assert "тысяча девятьсот девяносто девятом году" in a.chapters[0].text and "пять яблок" in a.chapters[0].text
    assert "[1]" not in a.chapters[0].text and "И так далее" in a.chapters[0].text
    assert book.chapters[0].title == "ГЛАВА I" and "[1]" in book.chapters[0].text            # the original is untouched
    assert rep.language == "ru" and tp.STEP_NUMBERS in rep.steps and rep.counts[tp.STEP_NUMBERS] >= 2


def test_language_is_detected_from_the_text_when_the_book_has_no_tag():
    assert tp.resolve_language(Book("t", chapters=[Chapter("", "Привет, мир, это русский текст.")])) == "ru"
    assert tp.resolve_language(Book("t", chapters=[Chapter("", "Hello world, this is English.")])) == "en"
    assert tp.resolve_language(Book("t", language="de-DE", chapters=[Chapter("", "x")])) == "de"
    assert tp.resolve_language(None, "russian") == "ru"


def test_no_steps_means_no_change_and_every_step_can_be_switched_off():
    text = "Mr. Smith paid $5 in 1999.[1]"
    assert prep(text, "en", []) == text
    assert prep(text, "en", [tp.STEP_NUMBERS]) == "Mr. Smith paid five dollars in nineteen ninety nine.[one]"
    assert prep(text, "en", [tp.STEP_ABBREV]) == "Mister Smith paid $5 in 1999.[1]"


def test_long_sentences_are_left_to_the_chunker_which_cuts_at_clauses():
    from core.chunker import chunk_text
    long = ", ".join(f"clause number {i} with several words in it" for i in range(12)) + "."
    assert len(long) > 400
    chunks = chunk_text(long, 200)
    assert len(chunks) >= 2 and all(len(c[0]) <= 200 for c in chunks)
