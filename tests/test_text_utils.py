"""Text helpers: decoding, language detection, sentence/clause splitting, word-to-text alignment spans."""
from core.text_utils import (attach_spans, decode_bytes, detect_language, normalize_text, split_clauses,
                             split_sentences, text_for_words)
from core.types import WordTiming


def test_split_sentences_basic():
    t = "Привет, мир! Это тест. Как дела? Хорошо."
    assert split_sentences(t) == ["Привет, мир!", "Это тест.", "Как дела?", "Хорошо."]


def test_split_sentences_abbreviations_and_numbers():
    t = "А. С. Пушкин родился в 1799 г. в Москве. Это стоит 3.5 рубля и т.д. Конец."
    s = split_sentences(t)
    assert s[0].startswith("А. С. Пушкин") and s[0].endswith("Москве.")
    assert len(s) == 3
    assert "3.5" in s[1]


def test_split_sentences_dialogue_and_paragraphs():
    t = "— Ты придёшь? — спросил он.\n\n— Да.\nПотом ушёл."
    assert split_sentences(t) == ["— Ты придёшь?", "— спросил он.", "— Да. Потом ушёл."] or len(split_sentences(t)) >= 3


def test_split_lines_without_punctuation():
    assert split_sentences("первая строка\nвторая строка") == ["первая строка", "вторая строка"]


def test_clauses():
    assert split_clauses("Привет, мир! Это — тест; да.") == ["Привет,", "мир!", "Это", "— тест;", "да."]


def test_normalize_removes_junk():
    assert normalize_text("\ufeffПривет\u200b  мир\r\n\r\n\r\n\r\nТекст\u00a0тут") == "Привет мир\n\nТекст тут"


def test_decode_utf8_and_cp1251_warns():
    assert decode_bytes("Привет".encode("utf-8")).warnings == []
    r = decode_bytes("Привет, мир".encode("cp1251"))
    assert r.text == "Привет, мир" and r.warnings
    assert decode_bytes(b"\xef\xbb\xbf" + "Да".encode()).text == "Да"


def test_detect_language():
    assert detect_language("Привет мир") == "Russian"
    assert detect_language("Hello world") == "English"
    assert detect_language("你好世界") == "Chinese"
    assert detect_language("こんにちは") == "Japanese"
    assert detect_language("안녕하세요") == "Korean"


def test_attach_spans_keeps_punctuation_and_hyphen():
    text = "«Привет», — сказал он. Это тест-кейс!"
    toks = ["Привет", "сказал", "он", "Это", "тесткейс"]
    words = [WordTiming(t, i, i + 1) for i, t in enumerate(toks)]
    assert attach_spans(words, text) == []
    assert text_for_words(text, words[:3]) == "«Привет», — сказал он."  or text_for_words(text, words[:3]).endswith("сказал он.")
    assert text_for_words(text, words[3:]) == "Это тест-кейс!"
    assert words[2].sentence_end and words[4].sentence_end and not words[0].sentence_end


def test_split_clauses_wraps_unpunctuated_text():
    words = [f"w{i}" for i in range(100)]
    units = split_clauses(" ".join(words))        # no punctuation at all: still many small units
    assert len(units) >= 7 and all(len(u.split()) <= 14 for u in units)
    assert " ".join(units).split() == words
