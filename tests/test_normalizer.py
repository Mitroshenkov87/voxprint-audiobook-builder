import pytest

from core import normalizer as nz


def test_number_to_words_basic():
    assert nz.number_to_words(0) == "ноль"
    assert nz.number_to_words(21) == "двадцать один"
    assert nz.number_to_words(2024) == "две тысячи двадцать четыре"
    assert nz.number_to_words(1001) == "одна тысяча один"
    assert nz.number_to_words(12000) == "двенадцать тысяч"
    assert nz.number_to_words(5000000) == "пять миллионов"
    assert nz.number_to_words(-3) == "минус три"


def test_builtin_abbreviations_and_digits():
    s = nz.builtin_normalize("Купили хлеб, молоко и т.д., т.е. всё. В т.ч. 15% скидки.")
    assert "так далее" in s and "то есть" in s and "том числе" in s and "пятнадцать процентов" in s


@pytest.mark.parametrize("engine", [("builtin", lambda s: s)])
def test_normalize_builtin_engine_keeps_sentence_ends_and_map(engine):
    raw = "В зале было 25 человек. Мы прошли 3 км, т.е. немало! Конец."
    r = nz.normalize_for_tts(raw, "Russian", engine)
    assert r.changed and "двадцать пять человек." in r.spoken
    assert r.spoken.count(".") + r.spoken.count("!") == 3
    assert not any(ch.isdigit() for ch in r.spoken)
    # карта: фрагмент «двадцать пять» -> исходное «25»
    i = r.spoken.index("двадцать пять")
    assert r.raw_for_span(i, i + len("двадцать пять")) == "25"
    j = r.spoken.index("Конец")
    assert r.raw_for_span(j, j + 5) == "Конец."


def test_unchanged_text_and_other_languages_are_identity():
    t = "Просто текст без чисел.\nВторая строка."
    r = nz.normalize_for_tts(t, "Russian", ("builtin", lambda s: s))
    assert not r.changed and r.spoken == t
    e = nz.normalize_for_tts("There are 5 apples.", "English")
    assert e.spoken == "There are 5 apples." and not e.changed


def test_real_engine_if_installed():
    pytest.importorskip("ru_normalizr")
    r = nz.normalize_for_tts("Цена 3,5 рубля. Это в 2024 году.", "Russian")
    assert r.engine == "ru-normalizr" and not any(c.isdigit() for c in r.spoken)
    assert r.spoken.count(".") == 2
