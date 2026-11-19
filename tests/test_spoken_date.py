"""Dates and names written in words by the recogniser (real Qwen3-ASR output of synthesized consent statements, RTX 4090 run)."""
import pytest

from core import consent, spoken_date

REAL_RU = [
    "Я Александр Митрошенков сегодня три октября две тысячи двадцать шесть года разрешаю использовать эту запись и характеристики моего голоса для обучения нероситей и синтазаречи в том числе в коммерческих целях а звучки можно продавать",
    "Я Александр Митрошенков. Сегодня три октября две тысячи двадцать шесть года разрешаю использовать эту запись и характеристики моего голоса для обучения нейросетей и синтеза речевника мерческих целях. А звучки можно публиковать в открытом доступе, но нельзя продавать.",
    "Я Александр Митрошенков сегодня три октября две тысячи двадцать шесть года разрешаю использовать оту запись и характеристики моего голоса для обучения неросителей синтеза речи только в личных целях а звучки нельзя публиковать они должны оставаться на компьютере пользователя",
]
REAL_EN = [
    "I, Alexander Mitroshinov, today, three October two thousand twenty-six, give permission to use this recording and the characteristics of my voice for training neural networks and for speech synthesis, including commercial purposes. The audio may be sold.",
    "I, Alexander Mitrshenko, today, three October twenty twenty-six, give permission to use this recording and the characteristics of my voice for training neural networks and for speech synthesis for non-commercial purposes. The audio may be published freely, but may not be sold.",
]


@pytest.mark.parametrize("text,scope", list(zip(REAL_RU, [consent.COMMERCIAL, consent.PUBLIC_NC, consent.PRIVATE])))
def test_real_ru_hypotheses_name_date_scope(text, scope):
    p = consent.parse_statement(text)
    assert (p.scope, p.name, p.date) == (scope, "Александр Митрошенков", "2026-10-03")
    assert p.confident


@pytest.mark.parametrize("text,scope", list(zip(REAL_EN, [consent.COMMERCIAL, consent.PUBLIC_NC])))
def test_real_en_hypotheses(text, scope):
    p = consent.parse_statement(text)
    assert p.scope == scope and p.date == "2026-10-03" and p.name.startswith("Alexander")


@pytest.mark.parametrize("text,iso", [
    ("сегодня третьего октября две тысячи двадцать шестого года", "2026-10-03"),
    ("двадцать первого марта тысяча девятьсот девяносто девятого года", "1999-03-21"),
    ("тридцать первого декабря две тысячи двадцать пятого", "2025-12-31"),
    ("the 31st of December two thousand and twenty six", "2026-12-31"),
    ("today October third, twenty twenty six", "2026-10-03"),
    ("nineteenth of May nineteen ninety nine", "1999-05-19"),
    ("heute am dritten Oktober zweitausendsechsundzwanzig", "2026-10-03"),
    ("am einunddreißigsten Dezember zweitausend sechsundzwanzig", "2026-12-31"),
    ("am zwanzigsten Juni neunzehnhundertneunundneunzig", "1999-06-20"),
    ("3 октября 2026 года", "2026-10-03"),
])
def test_find_date_words_and_digits(text, iso):
    assert consent._parse_date(consent._norm(text)) == iso


@pytest.mark.parametrize("text", ["31 февраля 2026", "тридцать второго октября две тысячи двадцать шестого", "октября", "Я Александр мая",
                                  "third of October", "ничего особенного не сказано"])
def test_no_date_when_invalid_or_incomplete(text):
    assert spoken_date.find_date(text) == ""


@pytest.mark.parametrize("text,name", [
    ("Я Александр Митрошенков сегодня три октября", "Александр Митрошенков"),
    ("Я, Александр Митрошенков, сегодня, 3 октября 2026 года, разрешаю", "Александр Митрошенков"),
    ("I Alexander Mitroshenkov today third of October give permission", "Alexander Mitroshenkov"),
    ("Ich Hans Müller erlaube heute", "Hans Müller"),
    ("я иван иванов разрешаю использовать", "Иван Иванов"),
    ("Я разрешаю использовать", ""),
])
def test_name_with_and_without_commas(text, name):
    assert consent._parse_name(text) == name
