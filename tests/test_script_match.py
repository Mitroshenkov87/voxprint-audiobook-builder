"""Recording script v3 markup + tolerant matching of recognised clips to script lines (stumbles, re-read sentences)."""
import re
from pathlib import Path

import pytest

import numpy as np

from core import audio_utils as au
from core.asr import FakeASR
from core.asr_dataset import build_from_audio
from core.consent import TEMPLATES
from core.dataset_builder import read_metadata_jsonl
from core.script_match import match_clips, parse_script_lines

ROOT = Path(__file__).resolve().parents[1]
V3 = (ROOT / "docs" / "voice-script-ru-v3.txt").read_text(encoding="utf-8")
V2 = (ROOT / "docs" / "voice-script-ru-v2.txt").read_text(encoding="utf-8")

SCRIPT = """=== БЛОК 1 ===
[Скажите чуть теплее.]
Программа запускается и открывает главное окно.
Слева находится список голосов.
Справа находится текст для чтения.

[Человек, живо]
Сколько времени займёт обучение?

[Программа, ровно]
Примерно двадцать минут.
"""


def test_parse_skips_headers_brackets_hints_and_labels():
    lines = parse_script_lines(SCRIPT + "(подсказка в скобках)\n")
    assert lines == ["Программа запускается и открывает главное окно.", "Слева находится список голосов.",
                     "Справа находится текст для чтения.", "Сколько времени займёт обучение?", "Примерно двадцать минут."]


def test_v3_script_markup_and_blocks():
    for title in ("БЛОК 16. НЕОБЯЗАТЕЛЬНО: БЛОК ТЕРМИНОВ", "БЛОК 17. СОГЛАСИЕ"):
        assert title in V3
    lines = parse_script_lines(V3)
    assert 140 <= len(lines) <= 260                          # about the coverage of v2
    assert len(parse_script_lines(V2)) > 40                  # the v2 markup (hints in round brackets) still parses
    assert all("[" not in s and "===" not in s for s in lines)
    assert "Человек, живо" not in " ".join(lines) and "Программа, ровно" not in " ".join(lines)   # speaker labels are never read
    assert "Сколько времени займёт обучение?" in lines and "Примерно двадцать минут." in lines
    assert not any("разрешаю" in s for s in lines)           # the consent block is not training text


def test_v3_keeps_the_consent_templates_for_the_parser():
    flat = " ".join(V3.split())
    for lang, d in TEMPLATES.items():
        for scope, tpl in d.items():
            # the templates are in the script word for word, with the name/date placeholders in square brackets
            head = tpl.split("{name}")[0].strip()
            assert " ".join(head.split()) in flat
    assert tpl.split("{date}")[1].strip()[:30] in flat


def test_stumbles_and_rereads_keep_the_best_reading_of_each_line():
    lines = parse_script_lines(SCRIPT)
    heard = [
        "Программа запускается и откры",                              # stumble, cut off
        "Программа запускается и открывает главное окно",            # the re-read
        "слева находится список голосов",
        "Слева находится список голосов",                             # the same line twice: only one is kept
        "Справа находится текст для чтения. Сколько времени займёт обучение",   # two lines in one piece
        "Меня зовут Иван и сегодня третье октября",                  # not in the script (consent / chatter)
        "Примерно двадцать минут",
    ]
    res = match_clips(heard, lines)
    # equal quality of two readings: the later one wins
    assert [r.status for r in res] == ["no_match", "matched", "repeat", "matched", "matched", "no_match", "matched"]
    assert res[1].text == lines[0] and res[4].text == lines[2] + " " + lines[3] and res[6].text == lines[4]
    kept_lines = sorted(r.line_start + k for r in res if r.status == "matched" for k in range(r.n_lines))
    assert kept_lines == [0, 1, 2, 3, 4]                              # every line exactly once


def test_build_from_audio_with_script_ignores_stumbles_and_uses_script_text(tmp_path):
    lines = parse_script_lines(SCRIPT)
    rnd = np.random.default_rng(0)
    d = tmp_path / "clips"
    d.mkdir()
    heard = ["Программа запускается и откры", "Программа запускается и открывает главное окно", "Слева находится список голосов",
             "Слева находится список голосов да", "Справа находится текст для чтения", "Просто болтовня о погоде"]
    for i in range(len(heard)):
        t = np.arange(16000 * 4) / 16000
        x = (0.2 * np.sin(2 * np.pi * (180 + 30 * i) * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
        x += rnd.normal(0, 0.003, len(x)).astype(np.float32)
        au.write_wav(d / f"c{i}.wav", x, 16000)
    asr = FakeASR(lambda x, sr: heard[asr.calls])
    res = build_from_audio([d], tmp_path / "ds", asr, script_text=SCRIPT)
    rows = read_metadata_jsonl(tmp_path / "ds" / "metadata.jsonl")
    assert [r["text"] for r in rows] == [lines[0], lines[1], lines[2]]
    drops = res.asr_report.dropped
    assert drops.get("not_in_script", 0) + drops.get("repeat", 0) == 3 and drops.get("repeat") == 1


def test_consent_header_stops_parsing_in_ru_en_de_any_case():
    for head in ("=== БЛОК 18. СОГЛАСИЕ ===", "=== Block 18. Consent ===", "=== consent statement ===", "=== BLOCK 18: EINWILLIGUNG ===", "=== Einwilligung ==="):
        assert parse_script_lines("Первая строка текста.\n" + head + "\nЯ разрешаю использовать мой голос.\n") == ["Первая строка текста."], head
    # an ordinary header is skipped, not a stop
    assert parse_script_lines("=== Block 2 ===\nПервая строка.\nВторая строка.\n") == ["Первая строка.", "Вторая строка."]


# ------------------------------------------------------------------ recording scripts v4 (docs/recording-scripts/)
V4_LANGS = ("ru", "en", "de")
_PROFANE = re.compile(r"^(ху[йяеи]|хуе|пизд|бля|ебан|ебат|ебл|ёб|заеб|уеб|наеб|мудак|мудил|говн|жоп|дерьм|хер|хрен|охр|нахр|сук[аиу]$|шлюх|"
                      r"fuck|shit|bitch|cunt|dick|piss|bastard|wank|asshole|scheiß|scheiss|arsch|fick|wichs|hure|schlampe|kacke)", re.I)


def _v4(lang):
    return (ROOT / "docs" / "recording-scripts" / f"Voxprint-RecordingScript-v4-{lang}.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("lang", V4_LANGS)
def test_v4_scripts_have_no_profanity_and_parse_with_consent_at_the_end(lang):
    text = _v4(lang)
    words = re.findall(r"[\wёÄÖÜäöüß]+", text)
    assert not [w for w in words if _PROFANE.match(w)]
    lines = parse_script_lines(text)
    assert 80 <= len(lines) <= 140
    assert all("[" not in s and "===" not in s for s in lines)
    assert not any(k in " ".join(lines).lower() for k in ("разрешаю", "give permission", "erlaube heute"))   # the consent block is not training text


@pytest.mark.parametrize("lang", V4_LANGS)
def test_v4_scripts_contain_only_their_own_consent_templates(lang):
    flat = " ".join(_v4(lang).split())
    for scope, tpl in TEMPLATES[lang].items():
        assert " ".join(tpl.split("{name}")[0].split()) in flat
        assert " ".join(tpl.split("{date}")[1].split())[:40] in flat
    for other in V4_LANGS:
        if other != lang:
            for tpl in TEMPLATES[other].values():
                assert " ".join(tpl.split("{name}")[0].split()) not in flat


# ------------------------------------------------------------------ recording scripts v5 (docs/recording-scripts/)
V5_LANGS = ("ru", "en", "de")


def _v5(lang):
    return (ROOT / "docs" / "recording-scripts" / f"Voxprint-RecordingScript-v5-{lang}.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("lang", V5_LANGS)
def test_v5_scripts_have_no_profanity_and_parse_with_consent_at_the_end(lang):
    text = _v5(lang)
    words = re.findall(r"[\wёÄÖÜäöüß]+", text)
    assert not [w for w in words if _PROFANE.match(w)]
    lines = parse_script_lines(text)
    assert 100 <= len(lines) <= 220
    assert all("[" not in s and "===" not in s for s in lines)
    assert not any(k in " ".join(lines).lower() for k in ("разрешаю", "give permission", "erlaube heute"))


@pytest.mark.parametrize("lang", V5_LANGS)
def test_v5_scripts_contain_only_their_own_consent_templates(lang):
    flat = " ".join(_v5(lang).split())
    for scope, tpl in TEMPLATES[lang].items():
        assert " ".join(tpl.split("{name}")[0].split()) in flat
        assert " ".join(tpl.split("{date}")[1].split())[:40] in flat
    for other in V5_LANGS:
        if other != lang:
            for tpl in TEMPLATES[other].values():
                assert " ".join(tpl.split("{name}")[0].split()) not in flat
