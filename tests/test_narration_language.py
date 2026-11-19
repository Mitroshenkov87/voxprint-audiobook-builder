"""The language token sent to Qwen3-TTS: the language of the spoken TEXT, normalised to the model's exact names."""
from __future__ import annotations

import pytest

from core import languages as lg
from core import narration as nr
from core.book_parsers import Book, Chapter
from core.events import CancelToken

EN = ("The old lighthouse keeper walked down to the shore every morning and he looked at the sea for a long time. "
      "It was the only thing that he really loved, and the people of the village knew it well.") * 3
RU = ("Старый смотритель маяка каждое утро спускался к берегу и долго смотрел на море. Это было единственное, что он "
      "по-настоящему любил, и в деревне об этом знали все.") * 3
UK = ("Старий доглядач маяка щоранку спускався до берега і довго дивився на море. Це було єдине, що він "
      "по-справжньому любив, і в селі про це знали всі.") * 3


@pytest.mark.parametrize("value, expected", [
    ("ru", "Russian"), ("ru-RU", "Russian"), ("ru_ru", "Russian"), ("russian", "Russian"), ("Russian", "Russian"),
    ("RUSSIAN", "Russian"), ("rus", "Russian"), ("русский", "Russian"), ("en-GB", "English"), ("eng", "English"),
    ("de", "German"), ("Deutsch", "German"), ("zh-CN", "Chinese"), ("ja", "Japanese"), ("ko", "Korean"),
    ("pt-BR", "Portuguese"), ("fr", "French"), ("es", "Spanish"), ("it", "Italian"), ("auto", "Auto"), ("AUTO", "Auto"),
    ("uk", ""), ("Polish", ""), ("", ""), (None, ""), ("klingon!!", ""),
])
def test_any_language_form_maps_to_the_exact_qwen_name(value, expected):
    assert lg.qwen_language(value) == expected


def test_qwen_names_are_the_models_own_list():
    # qwen_tts 0.1.1: "auto" + talker_config.codec_language_id keys of the Base models (case-insensitive check there)
    assert {n.lower() for n in lg.QWEN_TTS_LANGUAGES} == {"chinese", "english", "french", "german", "italian", "japanese",
                                                          "korean", "portuguese", "russian", "spanish"}
    assert all(lg.qwen_language(n) == n for n in lg.QWEN_TTS_LANGUAGES)


def test_russian_voice_reading_an_english_book_speaks_english():
    assert lg.narration_language(Book("T", "", "en", [Chapter("c", EN)]), "russian") == "English"
    assert lg.narration_language(Book("T", "", "en", [Chapter("c", EN)]), "ru") == "English"


def test_the_translation_target_wins():
    book = Book("T", "", "en", [Chapter("c", EN)])
    assert lg.narration_language(book, "en", translate_target="ru") == "Russian"
    assert lg.narration_language(book, "en", translate_target="de") == "German"


def test_missing_or_wrong_metadata_falls_back_to_the_text():
    assert lg.narration_language(Book("T", "", "", [Chapter("c", RU)]), "english") == "Russian"
    assert lg.narration_language(Book("T", "", "en", [Chapter("c", RU)]), "english") == "Russian"   # a wrong "en" stamp
    assert lg.narration_language(Book("T", "", "", [Chapter("c", "これは日本語の本です。" * 5)]), "ru") == "Japanese"


def test_metadata_is_used_when_the_text_says_nothing():
    assert lg.narration_language(Book("T", "", "fr-FR", [Chapter("c", "12345 67890")]), "ru") == "French"


def test_voice_language_is_the_last_fallback_then_auto():
    assert lg.narration_language(Book("T", "", "", [Chapter("c", "12345")]), "German") == "German"
    assert lg.narration_language(Book("T", "", "", [Chapter("c", "12345")]), "") == "Auto"
    assert lg.narration_language(Book("T", "", "", []), "uk") == "Auto"


def test_a_language_qwen_lacks_gets_auto_not_the_voice_language():
    assert lg.narration_language(Book("T", "", "uk", [Chapter("c", UK)]), "russian") == "Auto"


def test_runner_passes_the_text_language_to_the_engine_and_the_tag(monkeypatch, tmp_path):
    from workers import narration_runner as nrun
    seen = {}
    monkeypatch.setattr(nrun.tts_engine, "make_engine_factory", lambda voice, language: seen.setdefault("engine", language) or (lambda: None))
    monkeypatch.setattr(nrun.tts_engine, "engine_tag", lambda v, language="": seen.setdefault("tag", language) or "t")
    monkeypatch.setattr(nrun, "ensure_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(nrun.nr, "narrate_book", lambda *a, **k: seen.setdefault("nb", k["language"]))

    class V:
        language, name = "ru", "Anna"
    nrun.run_narration(nrun.NarrationJob(Book("T", "", "en", [Chapter("c", EN)]), V(), tmp_path), lambda p: None,
                       CancelToken(), nr.PauseToken())
    assert seen == {"engine": "English", "tag": "English", "nb": "English"}
    seen.clear()
    nrun.run_narration(nrun.NarrationJob(Book("T", "", "", []), type("W", (), {"language": "", "name": "x"})(), tmp_path),
                       lambda p: None, CancelToken(), nr.PauseToken())
    assert seen["engine"] == "Auto" and seen["nb"] == ""          # Auto: text prep / disclosure use the book's own tag


def test_engine_tag_keeps_old_caches_when_the_token_is_unchanged(tmp_path):
    from core import tts_engine
    from core.voice_library import VoiceRecord
    d = tmp_path / "v"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"w")
    rec = VoiceRecord("v", d, {"language": "ru", "base_model": "Qwen/B"})
    old = tts_engine.engine_tag(rec)
    assert tts_engine.engine_tag(rec, "Russian") == old            # same token as older versions used: cache stays valid
    assert tts_engine.engine_tag(rec, "English") != old            # a different token must not reuse Russian-token audio
