"""Automatic book translation before narration (core/translate.py, the registry of Opus-MT models, the Narrate window card).
Everything runs with a fake translator: no model, no network, no GPU."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import i18n
from core import narration as nr
from core import translate as tl
from core.book_parsers import Book, Chapter
from core.errors import CancelledByUser, ModelDownloadError
from core.events import CancelToken
from infra import text_models

RU = ("Это длинный русский текст, который рассказывает о старом маяке и о людях, которые жили рядом с ним много лет. "
      "Они не знали, что будет дальше.")
EN = "The old lighthouse stood on the cliff, and the people who lived near it did not know what would happen next."
DE = "Der alte Leuchtturm stand auf der Klippe, und die Menschen, die in der Nähe wohnten, wussten nicht, was als Nächstes kommen würde."


class FakeTranslator:
    """'Translates' by tagging the sentence with the direction; counts the sentences it was given."""

    def __init__(self, src, dst, log):
        self.src, self.dst, self.log, self.closed = src, dst, log, False
        self.tag = f"fake-{src}-{dst}@1"

    def translate(self, sentences):
        self.log.extend(sentences)
        return [f"[{self.dst}] {s}" for s in sentences]

    def close(self):
        self.closed = True


def plan_for(target, log, source="", missing=()):
    return tl.TranslatePlan(target, source, lambda s, d: None if (s, d) in missing else FakeTranslator(s, d, log))


def ru_book():
    return Book("Маяк", "Автор", "ru", [
        Chapter("Глава первая", "Первое предложение главы. Второе предложение главы.\n\nНовый абзац идёт здесь.\n\n* * *\n\nПосле разрыва сцены."),
        Chapter("Глава вторая", "Белеет парус одинокий\nВ тумане моря голубом\nЧто ищет он в стране далёкой"),
        Chapter("", RU)])


# ----------------------------------------------------------------------------- language detection and routes
def test_language_detection_tells_en_de_ru_and_refuses_the_rest():
    assert tl.detect_language(RU) == "ru" and tl.detect_language(EN) == "en" and tl.detect_language(DE) == "de"
    assert tl.detect_language("Це довгий український текст про старий маяк і людей, які жили поруч із ним їхнього життя.") == "uk"
    assert tl.detect_language("Le vieux phare se dressait sur la falaise et les gens qui habitaient près de lui ne savaient rien.") == ""
    assert tl.detect_language("Hi") == ""
    book = Book("x", "", "ru-RU", [Chapter("c", "12345 67890")])          # no letters: the file's own tag is the fallback
    assert tl.detect_book_language(book) == "ru" and tl.detect_book_language(Book("x", "", "fr", [Chapter("c", "123")])) == ""
    assert tl.detect_book_language(Book("x", "", "de", [Chapter("c", "123")])) == "de"


def test_routes_are_direct_or_through_english():
    assert tl.route("ru", "en") == [("ru", "en")] and tl.route("de", "en") == [("de", "en")]
    assert tl.route("ru", "de") == [("ru", "en"), ("en", "de")] and tl.route("de", "ru") == [("de", "en"), ("en", "ru")]
    assert tl.route("en", "en") == []
    with pytest.raises(tl.TranslateError):
        tl.route("uk", "en")
    assert [m.key for m in text_models.translate_models("ru", "de")] == ["opus-ru-en", "opus-en-de"]
    with pytest.raises(ValueError):
        text_models.translate_models("fr", "de")


# ----------------------------------------------------------------------------- structure
def test_paragraphs_keep_verse_lines_and_scene_breaks_and_long_sentences_are_cut():
    paras = tl.split_paragraphs("Раз. Два три.\n\nСтрока без точки\nвторая строка без точки\n\n* * *")
    assert [(p.joiner, p.units) for p in paras] == [(" ", ["Раз.", "Два три."]), ("\n", ["Строка без точки", "вторая строка без точки"]),
                                                    (" ", ["* * *"])]
    long = "слово, " * 200
    pieces = tl._cut_long(long.strip())
    assert len(pieces) > 2 and all(len(p) <= tl.MAX_SENTENCE_CHARS for p in pieces) and " ".join(pieces).replace(",", "") .split() == long.replace(",", "").split()


def test_translate_book_keeps_chapters_titles_paragraphs_and_scene_breaks():
    log = []
    book = ru_book()
    out = tl.translate_book(book, plan_for("en", log), "ru", tl.TranslationCache(None))
    assert out.language == "en" and out.author == "Автор" and len(out.chapters) == 3
    assert out.title == "[en] Маяк" and out.chapters[0].title == "[en] Глава первая" and out.chapters[2].title == ""
    paras = out.chapters[0].text.split("\n\n")
    assert paras == ["[en] Первое предложение главы. [en] Второе предложение главы.", "[en] Новый абзац идёт здесь.", "* * *",
                     "[en] После разрыва сцены."]
    assert out.chapters[1].text.split("\n")[0] == "[en] Белеет парус одинокий" and out.chapters[1].text.count("\n") == 2   # verse lines
    assert "* * *" not in log and len(log) == len(set(log))             # the scene break is not translated; duplicates sent once


def test_chapter_selection_translates_only_the_chosen_chapters():
    log = []
    out = tl.translate_book(ru_book(), plan_for("en", log), "ru", tl.TranslationCache(None), only={0})
    assert out.chapters[0].text.startswith("[en]") and out.chapters[1].text == ru_book().chapters[1].text
    assert not any("Белеет" in s for s in log)


def test_a_pair_through_english_uses_two_models_one_after_another():
    log, made = [], []
    plan = tl.TranslatePlan("de", "", lambda s, d: made.append((s, d)) or FakeTranslator(s, d, log))
    out = tl.translate_book(Book("T", "", "ru", [Chapter("c", "Привет мир. Как дела?")]), plan, "ru", tl.TranslationCache(None))
    assert made == [("ru", "en"), ("en", "de")]
    assert out.chapters[0].text == "[de] [en] Привет мир. [de] [en] Как дела?"


def test_missing_model_and_unsupported_pair_are_reported_clearly():
    with pytest.raises(tl.TranslateError) as e:
        tl.translate_book(ru_book(), plan_for("en", [], missing={("ru", "en")}), "ru", tl.TranslationCache(None))
    assert "ru" in e.value.user_message
    with pytest.raises(tl.TranslateError):
        tl.translate_book(ru_book(), plan_for("en", []), "uk", tl.TranslationCache(None))


# ----------------------------------------------------------------------------- cache, resume, the editable file
def test_the_sentence_cache_means_a_resumed_job_never_translates_twice(tmp_path):
    cache_file = tmp_path / "c.json"
    log1 = []
    cache = tl.TranslationCache(cache_file)
    token = CancelToken()
    done = []

    def progress(f, m):
        done.append(f)
        if len(done) >= 3:                                  # cancel in the middle of the book
            token.cancel()

    big = Book("T", "", "en", [Chapter("c", " ".join(f"Sentence number {i} is here." for i in range(60)))])
    with pytest.raises(CancelledByUser):
        tl.translate_book(big, plan_for("ru", log1), "en", cache, progress, token)
    assert 0 < len(log1) < 62 and cache_file.is_file()
    log2 = []
    out = tl.translate_book(big, plan_for("ru", log2), "en", tl.TranslationCache(cache_file))
    assert len(log1) + len(log2) <= 62 and not set(log1) & set(log2)      # nothing was translated twice
    assert out.chapters[0].text.count("[ru]") == 60


def test_a_damaged_cache_file_is_ignored(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{oops", encoding="utf-8")
    assert tl.TranslationCache(p).data == {}


def test_ensure_translation_writes_a_readable_file_and_reuses_the_edited_one(tmp_path):
    i18n.set_language("en")
    log = []
    book = ru_book()
    tr1, src = tl.ensure_translation(book, plan_for("en", log), tmp_path)
    assert src == "ru" and tr1.chapters[0].title == "[en] Глава первая"
    f = tmp_path / "translation_en.txt"
    text = f.read_text(encoding="utf-8")
    assert text.startswith("# Voxprint machine translation ru -> en") and "TITLE: [en] Маяк" in text and "=== [1] [en] Глава первая" in text
    first_calls = len(log)
    # the user edits the file; the next run narrates the edited text and does not call the model
    f.write_text(text.replace("[en] Новый абзац идёт здесь.", "A better paragraph, edited by hand."), encoding="utf-8")
    tr2, _ = tl.ensure_translation(book, plan_for("en", log, missing={("ru", "en")}), tmp_path)      # model not even needed
    assert "A better paragraph, edited by hand." in tr2.chapters[0].text and len(log) == first_calls
    # a different source text: translated anew, the old file is kept as .previous
    other = Book("Маяк", "Автор", "ru", [Chapter("Глава", "Совсем другой текст про другой маяк, который стоял далеко отсюда.")])
    tr3, _ = tl.ensure_translation(other, plan_for("en", log), tmp_path)
    assert (tmp_path / "translation_en.previous.txt").is_file() and "Совсем другой" in tr3.chapters[0].text


def test_a_book_already_in_the_target_language_is_not_translated(tmp_path):
    book = Book("T", "", "en", [Chapter("c", EN)])
    out, src = tl.ensure_translation(book, plan_for("en", []), tmp_path)
    assert out is book and src == "en" and not (tmp_path / "translation_en.txt").exists()
    with pytest.raises(tl.TranslateError):
        tl.ensure_translation(Book("T", "", "", [Chapter("c", "12345")]), plan_for("en", []), tmp_path)


def test_quotes_spacing_of_the_marian_output_is_tidied():
    assert tl.tidy_quotes('" Are you sure? " she asked, " ok ".') == '"Are you sure?" she asked, "ok".'
    assert tl.tidy_quotes('One " quote only') == 'One " quote only'


# ----------------------------------------------------------------------------- narration with translation
def test_narration_translates_first_narrates_the_target_language_and_names_the_folder(tmp_path):
    from tests.test_narration import FakeEngine, FakeFfmpeg
    i18n.set_language("en")
    log, engine, ff, events = [], FakeEngine(), FakeFfmpeg(), []
    book = Book("Маяк", "Автор", "ru", [Chapter("Глава первая", "Первое предложение главы. Второе предложение главы.")])
    opts = nr.NarrationOptions(translate=plan_for("en", log), keep_cache=True)
    res = nr.narrate_book(book, lambda: engine, engine.tag, tmp_path / "out", language="russian", narrator="Anna",
                          progress=events.append, ffmpeg="ffmpeg", run=ff, options=opts)
    assert res.out_dir.name == "Маяк (en)" and (res.out_dir / "translation_en.txt").is_file()
    assert engine.calls and all(c.startswith("[en]") or "[en]" in c for c in engine.calls)
    assert "title=[en] Глава первая" in ff.meta_texts[-1]               # the exported chapter titles are translated too
    assert any(e.phase == "prepare" and "Translating" in e.message for e in events)
    # the same job again: nothing is translated or synthesized twice
    log.clear()
    engine2 = FakeEngine()
    nr.narrate_book(book, lambda: engine2, engine2.tag, tmp_path / "out", language="russian", ffmpeg="ffmpeg", run=FakeFfmpeg(),
                    options=nr.NarrationOptions(translate=plan_for("en", log), keep_cache=True))
    assert log == [] and engine2.calls == []


def test_narration_without_translation_or_with_the_same_language_keeps_the_folder_name(tmp_path):
    from tests.test_narration import FakeEngine, FakeFfmpeg
    book = Book("The Book", "", "en", [Chapter("One", EN)])
    for opts in (nr.NarrationOptions(), nr.NarrationOptions(translate=plan_for("en", []))):
        engine = FakeEngine()
        res = nr.narrate_book(book, lambda: engine, engine.tag, tmp_path / "out", language="english", ffmpeg="ffmpeg",
                              run=FakeFfmpeg(), options=opts)
        assert res.out_dir.name == "The Book"


def test_runner_narrates_in_the_target_language(monkeypatch, tmp_path):
    from workers import narration_runner as nrun
    seen = {}
    monkeypatch.setattr(nrun.tts_engine, "make_engine_factory", lambda voice, language: seen.setdefault("lang", language) or (lambda: None))
    monkeypatch.setattr(nrun.tts_engine, "engine_tag", lambda v: "t")
    monkeypatch.setattr(nrun, "ensure_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(nrun.nr, "narrate_book", lambda *a, **k: seen.setdefault("nb", k["language"]))

    class V:
        language, name = "russian", "Anna"
    job = nrun.NarrationJob(Book("T", "", "ru", []), V(), tmp_path, nr.NarrationOptions(translate=plan_for("de", [])))
    nrun.run_narration(job, lambda p: None, CancelToken(), nr.PauseToken())
    assert seen["lang"] == "German" and seen["nb"] == "German"


# ----------------------------------------------------------------------------- the model registry and its hash check
def test_ensure_downloads_only_the_needed_files_and_checks_the_sha256(tmp_path, monkeypatch):
    import hashlib
    from infra import model_downloader as md
    m = text_models.get("opus-de-en")
    blob = b"weights"
    calls = []

    def fake_snapshot(repo_id, local_dir, revision=None, allow_patterns=None, **kw):
        calls.append((repo_id, revision, tuple(allow_patterns or ())))
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text("{}")
        (Path(local_dir) / "pytorch_model.bin").write_bytes(blob)

    monkeypatch.setenv("VOXPRINT_NO_MIRROR", "1")
    bad = text_models.TextModel(**{**m.__dict__, "sha256": (("pytorch_model.bin", "0" * 64),)})
    with pytest.raises(ModelDownloadError):
        text_models.ensure(bad, snapshot_download=fake_snapshot, get_remote_sha=lambda r: None)
    assert not m.local_dir.exists() and text_models.state(m) == text_models.STATE_NEEDS_DOWNLOAD     # a bad download is deleted
    good = text_models.TextModel(**{**m.__dict__, "sha256": (("pytorch_model.bin", hashlib.sha256(blob).hexdigest()),)})
    path = text_models.ensure(good, snapshot_download=fake_snapshot, get_remote_sha=lambda r: None)
    assert path == m.local_dir and calls[-1] == (m.repo, m.revision, text_models.OPUS_FILES)
    assert "tf_model.h5" not in text_models.OPUS_FILES and "rust_model.ot" not in text_models.OPUS_FILES
    assert text_models.state(m) == text_models.STATE_READY
    eng = text_models.make_translator("de", "en")
    assert eng is not None and eng.tag.startswith("opus-mt-de-en@1a922f3b") and text_models.make_translator("en", "ru") is None
    plan = text_models.build_translate_plan("en", "de")
    assert plan.target == "en" and plan.engine_factory is text_models.make_translator and text_models.build_translate_plan("") is None


# ----------------------------------------------------------------------------- the Narrate window
from tests.test_studio import app, lib, make_studio, add_voice, wait_for  # noqa: E402,F401


def book_file(tmp_path, text, name="b.txt"):
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return f


RU_TXT = "Глава 1\n\n" + RU + "\n\nГлава 2\n\n" + RU
EN_TXT = "Chapter 1\n\n" + EN + "\n\nChapter 2\n\n" + EN


def test_translate_card_states_download_and_the_plan(app, lib, tmp_path):
    i18n.set_language("en")
    ready = set()
    ensured = []
    s = make_studio(lib)
    n = s.narrate_window
    n.model_state = lambda m: text_models.STATE_READY if m.key in ready else text_models.STATE_NEEDS_DOWNLOAD
    n.model_ensure = lambda m, progress=None, **kw: (ensured.append(m.key), ready.add(m.key))
    n.show()
    add_voice(lib, tmp_path)
    n.refresh_voices()
    n.load_book_file(book_file(tmp_path, RU_TXT))
    assert n.chk_translate.isVisibleTo(n) and not n.tr_box.isVisibleTo(n) and n.translate_plan() is None
    assert "right to translate" in n.lbl_tr_note.text() and "quality varies" in n.lbl_tr_note.text()
    n.chk_translate.setChecked(True)
    n.cmb_translate.setCurrentIndex(n.cmb_translate.findData("en"))
    assert n.book_source_language() == "ru" and n.tr_box.isVisibleTo(n)
    assert "Russian" in n.lbl_tr_state.text() and "download" in n.lbl_tr_state.text() and n.btn_tr_download.isVisibleTo(n)
    assert not n.btn_start.isEnabled() and n.translate_plan() is None            # the model must be downloaded first
    assert n.download_translate_models()
    assert wait_for(lambda: not n.btn_tr_download.isVisibleTo(n))
    assert ensured == ["opus-ru-en"] and "downloaded" in n.lbl_tr_state.text() and n.btn_start.isEnabled()
    plan = n.translate_plan()
    assert plan.target == "en" and plan.source == "ru" and n.options().translate.target == "en"
    # ru -> de goes through English: both models, a note about it
    n.cmb_translate.setCurrentIndex(n.cmb_translate.findData("de"))
    assert [m.key for m in n.translate_models_needed()] == ["opus-ru-en", "opus-en-de"] and "English" in n.lbl_tr_state.text()
    assert not n.btn_start.isEnabled()
    # same language: nothing to do, the start button works
    n.cmb_translate.setCurrentIndex(n.cmb_translate.findData("ru"))
    assert "already in this language" in n.lbl_tr_state.text() and n.translate_plan() is None and n.btn_start.isEnabled()
    n.chk_translate.setChecked(False)
    assert n.options().translate is None


def test_translate_card_warns_when_the_voice_speaks_another_language(app, lib, tmp_path):
    i18n.set_language("en")
    s = make_studio(lib)
    n = s.narrate_window
    n.model_state = lambda m: text_models.STATE_READY
    n.show()
    rec = add_voice(lib, tmp_path)
    n.refresh_voices()
    n.load_book_file(book_file(tmp_path, EN_TXT))
    n.chk_translate.setChecked(True)
    n.cmb_translate.setCurrentIndex(n.cmb_translate.findData("de"))
    lang = rec.language
    if lang and lang.lower() not in ("de", "german"):
        assert n.lbl_tr_voice.isVisibleTo(n) and "narrated in German" in n.lbl_tr_voice.text()
    n.cmb_translate.setCurrentIndex(n.cmb_translate.findData("en"))
    assert "already in this language" in n.lbl_tr_state.text()


def test_translate_card_is_localized_in_en_ru_de(app, lib, tmp_path):
    s = make_studio(lib)
    n = s.narrate_window
    n.show()
    seen = set()
    for lang, word, right in (("en", "Translate the book", "right to translate"), ("ru", "Перевод книги", "отвечаете за наличие прав"),
                              ("de", "Buch übersetzen", "verantwortlich")):
        i18n.set_language(lang)
        n.retranslate()
        assert word in n.lbl_tr_title.text() and right in n.lbl_tr_note.text()
        names = [n.cmb_translate.itemText(i) for i in range(n.cmb_translate.count())]
        assert len(set(names)) == 3
        seen.add(tuple(names))
    assert len(seen) == 3
    i18n.set_language("en")
