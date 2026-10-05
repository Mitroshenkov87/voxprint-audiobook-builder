"""Pause strength: explicit silence between sentences, commas, ellipses, dashes, paragraphs and chapters."""
import numpy as np
import pytest
import soundfile as sf

from core import chunker
from core import narration as nr
from core import pauses as pz
from core.book_parsers import Book, Chapter
from tests.test_narration import SR, FakeEngine, run
from tests.test_studio import app, lib  # noqa: F401  (fixtures)

TEXT = ("He came into the room, looked around and said quietly: \"Hello\". Nobody answered... Then he - a stubborn man - "
        "repeated it louder! Yes.\n\n* * *\n\nA new scene begins, and it has commas, dashes - and an ellipsis\u2026 The end.")


def test_the_default_is_normal_and_longer_than_the_old_fixed_pauses():
    d = pz.PauseProfile()
    assert d.level == pz.DEFAULT_LEVEL and pz.LEVELS[d.level] == 1.0
    assert d.ms(pz.SENTENCE) > chunker.PAUSE_SENTENCE_MS and d.ms(pz.PARAGRAPH) > chunker.PAUSE_PARAGRAPH_MS
    assert d.ms(pz.CHAPTER) > nr.CHAPTER_TAIL_MS and d.ms(pz.TITLE) > chunker.PAUSE_AFTER_TITLE_MS
    assert d.ms(pz.SCENE) > chunker.PAUSE_SCENE_MS


def test_every_kind_grows_with_the_slider_and_the_order_of_kinds_is_kept():
    tables = [pz.PauseProfile(i).table() for i in range(len(pz.LEVELS))]
    for kind in pz.KINDS:
        values = [t[kind] for t in tables]
        assert values == sorted(values) and values[0] < values[-1], kind
    for t in tables:
        assert t[pz.COMMA] < t[pz.SENTENCE] < t[pz.ELLIPSIS] < t[pz.PARAGRAPH] < t[pz.SCENE]
        assert t[pz.COMMA] < t[pz.DASH] < t[pz.SENTENCE]


def test_separate_multipliers_and_bad_levels():
    p = pz.PauseProfile(2, {pz.ELLIPSIS: 2.0, pz.COMMA: 0.0})
    assert p.ms(pz.ELLIPSIS) == 2 * pz.BASE_MS[pz.ELLIPSIS] and p.ms(pz.COMMA) == 0
    assert p.ms(pz.SENTENCE) == pz.BASE_MS[pz.SENTENCE]
    assert pz.PauseProfile(99).level == len(pz.LEVELS) - 1 and pz.PauseProfile(-3).level == 0
    assert pz.clamp_level("x") == pz.DEFAULT_LEVEL and pz.clamp_level(None) == pz.DEFAULT_LEVEL
    assert max(pz.PauseProfile(4, {k: 100.0 for k in pz.KINDS}).table().values()) == pz.MAX_PAUSE_MS


def test_text_is_cut_at_every_kind_of_break():
    pieces = chunker.chunk_text_pauses(TEXT)
    kinds = [k for _, k in pieces]
    for kind in (pz.COMMA, pz.SENTENCE, pz.ELLIPSIS, pz.DASH, pz.PARAGRAPH, pz.SCENE):
        assert kind in kinds, kind
    assert kinds[-1] == pz.PARAGRAPH
    joined = " ".join(t for t, _ in pieces)
    for word in ("room,", "Nobody", "stubborn", "ellipsis\u2026", "end."):
        assert word in joined                                          # nothing is lost
    # no separate utterance is tiny (a comma after "Well" stays with the next words)
    assert all(len(t) >= 3 for t, _ in pieces)
    assert chunker.chunk_text_pauses("Well, yes, of course it is so.")[0][0].startswith("Well, yes")


def test_a_long_sentence_is_cut_but_keeps_all_words():
    sentence = " ".join(f"word{i}" for i in range(120)) + "."
    pieces = chunker.chunk_text_pauses(sentence, 100)
    assert all(len(t) <= 100 for t, _ in pieces) and len(pieces) > 1
    assert " ".join(t for t, _ in pieces).split() == sentence.split()
    assert [k for _, k in pieces[:-1]] == [pz.COMMA] * (len(pieces) - 1)


def test_chunk_book_uses_the_profile_and_legacy_mode_is_unchanged():
    book = Book("B", "A", "en", [Chapter("One", TEXT)])
    new = chunker.chunk_book(book, 260, speak_titles=True, pauses=pz.PauseProfile(4))
    assert new[0].pause_kind == pz.TITLE and new[0].pause_ms == pz.PauseProfile(4).ms(pz.TITLE)
    assert all(c.pause_ms == pz.PauseProfile(4).ms(c.pause_kind) for c in new)
    old = chunker.chunk_book(book, 260, speak_titles=True)
    assert old[0].pause_ms == chunker.PAUSE_AFTER_TITLE_MS and all(c.pause_kind == "" for c in old)
    assert len(new) > len(old)


def _silence_runs(path, thresh=1e-4):
    data, sr = sf.read(str(path), dtype="float32")
    quiet = np.abs(data) < thresh
    runs, n = [], 0
    for q in quiet:
        if q:
            n += 1
        elif n:
            runs.append(n / sr * 1000)
            n = 0
    if n:
        runs.append(n / sr * 1000)
    return runs


def test_narrate_book_cuts_every_sentence_into_its_own_chunk_when_pauses_are_on(tmp_path):
    book = Book("B", "A", "en", [Chapter("One", "First sentence is here. Second sentence is here.\n\nNew paragraph starts here.")])
    res, eng, ff, events = run(tmp_path, engine=FakeEngine(), book=book,
                               options=nr.NarrationOptions(speak_titles=False, pauses=pz.PauseProfile()))
    assert eng.calls == ["First sentence is here.", "Second sentence is here.", "New paragraph starts here."]
    _res, eng2, *_ = run(tmp_path / "legacy", engine=FakeEngine(), book=book,
                         options=nr.NarrationOptions(speak_titles=False, pauses=None))
    assert len(eng2.calls) == 2                       # the earlier packed chunks: one per paragraph


def test_assembly_inserts_exactly_the_profile_lengths(tmp_path):
    from core.narration import ChunkCache, assemble_chapters
    book = Book("B", "A", "en", [Chapter("One", "First sentence is here. Second sentence is here.\n\nNew paragraph starts here.")])
    chunks = chunker.chunk_book(book, 260, pauses=pz.PauseProfile(4))
    cache = ChunkCache(tmp_path / "c")
    texts = {c.index: c.text for c in chunks}
    for c in chunks:
        cache.save(cache.key("t", c.text), np.full(SR // 10, 0.5, dtype=np.float32), SR)
    audio = assemble_chapters(book, chunks, "t", cache, texts, tmp_path / "w", pz.PauseProfile(4))
    runs = _silence_runs(audio[0].wav)
    prof = pz.PauseProfile(4)
    expected = [prof.ms(pz.SENTENCE), prof.ms(pz.PARAGRAPH), prof.ms(pz.CHAPTER)]
    assert len(runs) == 3 and all(abs(r - e) < 5 for r, e in zip(runs, expected))
    shorter = assemble_chapters(book, chunks, "t", cache, texts, tmp_path / "w2", pz.PauseProfile(0))
    assert audio[0].duration > shorter[0].duration + 2.0


def test_level_is_remembered(tmp_path, monkeypatch):
    from infra import paths
    monkeypatch.setattr(paths, "state_dir", lambda: tmp_path)
    assert pz.load_level() == pz.DEFAULT_LEVEL
    pz.save_level(4)
    assert pz.load_level() == 4
    (tmp_path / "pause_level.txt").write_text("junk", encoding="utf-8")
    assert pz.load_level() == pz.DEFAULT_LEVEL


def test_narrate_window_has_the_slider_with_the_default_and_passes_it_to_the_job(tmp_path, monkeypatch, app, lib):
    from core import i18n
    from infra import paths
    from tests.test_voice_catalog_ui import narrate
    monkeypatch.setattr(paths, "state_dir", lambda: tmp_path)
    i18n.set_language("en")
    n = narrate(lib, tmp_path, [], runner=lambda *a: None)
    assert not n.chk_pauses.isChecked() and n.pause_profile() is None and not n.sld_pauses.isEnabled()   # opt-in
    n.chk_pauses.setChecked(True)
    assert pz.load_enabled() and n.sld_pauses.isEnabled()
    assert n.sld_pauses.value() == pz.DEFAULT_LEVEL and n.pause_profile().level == pz.DEFAULT_LEVEL
    assert n.lbl_pauses.text() == "Pauses" and "normal" in n.lbl_pauses_value.text() and "430 ms" in n.lbl_pauses_value.text()
    n.sld_pauses.setValue(4)
    assert n.options().pauses.level == 4 and "much longer" in n.lbl_pauses_value.text()
    assert pz.load_level() == 4                                    # remembered for the next start
    i18n.set_language("ru")
    n.retranslate()
    assert n.lbl_pauses.text() == "Паузы" and "мс" in n.lbl_pauses_value.text()
    i18n.set_language("de")
    n.retranslate()
    assert n.lbl_pauses.text() == "Pausen"
    n.shutdown()
    n2 = narrate(lib, tmp_path, [], runner=lambda *a: None)
    assert n2.sld_pauses.value() == 4
    n2.shutdown()
