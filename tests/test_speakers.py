"""Speaker marks and multi-voice narration with fake models and fake engines. No GPU, no download."""
from __future__ import annotations

from core import speakers as spk
from core.book_parsers import Book, Chapter
from core.chunker import chunk_book
from core.llm_text import LLMPlan
from tests.test_llm_text import FakeModel
from tests.test_narration import FakeEngine, FakeFfmpeg, run


def _book():
    return Book("Cast", "", "en", [Chapter("One",
        'He walked along the quiet road for a while.\n\n'
        '"Hello there," she said softly.\n\n'
        '"Good day to you," he answered at once.')])


def test_parse_and_tag_paragraphs():
    assert spk.parse_tags("NARRATOR\nFEMALE: Ann\nMALE: Tom\n", 3) == [
        spk.SpeakerLine("narrator"), spk.SpeakerLine("female", "Ann"), spk.SpeakerLine("male", "Tom")]
    assert spk.parse_tags("NARRATOR\nnope\n", 2) is None
    paras = ["He walked along the quiet road.", '"Hello," she said to him today.']

    def answer(prompt):
        text = prompt.rsplit("TEXT:\n", 1)[1]
        n = len([p for p in text.split("\n\n") if p.strip()])
        return "\n".join(["NARRATOR", "FEMALE: Ann"][:n])

    model = FakeModel(answer)
    lines = spk.tag_paragraphs(paras, "en", LLMPlan(lambda: model, "fake"))
    assert [ln.role for ln in lines] == ["narrator", "female"] and lines[1].name == "Ann" and model.closed


def test_assign_sets_voice_ids_and_a_mismatch_leaves_them_alone():
    book = _book()
    chunks = chunk_book(book, speak_titles=False)
    lines = [spk.SpeakerLine("narrator"), spk.SpeakerLine("female", "Ann"), spk.SpeakerLine("male", "Tom")]
    assigned, note = spk.assign(chunks, book, lines, {"male": "m", "female": "f"}, "narr")
    assert note == ""
    heard = " ".join(c.text for c in assigned if c.voice_id == "f")
    assert "Hello" in heard and any(c.voice_id == "m" and "Good day" in c.text for c in assigned)
    assert all(c.voice_id == "" for c in assigned if "walked" in c.text)
    same, note = spk.assign(chunks, book, lines[:1], {"male": "m", "female": "f"}, "narr")
    assert note == "mismatch" and all(c.voice_id == "" for c in same)


def test_narration_uses_each_voice_and_a_mismatch_stays_with_the_narrator(tmp_path):
    from core import narration as nr
    from core import audiobook_export as ex

    book = _book()
    lines = [spk.SpeakerLine("narrator"), spk.SpeakerLine("female", "Ann"), spk.SpeakerLine("male", "Tom")]
    cast = spk.SpeakerCast(lines=lines, male_id="m", female_id="f", narrator_id="narr")
    female, male = FakeEngine(), FakeEngine()
    female.tag, male.tag = "fem", "male"
    made = {"f": 0, "m": 0}

    def ff():
        made["f"] += 1
        return female

    def mf():
        made["m"] += 1
        return male

    opts = nr.NarrationOptions(speakers=cast, speak_titles=False, formats={ex.FORMAT_WAV_CHAPTERS})
    res, narr, _ff, _ev = run(tmp_path, book=book, options=opts, extra_engines={"f": (ff, "fem"), "m": (mf, "male")})
    assert any("Hello" in c for c in female.calls)
    assert any("Good day" in c for c in male.calls)
    assert any("walked" in c for c in narr.calls)
    assert narr.closed and female.closed and male.closed and made == {"f": 1, "m": 1}
    assert (res.out_dir / ".debug" / "speakers.txt").is_file()

    lone = FakeEngine()
    calls = {"m": 0}

    def unused():
        calls["m"] += 1
        return FakeEngine()

    run(tmp_path / "one", engine=lone, book=book, options=opts)
    assert calls["m"] == 0 and any("Hello" in c for c in lone.calls)

    bad = spk.SpeakerCast(lines=lines[:1], male_id="m", female_id="f", narrator_id="narr")
    narr2 = FakeEngine()
    res2, _e, _f, _ev = run(
        tmp_path / "bad", engine=narr2, book=book,
        options=nr.NarrationOptions(speakers=bad, speak_titles=False, formats={ex.FORMAT_WAV_CHAPTERS}),
        extra_engines={"m": (unused, "male")})
    assert calls["m"] == 0
    assert "mismatch" in (res2.out_dir / ".debug" / "speakers.txt").read_text(encoding="utf-8")
    assert narr2.calls
