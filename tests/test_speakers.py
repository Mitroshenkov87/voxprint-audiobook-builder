"""Speaker marks and multi-voice narration with fake models and fake engines. No GPU, no download."""
from __future__ import annotations

from pathlib import Path

from core import speakers as spk
from core.book_parsers import Book, Chapter, parse_txt
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
    note = (res2.out_dir / ".debug" / "speakers.txt").read_text(encoding="utf-8")
    assert "mismatch" in note and res2.speaker_warning == "mismatch"
    assert spk.load_marks(note) == lines[:1]
    assert narr2.calls


def test_marks_file_round_trip():
    lines = [spk.SpeakerLine("narrator"), spk.SpeakerLine("female", "Ann"), spk.SpeakerLine("male", "Tom")]
    text = spk.dump_marks(lines)
    assert spk.load_marks(text) == lines
    assert spk.load_marks("NARRATOR\nFEMALE: Ann\nMALE: Tom\n") == lines
    try:
        spk.load_marks("1. NARRATOR\nhello\n")
    except ValueError as exc:
        assert "unreadable" in str(exc)
    else:
        raise AssertionError("a bad mark line must be rejected")


_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "spor-multivoice-ru.txt"
# 1-based indexes from the real-PC dialogue: the rest are the narrator.
_MALE = {3, 5, 8, 10, 12, 15, 18, 21}
_FEMALE = {4, 6, 9, 11, 14, 16, 19}


def _expected_roles(n: int):
    roles = []
    for i in range(1, n + 1):
        if i in _MALE:
            roles.append(spk.SpeakerLine("male", "Марк"))
        elif i in _FEMALE:
            roles.append(spk.SpeakerLine("female", "Анна"))
        else:
            roles.append(spk.SpeakerLine("narrator"))
    return roles


def _gemma_reply(prompt: str, paras) -> str:
    """A Gemma 4 reply: a thought channel (with tag-shaped lines), a preamble, then numbered marks.

    This is what ``parse_tags`` used to reject, leaving every paragraph as the narrator.
    """
    body = prompt.rsplit("TEXT:\n", 1)[1]
    block = [p.strip() for p in body.split("\n\n") if p.strip()]
    start = next(i for i, para in enumerate(paras) if para == block[0])
    chosen = _expected_roles(len(paras))[start:start + len(block)]
    numbered = "\n".join(
        f"{n}. " + ("NARRATOR" if line.role == "narrator" else f"{line.role.upper()}: {line.name}")
        for n, line in enumerate(chosen, 1))
    return (
        "<|channel>thought\n"
        "A paragraph that starts with an em dash is speech. Марк is male, Анна is female.\n"
        "NARRATOR\n"
        "MALE: Марк\n"
        "<channel|>\n"
        "Here are the speaker marks:\n\n"
        + numbered + "\n"
    )


def test_a_realistic_gemma_reply_marks_the_russian_dialogue():
    """The parse path that returned 21 narrators on the real PC."""
    text = _FIXTURE.read_text(encoding="utf-8")
    book = parse_txt(text, "spor")
    paras = [para for _ci, para in spk.paragraphs(book)]
    assert len(paras) == 21
    assert len(spk._batches(paras)) >= 2
    bare = (
        "<|channel>thought\nNARRATOR\n<channel|>\nHere are the speaker marks:\n\n"
        "1. NARRATOR\n2. MALE: Марк\n"
    )
    assert spk.parse_tags(bare, 2) == [spk.SpeakerLine("narrator"), spk.SpeakerLine("male", "Марк")]
    assert spk.parse_tags("<think>\nMALE: Марк\n</think>\nРАССКАЗЧИК\nЖЕН: Анна\n", 2) == [
        spk.SpeakerLine("narrator"), spk.SpeakerLine("female", "Анна")]
    assert spk.parse_tags("МУЖСКОЙ: Иван\nЖЕНСКИЙ\nМУЖ: Пётр\n", 3) == [
        spk.SpeakerLine("male", "Иван"), spk.SpeakerLine("female"), spk.SpeakerLine("male", "Пётр")]
    assert spk.load_marks("МУЖСКОЙ: Иван\nЖЕНСКИЙ\n") == [
        spk.SpeakerLine("male", "Иван"), spk.SpeakerLine("female")]
    model = FakeModel(lambda prompt: _gemma_reply(prompt, paras))
    tagged = spk.tag_paragraphs(paras, "ru", LLMPlan(lambda: model, "fake"))
    assert tagged.warning == "" and model.closed
    assert model.calls == len(spk._batches(paras))
    assert list(tagged) == _expected_roles(21)
    assert tagged[2].role == "male" and tagged[3].name == "Анна"
    assert "<|channel>thought" in tagged.raw and "3. MALE: Марк" in tagged.raw
    flat = spk.tag_paragraphs(
        paras, "ru", LLMPlan(lambda: FakeModel(lambda prompt: "\n".join(
            ["NARRATOR"] * len([p for p in prompt.rsplit("TEXT:\n", 1)[1].split("\n\n") if p.strip()]))), "fake"))
    assert flat.warning == spk.WARN_NO_SPEAKERS and all(ln.role == "narrator" for ln in flat)


def test_an_unreadable_reply_warns_and_an_all_narrator_dialogue_warns(tmp_path):
    from core import audiobook_export as ex
    from core import narration as nr

    paras = ["He walked along the quiet road.", '"Hello," she said to him today.']

    def garbage(_prompt):
        return "Sure, I can help with that story."

    missed = spk.tag_paragraphs(paras, "en", LLMPlan(lambda: FakeModel(garbage), "fake"))
    assert missed.warning == spk.WARN_UNPARSED
    assert [ln.role for ln in missed] == ["narrator", "narrator"]
    assert "Sure, I can help" in missed.raw

    def all_narrator(prompt):
        n = len([p for p in prompt.rsplit("TEXT:\n", 1)[1].split("\n\n") if p.strip()])
        return "\n".join(["NARRATOR"] * n)

    flat = spk.tag_paragraphs(paras, "en", LLMPlan(lambda: FakeModel(all_narrator), "fake"))
    assert flat.warning == spk.WARN_NO_SPEAKERS and "NARRATOR" in flat.raw

    def boom(_prompt):
        raise RuntimeError("llama died")

    failed = spk.tag_paragraphs(["— Привет, Анна.", "Она молчала."], "ru", LLMPlan(lambda: FakeModel(boom), "fake"))
    assert failed.warning == spk.WARN_UNPARSED and "RuntimeError" in failed.raw
    assert all(ln.role == "narrator" for ln in failed)

    book = _book()
    cast = spk.SpeakerCast(tagger=LLMPlan(lambda: FakeModel(garbage), "fake"), male_id="m", female_id="f",
                           narrator_id="narr")
    opts = nr.NarrationOptions(speakers=cast, speak_titles=False, formats={ex.FORMAT_WAV_CHAPTERS})
    narr = FakeEngine()
    res, _eng, _ff, _ev = run(
        tmp_path, engine=narr, book=book, options=opts,
        extra_engines={"m": (FakeEngine, "m"), "f": (FakeEngine, "f")})
    assert res.speaker_warning == "unparsed"
    raw = (res.out_dir / ".debug" / "speakers-raw.txt").read_text(encoding="utf-8")
    assert "Sure, I can help" in raw
    assert any("Hello" in call for call in narr.calls)
    marks = (res.out_dir / ".debug" / "speakers.txt").read_text(encoding="utf-8")
    assert "mismatch" not in marks and spk.load_marks(marks)


def _three_men():
    return Book("Men", "", "en", [Chapter("One",
        'The three of them sat down.\n\n'
        '"I start," David said.\n\n'
        '"I follow," Michael said.\n\n'
        '"And I," Hannah said.\n\n'
        '"Me again," David said.\n\n'
        '"Third man," Saul said.')])


def _three_men_marks():
    return [spk.SpeakerLine("narrator"), spk.SpeakerLine("male", "David"), spk.SpeakerLine("male", "Michael"),
            spk.SpeakerLine("female", "Hannah"), spk.SpeakerLine("male", "david"), spk.SpeakerLine("male", "Saul")]


def test_second_male_voice_alternates_by_character_and_old_casts_still_work():
    lines = _three_men_marks()
    one = spk.SpeakerCast(lines=lines, male_id="m", female_id="f", narrator_id="narr")
    assert one.voices_for(lines) == ["", "m", "m", "f", "m", "m"]          # unchanged: one voice per role
    two = spk.SpeakerCast(lines=lines, male_id="m", female_id="f", male2_id="m2", narrator_id="narr")
    assert two.voices_for(lines) == ["", "m", "m2", "f", "m", "m"]         # David m, Michael m2, Saul m again
    assert two.extra_ids("narr") == ["m", "m2", "f"] and two.uses_several("narr")
    assert two.assignment(lines) == {"David": "m", "Michael": "m2", "Hannah": "f", "Saul": "m"}
    pinned = spk.SpeakerCast(lines=lines, male_id="m", male2_id="m2", narrator_id="narr",
                             characters={"SAUL": "s", "Michael": "m"})
    assert pinned.voices_for(lines) == ["", "m", "m", "", "m", "s"]       # pins win; no female voice -> narrator
    assert pinned.extra_ids("narr") == ["m", "m2", "s"]
    unnamed = [spk.SpeakerLine("male"), spk.SpeakerLine("male", "Tom")]
    assert two.voices_for(unnamed) == ["m", "m"]
    assert spk.parse_character_map(["David = asher", "Hannah=noa"]) == {"David": "asher", "Hannah": "noa"}
    for bad in (["David"], ["=x"], ["David="]):
        try:
            spk.parse_character_map(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_narration_gives_two_men_two_voices(tmp_path):
    from core import narration as nr
    from core import audiobook_export as ex

    book = _three_men()
    cast = spk.SpeakerCast(lines=_three_men_marks(), male_id="m", female_id="f", male2_id="m2", narrator_id="narr")
    engines = {k: FakeEngine() for k in ("m", "m2", "f")}
    extra = {k: (lambda e=e: e, k) for k, e in engines.items()}
    opts = nr.NarrationOptions(speakers=cast, speak_titles=False, formats={ex.FORMAT_WAV_CHAPTERS})
    _res, narr, _ff, _ev = run(tmp_path, book=book, options=opts, extra_engines=extra)
    said = {k: " ".join(e.calls) for k, e in engines.items()}
    assert "I start" in said["m"] and "Me again" in said["m"] and "Third man" in said["m"]
    assert "I follow" in said["m2"] and "I follow" not in said["m"]
    assert "And I" in said["f"] and any("sat down" in c for c in narr.calls)


class _Rec:
    def __init__(self, vid, name, gender, repo_id=""):
        self.id, self.name, self.info = vid, name, {"gender": gender, "repo_id": repo_id}


def test_role_lists_drop_boaz_and_default_to_three_different_men():
    from infra import bundled_voices

    lib = [_Rec("v1", "Boaz", "male", "boaz"), _Rec("v2", "Gideon", "male", "gideon"), _Rec("v3", "Asher", "male", "asher"),
           _Rec("v4", "Eitan", "male", "eitan"), _Rec("v5", "Noa", "female", "noa"), _Rec("v6", "Boaz copy", "male")]
    offered = bundled_voices.offered_for_roles(lib)
    assert [r.name for r in offered] == ["Gideon", "Asher", "Eitan", "Noa", "Boaz copy"]
    assert bundled_voices.is_retired(_Rec("x", "Boaz", "male"))                 # an old copy without a repo id
    picks = spk.default_role_picks(offered, narrator_id="v2")
    assert picks == {"male": "v3", "male2": "v4", "female": "v5"}
    assert spk.default_role_picks([_Rec("a", "Gideon", "male"), _Rec("b", "Noa", "female")], "a") == \
        {"male": "a", "male2": "", "female": "b"}


def test_a_merged_title_splits_the_block_instead_of_losing_it():
    """Build 702 bug: Gemma merged the title with the first paragraph (11 tags for 12 paragraphs) and the whole block fell
    back to the narrator (16 of 26 marks). Now the block is split and asked again; only the merged pair costs a retry."""
    paras = ["Беседа вторая", "Исход субботы, трое задержались в синагоге.",
             "— Ариэль, ты опять про модели? — спросила Яэль.", "— Про них, — ответил Ариэль.",
             "— Модель ошибается, как ученик, — сказал Барух.", "— И учится, — сказала Яэль."]
    roles = ["NARRATOR", "NARRATOR", "FEMALE: Яэль", "MALE: Ариэль", "MALE: Барух", "FEMALE: Яэль"]
    asked = []

    def answer(prompt):
        text = prompt.rsplit("TEXT:\n", 1)[1]
        block = [p for p in text.split("\n\n") if p.strip()]
        asked.append(len(block))
        idx = [next(i for i, p in enumerate(paras) if p in b) for b in block]
        tags = [roles[i] for i in idx]
        if 0 in idx and 1 in idx:          # the title and the intro come back as one line
            tags = tags[1:]
        return "\n".join(tags)

    model = FakeModel(answer)
    res = spk.tag_paragraphs(paras, "ru", LLMPlan(lambda: model, "fake"))
    assert [ln.role for ln in res] == ["narrator", "narrator", "female", "male", "male", "female"]
    assert [ln.name for ln in res][2:] == ["Яэль", "Ариэль", "Барух", "Яэль"]
    assert res.warning == "" and model.closed
    assert asked[0] == 6 and len(asked) > 1 and max(asked[1:]) < 6
    assert "[retry] 6 paragraphs -> 3 + 3" in res.raw


def test_only_a_paragraph_the_model_never_marks_stays_narrator():
    paras = ["— Привет, — сказала Анна.", "Тихо было.", "— Здравствуй, — ответил Том."]

    def answer(prompt):
        text = prompt.rsplit("TEXT:\n", 1)[1]
        block = [p for p in text.split("\n\n") if p.strip()]
        if any("Тихо" in p for p in block):
            return "This paragraph has no speaker."   # never a valid tag, not even alone
        return "\n".join("FEMALE: Анна" if "Анна" in p else "MALE: Том" for p in block)

    res = spk.tag_paragraphs(paras, "ru", LLMPlan(lambda: FakeModel(answer), "fake"))
    assert [ln.role for ln in res] == ["female", "narrator", "male"]
    assert res.warning == spk.WARN_UNPARSED


_DIALOG04 = Path(__file__).resolve().parent / "fixtures" / "yo" / "dialog-ai-torah-04.txt"


def _mark_line(line):
    if line.role == "narrator":
        return "NARRATOR"
    return f"{line.role.upper()}: {line.name}"


def test_a_one_paragraph_shift_on_dialogue_04_is_asked_again_until_19_match():
    """Gemma can return the right number of tags shifted by one paragraph (dialogue 4 scored 11/19).

    The marks file is the ground truth. A block longer than one paragraph comes back shifted; a single
    paragraph comes back right. The check must reject the shift and end at 19/19, and the raw log must
    say ``[validate]``. A reply that is already right is kept, with no validate line.
    """
    text = _DIALOG04.read_text(encoding="utf-8")
    truth = spk.load_marks((_DIALOG04.parent / "dialog-ai-torah-04.marks.txt").read_text(encoding="utf-8"))
    paras = [para for _ci, para in spk.paragraphs(parse_txt(text, "torah"))]
    assert len(paras) == len(truth) == 19

    def reply(prompt, shift):
        body = prompt.rsplit("TEXT:\n", 1)[1]
        block = [p.strip() for p in body.split("\n\n") if p.strip()]
        start = next(i for i, para in enumerate(paras) if para == block[0])
        chosen = truth[start:start + len(block)]
        assert len(chosen) == len(block)
        if shift and len(chosen) > 1:
            chosen = chosen[1:] + chosen[-1:]
        return "\n".join(_mark_line(line) for line in chosen)

    straight_model = FakeModel(lambda prompt: reply(prompt, False))
    straight = spk.tag_paragraphs(paras, "ru", LLMPlan(lambda: straight_model, "fake"))
    assert list(straight) == truth and straight.warning == "" and straight_model.closed
    assert "[validate]" not in straight.raw
    assert straight_model.calls == len(spk._batches(paras))

    shifted = FakeModel(lambda prompt: reply(prompt, True))
    res = spk.tag_paragraphs(paras, "ru", LLMPlan(lambda: shifted, "fake"))
    assert list(res) == truth and res.warning == "" and shifted.closed
    assert "[validate]" in res.raw
    assert shifted.calls > len(spk._batches(paras))
