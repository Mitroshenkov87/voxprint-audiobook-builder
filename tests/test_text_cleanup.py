"""Neural clean-up: the validator accepts only safe edits; the driver caches; the registry and the plan; narration wiring."""
import json
from collections import Counter
from pathlib import Path

import pytest

from core import book_prep
from core import narration as nr
from core import text_cleanup as tc
from core.book_parsers import Book, Chapter
from core.errors import CancelledByUser
from core.events import CancelToken
from core.text_prep import PrepOptions
from infra import text_models as tm

SRC = "Мы долго шли по лесной дорге и наконец вышли к реке где стояла старая мельница и мы очень устали. Он сказал что веселый день кончился."
GOOD = "Мы долго шли по лесной дороге и наконец вышли к реке, где стояла старая мельница, и мы очень устали. Он сказал, что весёлый день кончился."


def judge(src, prop, freq=None):
    st = tc.CleanupStats()
    return tc.validate(src, prop, freq or Counter(), st), st


def test_damerau_distance():
    assert tc.damerau("дорге", "дороге") == 1 and tc.damerau("ab", "ba") == 1 and tc.damerau("abc", "xyz") == 3


def test_good_proposal_is_accepted_piece_by_piece():
    out, st = judge(SRC, GOOD)
    assert out == GOOD and (st.typos, st.yo, st.commas) == (1, 1, 3) and st.rejected == 0


def test_rewording_deleting_and_changing_punctuation_is_rejected():
    out, _ = judge(SRC, "Мы долго шли по лесной дороге и вышли к реке, мельница старая и мы устали. Он сказал что веселый день кончился.")
    assert out == SRC.replace("дорге", "дороге")                           # the safe part is taken, the rewording is not
    for bad in (SRC.replace("долго шли", "шли"),                               # deleted word
                SRC.replace("кончился.", "кончился!"),                          # changed end mark
                SRC.replace("Мы долго", "мы долго"),                            # case change
                SRC.upper(), "Совсем другой текст."):
        out, _ = judge(SRC, bad)
        assert out == SRC, bad


def test_frequent_words_and_names_are_not_corrected():
    src = "Он встретил Петрова на улице Невской и поздоровался."
    out, _ = judge(src, src.replace("Петрова", "Петров").replace("Невской", "Невский"))
    assert out == src                                                      # capitalized mid-sentence = names
    freq = Counter({"дорге": 30})
    out, _ = judge("Он шёл по дорге домой.", "Он шёл по дороге домой.", freq)
    assert out == "Он шёл по дорге домой."                                  # the "typo" is frequent: probably dialect


def test_yo_is_accepted_but_not_for_vse():
    out, st = judge("Мы весело пели и веселый день шёл.", "Мы весело пели и весёлый день шёл.")
    assert "весёлый" in out and st.yo == 1
    out, _ = judge("Он сказал, что все сложилось.", "Он сказал, что всё сложилось.")
    assert out == "Он сказал, что все сложилось."


def test_a_block_with_too_many_typo_fixes_is_left_alone_but_commas_survive():
    src = "Каждй день делой и спена балеть."
    out, st = judge(src, "Каждый день делай и спина болеть.")
    assert out == src and st.untouched_blocks == 1


def test_comma_density_is_capped():
    src = "один два три четыре пять шесть семь восемь девять десять"
    out, _ = judge(src, "один, два, три, четыре, пять, шесть, семь, восемь, девять, десять")
    assert out == src


class FakeCleaner:
    tag = "fake-cleaner@1"

    def __init__(self, mapping=None):
        self.calls, self.closed, self.mapping = [], False, mapping or {}

    def correct(self, texts):
        self.calls.append(list(texts))
        return [self.mapping.get(t, t) for t in texts]

    def close(self):
        self.closed = True


def sample_book():
    return Book("Т", "А", "ru", [Chapter("Глава", SRC + "\n\nКороткий абзац.\n\nСтих без точки\nвторая строка")])


def test_cleanup_book_applies_validated_edits_caches_and_resumes(tmp_path):
    book = sample_book()
    eng = FakeCleaner({SRC.split(". ")[0] + ". " + SRC.split(". ")[1]: GOOD})
    blocks_before = len(tc.split_blocks(book.chapters[0].text.split("\n\n")[0]))
    cache = tmp_path / "cleanup.json"
    out, st = tc.cleanup_book(book, eng, cache)
    assert out.chapters[0].text.split("\n\n")[0] == GOOD and out.chapters[0].text.split("\n\n")[1] == "Короткий абзац."
    assert out.chapters[0].text.split("\n\n")[2] == "Стих без точки\nвторая строка"        # verse untouched
    assert sum(len(c) for c in eng.calls) == st.blocks - st.cached and cache.exists()
    assert book.chapters[0].text.startswith("Мы долго шли по лесной дорге")                   # the input is not modified
    eng2 = FakeCleaner()
    out2, st2 = tc.cleanup_book(book, eng2, cache)
    assert eng2.calls == [] and st2.cached == st2.blocks and out2.chapters[0].text == out.chapters[0].text
    assert blocks_before >= 1


def test_cleanup_cancel_and_progress(tmp_path):
    book = Book("T", chapters=[Chapter("", "\n\n".join(f"Абзац номер {i}." for i in range(30)))])
    seen, cancel = [], CancelToken()
    eng = FakeCleaner()

    def prog(f, m):
        seen.append(f)
        cancel.cancel()
    with pytest.raises(CancelledByUser):
        tc.cleanup_book(book, eng, tmp_path / "c.json", prog, cancel)
    assert seen and seen[0] < 1.0


def test_a_damaged_cache_file_is_ignored(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{not json", encoding="utf-8")
    assert tc.BlockCache(p).data == {}


def test_blocks_are_whole_sentences_within_the_limit():
    text = " ".join(f"Это предложение номер {i} в длинном абзаце." for i in range(40))
    blocks = tc.split_blocks(text, 200)
    assert all(len(b) <= 200 for b in blocks) and " ".join(blocks) == text


# ----------------------------------------------------------------------------- registry and plan

def test_registry_is_consistent_and_placeholders_are_marked():
    keys = [m.key for m in tm.REGISTRY]
    assert len(keys) == len(set(keys))
    sage = tm.get("sage-ru")
    assert sage.integrated and sage.license == "MIT" and len(sage.revision) == 40 and sage.repo.startswith("ai-forever/")
    for step in (tm.STEP_STRESS, tm.STEP_ROLES, tm.STEP_PUNCT):
        m = tm.for_step(step)
        assert m is not None and not m.integrated and tm.state(m) == tm.STATE_PLANNED
    # translation: one integrated Opus-MT model per direction, a pinned commit and the SHA-256 of the weights
    for key, pair in (("opus-ru-en", ("ru", "en")), ("opus-en-ru", ("en", "ru")), ("opus-de-en", ("de", "en")), ("opus-en-de", ("en", "de"))):
        m = tm.get(key)
        assert m.integrated and m.pair == pair and len(m.revision) == 40 and m.license in ("CC-BY-4.0", "Apache-2.0")
        assert len(dict(m.sha256)["pytorch_model.bin"]) == 64 and "pytorch_model.bin" in m.files and tm.translate_model(*pair) is m
    assert tm.for_step(tm.STEP_SPELLFIX, "ru").key == "sage-ru" and tm.for_step(tm.STEP_SPELLFIX, "en").key == "spell-en"


def test_state_and_ensure_with_a_fake_download(tmp_path, monkeypatch):
    from infra import model_downloader as md
    sage = tm.get("sage-ru")
    assert tm.state(sage) == tm.STATE_NEEDS_DOWNLOAD and tm.cleanup_engine_for("ru") is None
    calls = []

    def fake_snapshot(repo_id, local_dir, revision=None, **kw):
        calls.append((repo_id, revision))
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text("{}")
        (Path(local_dir) / "model.safetensors").write_bytes(b"x")

    monkeypatch.setenv("VOXPRINT_NO_MIRROR", "1")
    monkeypatch.setattr(md, "manifest_bad_files", lambda *a, **k: None)   # fake files: the downloader's hash check is tested elsewhere
    path = tm.ensure(sage, snapshot_download=fake_snapshot, get_remote_sha=lambda r: None)
    assert path == sage.local_dir and calls == [(sage.repo, sage.revision)]
    assert tm.state(sage) == tm.STATE_READY
    eng = tm.cleanup_engine_for("ru")
    assert eng is not None and eng.tag.endswith(sage.revision) and tm.cleanup_engine_for("en") is None
    with pytest.raises(ValueError):
        tm.ensure(tm.get("stress-ru"))


def test_build_plan_keeps_only_integrated_neural_steps():
    plan = tm.build_plan({"numbers", "layout"}, {"spellfix", "stress", "translate"})
    assert plan.rules.steps == {"numbers", "layout"} and plan.neural == {"spellfix"} and plan.engine_factory is not None
    assert plan.enabled and plan.spells_out_numbers
    off = tm.build_plan(set(), set())
    assert not off.enabled and off.engine_factory is None


# ----------------------------------------------------------------------------- narration wiring

def test_preparation_is_part_of_narration_and_is_saved_for_debugging(tmp_path):
    from tests.test_narration import FakeEngine, FakeFfmpeg
    book = Book("Книга", "Автор", "ru", [Chapter("ГЛАВА I", "В 1999 г. было 5 яблок[1]. Мы шли по дорге.")])
    cleaner = FakeCleaner({"В тысяча девятьсот девяносто девятом году было пять яблок. Мы шли по дорге.":
                           "В тысяча девятьсот девяносто девятом году было пять яблок. Мы шли по дороге."})
    plan = book_prep.PrepPlan(PrepOptions(), frozenset({"spellfix"}), lambda lang: cleaner if lang == "ru" else None)
    eng, ff, events = FakeEngine(), FakeFfmpeg(), []
    opts = nr.NarrationOptions(prep=plan, speak_titles=True)
    res = nr.narrate_book(book, lambda: eng, eng.tag, tmp_path / "out", language="russian", options=opts,
                          progress=events.append, ffmpeg="ffmpeg", run=ff)
    spoken = " ".join(eng.calls)
    assert "Глава первая" in spoken and "пять яблок" in spoken and not any(ch.isdigit() for ch in spoken)
    assert "дороге" in spoken and "[1]" not in spoken and cleaner.closed
    debug = res.out_dir / ".debug"
    text = (debug / "prepared_text.txt").read_text(encoding="utf-8")
    assert "=== [1] Глава первая" in text and "дороге" in text
    report = json.loads((debug / "prep_report.json").read_text(encoding="utf-8"))
    assert report["language"] == "ru" and report["neural"][0]["model"] == "fake-cleaner@1" and report["neural"][0]["typos"] == 1
    assert any(e.phase == "prepare" for e in events)
    assert "title=ГЛАВА I" in ff.meta_texts[-1]                       # exported chapter titles keep the original spelling


def test_prep_plan_off_changes_nothing(tmp_path):
    from tests.test_narration import FakeEngine, FakeFfmpeg
    book = Book("B", "", "en", [Chapter("One", "Mr. Smith had 5 apples.")])
    eng = FakeEngine()
    nr.narrate_book(book, lambda: eng, eng.tag, tmp_path / "o1", language="english", ffmpeg="ffmpeg", run=FakeFfmpeg())
    assert "Mr. Smith had 5 apples." in " ".join(eng.calls)
    eng2 = FakeEngine()
    plan = book_prep.PrepPlan(PrepOptions())
    res = nr.narrate_book(book, lambda: eng2, eng2.tag, tmp_path / "o2", language="english", ffmpeg="ffmpeg", run=FakeFfmpeg(),
                          options=nr.NarrationOptions(prep=plan))
    assert "Mister Smith had five apples." in " ".join(eng2.calls) and (res.out_dir / ".debug" / "prepared_text.txt").exists()


def test_a_missing_model_skips_the_neural_step_with_a_note(tmp_path):
    book = Book("B", "", "ru", [Chapter("", "Текст про дорогу и лес.")])
    plan = book_prep.PrepPlan(PrepOptions.none(), frozenset({"spellfix"}), lambda lang: None)
    _, rep = book_prep.run_preparation(book, plan, debug_dir=tmp_path / "d")
    assert "spellfix" in rep["neural_skipped"] and rep["neural"] == []
