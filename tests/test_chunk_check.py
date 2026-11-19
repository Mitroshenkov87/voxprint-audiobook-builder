"""Per-chunk speech-recognition check during narration (core/chunk_check.py) and its use in narrate_book / the runner / the
Narrate window.  Fake engine and fake recogniser: no GPU, no model."""
from __future__ import annotations

import numpy as np
import soundfile as sf

from core import chunk_check as cc
from core import narration as nr
from core.asr import AsrResult
from tests.test_narration import SR, FakeEngine, FakeFfmpeg, book3, run  # noqa: F401
from tests.test_studio import app, lib, make_studio  # noqa: F401


class MarkASR:
    """'Hears' the text encoded in the audio: chunks are tagged by their first sample (see TaggedEngine)."""

    def __init__(self, heard):
        self.heard, self.loaded, self.unloaded, self.calls = heard, 0, 0, []

    def load(self):
        self.loaded += 1

    def unload(self):
        self.unloaded += 1

    def transcribe(self, audio, sr, language=None):
        tag = int(round(float(audio[0]) * 1000))
        self.calls.append(tag)
        h = self.heard(tag)
        if isinstance(h, Exception):
            raise h
        return AsrResult(h)


class TaggedEngine(FakeEngine):
    """synthesize() -> tag 0; synthesize_sampled(seed) -> tag 1, 2, ... in call order; records the seeds and sampling."""

    def __init__(self):
        super().__init__()
        self.sampled = []

    def _audio(self, text, tag):
        a = np.full(int(SR * 0.01 * max(1, len(text))), 0.0, dtype=np.float32)
        a[0] = tag / 1000
        return a

    def synthesize(self, text):
        self.calls.append(text)
        return self._audio(text, 0)

    def synthesize_sampled(self, text, seed=None, **sampling):
        self.sampled.append((text, seed, sampling))
        return self._audio(text, len(self.sampled))


TEXT = "the quick brown fox jumps"


def test_seed_is_deterministic_per_chunk_and_attempt():
    assert cc.chunk_seed("k", 1) == cc.chunk_seed("k", 1)
    assert len({cc.chunk_seed("k", 1), cc.chunk_seed("k", 2), cc.chunk_seed("j", 1)}) == 3
    assert 0 <= cc.chunk_seed("k", 1) < 2 ** 31


def test_recogniser_goes_to_gpu_only_with_room_next_to_tts():
    assert cc.asr_device(1.5, lambda: 6.0) == "cuda"               # 1.8 + 2 GB reserve fits
    assert cc.asr_device(1.5, lambda: 3.5) == "cpu"
    assert cc.asr_device(1.5, lambda: 0.0) == "cpu"                # no CUDA


def test_good_chunk_is_kept_without_regeneration():
    asr = MarkASR(lambda tag: TEXT)
    ch = cc.ChunkChecker(lambda: asr)
    eng = TaggedEngine()
    a = eng._audio(TEXT, 0)
    assert ch.check(eng, TEXT, "key", 0, a, SR) is a
    assert eng.sampled == [] and ch.stats["checked"] == 1 and ch.stats["regenerated"] == 0
    ch.close()
    assert asr.loaded == 1 and asr.unloaded == 1


def test_bad_chunk_is_regenerated_with_seeded_sampling_until_it_passes():
    heard = {0: "the quick", 1: "the quick brown fax jumps", 2: TEXT}     # 1st: 2 words lost, retry 1: 1 letter (4 %)
    asr = MarkASR(heard.get)
    ch = cc.ChunkChecker(lambda: asr, options=cc.ChunkCheckOptions(max_cer=0.15, retries=2))
    eng = TaggedEngine()
    out = ch.check(eng, TEXT, "key", 3, eng._audio(TEXT, 0), SR)
    assert round(out[0] * 1000) == 1                                       # retry 1 passed: no second retry
    assert [(s, kw) for _, s, kw in eng.sampled] == [(cc.chunk_seed("key", 1), cc.RETRY_SAMPLING)]
    assert cc.RETRY_SAMPLING == {"temperature": 0.8, "top_p": 0.85, "top_k": 30}
    assert ch.stats == {"checked": 1, "regenerated": 1, "fixed": 1, "still_bad": 0, "attempts": 1, "failed": 0}


def test_best_attempt_is_kept_when_none_passes():
    heard = {0: "the quick brown", 1: "a slow dog", 2: "the quick brown fox"}   # retry 2 is the best but still over 15 %
    ch = cc.ChunkChecker(lambda: MarkASR(heard.get), options=cc.ChunkCheckOptions(max_cer=0.15, retries=2))
    eng = TaggedEngine()
    out = ch.check(eng, TEXT, "key", 0, eng._audio(TEXT, 0), SR)
    assert round(out[0] * 1000) == 2
    assert [s for _, s, _ in eng.sampled] == [cc.chunk_seed("key", 1), cc.chunk_seed("key", 2)]
    assert ch.stats["still_bad"] == 1 and ch.stats["attempts"] == 2


def test_retries_are_capped_and_recognition_failure_accepts_the_chunk():
    assert cc.ChunkChecker(lambda: None, options=cc.ChunkCheckOptions(retries=50)).options.retries == cc.MAX_RETRIES
    ch = cc.ChunkChecker(lambda: MarkASR(lambda tag: RuntimeError("asr died")))
    eng = TaggedEngine()
    a = eng._audio(TEXT, 0)
    assert ch.check(eng, TEXT, "key", 0, a, SR) is a and eng.sampled == [] and ch.stats["failed"] == 1


def test_engine_without_sampled_api_falls_back_to_synthesize():
    eng = FakeEngine()
    ch = cc.ChunkChecker(lambda: MarkASR(lambda tag: "nothing alike at all"), options=cc.ChunkCheckOptions(retries=1))
    ch.check(eng, TEXT, "key", 0, np.zeros(100, dtype=np.float32) + 0.0, SR)
    assert eng.calls == [TEXT] and ch.stats["attempts"] == 1


def test_narrate_book_keeps_the_winner_in_the_cache_and_assembles_chapters(tmp_path):
    """The CPU overlap (chapter assembly while synthesis goes on) and the cache key are unchanged; the checker is closed;
    a resumed job does not check cached chunks again."""
    eng = TaggedEngine()
    texts = {}
    asr = MarkASR(lambda tag: "" if tag == 0 else texts["last"])   # every first attempt says nothing -> one good retry each
    orig = eng.synthesize_sampled

    def sampled(text, seed=None, **kw):
        texts["last"] = text
        return orig(text, seed=seed, **kw)

    eng.synthesize_sampled = sampled
    checker = cc.ChunkChecker(lambda: asr, options=cc.ChunkCheckOptions(retries=1))
    opts = nr.NarrationOptions(keep_cache=True)
    res, _, ff, _ = run(tmp_path, engine=eng, options=opts, checker=checker)
    assert res.chapters == 3 and res.chunk_check["checked"] == res.chunks == len(eng.calls)
    assert res.chunk_check["fixed"] == res.chunks and len(eng.sampled) == res.chunks
    assert asr.unloaded == 1 and ff.meta_texts[-1].count("[CHAPTER]") == 3
    cache = nr.ChunkCache(tmp_path / "out" / "The Test Book" / ".cache")
    for text, _, _ in eng.sampled:                       # the cache key is still engine tag + text, holding the retry
        data, _ = sf.read(str(cache.path(cache.key(eng.tag, text))), dtype="float32")
        assert data[0] > 0
    asr2 = MarkASR(lambda tag: "")
    res2, eng2, _, _ = run(tmp_path, engine=TaggedEngine(), options=opts,
                           checker=cc.ChunkChecker(lambda: asr2))
    assert res2.resumed_chunks == res2.chunks and asr2.calls == [] and res2.chunk_check["checked"] == 0


def test_runner_builds_a_checker_only_when_asked(monkeypatch):
    from workers import narration_runner as runner

    made = []
    monkeypatch.setattr(cc, "make_default_checker", lambda lang, opts: made.append((lang, opts.max_cer, opts.retries)) or "C")
    seen = {}
    monkeypatch.setattr(runner.nr, "narrate_book", lambda *a, **kw: seen.setdefault("checker", kw["checker"]))
    monkeypatch.setattr(runner, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(runner.tts_engine, "make_engine_factory", lambda v, lang: None)
    monkeypatch.setattr(runner.tts_engine, "engine_tag", lambda v, lang: "t")

    class Voice:
        language, name = "English", "V"

    def job(check):
        return runner.NarrationJob(book3(), Voice(), "out", nr.NarrationOptions(check_chunks=check, check_retries=3))

    runner.run_narration(job(False), lambda p: None, None, None)
    assert seen.pop("checker") is None and made == []
    runner.run_narration(job(True), lambda p: None, None, None)
    assert seen.pop("checker") == "C" and made == [("English", 0.15, 3)]


def test_default_checker_prefers_the_small_model_and_never_downloads():
    from infra import asr_choice

    assert cc.make_default_checker("English", ready=lambda repo: None) is None
    ch = cc.make_default_checker("Russian", ready=lambda repo: f"/m/{repo}")
    assert ch is not None and ch.language == "Russian" and ch.normalize is not None
    asked = []
    cc.make_default_checker("Auto", ready=lambda repo: asked.append(repo) or (f"/m/{repo}" if repo == asr_choice.LARGE else None))
    assert asked == [asr_choice.SMALL, asr_choice.LARGE]
    assert cc.make_default_checker("Auto", ready=lambda repo: "/m").language is None


def test_high_preset_switches_the_check_on(app, lib):
    from core import i18n

    i18n.set_language("en")
    n = make_studio(lib).narrate_window
    assert not n.chk_check_chunks.isChecked() and not n.options().check_chunks       # Standard is the default
    n.apply_preset("high")
    assert n.chk_check_chunks.isChecked() and n.options().check_chunks
    n.chk_check_chunks.setChecked(False)                                              # the user decides
    assert not n.options().check_chunks
    n.apply_preset("compact")
    assert not n.chk_check_chunks.isChecked()
    assert "speech recognition" in n.chk_check_chunks.text()
