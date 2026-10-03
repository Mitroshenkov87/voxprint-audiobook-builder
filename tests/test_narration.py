"""Narration core with a fake TTS engine and a fake ffmpeg: chunk caching/resume, pause/cancel, ETA, chapter assembly,
export commands, chapter metadata, file naming and the AAC switch.  Nothing here needs a GPU, a model or ffmpeg."""
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core import audiobook_export as ex
from core import narration as nr
from core.book_parsers import Book, Chapter
from core.errors import CancelledByUser, NarrationError
from core.events import CancelToken

SR = 24000


@pytest.fixture(autouse=True)
def _english():
    """File names and tags contain localized words ("Chapter 3"); the suite default language is Russian."""
    from core import i18n
    i18n.set_language("en")


class FakeEngine:
    """Silent-ish 'speech': 10 ms of a tone per character; fails from the n-th call on."""
    sample_rate = SR
    tag = "fake-voice-1"

    def __init__(self, fail_from=None, delay=0.0):
        self.calls, self.fail_from, self.delay, self.closed = [], fail_from, delay, False

    def synthesize(self, text):
        self.calls.append(text)
        if self.fail_from is not None and len(self.calls) >= self.fail_from:
            raise RuntimeError("CUDA exploded")
        time.sleep(self.delay)
        n = int(SR * 0.01 * len(text))
        return (0.1 * np.sin(np.linspace(0, 200, n))).astype(np.float32)

    def close(self):
        self.closed = True


class FakeFfmpeg:
    """Records commands; creates the output file (last argument) like ffmpeg would; knows its encoders."""
    def __init__(self, encoders=("aac", "libmp3lame", "libopus", "flac"), fail_on=None):
        self.cmds, self.meta_texts, self.encoders, self.fail_on = [], [], encoders, fail_on

    def __call__(self, cmd):
        self.cmds.append(cmd)
        if "-encoders" in cmd:
            return 0, "".join(f" A....D {e:<18} codec\n" for e in self.encoders)
        if self.fail_on and self.fail_on in cmd:
            return 1, "boom: something failed in the encoder"
        if "-i" in cmd:
            for i, a in enumerate(cmd):
                if a == "-i" and str(cmd[i + 1]).endswith(".ffmetadata"):
                    self.meta_texts.append(Path(cmd[i + 1]).read_text(encoding="utf-8"))
        Path(cmd[-1]).write_bytes(b"fake")
        return 0, ""


def book3():
    return Book("The Test Book", "Jane Roe", "en", [
        Chapter("Beginning", "First sentence here. Second sentence follows.\n\nNew paragraph in one."),
        Chapter("Middle: part/2", "Middle chapter text goes on and on."),
        Chapter("", "Last one.")])


def run(tmp_path, engine=None, **kw):
    engine = engine or FakeEngine()
    ff = kw.pop("ff", FakeFfmpeg())
    events = kw.pop("events", [])
    res = nr.narrate_book(kw.pop("book", book3()), lambda: engine, kw.pop("tag", engine.tag), tmp_path / "out",
                          language=kw.pop("language", "english"), narrator="Anna", progress=events.append,
                          ffmpeg="ffmpeg", run=ff, **kw)
    return res, engine, ff, events


# ----------------------------------------------------------------------------- defaults and the main flow

def test_default_is_a_single_opus_with_chapters_and_aac_is_not_default():
    assert ex.DEFAULT_FORMATS == (ex.FORMAT_OPUS_SINGLE,)
    assert ex.FORMAT_M4B not in nr.NarrationOptions().formats
    assert ex.ALL_FORMATS[:2] == (ex.FORMAT_OPUS_SINGLE, ex.FORMAT_MP3_CHAPTERS)      # order of the UI
    assert ex.ENCODER_FOR_FORMAT[ex.FORMAT_M4B] == "aac" and ex.ENCODER_FOR_FORMAT[ex.FORMAT_WAV_CHAPTERS] is None


def test_end_to_end_default_produces_one_opus_with_chapter_metadata(tmp_path):
    res, engine, ff, events = run(tmp_path)
    assert [f.name for f in res.files] == ["Jane Roe - The Test Book.opus"]
    assert res.chapters == 3 and res.chunks == len(engine.calls) and res.seconds > 3
    cmd = ff.cmds[-1]
    assert "libopus" in cmd and "32k" in cmd and cmd[cmd.index("-f", cmd.index("-map_chapters")) + 1] == "opus"
    meta = ff.meta_texts[-1]
    assert meta.splitlines()[0] == ";FFMETADATA1" and meta.count("[CHAPTER]") == 3
    assert "title=Beginning" in meta and "title=Middle: part/2" in meta and "title=Chapter 3" in meta   # untitled -> numbered
    assert engine.closed and events[-1].phase == "done" and events[-1].fraction == 1.0
    # the working files are gone, the cache too (default), the result is in the book folder
    job = tmp_path / "out" / "The Test Book"
    assert sorted(p.name for p in job.iterdir()) == ["Jane Roe - The Test Book.opus"]


def test_chapter_titles_are_spoken_when_requested(tmp_path):
    opts = nr.NarrationOptions(speak_titles=True)
    _, engine, _, _ = run(tmp_path, options=opts)
    assert engine.calls[0] == "Beginning." and "Middle: part/2." in engine.calls
    plain = FakeEngine()
    run(tmp_path / "b", engine=plain, options=nr.NarrationOptions(speak_titles=False))
    assert "Beginning." not in plain.calls


def test_all_formats_and_layout(tmp_path):
    opts = nr.NarrationOptions(formats=set(ex.ALL_FORMATS), keep_cache=True)
    res, _, ff, _ = run(tmp_path, options=opts)
    job = tmp_path / "out" / "The Test Book"
    names = sorted(p.relative_to(job).as_posix() for p in res.files)
    assert names == sorted([
        "Jane Roe - The Test Book.opus", "Jane Roe - The Test Book.m4b", "Jane Roe - The Test Book (Opus).m4b",
        "Jane Roe - The Test Book.mp3",
        *[f"Jane Roe - The Test Book - MP3/{n}" for n in ("01 - Beginning.mp3", "02 - Middle part 2.mp3",
                                                       "03 - Chapter 3.mp3", "Jane Roe - The Test Book.m3u8")],
        *[f"Jane Roe - The Test Book - Opus/{n}" for n in ("01 - Beginning.opus", "02 - Middle part 2.opus",
                                                        "03 - Chapter 3.opus", "Jane Roe - The Test Book.m3u8")],
        *[f"Jane Roe - The Test Book - FLAC/{n}" for n in ("01 - Beginning.flac", "02 - Middle part 2.flac",
                                                        "03 - Chapter 3.flac", "Jane Roe - The Test Book.m3u8")],
        *[f"Jane Roe - The Test Book - WAV/{n}" for n in ("01 - Beginning.wav", "02 - Middle part 2.wav",
                                                       "03 - Chapter 3.wav", "Jane Roe - The Test Book.m3u8")]])
    assert (job / ".cache").exists()                                    # keep_cache
    # the WAV archive is a real copy of the chapter audio, not an ffmpeg product
    wav = job / "Jane Roe - The Test Book - WAV" / "01 - Beginning.wav"
    assert sf.info(str(wav)).samplerate == SR
    playlist = (job / "Jane Roe - The Test Book - MP3" / "Jane Roe - The Test Book.m3u8").read_text(encoding="utf-8")
    assert playlist.startswith("#EXTM3U\n#EXTINF:") and "01 - Beginning.mp3\n" in playlist and ",Beginning\n" in playlist


# ----------------------------------------------------------------------------- resume / pause / cancel / ETA

def test_resume_after_a_failure_reuses_finished_chunks(tmp_path):
    flaky = FakeEngine(fail_from=4)
    with pytest.raises(NarrationError) as ei:
        run(tmp_path, engine=flaky)
    assert "4" in ei.value.user_message and flaky.closed
    job = tmp_path / "out" / "The Test Book"
    assert len(list((job / ".cache").glob("*.flac"))) == 3 and not list((job / ".cache").glob("*.part.flac"))
    again = FakeEngine()
    res, _, _, events = run(tmp_path, engine=again)
    total = res.chunks
    assert res.resumed_chunks == 3 and len(again.calls) == total - 3
    assert any(e.message and "3" in e.message for e in events[:1])        # "continuing: 3 fragments were finished"


def test_fully_cached_job_never_loads_the_model(tmp_path):
    first = FakeEngine()
    run(tmp_path, engine=first, options=nr.NarrationOptions(keep_cache=True, formats={ex.FORMAT_WAV_CHAPTERS}))

    def boom():
        raise AssertionError("engine must not be created")
    res = nr.narrate_book(book3(), boom, first.tag, tmp_path / "out", language="english", ffmpeg=None,
                          options=nr.NarrationOptions(formats={ex.FORMAT_WAV_CHAPTERS}))
    assert res.resumed_chunks == res.chunks and len(res.files) == 4        # 3 WAV + playlist; no ffmpeg needed for WAV


def test_changed_voice_or_text_invalidates_the_cache(tmp_path):
    run(tmp_path, options=nr.NarrationOptions(keep_cache=True))
    other = FakeEngine()
    res, _, _, _ = run(tmp_path, engine=other, tag="another-voice")
    assert res.resumed_chunks == 0 and len(other.calls) == res.chunks


def test_cancel_stops_after_the_current_chunk_and_can_resume(tmp_path):
    cancel = CancelToken()
    seen = []

    def progress(p):
        seen.append(p)
        if p.done == 2:
            cancel.cancel()
    engine = FakeEngine()
    with pytest.raises(CancelledByUser):
        nr.narrate_book(book3(), lambda: engine, engine.tag, tmp_path / "out", language="english", ffmpeg="ffmpeg",
                        run=FakeFfmpeg(), progress=progress, cancel=cancel)
    assert len(engine.calls) == 2 and engine.closed
    again = FakeEngine()
    res, _, _, _ = run(tmp_path, engine=again)
    assert res.resumed_chunks == 2


def test_pause_blocks_until_resume(tmp_path):
    pause = nr.PauseToken()
    stamps = {}

    def progress(p):
        if p.phase == "synth" and p.done == 2 and "t" not in stamps:
            stamps["t"] = time.monotonic()
            pause.pause()
            threading.Timer(0.4, pause.resume).start()
    engine = FakeEngine()
    nr.narrate_book(book3(), lambda: engine, engine.tag, tmp_path / "out", language="english", ffmpeg="ffmpeg",
                    run=FakeFfmpeg(), progress=progress, pause=pause)
    assert time.monotonic() - stamps["t"] >= 0.35 and not pause.paused


def test_pause_wakes_up_on_cancel():
    pause, cancel = nr.PauseToken(), CancelToken()
    pause.pause()
    threading.Timer(0.2, cancel.cancel).start()
    with pytest.raises(CancelledByUser):
        pause.wait(cancel, poll=0.05)


def test_eta_appears_after_the_first_chunk_and_reaches_zero(tmp_path):
    _, engine, _, events = run(tmp_path, engine=FakeEngine(delay=0.02))
    synth = [e for e in events if e.phase == "synth"]
    assert synth[0].eta is None
    assert all(e.eta is not None for e in synth if e.done >= 1)
    assert synth[-1].eta == 0.0 and synth[-1].done == synth[-1].total
    fractions = [e.fraction for e in events]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0


def test_engine_errors_empty_audio_and_retry(tmp_path):
    class Silent(FakeEngine):
        def synthesize(self, text):
            return np.zeros(0, dtype=np.float32)
    with pytest.raises(NarrationError):
        run(tmp_path, engine=Silent())

    class Hiccup(FakeEngine):
        def synthesize(self, text):
            self.calls.append(text)
            if len(self.calls) == 1:
                raise RuntimeError("transient")
            return super().synthesize(text) if False else np.ones(2400, dtype=np.float32) * 0.1
    res, e, _, _ = run(tmp_path / "r", engine=Hiccup())
    assert len(e.calls) == res.chunks + 1                                  # exactly one retry


def test_format_eta():
    from workers.narration_runner import format_eta
    assert format_eta(None) == "" and format_eta(-1) == "" and format_eta(0) == "1 s" and format_eta(40) == "40 s"
    assert format_eta(300) == "5 min" and format_eta(3900) == "1 h 05 min" and format_eta(7260) == "2 h 01 min"


# ----------------------------------------------------------------------------- assembly

def test_chapter_audio_has_exact_pauses_and_tail(tmp_path):
    from core.chunker import Chunk
    book = Book("B", chapters=[Chapter("One", "x")])
    chunks = [Chunk(0, 0, "aaaa", 300), Chunk(1, 0, "bbbbbb", 750)]
    cache = nr.ChunkCache(tmp_path / "c")
    texts = {0: "aaaa", 1: "bbbbbb"}
    for c in chunks:
        cache.save(cache.key("t", texts[c.index]), np.full(SR * c.index + SR // 2, 0.1, np.float32), SR)
    audio = nr.assemble_chapters(book, chunks, "t", cache, texts, tmp_path / "w")
    expected = (SR // 2 + SR * 0.3 + (SR + SR // 2) + SR * nr.CHAPTER_TAIL_MS / 1000) / SR
    assert len(audio) == 1 and audio[0].duration == pytest.approx(expected, abs=1e-6)
    assert sf.info(str(audio[0].wav)).duration == pytest.approx(expected, abs=1e-3)


def test_missing_cache_entry_is_an_error(tmp_path):
    from core.chunker import Chunk
    with pytest.raises(NarrationError):
        nr.assemble_chapters(Book("B", chapters=[Chapter("One", "x")]), [Chunk(0, 0, "a", 100)], "t",
                             nr.ChunkCache(tmp_path / "c"), {0: "a"}, tmp_path / "w")


def test_cache_atomic_and_corrupt_files(tmp_path):
    cache = nr.ChunkCache(tmp_path / "c")
    key = cache.key("t", "hello")
    assert not cache.has(key) and cache.load(key) is None
    cache.save(key, np.full(100, 0.2, np.float32), SR)
    data, sr = cache.load(key)
    assert cache.has(key) and sr == SR and len(data) == 100 and not list(cache.dir.glob("*.part.flac"))
    cache.path(key).write_bytes(b"corrupt")
    assert cache.load(key) is None and not cache.has(key)                  # removed so it is synthesized again
    assert cache.key("t", "hello") != cache.key("u", "hello") != cache.key("t", "hello!")


# ----------------------------------------------------------------------------- text preparation

def test_preprocessor_hook_and_russian_normalizer():
    opts = nr.NarrationOptions(preprocessors=[lambda t: t.replace("foo", "bar"), str.upper])
    assert nr.prepare_text("foo baz", opts) == "BAR BAZ"
    ru = nr.default_normalizer("Russian")
    assert ru is not None and nr.default_normalizer("english") is None
    spoken = nr.prepare_text("У меня 3 яблока.", nr.NarrationOptions(), ru)
    assert "3" not in spoken and "яблока" in spoken


# ----------------------------------------------------------------------------- file names and chapter metadata

def test_safe_filename_and_chapter_names():
    assert ex.safe_filename('A<b>:c/d\\e|f?g*"h') == "Ab c d e fgh"
    assert ex.safe_filename("  ..Title.. ") == "Title" and ex.safe_filename("") == "untitled"
    assert ex.safe_filename("CON") == "_CON" and ex.safe_filename("nul.txt") == "_nul.txt"
    assert len(ex.safe_filename("x" * 300)) == 80
    assert ex.chapter_filename(0, 9, "Intro", "mp3") == "01 - Intro.mp3"
    assert ex.chapter_filename(9, 120, "Ten", ".opus") == "010 - Ten.opus"
    assert ex.chapter_filename(2, 5, "", "mp3") == "03 - Chapter 3.mp3"
    assert ex.chapter_filename(0, 3, "What?!", "flac") == "01 - What!.flac"
    meta = ex.BookMeta("T: sub", "A/B")
    assert ex.book_basename(meta) == "A B - T sub" and ex.book_basename(ex.BookMeta("Only")) == "Only"


def chapters_of(*durs):
    return [ex.ChapterAudio(i, f"Ch {i + 1}", Path(f"/w/{i}.wav"), d) for i, d in enumerate(durs)]


def test_chapter_timings_are_contiguous_and_in_milliseconds():
    plain = ex.chapter_timings(chapters_of(5.0, 7.5, 2.25))
    assert plain == [(0, 5000, "Ch 1"), (5000, 12500, "Ch 2"), (12500, 14750, "Ch 3")]
    tiny = ex.chapter_timings(chapters_of(10.0, 0.0004, 3.2504))
    assert tiny[1][1] > tiny[1][0]                                       # a chapter is never zero-length
    assert tiny[0] == (0, 10000, "Ch 1") and tiny[2][1] >= 13250


def test_ffmetadata_document_and_escaping():
    meta = ex.BookMeta("Tom; Jerry = #1", "A. Author", "Anna", "ru")
    doc = ex.build_ffmetadata(meta, [ex.ChapterAudio(0, "Part=1\nline", Path("a.wav"), 1.5),
                                     ex.ChapterAudio(1, "Back\\slash", Path("b.wav"), 2.0)])
    lines = doc.splitlines()
    assert lines[0] == ";FFMETADATA1" and "title=Tom\\; Jerry \\= \\#1" in lines and "artist=A. Author" in lines
    assert "composer=Anna" in lines and "genre=Audiobook" in lines and "language=ru" in lines
    assert "comment=Narrated by the AI voice “Anna” (Voxprint AI Audiobook Builder)" in lines
    assert doc.count("[CHAPTER]") == 2 and "TIMEBASE=1/1000" in lines
    assert "START=0" in lines and "END=1500" in lines and "START=1500" in lines and "END=3500" in lines
    assert "title=Part\\=1\\\nline" in doc and "title=Back\\\\slash" in doc
    assert doc.endswith("\n")


def test_artist_falls_back_to_narrator():
    tags = ex.common_tags(ex.BookMeta("T", "", "Voice"))
    assert tags["artist"] == "Voice" and tags["album"] == "T" and "composer" in tags
    assert "artist" not in ex.common_tags(ex.BookMeta("T"))


def test_m3u8_and_concat_list():
    assert ex.build_m3u8([("01 - A.mp3", 12.4, "A"), ("02 - B.mp3", 3600.6, "B\nC")]) == (
        "#EXTM3U\n#EXTINF:12,A\n01 - A.mp3\n#EXTINF:3601,B C\n02 - B.mp3\n")
    assert ex.concat_list([Path("/a b/it's.wav")]) == "file '/a b/it'\\''s.wav'\n"


# ----------------------------------------------------------------------------- ffmpeg command lines

def test_cmd_single_m4b_has_aac_chapters_and_cover():
    rates = ex.Bitrates()
    meta = ex.BookMeta("T", "A", "V", cover=Path("/w/cover.jpg"))
    c = ex.cmd_single("ff", ex.FORMAT_M4B, Path("/w/l.txt"), Path("/w/m.ffmetadata"), meta, Path("/o/T.m4b"), rates)
    s = " ".join(c).replace("\\", "/")      # Windows paths use backslashes
    assert c[0] == "ff" and "-f concat -safe 0 -i /w/l.txt -i /w/m.ffmetadata -i /w/cover.jpg" in s
    assert "-map 0:a -map 2:v -map_metadata 1 -map_chapters 1" in s
    assert "-c:a aac -b:a 64k -ac 1" in s and "-disposition:v:0 attached_pic" in s and s.endswith("-f ipod /o/T.m4b")
    nocover = " ".join(ex.cmd_single("ff", ex.FORMAT_M4B, Path("l"), Path("m"), ex.BookMeta("T"), Path("o.m4b"), rates))
    assert "-map 2:v" not in nocover and "attached_pic" not in nocover and "libmp3lame" not in nocover


def test_cmd_single_opus_mp3_and_m4b_opus():
    r = ex.Bitrates(opus_kbps=24, mp3_kbps=128)
    opus = " ".join(ex.cmd_single("ff", ex.FORMAT_OPUS_SINGLE, Path("l"), Path("m"), ex.BookMeta("T", cover=Path("c.jpg")), Path("o.opus"), r))
    assert "-c:a libopus -b:a 24k" in opus and "-f opus" in opus and "-map_chapters 1" in opus and "c.jpg" not in opus
    mp3 = " ".join(ex.cmd_single("ff", ex.FORMAT_MP3_SINGLE, Path("l"), Path("m"), ex.BookMeta("T", cover=Path("c.jpg")), Path("o.mp3"), r))
    assert "-c:a libmp3lame -b:a 128k" in mp3 and "-id3v2_version 3" in mp3 and "-map_chapters 1" in mp3 and "c.jpg" in mp3
    mo = " ".join(ex.cmd_single("ff", ex.FORMAT_M4B_OPUS, Path("l"), Path("m"), ex.BookMeta("T"), Path("o.m4b"), r))
    assert "libopus" in mo and "-f mp4" in mo and "aac" not in mo
    with pytest.raises(ValueError):
        ex.cmd_single("ff", "nope", Path("l"), Path("m"), ex.BookMeta("T"), Path("o"), r)


def test_cmd_chapter_tags_and_cover():
    meta = ex.BookMeta("Book", "Auth", "Voice", "en", cover=Path("/w/cover.png"))
    ch = ex.ChapterAudio(1, "Second", Path("/w/chapter_0002.wav"), 5)
    s = " ".join(ex.cmd_chapter("ff", ex.FORMAT_MP3_CHAPTERS, ch, 12, meta, Path("/o/02 - Second.mp3"), ex.Bitrates()))
    assert "-i /w/chapter_0002.wav -i /w/cover.png -map 0:a -map 1:v -c:v copy" in s
    assert "-c:a libmp3lame -b:a 96k -ac 1 -id3v2_version 3" in s
    for tag in ("title=Second", "album=Book", "artist=Auth", "track=2/12", "composer=Voice"):
        assert f"-metadata {tag}" in s
    flac = " ".join(ex.cmd_chapter("ff", ex.FORMAT_FLAC_CHAPTERS, ch, 12, meta, Path("o.flac"), ex.Bitrates()))
    assert "-c:a flac" in flac and "cover" not in flac
    opus = " ".join(ex.cmd_chapter("ff", ex.FORMAT_OPUS_CHAPTERS, ch, 12, meta, Path("o.opus"), ex.Bitrates()))
    assert "libopus -b:a 32k" in opus and "-f opus" in opus
    with pytest.raises(ValueError):
        ex.cmd_chapter("ff", ex.FORMAT_M4B, ch, 1, meta, Path("o"), ex.Bitrates())


def test_bitrates_are_clamped():
    c = ex.Bitrates(aac_kbps=9999, mp3_kbps=1, opus_kbps=-5).clamp()
    assert (c.aac_kbps, c.mp3_kbps, c.opus_kbps) == (256, 32, 12)
    assert ex.Bitrates().clamp() == ex.Bitrates()


# ----------------------------------------------------------------------------- encoders, failures, the AAC switch

def test_missing_encoders_are_detected_before_synthesis(tmp_path):
    ff = FakeFfmpeg(encoders=("flac", "aac"))
    assert ex.missing_encoders("ff", [ex.FORMAT_OPUS_SINGLE, ex.FORMAT_M4B, ex.FORMAT_WAV_CHAPTERS], ff) == ["libopus"]
    assert ex.missing_encoders("ff", [ex.FORMAT_WAV_CHAPTERS], lambda c: pytest.fail("no ffmpeg needed")) == []
    engine = FakeEngine()
    with pytest.raises(NarrationError) as ei:
        run(tmp_path, engine=engine, ff=ff)
    assert "libopus" in ei.value.user_message and engine.calls == []          # failed fast, nothing was synthesized
    assert ex.missing_encoders("ff", [ex.FORMAT_M4B], lambda c: (1, "")) == ["aac"]


def test_ffmpeg_not_found_and_ffmpeg_failure(tmp_path):
    with pytest.raises(NarrationError):
        nr.narrate_book(book3(), FakeEngine, "t", tmp_path / "o", ffmpeg=None)
    ff = FakeFfmpeg(fail_on="libopus")
    ff.cmds = []
    # the failure is reported as NarrationError with the stderr tail as details; the cache survives for a retry
    class Ok(FakeFfmpeg):
        def __call__(self, cmd):
            if "-encoders" in cmd:
                return super().__call__(cmd)
            return 1, "Error initializing the muxer\nreal reason here"
    with pytest.raises(NarrationError) as ei:
        run(tmp_path, ff=Ok(), options=nr.NarrationOptions(keep_cache=False))
    assert "real reason here" in ei.value.details and ".opus" in ei.value.user_message
    assert (tmp_path / "out" / "The Test Book" / ".cache").exists()


def test_aac_can_be_switched_off(tmp_path, monkeypatch):
    from infra import features
    opts = nr.NarrationOptions(formats={ex.FORMAT_M4B}, allow_aac=False)
    engine = FakeEngine()
    with pytest.raises(NarrationError) as ei:
        run(tmp_path, engine=engine, options=opts)
    assert engine.calls == [] and ei.value.user_message
    ok, _, ff, _ = run(tmp_path / "ok", options=nr.NarrationOptions(formats={ex.FORMAT_M4B}, allow_aac=True))
    assert ok.files[0].suffix == ".m4b" and "aac" in ff.cmds[-1]
    # the feature flag: default on, environment and state file switch it off
    monkeypatch.delenv(features.ENV_AAC, raising=False)
    assert features.aac_enabled() is True
    (features.paths.state_dir() / "features.json").write_text('{"aac_m4b": false}', encoding="utf-8")
    assert features.aac_enabled() is False
    monkeypatch.setenv(features.ENV_AAC, "1")
    assert features.aac_enabled() is True                                      # environment wins
    monkeypatch.setenv(features.ENV_AAC, "0")
    (features.paths.state_dir() / "features.json").write_text('{"aac_m4b": true}', encoding="utf-8")
    assert features.aac_enabled() is False
    (features.paths.state_dir() / "features.json").write_text("garbage", encoding="utf-8")
    monkeypatch.delenv(features.ENV_AAC)
    assert features.aac_enabled() is features.AAC_DEFAULT


def test_run_narration_wires_flag_and_engine(tmp_path, monkeypatch):
    from core.voice_library import VoiceRecord
    from workers import narration_runner as rn
    seen = {}

    def fake_narrate(book, factory, tag, out_dir, **kw):
        seen.update(tag=tag, allow_aac=kw["options"].allow_aac, narrator=kw["narrator"], language=kw["language"])
        return nr.NarrationResult(Path(out_dir))
    monkeypatch.setattr(rn.nr, "narrate_book", fake_narrate)
    monkeypatch.setattr(rn, "ensure_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setenv("VOXPRINT_ENABLE_AAC", "0")
    voice = VoiceRecord("v1", tmp_path / "v1", {"name": "Anna", "language": "russian"})
    rn.run_narration(rn.NarrationJob(book3(), voice, tmp_path), lambda p: None, CancelToken(), nr.PauseToken())
    assert seen["allow_aac"] is False and seen["narrator"] == "Anna" and seen["language"] == "russian" and seen["tag"]


def test_engine_tag_changes_with_the_adapter(tmp_path):
    from core import tts_engine
    from core.voice_library import VoiceRecord
    d = tmp_path / "v"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"a" * 10)
    rec = VoiceRecord("v", d, {"base_model": "Qwen/B"})
    t1 = tts_engine.engine_tag(rec)
    (d / "adapter_model.safetensors").write_bytes(b"a" * 11)
    assert tts_engine.engine_tag(rec) != t1 and tts_engine.engine_tag(VoiceRecord("w", d, {"base_model": "Qwen/B"})) != tts_engine.engine_tag(rec)


def test_max_tokens_is_bounded_by_the_text_length():
    """A one-word chapter title must not be allowed to run to the library default of 2048 frames (found on the GPU)."""
    from core import tts_engine as te

    assert te.max_tokens_for("Знакомство") < 100
    assert te.max_tokens_for("") == te.MIN_TOKENS
    assert te.max_tokens_for("а" * 260) == round((3 + 0.19 * 260) * 12.5)
    assert te.max_tokens_for("а" * 100000) == te.MAX_TOKENS
    assert te.max_tokens_for("a" * 20) < te.max_tokens_for("a" * 200)
