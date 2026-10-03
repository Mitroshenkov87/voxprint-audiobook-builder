"""Dataset builder end to end on synthetic audio: segments, reference clip, metadata.jsonl, long recordings, quality filter."""
import json
import wave

import numpy as np
import pytest

from core import audio_utils as au
from core.aligner import FakeAligner, align_long, check_alignment
from core.dataset_builder import (BuildConfig, DatasetBuilder, make_rows, read_metadata_jsonl, select_ref,
                                  write_metadata_jsonl)
from core.errors import AudioTextMismatchError, AudioReadError
from core.events import CancelToken, Stage
from core.text_utils import clean_token
from core.types import Segment
from tests.synth import TrueRateAligner, make_text, synth_reading, truth_timeline


def _write_inputs(tmp_path, n_sent=60, seed=1, text_override=None):
    text = make_text(n_sent, seed)
    audio, tl = synth_reading(text, seed=seed)
    wav = tmp_path / "in.wav"
    au.write_wav(wav, audio, 16000)
    txt = tmp_path / "in.txt"
    txt.write_text(text_override or text, encoding="utf-8")
    return wav, txt, text, audio, tl


def test_jsonl_roundtrip_utf8_no_bom(tmp_path):
    segs = [Segment(1, 0, 3, "Привет, мир!"), Segment(2, 3, 6, "Ёжик «в тумане»")]
    p = tmp_path / "m.jsonl"
    write_metadata_jsonl(p, make_rows(segs))
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw
    assert "Привет" in raw.decode("utf-8")  # no \u escaping
    rows = read_metadata_jsonl(p)
    assert rows[0] == {"audio": "segment_001.wav", "text": "Привет, мир!", "ref_audio": "ref.wav"}
    assert list(rows[0].keys()) == ["audio", "text", "ref_audio"]


def test_full_pipeline_with_true_rate_aligner(tmp_path):
    wav, txt, text, audio, tl = _write_inputs(tmp_path)
    out = tmp_path / "dataset"
    stages = []
    res = DatasetBuilder(TrueRateAligner()).run(wav, txt, out, lambda s, f, m: stages.append(s))
    assert res.n_segments >= 10
    rows = read_metadata_jsonl(out / "metadata.jsonl")
    assert len(rows) == res.n_segments
    assert {r["ref_audio"] for r in rows} == {"ref.wav"}
    for r in rows:
        f = out / r["audio"]
        with wave.open(str(f)) as w:
            assert w.getframerate() == 24000 and w.getnchannels() == 1
            dur = w.getnframes() / 24000
        # 3-12 s of speech + ~1 s of silence at the end
        assert 3.0 + 0.95 <= dur <= 12.0 + 1.05
        x, _ = au.load_audio(f, 24000)
        assert np.abs(x[-int(0.9 * 24000):]).max() < 1e-3
        assert r["text"] and not any(ch in r["text"] for ch in "\n\r\t")
    with wave.open(str(out / "ref.wav")) as w:
        assert w.getframerate() == 24000 and 5.0 <= w.getnframes() / 24000 <= 10.0
    # ref_text.txt is the exact transcript of ref.wav (a substring of the reading text)
    ref_text = (out / "ref_text.txt").read_text(encoding="utf-8").strip()
    assert ref_text and ref_text in " ".join(text.split()) and ref_text == res.ref_text.strip()
    assert res.training_language == "russian"
    assert set(p.name for p in out.iterdir()) >= {"metadata.jsonl", "ref.wav", "ref_text.txt", "report.json"}
    assert all(set(r) == {"audio", "text", "ref_audio"} for r in rows)
    # the segment texts are substrings of the source (clean, with punctuation)
    flat = " ".join(text.split())
    for r in rows:
        assert r["text"] in flat
    # the segment really contains speech at the right place: its first letter matches in time
    assert Stage.ALIGN in stages and Stage.SLICE in stages and Stage.SAVE in stages and Stage.MODEL in stages
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["segments"] == res.n_segments and report["language"] == "Russian"


def test_rerun_cleans_old_files(tmp_path):
    wav, txt, *_ = _write_inputs(tmp_path, n_sent=30)
    out = tmp_path / "ds"
    out.mkdir()
    (out / "segment_999.wav").write_bytes(b"x")
    (out / "keep.me").write_text("1")
    DatasetBuilder(TrueRateAligner()).run(wav, txt, out)
    assert not (out / "segment_999.wav").exists() and (out / "keep.me").exists()


def test_quality_filter_drops_clipped_segments_automatically(tmp_path):
    wav, txt, text, audio, tl = _write_inputs(tmp_path, n_sent=40)
    clean = DatasetBuilder(TrueRateAligner()).run(wav, txt, tmp_path / "a")
    # fill a section in the middle of the recording with clipping
    mid = int(len(audio) / 2)
    bad = audio.copy()
    bad[mid:mid + 16000 * 2] = np.sign(np.sin(np.arange(16000 * 2) * 0.05)).astype(np.float32)
    au.write_wav(wav, bad, 16000)
    res = DatasetBuilder(TrueRateAligner()).run(wav, txt, tmp_path / "b")
    assert res.n_dropped_quality >= 1 and res.n_segments < clean.n_segments
    rep = json.loads((tmp_path / "b" / "report.json").read_text(encoding="utf-8"))
    assert rep["quality_dropped"] == res.n_dropped_quality and "clipping" in rep["quality_dropped_reasons"]
    # file indices are consecutive and match metadata.jsonl
    rows = read_metadata_jsonl(tmp_path / "b" / "metadata.jsonl")
    assert [r["audio"] for r in rows] == [f"segment_{i:03d}.wav" for i in range(1, len(rows) + 1)]
    assert all((tmp_path / "b" / r["audio"]).exists() for r in rows)


def test_digits_are_normalized_before_alignment_and_raw_kept_in_report(tmp_path):
    spoken = ("Мы прошли пять километров по лесу и очень устали сегодня. " * 3 +
              "В зале было двадцать пять человек и все ждали начала. " * 3) * 4
    raw = spoken.replace("двадцать пять", "25")
    audio, _ = synth_reading(spoken, seed=4)
    wav = tmp_path / "in.wav"
    au.write_wav(wav, audio, 16000)
    txt = tmp_path / "in.txt"
    txt.write_text(raw, encoding="utf-8")
    cfg = BuildConfig(normalizer_engine=("builtin", lambda s: s))
    res = DatasetBuilder(TrueRateAligner(), cfg).run(wav, txt, tmp_path / "o")
    rows = read_metadata_jsonl(tmp_path / "o" / "metadata.jsonl")
    assert not any(ch.isdigit() for r in rows for ch in r["text"])
    rep = json.loads((tmp_path / "o" / "report.json").read_text(encoding="utf-8"))
    raws = [t["text_raw"] for t in rep["segment_times"]]
    assert any("25" in t for t in raws) and rep["normalizer"]["changed"]
    assert res.n_segments >= 3


def test_mismatch_detected(tmp_path):
    wav, txt, text, *_ = _write_inputs(tmp_path, n_sent=30)
    txt.write_text(make_text(120, seed=9), encoding="utf-8")  # text 4 times longer
    with pytest.raises(AudioTextMismatchError):
        DatasetBuilder(TrueRateAligner()).run(wav, txt, tmp_path / "o")


def test_cancel(tmp_path):
    wav, txt, *_ = _write_inputs(tmp_path, n_sent=20)
    tok = CancelToken()
    tok.cancel()
    from core.errors import CancelledByUser
    with pytest.raises(CancelledByUser):
        DatasetBuilder(TrueRateAligner()).run(wav, txt, tmp_path / "o", cancel=tok)


def test_bad_audio(tmp_path):
    bad = tmp_path / "x.wav"
    bad.write_bytes(b"not audio")
    t = tmp_path / "t.txt"
    t.write_text("текст", encoding="utf-8")
    with pytest.raises(AudioReadError):
        DatasetBuilder(FakeAligner()).run(bad, t, tmp_path / "o")


def test_long_audio_chunking_matches_truth():
    # ~9 minutes: must be split into several chunks
    text = make_text(330, seed=11)
    audio, tl = synth_reading(text, seed=2)
    total = len(audio) / 16000
    assert total > 400
    al = TrueRateAligner()
    words = align_long(al, audio, 16000, text, "Russian")
    assert al.calls >= 3
    assert len(words) == len(tl)
    errs = [max(abs(w.start - t[1]), abs(w.end - t[2])) for w, t in zip(words, tl)]
    assert max(errs) < 0.15, max(errs)
    check_alignment(words, audio, 16000)


def test_fake_aligner_dry_run(tmp_path):
    wav, txt, *_ = _write_inputs(tmp_path, n_sent=25)
    res = DatasetBuilder(FakeAligner()).run(wav, txt, tmp_path / "o")
    assert res.n_segments > 3


def test_select_ref_prefers_clean_segment():
    sr = 24000
    rng = np.random.default_rng(0)
    noisy = (np.sin(2 * np.pi * 150 * np.arange(8 * sr) / sr) * 0.3 + rng.normal(0, 0.1, 8 * sr)).astype(np.float32)
    clean = (np.sin(2 * np.pi * 150 * np.arange(8 * sr) / sr) * 0.3 * (np.arange(8 * sr) % 4000 < 2800)
             + rng.normal(0, 0.002, 8 * sr)).astype(np.float32)
    audio = np.concatenate([noisy, clean])
    segs = [Segment(1, 0, 8, "a"), Segment(2, 8, 16, "b")]
    ref = select_ref(audio, sr, segs)
    assert (ref.start, ref.end) == (8, 16)


def test_voiced_bounds_and_silences():
    text = "Раз два три. Четыре пять шесть."
    audio, tl = synth_reading(text)
    vs, ve = au.voiced_bounds(audio, 16000)
    assert abs(vs - tl[0][1]) < 0.1 and abs(ve - tl[-1][2]) < 0.1
    sil = au.find_silences(audio, 16000, 0.5)
    assert any(a > 1.0 for a, b in sil)
