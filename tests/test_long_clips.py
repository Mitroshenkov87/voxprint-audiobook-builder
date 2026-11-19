"""Longer training clips (core/slicer.long_clip_config): a share of 12-20 s clips from whole sentences with commas, capped
by the machine's training memory; the classic 12 s cut stays available and unchanged."""
from __future__ import annotations

import json

import pytest

from core import slicer
from core.slicer import SliceConfig, long_clip_config, slice_words
from core.text_utils import attach_spans
from core.types import WordTiming
from infra.vram_optimizer import GpuInfo, safe_max_clip_seconds
from tests.synth import make_text, truth_timeline
from tests.test_ui import app  # noqa: F401


def _comma_book(clauses_per_sentence=(4, 3, 2, 4, 1, 3, 4, 2)):
    """Sentences of 1-4 clauses (8 words, 4 s each) joined by commas with 0.4 s pauses; 0.7 s between sentences."""
    words, toks, t = [], [], 0.0
    for si, nc in enumerate(clauses_per_sentence):
        for ci in range(nc):
            for wi in range(8):
                tok = f"слово{chr(1072 + (si * 7 + ci * 3 + wi) % 30)}"
                last_clause, last_word = ci == nc - 1, wi == 7
                toks.append(tok + ("." if last_clause and last_word else "," if last_word else ""))
                words.append(WordTiming(tok, t, t + 0.4))
                t += 0.5
            t += 0.7 if ci == nc - 1 else 0.4
    text = " ".join(toks)
    attach_spans(words, text)
    return words, text, t + 0.5


def test_classic_cut_is_unchanged_at_12_seconds():
    assert long_clip_config(12) == SliceConfig() and long_clip_config(5) == SliceConfig()
    assert SliceConfig().long_slope == 0.08 and SliceConfig().max_dur == 12.0
    text = make_text(60, seed=3)
    tl = truth_timeline(text)
    ws = [WordTiming(w, a, b) for w, a, b in tl]
    attach_spans(ws, text)
    total = tl[-1][2] + 0.8
    a = slice_words(ws, text, total)
    b = slice_words(ws, text, total, long_clip_config(12))
    assert [(s.start, s.end) for s in a.segments] == [(s.start, s.end) for s in b.segments]


def test_long_mode_keeps_whole_comma_sentences_up_to_the_limit():
    ws, text, total = _comma_book()
    classic = slice_words(ws, text, total)
    assert classic.segments and all(s.duration <= 12.0 for s in classic.segments)
    for limit in (15, 20):
        res = slice_words(ws, text, total, long_clip_config(limit))
        durs = [s.duration for s in res.segments]
        assert all(3.0 <= d <= limit + 1e-6 for d in durs) and res.dropped == 0
        assert any(d > 12.0 for d in durs)                                 # a share of long clips appears
        long_ones = [s for s in res.segments if s.duration > 12.0]
        assert all(s.text.endswith(".") for s in long_ones)                # only whole sentences get long
    res20 = slice_words(ws, text, total, long_clip_config(20))
    assert sum(s.duration > 15.0 for s in res20.segments) >= 1            # the 17 s four-clause sentences stay whole
    assert sum(s.duration <= 12.0 for s in res20.segments) >= 1            # short ones are not glued together


def test_limit_is_clamped_to_12_20_seconds():
    assert long_clip_config(30).max_dur == 20.0 and long_clip_config(15).max_dur == 15.0
    assert long_clip_config(15).long_slope == slicer.LONG_SLOPE and long_clip_config("x").max_dur == 15.0


def test_memory_cap_by_gpu():
    assert safe_max_clip_seconds(GpuInfo(True, "RTX", 16.0, 15.0)) == 20.0
    assert safe_max_clip_seconds(GpuInfo(True, "RTX", 8.0, 7.5)) == 15.0           # 0.6B / 8-bit Adam tier
    assert safe_max_clip_seconds(GpuInfo(False)) == 15.0
    assert safe_max_clip_seconds(GpuInfo(True, "RTX", 16.0, 15.0), force_cpu=True) == 15.0


def test_runner_caps_the_requested_length(monkeypatch):
    from infra import vram_optimizer
    from workers import pipeline_runner as pr

    big, small = GpuInfo(True, "a", 16.0, 15.0), GpuInfo(True, "b", 8.0, 7.0)
    assert pr._slice_config(pr.TaskRequest(kind=pr.KIND_LORA), big).max_dur == 15.0          # default
    assert pr._slice_config(pr.TaskRequest(kind=pr.KIND_LORA, max_clip_s=20), big).max_dur == 20.0
    assert pr._slice_config(pr.TaskRequest(kind=pr.KIND_LORA, max_clip_s=20), small).max_dur == 15.0
    monkeypatch.setattr(vram_optimizer, "detect_gpu", lambda: pytest.fail("no GPU probe for the classic cut"))
    assert pr._slice_config(pr.TaskRequest(kind=pr.KIND_LORA, max_clip_s=12)) == SliceConfig()


def test_dataset_report_records_the_limit(tmp_path):
    from core.dataset_builder import BuildConfig, DatasetBuilder
    from tests.synth import TrueRateAligner
    from tests.test_dataset_builder import _write_inputs

    wav, txt, *_ = _write_inputs(tmp_path, n_sent=30)
    DatasetBuilder(TrueRateAligner(), BuildConfig(slice=long_clip_config(18))).run(wav, txt, tmp_path / "o")
    report = json.loads((tmp_path / "o" / "report.json").read_text(encoding="utf-8"))
    assert report["max_clip_seconds"] == 18.0 and "segments_over_12s" in report
    assert all(t["end"] - t["start"] <= 18.0 + 1e-3 for t in report["segment_times"])


def test_train_window_passes_the_longest_clip(app):
    from tests.test_ui import make_window

    w = make_window(lambda *a: None)
    assert w._task_extras()["max_clip_s"] == 15.0
    w.sp_clipmax.setValue(20)
    assert w._task_extras()["max_clip_s"] == 20.0
    w.sp_clipmax.setValue(30)
    assert w.sp_clipmax.value() == 20
    w.reload_auto_steps()
    assert w.sp_clipmax.value() == 15
    w.set_no_transcript(True)
    assert w.clipmax_row.isHidden()
    w.close()
