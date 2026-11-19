"""Speech-recognition cross-check of dataset clips (core/clip_check.py) and its use in the dataset builder / runner / Train window."""
from __future__ import annotations

import json

import numpy as np
import pytest

from core import clip_check as cc
from core.asr import AsrResult, FakeASR
from tests.test_ui import app  # noqa: F401


def _clips(n):
    return [np.zeros(2400, dtype=np.float32) for _ in range(n)], [f"фраза номер {i} про кота" for i in range(n)]


class ListASR(FakeASR):
    """Returns ``texts[i]`` for the i-th clip; an Exception instance is raised instead."""

    def __init__(self, texts):
        super().__init__(texts)
        self.loaded = self.unloaded = 0

    def load(self):
        self.loaded += 1

    def unload(self):
        self.unloaded += 1

    def transcribe(self, audio, sr, language=None):
        t = self.texts[self.calls]
        self.calls += 1
        if isinstance(t, Exception):
            raise t
        return AsrResult(t)


def test_threshold_is_clamped_to_10_15_percent_and_zero_means_off():
    assert cc.clamp_threshold(0.12) == 0.12
    assert cc.clamp_threshold(0.3) == 0.15 and cc.clamp_threshold(0.05) == 0.10
    assert cc.clamp_threshold(0) == 0.0 and cc.clamp_threshold(None) == 0.0 and cc.clamp_threshold("x") == 0.0


def test_mismatching_clips_are_dropped_and_the_model_is_freed():
    pieces, texts = _clips(25)
    heard = list(texts)
    heard[3] = "совсем другая фраза"                    # a slip / misalignment
    heard[7] = texts[7] + " и ещё лишние слова"          # repeated / extra words
    heard[9] = texts[9].replace("кота", "кото")          # one letter: 1/18 = 5.6 % < 12 %, kept
    asr = ListASR(heard)
    keep, rep = cc.check_clips(pieces, 24000, texts, asr, "Russian", 0.12, min_keep=20)
    assert [i for i, k in enumerate(keep) if not k] == [3, 7]
    assert rep.dropped == 2 and rep.kept_over == 0 and rep.checked == 25 and rep.failed == 0
    assert rep.cers[9] == pytest.approx(1 / 18, abs=0.01) and asr.loaded == 1 and asr.unloaded == 1
    assert rep.as_dict()["min_keep"] == 20


def test_never_below_the_safe_minimum_drops_the_worst_first():
    pieces, texts = _clips(22)
    heard = list(texts)
    heard[0] = "полностью не то"                         # worst
    heard[1] = texts[1] + " лишнее"                      # milder
    heard[2] = "совсем иное предложение тут"             # bad
    heard[3] = texts[3] + " лишнее слово"                # 0.61
    keep, rep = cc.check_clips(pieces, 24000, texts, ListASR(heard), "Russian", 0.12, min_keep=20)
    assert sum(keep) == 20 and rep.dropped == 2 and rep.kept_over == 2
    assert not keep[0] and not keep[2] and keep[1] and keep[3]
    pieces, texts = _clips(20)
    keep, rep = cc.check_clips(pieces, 24000, texts, ListASR(["???"] * 20), "Russian", 0.12, min_keep=20)
    assert all(keep) and rep.dropped == 0 and rep.kept_over == 20         # small dataset: warn, never shrink


def test_failed_recognition_keeps_the_clip_and_normalization_avoids_false_drops():
    pieces, texts = _clips(3)
    texts[1] = "в две тысячи шестом году"
    heard = [texts[0], "в 2006 году", RuntimeError("cuda error")]
    norm = lambda t: t.replace("2006", "две тысячи шестом")
    keep, rep = cc.check_clips(pieces, 24000, texts, ListASR(heard), "Russian", 0.12, min_keep=0, normalize=norm)
    assert keep == [True, True, True] and rep.failed == 1 and rep.cers[1] == 0.0 and rep.cers[2] is None


# ------------------------------------------------------------------------------------------- dataset builder
def test_dataset_builder_drops_mismatched_clips_and_reports_them(tmp_path):
    from core.dataset_builder import BuildConfig, DatasetBuilder, read_metadata_jsonl
    from tests.synth import TrueRateAligner
    from tests.test_dataset_builder import _write_inputs

    wav, txt, *_ = _write_inputs(tmp_path, n_sent=60)
    plain = DatasetBuilder(TrueRateAligner()).run(wav, txt, tmp_path / "a")
    texts = [r["text"] for r in read_metadata_jsonl(tmp_path / "a" / "metadata.jsonl")]
    assert plain.clip_check is None and len(texts) >= 8
    heard = list(texts)
    heard[1] = "что-то совсем другое"
    heard[4] = "и это тоже не то"

    class Unloading(TrueRateAligner):
        unloaded = 0

        def unload(self):
            Unloading.unloaded += 1

    asr = ListASR(heard)
    res = DatasetBuilder(Unloading(), BuildConfig(clip_max_cer=0.12, clip_min_keep=2), asr=asr).run(wav, txt, tmp_path / "b")
    rows = read_metadata_jsonl(tmp_path / "b" / "metadata.jsonl")
    assert [r["text"] for r in rows] == [t for i, t in enumerate(texts) if i not in (1, 4)]
    assert res.n_segments == len(texts) - 2 and res.clip_check["dropped"] == 2
    assert Unloading.unloaded >= 1 and asr.loaded == 1 and asr.unloaded == 1      # aligner freed before the ASR loads
    assert any("2" in w and str(len(texts)) in w for w in res.warnings)
    report = json.loads((tmp_path / "b" / "report.json").read_text(encoding="utf-8"))
    assert report["clip_check"]["dropped"] == 2 and len(report["clip_check"]["dropped_clips"]) == 2
    assert report["clip_check"]["dropped_clips"][0]["heard"] == "что-то совсем другое"
    assert all(t["cer"] == 0.0 for t in report["segment_times"])
    ref_text = (tmp_path / "b" / "ref_text.txt").read_text(encoding="utf-8").strip()
    assert texts[1] not in ref_text and texts[4] not in ref_text


# ------------------------------------------------------------------------------------------- runner + window
def test_runner_uses_the_check_only_for_full_runs_with_a_text(monkeypatch):
    from infra import model_downloader as md
    from workers import pipeline_runner as pr

    fake = ListASR([])
    on = pr.TaskRequest(kind=pr.KIND_LORA, clip_max_cer=0.12)
    assert pr._asr_for_clip_check(on, lambda: fake, preview=False) == (fake, "")
    assert pr._asr_for_clip_check(on, lambda: fake, preview=True) == (None, "")
    assert pr._asr_for_clip_check(pr.TaskRequest(kind=pr.KIND_LORA), lambda: fake, False) == (None, "")
    assert pr._asr_for_clip_check(pr.TaskRequest(kind=pr.KIND_LORA, clip_max_cer=0.12, no_transcript=True),
                                  lambda: fake, False) == (None, "")
    monkeypatch.setattr(md, "ready_model_path", lambda repo: None)
    asr, warn = pr._asr_for_clip_check(on, None, False)
    assert asr is None and warn                                       # not installed: a warning, never a download


def test_train_window_passes_the_threshold(app):
    from tests.test_ui import make_window

    w = make_window(lambda *a: None)
    assert w._task_extras()["clip_max_cer"] == pytest.approx(0.12)
    w.sp_clipcer.setValue(15)
    assert w._task_extras()["clip_max_cer"] == pytest.approx(0.15)
    w.chk_clipcheck.setChecked(False)
    assert w._task_extras()["clip_max_cer"] == 0.0 and not w.sp_clipcer.isEnabled()
    w.reload_auto_steps()
    assert w.chk_clipcheck.isChecked()
    w.set_no_transcript(True)
    assert w.clipcheck_row.isHidden()
    w.close()
