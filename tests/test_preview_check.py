"""Voice checks (pitch / WER / stop), quick previews with fake training + engine, the post-training check and the Train window flow."""
import time

import numpy as np
import pytest

from core import audio_utils as au
from core import train_presets, voice_check as vc
from core.asr import FakeASR
from core.events import CancelToken
from infra.vram_optimizer import GpuInfo, plan_training
from tests.synth import TrueRateAligner, make_text, synth_reading

G = GpuInfo(True, "NVIDIA GeForce RTX 4090", 24.0, 22.0)


def tone(f, sec=2.0, sr=24000, amp=0.3):
    t = np.arange(int(sec * sr)) / sr
    return (amp * (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t))).astype(np.float32)


def test_f0_and_semitones():
    assert abs(vc.f0_median(tone(120), 24000) - 120) < 3
    assert abs(vc.f0_median(tone(220), 24000) - 220) < 5
    assert vc.f0_median(np.zeros(24000, dtype=np.float32), 24000) == 0.0
    assert abs(vc.semitones(240, 120) - 12) < 1e-6 and vc.semitones(0, 120) is None


def test_wer():
    assert vc.wer("раз два три", "раз два три") == 0 and vc.wer("раз два три", "раз три") == pytest.approx(1 / 3)
    assert vc.wer("a b", "") == 1.0


def test_check_sample_verdicts():
    ref = tone(120)
    ok = vc.check_sample(tone(122), 24000, "раз два", ref, 24000, asr_text="раз два", max_seconds=10)
    assert ok.verdict == vc.GOOD and not ok.issues
    high = vc.check_sample(tone(190), 24000, "раз два", ref, 24000, asr_text="раз два", max_seconds=10)       # ~7.9 semitones up
    assert high.verdict == vc.BAD and "pitch" in high.issues and high.semitones > 6
    babble = vc.check_sample(tone(120, 10.0), 24000, "раз два", ref, 24000, asr_text="раз два", max_seconds=9.8)
    assert babble.verdict == vc.BAD and "no_stop" in babble.issues
    wrong = vc.check_sample(tone(120), 24000, "раз два три четыре", ref, 24000, asr_text="что-то другое совсем", max_seconds=10)
    assert "wer" in wrong.issues and wrong.verdict == vc.BAD
    quiet = vc.check_sample(tone(120, amp=0.0005), 24000, "x", ref, 24000, asr_text="x", max_seconds=10)
    assert "quiet" in quiet.issues
    assert ok.as_dict()["verdict"] == "good" and ok.as_dict()["wer"] == 0.0


def _dataset(tmp_path, n=20):
    d = tmp_path / "ds"
    d.mkdir()
    from core.dataset_builder import write_metadata_jsonl
    rows = []
    for i in range(1, n + 1):
        au.write_wav(d / f"segment_{i:03d}.wav", tone(120 + i, 3.0), 24000)
        rows.append({"audio": f"segment_{i:03d}.wav", "text": f"фраза {i}", "ref_audio": "ref.wav"})
    au.write_wav(d / "ref.wav", tone(120, 6.0), 24000)
    (d / "ref_text.txt").write_text("опорная фраза\n", encoding="utf-8")
    write_metadata_jsonl(d / "metadata.jsonl", rows)
    return d


class FakeEngine:
    def __init__(self, f0=120.0, secs=8.0):
        self.f0, self.secs, self.sample_rate, self.closed = f0, secs, 24000, False
    def synthesize(self, text):
        return tone(self.f0, self.secs)
    def close(self):
        self.closed = True


def _train(calls):
    def train(dataset, out, progress, cancel, force_cpu, plan=None, language=None):
        from core.dataset_builder import read_metadata_jsonl
        calls.append((len(read_metadata_jsonl(dataset / "metadata.jsonl")), plan))
        out.mkdir(parents=True, exist_ok=True)
        (out / "training_meta.json").write_text("{}", encoding="utf-8")
        return out
    return train


def test_variants_and_time_cap():
    base = plan_training(G, 70)
    assert [v.key for v in __import__("workers.preview_runner", fromlist=["x"]).make_variants(base, False)] == ["A"]
    from workers import preview_runner as pr
    a, b = pr.make_variants(base, True)
    assert a.plan.epochs <= pr.MAX_EPOCHS and b.plan.lora_r == 2 * a.plan.lora_r and b.plan.epochs > a.plan.epochs
    assert a.plan.lr == b.plan.lr == base.lr                             # the learning rate is never raised
    slow = GpuInfo(False)
    assert pr.clips_for_time_cap(a.plan, slow, 12) == pr.MIN_CLIPS < pr.clips_for_time_cap(a.plan, G, 12) + 1


def test_run_previews_trains_on_a_subset_and_checks_each_sample(tmp_path):
    from workers import preview_runner as pr
    calls = []
    base = plan_training(G, 70)
    items = pr.run_previews(_dataset(tmp_path), tmp_path / "out", base, compare=True, language="russian", gpu=G,
                            train_fn=_train(calls), engine_factory=lambda adapter, lang: FakeEngine(),
                            asr=FakeASR([pr.SAMPLE_TEXT["russian"]]))
    assert [i.key for i in items] == ["A", "B"] and len(calls) == 2
    assert all(n <= pr.MAX_CLIPS for n, _ in calls) and calls[1][1].lora_r == 2 * calls[0][1].lora_r
    for it in items:
        assert it.wav.is_file() and it.check["verdict"] == "good" and it.check["wer"] == 0.0 and abs(it.seconds - 8.0) < 0.1


def test_run_previews_flags_a_babbling_variant_and_can_be_cancelled(tmp_path):
    from core.errors import CancelledByUser
    from workers import preview_runner as pr
    base = plan_training(G, 70)
    items = pr.run_previews(_dataset(tmp_path), tmp_path / "o1", base, compare=False, language="english", gpu=G,
                            train_fn=_train([]), engine_factory=lambda a, l: FakeEngine(secs=40.0),
                            asr=FakeASR([pr.SAMPLE_TEXT["english"]]))
    assert items[0].check["verdict"] == "bad" and "no_stop" in items[0].check["issues"]
    tok = CancelToken(); tok.cancel()
    with pytest.raises(CancelledByUser):
        pr.run_previews(_dataset(tmp_path / "x") if (tmp_path / "x").mkdir() is None else None, tmp_path / "o2", base, compare=True,
                        language="english", gpu=G, cancel=tok, train_fn=_train([]), engine_factory=lambda a, l: FakeEngine())


class _NoUpd:
    def should_autocheck(self):
        return False


def _inputs(tmp_path):
    text = make_text(40, 1)
    audio, _ = synth_reading(text, seed=1)
    au.write_wav(tmp_path / "in.wav", audio, 16000)
    (tmp_path / "in.txt").write_text(text, encoding="utf-8")


def test_preview_task_end_to_end_registers_nothing(tmp_path):
    from core.voice_library import VoiceLibrary
    from workers.pipeline_runner import KIND_PREVIEW, TaskRequest, run_task
    _inputs(tmp_path)
    lib = VoiceLibrary(tmp_path / "lib")
    res = run_task(TaskRequest(kind=KIND_PREVIEW, audio=tmp_path / "in.wav", text=tmp_path / "in.txt", out_root=tmp_path / "out",
                               compare=True), updater=_NoUpd(), aligner_factory=TrueRateAligner, voice_library=lib,
                   asr_factory=lambda: FakeASR(["x"]),
                   synth_deps=dict(train_fn=_train([]), engine_factory=lambda a, l: FakeEngine()))
    assert [p.key for p in res.previews] == ["A", "B"] and lib.list_voices() == [] and res.adapter_path is None


def test_post_training_check_reports_quality_and_suggests(tmp_path, monkeypatch):
    import core.lora_trainer as lt
    from core.voice_library import VoiceLibrary
    from workers.pipeline_runner import KIND_LORA, TaskRequest, run_task
    _inputs(tmp_path)

    def fake_train(d, o, *a, **k):
        o.mkdir(parents=True, exist_ok=True)
        for f in ("adapter_model.safetensors", "adapter_config.json"):
            (o / f).write_bytes(b"x")
        au.write_wav(o / "ref_sample.wav", tone(120, 6.0), 24000)
        (o / "training_meta.json").write_text('{"epochs": 1, "ref_sample_text": "t"}', encoding="utf-8")
        return o
    monkeypatch.setattr(lt, "train_lora_from_dataset", fake_train)
    from workers import preview_runner as pr
    for engine, expect in ((FakeEngine(), "good"), (FakeEngine(f0=200.0), "bad")):
        res = run_task(TaskRequest(kind=KIND_LORA, audio=tmp_path / "in.wav", text=tmp_path / "in.txt", out_root=tmp_path / expect,
                                   quality_check=True), updater=_NoUpd(), aligner_factory=TrueRateAligner,
                       voice_library=VoiceLibrary(tmp_path / ("lib" + expect)), asr_factory=lambda: FakeASR([pr.SAMPLE_TEXT["russian"]]),
                       synth_deps=dict(engine_factory=lambda a, l, e=engine: e))
        assert res.quality["verdict"] == expect
        assert (expect == "good") == (not any("проверк" in w.lower() for w in res.warnings))


pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_train_window_preview_flow(app, tmp_path):
    from types import SimpleNamespace
    from ui.main_window import MainWindow
    from workers.preview_runner import PreviewItem
    seen = []
    w = MainWindow(runner=lambda req, p, c: seen.append(req), autocheck=False, auto_open_folder=False, gpu_fn=lambda: G)
    played = []
    w.previewer = SimpleNamespace(play=lambda p: played.append(p))
    assert not w.btn_preview.isEnabled() and "около" in w.lbl_preview_estimate.text()
    (tmp_path / "a.wav").write_bytes(b"x"); (tmp_path / "t.txt").write_text("x", encoding="utf-8")
    w.set_audio(tmp_path / "a.wav"); w.set_text(tmp_path / "t.txt")
    one = w.lbl_preview_estimate.text()
    w.chk_compare.setChecked(True)
    assert w.lbl_preview_estimate.text() != one and w.btn_preview.isEnabled()
    w.btn_preview.click()
    end = time.time() + 5
    while not seen and time.time() < end:
        QApplication.processEvents(); time.sleep(0.01)
    assert seen[0].kind == "preview" and seen[0].compare and seen[0].quality_check
    w.worker.wait(3000)
    items = [PreviewItem("A", "A", 3, 32, 128, 1e-6, 4, tmp_path / "pa.wav", 9.5, 40.0, {"verdict": "good", "wer": 0.1, "semitones": 0.4, "issues": []}, 12),
             PreviewItem("B", "B", 5, 64, 256, 1e-6, 4, tmp_path / "pb.wav", 31.0, 60.0, {"verdict": "bad", "wer": 0.9, "semitones": 7.0, "issues": ["no_stop", "pitch"]}, 12)]
    w.on_done(SimpleNamespace(kind="preview", previews=items, open_dir=tmp_path))
    assert not w.preview_box.isHidden() and len(w.preview_rows) == 2
    w.preview_rows[1][1].click()
    assert played == [tmp_path / "pb.wav"]
    w.preview_rows[1][2].click()
    assert w.preset == "manual" and w.sp_rank.value() == 64 and w.sp_alpha.value() == 256


def test_preview_and_check_texts_in_every_language():
    from core import i18n
    from core.i18n import tr
    for lang in ("en", "ru", "de"):
        i18n.set_language(lang, persist=False)
        for k in ("preview.button", "preview.estimate", "preview.row", "check.checkbox", "check.sugg_no_stop", "check.verdict_bad"):
            assert tr(k) != k


def test_estimate_matches_the_measured_rtx4090_previews():
    """Measured on the real GPU (RTX 4090, 1.7B, 12 clips): A (4 epochs, r32) 22.0-26.7 s, B (6 epochs, r64) 31.8 s of training."""
    from core import train_presets
    from infra.vram_optimizer import GpuInfo, plan_training
    from workers import preview_runner as pr

    gpu = GpuInfo(True, "NVIDIA GeForce RTX 4090", 22.5, 21.0)
    plan = plan_training(gpu, 70)
    a, b = pr.make_variants(plan, True)
    ea = train_presets.estimate_seconds(a.plan, 12, gpu)
    eb = train_presets.estimate_seconds(b.plan, 12, gpu)
    assert 22.0 * 0.95 <= ea <= 26.7 * 1.25 and 31.8 * 0.95 <= eb <= 31.8 * 1.25
    assert ea < eb                                     # the bigger variant is estimated slower
    total = ea + eb + 2 * pr.SYNTH_CHECK_SEC + pr.ASR_LOAD_SEC   # measured whole compare run: ~125 s incl. ASR load
    assert 105 <= total <= 150


def test_gpu_factor_table_is_ordered_by_speed():
    """Sanity of the scaling table: a faster card never gets a bigger factor (spec-derived, only the 4090 is measured)."""
    from core import train_presets as tp

    order = ["5090", "4090", "4080", "3090", "4070 ti", "3080", "4070", "4060 ti", "3070", "4060", "2080", "3060", "2070", "2060", "1080", "1070", "1060"]
    f = dict(tp.GPU_FACTORS)
    assert f["4090"] == 1.0
    vals = [f[k] for k in order]
    assert vals == sorted(vals)


def test_demo_text_is_technical_gender_neutral_and_fits_the_durations():
    import re

    from workers import preview_runner as pr

    ru, short = pr.DEMO_TEXT["russian"], pr.SAMPLE_TEXT["russian"]
    assert ru.startswith(short) and set(pr.DEMO_TEXT) == set(pr.SAMPLE_TEXT) == {"russian", "english", "german"}
    assert 8 <= len(short.split()) / 2.4 <= 13                          # ~10 s at ~2.4 words/s
    assert 45 <= len(ru.split()) <= 65                                    # ~15-25 s when read
    for term in ("LoRA", "Qwen3-TTS", "WER"):
        assert term in ru
    for w in ("согласие", "адаптер", "пайплайн"):
        assert w in ru
    # gender-neutral: no first-person past tense / short adjectives that show the speaker's gender
    assert not re.search(r"\bя\b[^.]*?\w+(?:л|ла|лся|лась)\b", ru.lower())
    for lang in ("english", "german"):
        assert pr.DEMO_TEXT[lang].startswith(pr.SAMPLE_TEXT[lang]) and "LoRA" in pr.DEMO_TEXT[lang]
