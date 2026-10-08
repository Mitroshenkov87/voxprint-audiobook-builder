"""Training presets: plans, GPU-scaled time estimate (calibrated on the measured RTX 4090 run), Train window controls."""
import time

import pytest

from core import train_presets as tp
from infra.vram_optimizer import GpuInfo, plan_training

G4090 = GpuInfo(True, "NVIDIA GeForce RTX 4090", 24.0, 22.0)
G3060 = GpuInfo(True, "NVIDIA GeForce RTX 3060", 12.0, 11.0)
NOGPU = GpuInfo(False)


def test_balanced_is_exactly_the_automatic_plan_and_others_scale_epochs_and_rank():
    base = plan_training(G4090, 70)
    assert tp.build_plan(tp.BALANCED, G4090, 70) == base and base.epochs == 5
    fast, mx = tp.build_plan(tp.FAST, G4090, 70), tp.build_plan(tp.MAXIMUM, G4090, 70)
    assert fast.epochs < base.epochs < mx.epochs and fast.lora_r < base.lora_r < mx.lora_r
    assert fast.lr == base.lr == mx.lr          # the learning rate is never raised by a preset (real tests: higher lr babbles)
    assert fast.lora_alpha == fast.lora_r * 4 and mx.lora_alpha == mx.lora_r * 4


def test_manual_values_are_applied_and_sanitised():
    m = tp.Manual(epochs=9, lora_r=8, lora_alpha=16, lr=2e-6, grad_accum=2)
    p = tp.build_plan(tp.MANUAL, G4090, 70, manual=m)
    assert (p.epochs, p.lora_r, p.lora_alpha, p.lr, p.grad_accum) == (9, 8, 16, 2e-6, 2)
    assert tp.build_plan(tp.MANUAL, G4090, 70, manual=tp.Manual(epochs=0)).epochs == 1
    assert tp.build_plan(tp.MANUAL, G4090, 70) == plan_training(G4090, 70)      # no values -> automatic


def test_estimate_reproduces_the_measured_4090_run_and_scales_with_the_gpu():
    plan = plan_training(G4090, 70)
    est = tp.estimate_seconds(plan, 70, G4090)
    assert 12 + 5 * 24.0 - 6 < est < 12 + 5 * 24.0 + 6           # measured: 24 s/epoch for 70 clips, 12 s load
    assert tp.estimate_seconds(plan_training(G3060, 70), 70, G3060) > 2.5 * est
    assert tp.estimate_seconds(plan_training(NOGPU, 70), 70, NOGPU) > 10 * est
    f, b, m = (tp.estimate_seconds(tp.build_plan(x, G4090, 70), 70, G4090) for x in (tp.FAST, tp.BALANCED, tp.MAXIMUM))
    assert f < b < m
    assert tp.speed_factor(GpuInfo(True, "Mystery GPU", 8.0)) == 5.0           # unknown card -> by VRAM tier


def test_clips_from_audio_and_duration_text():
    assert tp.clips_from_audio(609) == 70 or abs(tp.clips_from_audio(609) - 70) <= 1     # the measured recording
    assert tp.clips_from_audio(0) == 1
    assert tp.format_duration(40) == "40 s" and tp.format_duration(150) in ("2 min", "3 min")
    assert tp.format_duration(3 * 3600 + 20 * 60, "min", "s", "h") == "3 h 20 min"


pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _window(seen=None):
    from ui.main_window import MainWindow
    return MainWindow(runner=(lambda req, p, c: seen.append(req)) if seen is not None else (lambda *a: None),
                      autocheck=False, auto_open_folder=False, gpu_fn=lambda: G4090)


def test_train_window_shows_presets_estimate_and_collapsed_advanced_panel(app, tmp_path):
    import numpy as np
    from core import audio_utils as au
    w = _window()
    assert [w.cmb_preset.itemData(i) for i in range(4)] == ["fast", "balanced", "maximum", "manual"]
    assert w.preset == "balanced" and w.adv_box.isHidden()
    assert "RTX 4090" in w.lbl_estimate.text() and "10" in w.lbl_estimate.text()       # example for a 10-minute recording
    wav = tmp_path / "r.wav"
    au.write_wav(wav, np.zeros(16000 * 300, dtype=np.float32), 16000)                  # 5 minutes
    w.set_audio(wav)
    assert "5" in w.lbl_estimate.text() or "clips" in w.lbl_estimate.text()
    n5 = w.lbl_estimate.text()
    w.cmb_preset.setCurrentIndex(2)
    assert w.lbl_estimate.text() != n5 and "RTX 4090" in w.lbl_estimate.text()
    assert not w.sp_epochs.isEnabled()                                                  # only Manual edits the numbers
    w.cmb_preset.setCurrentIndex(3)
    assert w.sp_epochs.isEnabled() and not w.adv_box.isHidden()
    before = w.lbl_estimate.text()
    w.sp_epochs.setValue(w.sp_epochs.value() * 3)
    assert w.lbl_estimate.text() != before


def test_chosen_preset_and_manual_values_reach_the_task_request(app, tmp_path):
    seen = []
    w = _window(seen)
    (tmp_path / "a.wav").write_bytes(b"x"); (tmp_path / "t.txt").write_text("hello", encoding="utf-8")
    w.set_audio(tmp_path / "a.wav"); w.set_text(tmp_path / "t.txt")
    w.cmb_preset.setCurrentIndex(1)
    w.cmb_preset.setCurrentIndex(3)
    w.sp_epochs.setValue(7); w.sp_rank.setValue(24)
    w.start("lora")
    end = time.time() + 5
    while not seen and time.time() < end:
        QApplication.processEvents(); time.sleep(0.01)
    assert seen[0].preset == "manual"
    assert seen[0].manual.epochs == 7 and seen[0].manual.lora_r == 24
    w.worker.wait(3000)


def test_preset_texts_in_every_language():
    from core import i18n
    from core.i18n import tr
    for lang in i18n.LANGS:
        i18n.set_language(lang, persist=False)
        for k in ("preset.label", "preset.fast", "preset.desc_maximum", "preset.estimate", "preset.adv_lr", "preset.adv_hint"):
            assert tr(k) != k
    i18n.set_language("en", persist=False)
    assert "about 2 min" in tr("preset.estimate", time="2 min", epochs=5, rank=32, n=70, gpu="X")


def test_run_task_passes_a_preset_plan_to_the_trainer_only_when_not_balanced(tmp_path, monkeypatch):
    import core.lora_trainer as lt
    from tests.synth import TrueRateAligner, make_text, synth_reading
    from core import audio_utils as au
    from workers.pipeline_runner import KIND_LORA, TaskRequest, run_task
    text = make_text(60, 1); audio, _ = synth_reading(text, seed=1)
    au.write_wav(tmp_path / "in.wav", audio, 16000); (tmp_path / "in.txt").write_text(text, encoding="utf-8")
    calls = []
    def fake_train(d, o, *a, **k):
        calls.append(k.get("plan"))
        o.mkdir(parents=True, exist_ok=True)
        return o
    monkeypatch.setattr(lt, "train_lora_from_dataset", fake_train)
    class NoUpd:
        def should_autocheck(self): return False
    for preset in ("balanced", "fast"):
        run_task(TaskRequest(kind=KIND_LORA, audio=tmp_path / "in.wav", text=tmp_path / "in.txt", out_root=tmp_path / preset,
                             preset=preset), updater=NoUpd(), aligner_factory=TrueRateAligner, voice_library=_Lib())
    assert calls[0] is None and calls[1] is not None and calls[1].lora_r == 16


class _Lib:
    def add_from_adapter(self, d, info=None):
        class V: id = "x"
        return V()
