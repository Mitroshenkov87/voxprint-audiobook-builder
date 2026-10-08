"""Automatic checkpoint + strength pick: holdout split, validation loss, phrase choice, scoring, the pick with a fake engine,
applying it (winner kept, losers deleted), and the run_task / Train window wiring."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from core import adapter_strength as ast
from core import audio_utils as au
from core import checkpoint_pick as cp
from core import voice_check as vc
from core.asr import FakeASR
from tests.test_preview_check import tone


def test_split_holdout():
    from core.lora_trainer import HOLDOUT_MAX, split_holdout
    rows = [{"i": i} for i in range(100)]
    tr_, ho = split_holdout(rows, 0.05)
    assert len(ho) == 5 and len(tr_) == 95 and not ({r["i"] for r in ho} & {r["i"] for r in tr_})
    assert max(r["i"] for r in ho) - min(r["i"] for r in ho) > 60      # spread over the recording
    assert split_holdout(rows[:19], 0.05) == (rows[:19], [])            # small datasets keep every clip
    assert len(split_holdout(rows[:20], 0.05)[1]) == 2                   # at least 2
    assert len(split_holdout([{"i": i} for i in range(1000)], 0.05)[1]) == HOLDOUT_MAX
    assert split_holdout(rows, 0.0)[1] == []


def test_score_mixes_available_metrics_and_penalises_babbling():
    full = cp.score(0.0, 1.0, 5.0, 0.0)
    assert full == pytest.approx(1.0)
    assert cp.score(0.1, None, None, 0.0) == pytest.approx(0.9)          # only CER available
    assert cp.score(0.0, 0.5, None, 0.0) == pytest.approx((0.5 + 0.3 * 0.5) / 0.8)
    assert cp.score(0.0, 1.0, 5.0, 1.0) == pytest.approx(1.0 - cp.NO_STOP_PENALTY)
    assert cp.score(None, None, None, 0.0) == 0.0


def _adapter(tmp_path, epochs=4, holdout=("Short.", "A medium sentence for the test.",
                                          "A rather long sentence that the training never saw, read slowly.",
                                          "Another unseen sentence of moderate length here.", "Tiny.")):
    d = tmp_path / "voice"
    for e in range(1, epochs + 1):
        f = d / "checkpoints" / f"epoch_{e:02d}"
        f.mkdir(parents=True)
        (f / "adapter_model.safetensors").write_bytes(f"weights-{e}".encode())
        (f / "adapter_config.json").write_text(json.dumps({"epoch": e}))
    (d / "adapter_model.safetensors").write_bytes(f"weights-{epochs}".encode())
    (d / "adapter_config.json").write_text(json.dumps({"epoch": epochs}))
    au.write_wav(d / "ref_sample.wav", tone(120, 6.0), 24000)
    (d / "training_meta.json").write_text(json.dumps({"epochs": epochs, "ref_sample_text": "t", "model_name": "Qwen/B"}))
    if holdout:
        (d / "checkpoints" / "holdout.json").write_text(json.dumps({"rows": [{"audio": "x.wav", "text": t} for t in holdout]}))
    return d


def test_pick_phrases_spread_from_short_to_long(tmp_path):
    d = _adapter(tmp_path)
    ph = cp.pick_phrases(d, "english", n=2)
    assert ph[0] == "A medium sentence for the test." and ph[-1].startswith("A rather long")   # too short ones dropped
    assert len(cp.pick_phrases(d, "english")) == 3
    d2 = _adapter(tmp_path / "b", holdout=None)
    from workers.preview_runner import SAMPLE_TEXT
    assert cp.pick_phrases(d2, "german") == [SAMPLE_TEXT["german"]]


class PickEngine:
    """Epoch 3 at strength 0.5 sounds like the speaker; 1.0 never stops; other combinations drift in pitch."""
    sample_rate = 24000

    def __init__(self):
        self.epoch, self.scale, self.closed, self.switches = 4, 1.0, False, []

    def switch_adapter(self, folder):
        self.epoch = int(json.loads((folder / "adapter_config.json").read_text())["epoch"])
        self.switches.append(self.epoch)

    def set_adapter_scale(self, s):
        self.scale = s

    def synthesize(self, text):
        if self.scale == 1.0:                       # runs to the length cap (kept short: the pitch tracker is slow)
            from core.tts_engine import FRAMES_PER_SECOND, max_tokens_for
            return tone(120, max_tokens_for(text) / FRAMES_PER_SECOND)
        f = 120 if (self.epoch, self.scale) == (3, 0.5) else 120 + 25 * abs(self.epoch - 3) + 60 * abs(self.scale - 0.5)
        return tone(f, 3.0)

    def speaker_embedding(self, audio, sr):
        spec = np.abs(np.fft.rfft(audio[: sr * 2]))                  # dominant frequency (fast stand-in for an encoder)
        f = float(np.argmax(spec)) * sr / (2 * min(len(audio), sr * 2))
        return np.array([np.cos(f / 40.0), np.sin(f / 40.0)])

    def close(self):
        self.closed = True


class LoadASR(FakeASR):
    def __init__(self, texts):
        super().__init__(texts)
        self.loaded = []

    def load(self):
        self.loaded.append("load")

    def unload(self):
        self.loaded.append("unload")


def test_run_pick_scores_every_candidate_and_keeps_the_best(tmp_path):
    d = _adapter(tmp_path)
    eng = PickEngine()
    asr = LoadASR(lambda a, sr: "A medium sentence for the test.")
    msgs = []
    res = cp.run_pick(d, "english", engine_factory=lambda a, l: eng, asr=asr, mos=None,
                      progress=lambda s, f, m: msgs.append(m))
    assert eng.closed and eng.switches == [2, 3, 4] and asr.loaded == ["load", "unload"]
    assert len(res["candidates"]) == 3 * len(ast.PREVIEW_SCALES) and res["phrases"][0] == "A medium sentence for the test."
    assert (res["best_epoch"], res["best_scale"]) == (3, 0.5)
    babble = [c for c in res["candidates"] if c["scale"] == 1.0]
    assert all(c["no_stop"] == 1.0 for c in babble)
    assert any("3" in m and "0.50" in m for m in msgs[-1:])
    # an engine that cannot switch: nothing to pick
    assert cp.run_pick(d, "english", engine_factory=lambda a, l: SimpleNamespace(close=lambda: None)) is None
    assert cp.run_pick(tmp_path / "nothing", "english", engine_factory=lambda a, l: eng) is None


def test_apply_pick_keeps_the_winner_and_deletes_the_losers(tmp_path):
    d = _adapter(tmp_path)
    result = {"best_epoch": 3, "best_scale": 0.5, "phrases": ["x"], "candidates": []}
    cp.apply_pick(d, result)
    assert (d / "adapter_model.safetensors").read_bytes() == b"weights-3"
    assert json.loads((d / "adapter_config.json").read_text())["epoch"] == 3
    assert json.loads((d / "training_meta.json").read_text())["epochs"] == 3
    assert json.loads((d / "checkpoints" / cp.PICK_FILE).read_text())["best_scale"] == 0.5
    assert [e for e, _ in cp.epoch_folders(d)] == [3]                    # disk: one checkpoint left
    with pytest.raises(FileNotFoundError):
        cp.apply_pick(d, dict(result, best_epoch=1))


def test_run_task_auto_pick_end_to_end(tmp_path, monkeypatch):
    import core.lora_trainer as lt
    from core.voice_library import VoiceLibrary
    from tests.test_preview_check import TrueRateAligner, _inputs, _NoUpd
    from workers.pipeline_runner import HOLDOUT_FRACTION, KIND_LORA, TaskRequest, run_task
    _inputs(tmp_path)
    seen = {}

    def fake_train(dset, out, *a, **k):
        seen.update(k)
        src = _adapter(tmp_path / f"src{len(seen)}_{out.name}")
        import shutil
        shutil.copytree(src, out, dirs_exist_ok=True)
        return out
    monkeypatch.setattr(lt, "train_lora_from_dataset", fake_train)
    lib = VoiceLibrary(tmp_path / "lib")
    asr = lambda: FakeASR(lambda a, sr: "A medium sentence for the test.")
    res = run_task(TaskRequest(kind=KIND_LORA, audio=tmp_path / "in.wav", text=tmp_path / "in.txt", out_root=tmp_path / "o",
                               auto_pick=True), updater=_NoUpd(), aligner_factory=TrueRateAligner, voice_library=lib,
                   asr_factory=asr, synth_deps=dict(engine_factory=lambda a, l: PickEngine(), mos=None))
    assert seen["holdout_fraction"] == HOLDOUT_FRACTION
    assert res.pick["best_epoch"] == 3 and res.pick["best_scale"] == 0.5
    rec = lib.get(res.voice_id)
    assert rec.adapter_scale == 0.5 and (rec.path / "adapter_model.safetensors").read_bytes() == b"weights-3"
    # a strength chosen in the preview is kept: only the epoch is picked, at that strength
    res2 = run_task(TaskRequest(kind=KIND_LORA, audio=tmp_path / "in.wav", text=tmp_path / "in.txt", out_root=tmp_path / "o2",
                                auto_pick=True, adapter_scale=0.35, voice_display_name="Second"),
                    updater=_NoUpd(), aligner_factory=TrueRateAligner, voice_library=lib, asr_factory=asr,
                    synth_deps=dict(engine_factory=lambda a, l: PickEngine(), mos=None))
    assert {c["scale"] for c in res2.pick["candidates"]} == {0.35} and lib.get(res2.voice_id).adapter_scale == 0.35
    # a failing pick never fails the training run
    def broken(a, l):
        raise RuntimeError("no GPU")
    res3 = run_task(TaskRequest(kind=KIND_LORA, audio=tmp_path / "in.wav", text=tmp_path / "in.txt", out_root=tmp_path / "o3",
                                auto_pick=True, voice_display_name="Third"),
                    updater=_NoUpd(), aligner_factory=TrueRateAligner, voice_library=lib, asr_factory=asr,
                    synth_deps=dict(engine_factory=broken, mos=None))
    assert res3.pick is None and res3.voice_id and lib.get(res3.voice_id).adapter_scale == ast.DEFAULT_SCALE
    assert any("контрольн" in w.lower() for w in res3.warnings)


# --------------------------------------------------------------------------- tiny real model: holdout + validation loss
def test_training_holds_out_clips_and_records_validation_loss(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from core.events import CancelToken
    from core.lora_trainer import load_training_rows, train_on_model
    from tests.test_lora_trainer import _dataset, _encode, _plan, _tok, tiny_model
    d = _dataset(tmp_path, n=20)
    out = tmp_path / "out"
    train_on_model(tiny_model(), _tok, _encode, load_training_rows(d), out, _plan(holdout_fraction=0.05), "Qwen/B",
                   lambda *a: None, CancelToken(), [])
    losses = json.loads((out / "checkpoints" / "losses.json").read_text())
    assert len(losses["epoch_val_loss"]) == 2 and all(v > 0 for v in losses["epoch_val_loss"])
    hold = json.loads((out / "checkpoints" / "holdout.json").read_text(encoding="utf-8"))
    assert len(hold["rows"]) == 2 and all(r["text"].startswith("Текст номер") for r in hold["rows"])
    assert json.loads((out / "training_meta.json").read_text())["num_samples"] == 18
    train_on_model(tiny_model(), _tok, _encode, load_training_rows(d), out, _plan(), "Qwen/B", lambda *a: None,
                   CancelToken(), [])
    assert not (out / "checkpoints" / "holdout.json").exists()        # no holdout: no stale file
    assert json.loads((out / "checkpoints" / "losses.json").read_text())["epoch_val_loss"] == []


def test_train_window_pick_checkbox(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from tests.test_preview_check import G
    from ui.main_window import MainWindow
    w = MainWindow(runner=lambda *a: None, autocheck=False, auto_open_folder=False, gpu_fn=lambda: G)
    assert w.chk_pick.isChecked() and w._task_extras()["auto_pick"] is True
    w.chk_pick.setChecked(False)
    assert w._task_extras()["auto_pick"] is False
    w.reload_auto_steps()
    assert w.chk_pick.isChecked()
