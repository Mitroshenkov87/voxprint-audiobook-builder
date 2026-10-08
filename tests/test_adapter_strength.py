"""LoRA adapter strength at inference: the value, voice.json / library, cache key, scaled merge, engine switching,
the preview's per-strength samples and the UI (Train window preview rows, voice Properties)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from core import adapter_strength as ast
from core import audio_utils as au
from core import voice_info
from tests.test_voice_library import make_adapter


def test_clamp_and_full():
    assert ast.clamp("0.333") == 0.33 and ast.clamp(5) == ast.MAX_SCALE and ast.clamp(0) == ast.MIN_SCALE
    assert ast.clamp("x") is None and ast.clamp(None, 1.0) == 1.0 and ast.clamp(float("nan"), 0.5) == 0.5
    assert ast.is_full(None) and ast.is_full(1.0) and not ast.is_full(0.5)
    assert ast.MIN_SCALE <= ast.DEFAULT_SCALE < ast.LEGACY_SCALE == 1.0
    assert 1.0 in ast.PREVIEW_SCALES and ast.DEFAULT_SCALE in ast.PREVIEW_SCALES   # the old behaviour stays reachable


def test_voice_json_field_is_optional_and_clamped(tmp_path):
    from core.voice_library import VoiceLibrary, VoiceRecord
    info = voice_info.build_voice_info("A", "english", 600, 3, "Qwen/B")
    assert "adapter_scale" not in info                                  # older voices: no field = full strength
    assert voice_info.build_voice_info("A", "english", 600, 3, "Qwen/B", adapter_scale=0.456)["adapter_scale"] == 0.46
    assert voice_info.normalize_info({"name": "A", "adapter_scale": "junk"}, "a").get("adapter_scale") is None
    assert voice_info.normalize_info({"name": "A", "adapter_scale": 3}, "a")["adapter_scale"] == 1.0
    assert VoiceRecord("v", tmp_path, {}).adapter_scale == 1.0
    lib = VoiceLibrary(tmp_path / "lib")
    rec = lib.add_from_adapter(make_adapter(tmp_path / "a"))
    assert rec.adapter_scale == 1.0
    rec = lib.update(rec.id, adapter_scale=0.35)
    assert rec.adapter_scale == 0.35
    assert json.loads((rec.path / "voice.json").read_text(encoding="utf-8"))["adapter_scale"] == 0.35


def test_engine_tag_includes_the_strength_but_keeps_old_caches(tmp_path):
    from core import tts_engine
    from core.voice_library import VoiceRecord
    d = tmp_path / "v"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"w")
    old = tts_engine.engine_tag(VoiceRecord("v", d, {"language": "ru", "base_model": "Qwen/B"}))
    assert tts_engine.engine_tag(VoiceRecord("v", d, {"language": "ru", "base_model": "Qwen/B", "adapter_scale": 1.0})) == old
    half = tts_engine.engine_tag(VoiceRecord("v", d, {"language": "ru", "base_model": "Qwen/B", "adapter_scale": 0.5}))
    assert half != old
    assert tts_engine.engine_tag(VoiceRecord("v", d, {"language": "ru", "base_model": "Qwen/B"}), scale=0.5) == half
    assert tts_engine.engine_tag(VoiceRecord("v", d, {"language": "ru", "base_model": "Qwen/B", "adapter_scale": 0.35})) != half


def test_write_voice_json_strength_order(tmp_path):
    from workers.pipeline_runner import TaskRequest, _write_voice_json
    d = make_adapter(tmp_path / "a", with_voice_json=False)
    assert _write_voice_json(TaskRequest(kind="lora"), d, "english", 60)["adapter_scale"] == ast.DEFAULT_SCALE
    assert _write_voice_json(TaskRequest(kind="lora"), d, "english", 60, picked_scale=0.7)["adapter_scale"] == 0.7
    info = _write_voice_json(TaskRequest(kind="lora", adapter_scale=1.0), d, "english", 60, picked_scale=0.7)
    assert info["adapter_scale"] == 1.0                                 # the user's preview choice wins
    assert json.loads((d / "voice.json").read_text(encoding="utf-8"))["adapter_scale"] == 1.0


# --------------------------------------------------------------------------- real (tiny) peft model
torch_mod = pytest.importorskip("torch")
pytest.importorskip("peft")


def _q_weight(talker):
    return talker.model.layers[0].self_attn.q_proj.weight.detach().clone()


def test_scaled_merge_carries_the_scaled_delta(tmp_path):
    from peft import PeftModel
    from core.model_export import adapter_scale_of, merge_adapter_into_model
    from tests.test_lora_trainer import tiny_model
    from tests.test_merge import SPK
    base = _q_weight(tiny_model().talker)
    ad = _train_adapter(tmp_path)
    full = tiny_model()
    merge_adapter_into_model(full, ad, spk_id=SPK, adapter_scale=1.0)
    half = tiny_model()
    merge_adapter_into_model(half, ad, spk_id=SPK, adapter_scale=0.5)
    d_full, d_half = _q_weight(full.talker) - base, _q_weight(half.talker) - base
    assert float(d_full.abs().sum()) > 0 and torch_mod.allclose(d_half, 0.5 * d_full, atol=1e-6)
    # without an explicit value the strength comes from the folder's voice.json (none -> 1.0)
    assert adapter_scale_of(ad) == 1.0
    (ad / "voice.json").write_text(json.dumps({"adapter_scale": 0.5}), encoding="utf-8")
    assert adapter_scale_of(ad) == 0.5
    auto = tiny_model()
    merge_adapter_into_model(auto, ad, spk_id=SPK)
    assert torch_mod.allclose(_q_weight(auto.talker), _q_weight(half.talker), atol=1e-6)
    # the live (unmerged) model at 0.5 gives the same output as the merged one
    live = PeftModel.from_pretrained(tiny_model().talker, str(ad)).eval()
    ast.apply(live, 0.5)
    x = torch_mod.randn(1, 5, 32)
    with torch_mod.no_grad():
        assert torch_mod.allclose(live.base_model.model.model(inputs_embeds=x).last_hidden_state,
                                  half.talker.model(inputs_embeds=x).last_hidden_state, atol=1e-4)


def _train_adapter(tmp_path):
    from core.events import CancelToken
    from core.lora_trainer import load_training_rows, train_on_model
    from tests.test_lora_trainer import _dataset, _encode, _plan, _tok, tiny_model
    out = tmp_path / "output" / "voice"
    train_on_model(tiny_model(), _tok, _encode, load_training_rows(_dataset(tmp_path)), out, _plan(),
                   "Qwen/Qwen3-TTS-12Hz-0.6B-Base", lambda *a: None, CancelToken(), [])
    return out


def test_unmerged_engine_changes_strength_and_switches_checkpoints(tmp_path):
    """The engine methods the preview / checkpoint pick use, on a tiny model (the full engine needs the real base model)."""
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    from core.tts_engine import Qwen3AdapterEngine
    from tests.test_lora_trainer import tiny_model
    ad = _train_adapter(tmp_path)
    eng = Qwen3AdapterEngine.__new__(Qwen3AdapterEngine)
    eng._peft, eng._adapter_no, eng.adapter_scale = PeftModel.from_pretrained(tiny_model().talker, str(ad)), 0, 1.0

    def scalings():
        return {round(v, 6) for m in eng._peft.modules() if isinstance(m, LoraLayer) for v in m.scaling.values()}
    assert scalings() == {4.0}                                         # alpha 16 / r 4
    eng.set_adapter_scale(0.5)
    assert scalings() == {2.0} and eng.adapter_scale == 0.5
    eng.switch_adapter(ad / "checkpoints" / "epoch_01")
    assert list(eng._peft.peft_config) == ["candidate_1"] and scalings() == {2.0}   # one adapter in memory, strength kept
    from safetensors import safe_open
    with safe_open(str(ad / "checkpoints" / "epoch_01" / "adapter_model.safetensors"), "pt") as f:
        key = next(k for k in f.keys() if k.startswith("base_model.model.model.layers.0.self_attn.q_proj.lora_B"))
        want = f.get_tensor(key)
    got = eng._peft.base_model.model.model.layers[0].self_attn.q_proj.lora_B["candidate_1"].weight
    assert torch_mod.allclose(got, want)
    eng._peft = None
    with pytest.raises(RuntimeError):
        eng.set_adapter_scale(0.5)                                      # merged engines cannot change it


# --------------------------------------------------------------------------- quick preview with per-strength samples
from tests.test_preview_check import G, FakeEngine, _dataset as _pdataset, _train, tone  # noqa: E402


class ScalableEngine(FakeEngine):
    """Full strength "babbles" (no stop), 0.35 shifts the pitch, 0.5 is fine."""
    def __init__(self):
        super().__init__()
        self.scale, self.seen = 1.0, []

    def set_adapter_scale(self, s):
        self.scale = s
        self.seen.append(s)

    def synthesize(self, text):
        if self.scale == 1.0:
            return tone(120, 40.0)
        return tone(200 if self.scale < 0.4 else 120, 8.0)


def test_preview_synthesizes_every_strength_and_preselects_the_best(tmp_path):
    from core.asr import FakeASR
    from infra.vram_optimizer import plan_training
    from workers import preview_runner as pr
    engines = []

    def factory(adapter, lang):
        engines.append(ScalableEngine())
        return engines[-1]
    items = pr.run_previews(_pdataset(tmp_path), tmp_path / "out", plan_training(G, 70), compare=False, language="english",
                            gpu=G, train_fn=_train([]), engine_factory=factory, asr=FakeASR([pr.SAMPLE_TEXT["english"]] * 3))
    it = items[0]
    assert engines[0].seen == list(ast.PREVIEW_SCALES) and engines[0].closed
    assert [e["scale"] for e in it.scales] == list(ast.PREVIEW_SCALES) and all(e["wav"].is_file() for e in it.scales)
    assert it.scale == 0.5 and it.check["verdict"] == "good" and it.wav == it.scales[1]["wav"]
    it.use_scale(1.0)
    assert it.scale == 1.0 and "no_stop" in it.check["issues"] and it.wav == it.scales[2]["wav"]
    # an engine without set_adapter_scale (older fakes): one sample as before, no strength list
    (tmp_path / "b").mkdir()
    items = pr.run_previews(_pdataset(tmp_path / "b"), tmp_path / "out2", plan_training(G, 70), compare=False,
                            language="english", gpu=G, train_fn=_train([]), engine_factory=lambda a, l: FakeEngine())
    assert items[0].scale is None and items[0].scales == [] and items[0].wav.name == "preview_A.wav"


def test_best_scale_ties_go_to_the_default():
    from workers.preview_runner import best_scale
    good = {"verdict": "good", "wer": 0.1, "semitones": 0.2}
    assert best_scale([{"scale": s, "check": dict(good)} for s in (0.35, 0.5, 1.0)]) == ast.DEFAULT_SCALE
    assert best_scale([{"scale": 0.35, "check": dict(good, wer=0.0)}, {"scale": 0.5, "check": dict(good, wer=0.2)}]) == 0.35
    assert best_scale([]) is None


# --------------------------------------------------------------------------- UI
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_train_window_strength_selector_and_choice(app, tmp_path):
    from ui.main_window import MainWindow
    from workers.preview_runner import PreviewItem
    w = MainWindow(runner=lambda *a: None, autocheck=False, auto_open_folder=False, gpu_fn=lambda: G)
    played = []
    w.previewer = SimpleNamespace(play=lambda p: played.append(p))
    good = {"verdict": "good", "wer": 0.1, "semitones": 0.4, "issues": []}
    bad = {"verdict": "bad", "wer": 0.9, "semitones": 0.4, "issues": ["no_stop"]}
    scales = [{"scale": 0.35, "wav": tmp_path / "a35.wav", "seconds": 9.0, "check": good},
              {"scale": 0.5, "wav": tmp_path / "a50.wav", "seconds": 9.0, "check": good},
              {"scale": 1.0, "wav": tmp_path / "a100.wav", "seconds": 30.0, "check": bad}]
    item = PreviewItem("A", "A", 3, 32, 128, 1e-6, 4, tmp_path / "a50.wav", 9.0, 40.0, good, 12, scale=0.5, scales=scales)
    w.show_previews([item])
    row = w.preview_rows[0][0]
    from PySide6.QtWidgets import QComboBox, QLabel
    combo = row.findChild(QComboBox)
    assert combo.count() == 3 and combo.currentData() == 0.5 and "1.00" in combo.itemText(2)
    combo.setCurrentIndex(2)
    w.preview_rows[0][1].click()
    assert played == [tmp_path / "a100.wav"] and item.scale == 1.0
    assert any(" 30 " in l.text() for l in row.findChildren(QLabel))      # the checks shown follow the selection
    w.preview_rows[0][2].click()
    assert w.preview_scale == 1.0 and w._task_extras()["adapter_scale"] == 1.0 and "1.00" in w.lbl_status.text()
    (tmp_path / "n.wav").write_bytes(b"x")
    w.set_audio(tmp_path / "n.wav")
    assert w._task_extras()["adapter_scale"] is None                   # a new recording forgets the previous choice


def test_voice_properties_edit_the_strength(app, tmp_path):
    from core.voice_library import VoiceLibrary
    from ui.voices_window import VoiceEditDialog
    lib = VoiceLibrary(tmp_path / "lib")
    rec = lib.add_from_adapter(make_adapter(tmp_path / "a"))
    dlg = VoiceEditDialog(rec)
    assert dlg.sp_strength.value() == 1.0                               # older voices: as trained
    dlg.sp_strength.setValue(0.45)
    vals = dlg.values()
    assert vals["adapter_scale"] == 0.45
    assert lib.update(rec.id, **vals).adapter_scale == 0.45
