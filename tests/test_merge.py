"""Универсальная (merged) модель: слияние адаптера, формат папки, диск, запуск через runner и кнопка в UI."""
import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

from core.errors import ExportError
from core.events import CancelToken, Stage
from tests.test_lora_trainer import _dataset, _encode, _plan, _tok, tiny_model

REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
SPK = 100   # < vocab_size(128) крошечной модели


@pytest.fixture()
def adapter(tmp_path):
    from core.lora_trainer import load_training_rows, train_on_model
    d = _dataset(tmp_path)
    out = tmp_path / "output" / "Мой голос"
    train_on_model(tiny_model(), _tok, _encode, load_training_rows(d), out, _plan(), REPO,
                   lambda *a: None, CancelToken(), [])
    return out


@pytest.fixture()
def base_dir(tmp_path):
    d = tmp_path / "base"
    tiny_model().save_pretrained(d)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    cfg["speaker_encoder_config"].pop("model_type", None)       # у настоящих config.json этого ключа нет
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (d / "speech_tokenizer").mkdir()
    (d / "speech_tokenizer" / "config.json").write_text("{}", encoding="utf-8")
    (d / "speech_tokenizer" / "model.safetensors").write_bytes(b"0")
    (d / "generation_config.json").write_text("{}", encoding="utf-8")
    (d / "README.md").write_text("x", encoding="utf-8")
    return d


def _load(d):
    # Загружаем веса без речевого токенайзера (в тесте он - заглушка): родительский from_pretrained.
    from qwen_tts.core.models import Qwen3TTSForConditionalGeneration as M
    return super(M, M).from_pretrained(str(d), dtype=torch.float32)


def test_merged_talker_matches_adapter_model(adapter):
    from peft import PeftModel
    from core.model_export import merge_adapter_into_model
    from safetensors import safe_open
    with safe_open(str(adapter / "adapter_model.safetensors"), "pt") as f:
        assert any(float(f.get_tensor(k).abs().sum()) > 0 for k in f.keys() if "lora_B" in k)  # адаптер обучен
    x = torch.randn(1, 5, 32)
    ref = PeftModel.from_pretrained(tiny_model().talker, str(adapter)).eval()
    with torch.no_grad():
        want = ref.base_model.model.model(inputs_embeds=x).last_hidden_state
    merged = tiny_model()
    state, spk = merge_adapter_into_model(merged, adapter, spk_id=SPK)
    with torch.no_grad():
        got = merged.talker.model(inputs_embeds=x).last_hidden_state
    assert torch.allclose(want, got, atol=1e-4)
    base = tiny_model().talker
    with torch.no_grad():
        plain = base.model(inputs_embeds=x).last_hidden_state
    assert not torch.allclose(plain, got, atol=1e-4)           # слияние реально изменило модель
    assert not any("lora_" in k or k.startswith("speaker_encoder") for k in state)
    assert torch.allclose(state["talker.model.codec_embedding.weight"][SPK], spk[0])


def test_export_folder_format_and_reload(adapter, base_dir, tmp_path):
    from safetensors import safe_open
    from core.model_export import export_merged_model
    out = adapter / "merged_model"
    prog = []
    res = export_merged_model(adapter, out, base_dir=base_dir, voice_name="Мой голос", load_model=_load,
                              spk_id=SPK, skip_disk_check=True, progress=lambda s, f, m: prog.append((s, f, m)))
    assert res == out and not (adapter / "merged_model.partial").exists()
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["tts_model_type"] == "custom_voice"
    assert cfg["talker_config"]["spk_id"] == {"мой_голос": SPK}
    assert cfg["talker_config"]["spk_is_dialect"] == {"мой_голос": False}
    with safe_open(str(out / "model.safetensors"), "pt") as f:
        keys = list(f.keys())
        emb = f.get_tensor("talker.model.codec_embedding.weight")[SPK].float()
        sp = None
    assert keys and not any("lora_" in k or k.startswith("speaker_encoder") for k in keys)
    with safe_open(str(out / "speaker_embedding.safetensors"), "pt") as f:
        sp = f.get_tensor("speaker_embedding")[0].float()
    assert torch.allclose(emb, sp, atol=2e-2)                   # bf16
    assert (out / "speech_tokenizer" / "config.json").exists() and (out / "generation_config.json").exists()
    assert not (out / "README.md").exists()
    for n in ("ref_sample.wav", "ref_text.txt", "voxprint_voice.json", "USAGE.txt"):
        assert (out / n).exists(), n
    meta = json.loads((out / "voxprint_voice.json").read_text(encoding="utf-8"))
    assert meta["speaker"] == "мой_голос" and meta["base_model"] == REPO
    assert "generate_custom_voice" in (out / "USAGE.txt").read_text(encoding="utf-8")
    assert _load(out).config.tts_model_type == "custom_voice"   # папка снова грузится как обычная модель
    assert any(s is Stage.MODEL for s, _, _ in prog) and prog[-1][:2] == (Stage.SAVE, 1.0)
    assert (adapter / "adapter_model.safetensors").exists()      # исходный адаптер не тронут


def test_export_cancel_leaves_no_partial(adapter, base_dir):
    from core.errors import CancelledByUser
    from core.model_export import export_merged_model
    tok = CancelToken()

    def progress(stage, f, m):
        if stage is Stage.SAVE and f >= 0.4:
            tok.cancel()
    with pytest.raises(CancelledByUser):
        export_merged_model(adapter, adapter / "merged_model", base_dir=base_dir, load_model=_load, spk_id=SPK,
                            skip_disk_check=True, progress=progress, cancel=tok)
    assert not (adapter / "merged_model").exists() and not (adapter / "merged_model.partial").exists()


def test_export_disk_space_error(adapter, base_dir, monkeypatch):
    from core import model_export as mx
    need, repo = mx.required_free_gb(adapter)
    assert repo == REPO and need > 1
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10**12, 10**12 - 10**9, 10**9))
    with pytest.raises(ExportError) as ei:
        mx.export_merged_model(adapter, adapter / "merged_model", base_dir=base_dir, load_model=_load)
    assert ei.value.kind == "export" and "ГБ" in ei.value.user_message
    assert not (adapter / "merged_model").exists()


def test_export_without_adapter(tmp_path):
    from core.model_export import read_adapter_meta
    with pytest.raises(ExportError):
        read_adapter_meta(tmp_path)


def test_runner_merge_kind_and_last_adapter(adapter, monkeypatch, tmp_path):
    import core.model_export as mx
    from workers import pipeline_runner as pr
    assert pr.last_adapter() is None
    pr.remember_adapter(adapter, "Мой голос")
    assert pr.last_adapter() == adapter
    assert pr.plan_for(pr.KIND_MERGE) == [Stage.MODEL, Stage.SAVE]
    seen = {}

    def fake_export(adapter_dir, out_dir, progress=None, cancel=None, voice_name=None, **kw):
        seen.update(a=adapter_dir, o=out_dir, v=voice_name)
        out_dir.mkdir()
        return out_dir

    monkeypatch.setattr(mx, "export_merged_model", fake_export)
    res = pr.run_task(pr.TaskRequest(pr.KIND_MERGE, adapter_dir=adapter))
    assert seen == {"a": adapter, "o": adapter / "merged_model", "v": "Мой голос"}
    assert res.merged_path == adapter / "merged_model" and res.open_dir == res.merged_path
    assert res.speaker == "мой_голос"
    shutil.rmtree(adapter)
    assert pr.last_adapter() is None                              # папку удалили - кнопка снова неактивна
    with pytest.raises(ExportError):
        pr.run_task(pr.TaskRequest(pr.KIND_MERGE))


def test_voice_name_and_output_folder(tmp_path):
    from workers.pipeline_runner import KIND_LORA, TaskRequest
    r = TaskRequest(KIND_LORA, tmp_path / "Мой:голос*.wav", tmp_path / "t.txt")
    assert r.voice_name() == "Мой_голос_"


# ------------------------------------------------------------------ UI
@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_merge_button_state_hints_and_flow(app, adapter, monkeypatch, tmp_path):
    from PySide6.QtCore import QCoreApplication
    import time
    from core import i18n
    from ui import main_window as mw
    from workers import pipeline_runner as pr

    got = {}

    def runner(req, progress, cancel):
        got["req"] = req
        for st in pr.plan_for(req.kind):
            progress(st, 1.0, "x")
        out = req.adapter_dir / "merged_model"
        return pr.TaskResult(req.kind, out, req.adapter_dir, req.adapter_dir, merged_path=out, speaker="мой_голос")

    w = mw.MainWindow(runner=runner, autocheck=False, auto_open_folder=False)
    w.show()
    assert w.btn_merge.text() == "Собрать универсальную модель (~4 ГБ)"
    assert not w.btn_merge.isEnabled() and w.btn_merge.toolTip()
    assert "десятки МБ" in w.lbl_hint_lora.text() and "Alexandria" in w.lbl_hint_lora.text()
    assert "4 ГБ" in w.lbl_hint_merge.text() and "Qwen3-TTS" in w.lbl_hint_merge.text()
    pr.remember_adapter(adapter, "Мой голос")
    w._refresh_buttons()
    assert w.btn_merge.isEnabled()
    monkeypatch.setattr(mw.model_export, "required_free_gb", lambda a: (1.0, REPO))
    assert w.start_merge() is True
    deadline = time.time() + 20
    while time.time() < deadline and w.lbl_ready.isHidden():
        QCoreApplication.processEvents()
        time.sleep(0.02)
    assert got["req"].kind == pr.KIND_MERGE and got["req"].adapter_dir == adapter
    assert not w.lbl_ready.isHidden() and "merged_model" in w.lbl_status.text()
    assert "мой_голос" in w.lbl_status.text() and "USAGE.txt" in w.lbl_status.text()
    w.close()


def test_merge_button_blocks_on_low_disk(app, adapter, monkeypatch):
    from ui import main_window as mw
    from workers import pipeline_runner as pr
    pr.remember_adapter(adapter, "v")
    called = []
    w = mw.MainWindow(runner=lambda *a: called.append(a), autocheck=False, auto_open_folder=False)
    monkeypatch.setattr(mw.model_export, "required_free_gb", lambda a: (10**6, REPO))
    assert w.start_merge() is False and not called
    assert "ГБ" in w.last_error_text
    w.close()
