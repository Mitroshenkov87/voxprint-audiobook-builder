"""voice.json: pure helpers and the pipeline integration (written next to the adapter after training)."""
import json
from datetime import datetime, timezone

from core import voice_info
from workers.pipeline_runner import KIND_LORA, TaskRequest, run_task
from tests.test_runner_cli import NotDueUpdater, TrueRateAligner, _inputs


def test_build_voice_info_fields_and_normalization():
    now = datetime(2026, 10, 3, 9, 30, 5, tzinfo=timezone.utc)
    info = voice_info.build_voice_info("Anna", "english", 154.94, 15, "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                       voice_type=" Female ", description="  warm\n alto  ", now=now)
    assert info == {"schema": 1, "voice_name": "Anna", "language": "english", "created": "2026-10-03T09:30:05Z",
                    "speech_seconds": 154.9, "epochs": 15, "base_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "voice_type": "female", "description": "warm alto"}
    assert voice_info.normalize_voice_type("robot") == "" and voice_info.normalize_voice_type(None) == ""
    assert len(voice_info.clean_description("x" * 900)) == voice_info.MAX_DESCRIPTION_CHARS


def test_write_and_read_roundtrip_unicode(tmp_path):
    info = voice_info.build_voice_info("Голос", "russian", 60, 3, "base", description="тёплый голос")
    path = voice_info.write_voice_json(tmp_path / "new_dir", info)
    assert path.name == "voice.json" and json.loads(path.read_text(encoding="utf-8")) == info
    assert voice_info.read_voice_json(tmp_path / "new_dir") == info
    assert voice_info.read_voice_json(tmp_path / "missing") is None
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "voice.json").write_text("{not json", encoding="utf-8")
    assert voice_info.read_voice_json(tmp_path / "bad") is None


def test_run_task_writes_voice_json_next_to_adapter(tmp_path, monkeypatch):
    wav, txt = _inputs(tmp_path)

    def fake_train(dataset_dir, output_dir, progress, cancel, force_cpu, language=None, warnings_out=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "training_meta.json").write_text(
            json.dumps({"epochs": 7, "model_name": "Qwen/Base"}), encoding="utf-8")
        return output_dir

    import core.lora_trainer as lt
    monkeypatch.setattr(lt, "train_lora_from_dataset", fake_train)
    res = run_task(TaskRequest(KIND_LORA, wav, txt, out_root=tmp_path / "out", voice_type="male",
                               voice_description="radio voice"),
                   updater=NotDueUpdater(), aligner_factory=TrueRateAligner)
    info = voice_info.read_voice_json(res.adapter_path)
    assert info["voice_name"] == res.adapter_path.name and info["language"]
    assert info["epochs"] == 7 and info["base_model"] == "Qwen/Base"
    assert info["voice_type"] == "male" and info["description"] == "radio voice"
    assert info["speech_seconds"] > 0 and info["created"].endswith("Z")


def test_voice_json_optional_fields_default_to_empty(tmp_path, monkeypatch):
    wav, txt = _inputs(tmp_path)
    import core.lora_trainer as lt
    monkeypatch.setattr(lt, "train_lora_from_dataset", lambda d, o, *a, **k: o)   # no training_meta.json at all
    res = run_task(TaskRequest(KIND_LORA, wav, txt, out_root=tmp_path / "o2"),
                   updater=NotDueUpdater(), aligner_factory=TrueRateAligner)
    info = voice_info.read_voice_json(res.adapter_path)
    assert info["voice_type"] == "" and info["description"] == "" and info["epochs"] == 0
