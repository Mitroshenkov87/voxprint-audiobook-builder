"""voice.json (schema 2): pure helpers, licence logic, old-schema migration and the pipeline integration."""
import json
from datetime import datetime, timezone

from core import voice_info
from workers.pipeline_runner import KIND_LORA, TaskRequest, run_task
from tests.test_runner_cli import NotDueUpdater, TrueRateAligner, _inputs


def test_build_voice_info_fields_and_normalization():
    now = datetime(2026, 10, 3, 9, 30, 5, tzinfo=timezone.utc)
    info = voice_info.build_voice_info("Anna", "english", 154.94, 15, "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                       voice_type=" Female ", description="  warm\n alto  ", now=now,
                                       voice_id="anna", author="Anna K.", license="CC-BY-4.0")
    assert info == {"schema": 2, "id": "anna", "name": "Anna", "language": "english", "created": "2026-10-03T09:30:05Z",
                    "duration": 154.9, "epochs": 15, "base_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "author": "Anna K.", "license": "CC-BY-4.0",
                    "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "voice_type": "female", "description": "warm alto", "commercial_use": True}
    assert voice_info.normalize_voice_type("robot") == "" and voice_info.normalize_voice_type(None) == ""
    assert len(voice_info.clean_description("x" * 900)) == voice_info.MAX_DESCRIPTION_CHARS


def test_default_license_is_personal_only():
    info = voice_info.build_voice_info("Me", "russian", 10, 1, "base")
    assert info["license"] == "custom/personal-only" and info["commercial_use"] is False and info["license_url"] == ""


def test_commercial_use_is_derived_from_the_license():
    ok = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0"}
    for lic in voice_info.LICENSES:
        assert voice_info.license_allows_commercial(lic) == (lic in ok), lic
    for unknown in ("", None, "MIT", "custom/anything", "cc-by-4.0 "):
        assert voice_info.license_allows_commercial(unknown) is False       # unknown = personal only


def test_normalize_recomputes_commercial_use_and_cleans_urls():
    forged = {"name": "X", "license": "CC-BY-NC-4.0", "commercial_use": True, "license_url": "javascript:alert(1)"}
    info = voice_info.normalize_info(forged, fallback_id="x")
    assert info["commercial_use"] is False
    assert info["license_url"] == "https://creativecommons.org/licenses/by-nc/4.0/"      # unsafe URL replaced
    custom = voice_info.normalize_info({"license": "custom/personal-only", "license_url": "https://example.org/terms"})
    assert custom["license_url"] == "https://example.org/terms" and custom["commercial_use"] is False


def test_schema1_files_are_upgraded():
    old = {"schema": 1, "voice_name": "Old", "language": "russian", "created": "2026-01-01T00:00:00Z",
           "speech_seconds": 120.5, "epochs": 5, "base_model": "b", "voice_type": "male", "description": "d"}
    new = voice_info.normalize_info(old, fallback_id="old")
    assert new["schema"] == 2 and new["name"] == "Old" and new["duration"] == 120.5 and new["id"] == "old"
    assert new["license"] == voice_info.DEFAULT_LICENSE and new["commercial_use"] is False and new["author"] == ""
    assert voice_info.normalize_info({"epochs": "many", "duration": "x"})["epochs"] == 0


def test_write_and_read_roundtrip_unicode(tmp_path):
    info = voice_info.build_voice_info("Голос", "russian", 60, 3, "base", description="тёплый голос")
    path = voice_info.write_voice_json(tmp_path / "new_dir", info)
    assert path.name == "voice.json" and json.loads(path.read_text(encoding="utf-8")) == info
    assert voice_info.read_voice_json(tmp_path / "new_dir") == info
    assert voice_info.read_voice_json(tmp_path / "missing") is None
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "voice.json").write_text("{not json", encoding="utf-8")
    assert voice_info.read_voice_json(tmp_path / "bad") is None


def _fake_train(epochs=7):
    def fake_train(dataset_dir, output_dir, progress, cancel, force_cpu, language=None, warnings_out=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "training_meta.json").write_text(
            json.dumps({"epochs": epochs, "model_name": "Qwen/Base", "ref_sample_text": "hello"}), encoding="utf-8")
        (output_dir / "adapter_model.safetensors").write_bytes(b"x")
        (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (output_dir / "ref_sample.wav").write_bytes(b"RIFF")
        return output_dir
    return fake_train


def test_run_task_writes_voice_json_and_registers_the_voice(tmp_path, monkeypatch):
    from core.voice_library import VoiceLibrary

    wav, txt = _inputs(tmp_path)
    import core.lora_trainer as lt
    monkeypatch.setattr(lt, "train_lora_from_dataset", _fake_train())
    lib = VoiceLibrary(tmp_path / "lib")
    res = run_task(TaskRequest(KIND_LORA, wav, txt, out_root=tmp_path / "out", voice_type="male",
                               voice_description="radio voice"),
                   updater=NotDueUpdater(), aligner_factory=TrueRateAligner, voice_library=lib)
    info = voice_info.read_voice_json(res.adapter_path)
    assert info["name"] == res.adapter_path.name and info["language"] and info["schema"] == 2
    assert info["epochs"] == 7 and info["base_model"] == "Qwen/Base"
    assert info["voice_type"] == "male" and info["description"] == "radio voice"
    assert info["duration"] > 0 and info["created"].endswith("Z")
    # the new voice is in the library automatically, with its own voice.json and the safe default licence
    assert res.voice_id and [v.id for v in lib.list_voices()] == [res.voice_id]
    rec = lib.get(res.voice_id)
    assert rec.commercial_use is False and rec.info["voice_type"] == "male" and rec.base_model == "Qwen/Base"
    assert (rec.path / "ref_sample.wav").exists() and rec.ref_text == "hello"


def test_voice_json_optional_fields_default_to_empty_and_missing_adapter_only_warns(tmp_path, monkeypatch):
    from core.voice_library import VoiceLibrary

    wav, txt = _inputs(tmp_path)
    import core.lora_trainer as lt
    monkeypatch.setattr(lt, "train_lora_from_dataset", lambda d, o, *a, **k: o)   # no training_meta.json, no adapter files
    lib = VoiceLibrary(tmp_path / "lib2")
    res = run_task(TaskRequest(KIND_LORA, wav, txt, out_root=tmp_path / "o2"),
                   updater=NotDueUpdater(), aligner_factory=TrueRateAligner, voice_library=lib)
    info = voice_info.read_voice_json(res.adapter_path)
    assert info["voice_type"] == "" and info["description"] == "" and info["epochs"] == 0
    assert res.voice_id == "" and lib.is_empty() and res.warnings       # not registered, but the run did not fail
