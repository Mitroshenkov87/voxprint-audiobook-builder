"""voice.json (schema 3): pure helpers, licence logic, old-schema migration and the pipeline integration."""
import json
from datetime import datetime, timezone

import pytest

from core import voice_info
from workers.pipeline_runner import KIND_LORA, TaskRequest, run_task
from tests.test_runner_cli import NotDueUpdater, TrueRateAligner, _inputs


def test_build_voice_info_fields_and_normalization():
    now = datetime(2026, 10, 3, 9, 30, 5, tzinfo=timezone.utc)
    info = voice_info.build_voice_info("Anna", "english", 154.94, 15, "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                       voice_type=" Female ", description="  warm\n alto  ", now=now,
                                       voice_id="anna", author="Anna K.", license="CC-BY-4.0")
    assert info == {"schema": 3, "id": "anna", "name": "Anna", "language": "en", "created": "2026-10-03T09:30:05Z",
                    "duration": 154.9, "epochs": 15, "base_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "author": "Anna K.", "speaker": "", "prepared_by": "", "organization": "", "project_url": "",
                    "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "gender": "female", "age_group": "", "voice_type": "female", "description": "warm alto",
                    "commercial_use": True}
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
    assert new["schema"] == voice_info.VOICE_SCHEMA and new["name"] == "Old" and new["duration"] == 120.5 and new["id"] == "old"
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
    assert res.adapter_path.name == info["name"] + "_male" and info["language"] and info["schema"] == voice_info.VOICE_SCHEMA
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


# ----------------------------------------------------------------------------- schema 3: who / where, gender, age, codes
SCHEMA2 = {"schema": 2, "id": "anna", "name": "Anna", "language": "Russian", "created": "2026-10-03T12:30:05+03:00",
           "duration": 60, "epochs": 3, "base_model": "Qwen/B", "author": "A", "license": "CC0-1.0", "license_url": "",
           "voice_type": "female", "description": "d", "commercial_use": True}


@pytest.mark.parametrize("old_type, gender, age, derived", [
    ("male", "male", "", "male"), ("female", "female", "", "female"), ("child", "", "child", "child"),
    ("other", "", "", "other"), ("", "", "", ""), ("robot", "", "", ""),
])
def test_schema2_voice_type_migrates_to_gender_and_age(old_type, gender, age, derived):
    new = voice_info.normalize_info({**SCHEMA2, "voice_type": old_type})
    assert (new["gender"], new["age_group"], new["voice_type"]) == (gender, age, derived)
    assert voice_info.type_suffix(new["voice_type"]) == voice_info.type_suffix(old_type)    # folder names unchanged


def test_schema2_file_is_upgraded_with_codes_and_utc_time():
    new = voice_info.normalize_info(SCHEMA2)
    assert new["schema"] == 3 and new["language"] == "ru" and new["created"] == "2026-10-03T09:30:05Z"
    assert all(new[k] == "" for k in ("speaker", "prepared_by", "organization", "project_url"))
    for raw, code in (("russian", "ru"), ("English", "en"), ("Deutsch", "de"), ("ru_RU", "ru-RU"), ("pt-br", "pt-BR"),
                      ("Cantonese", "yue"), ("", ""), ("auto", "")):
        assert voice_info.normalize_info({"language": raw})["language"] == code, raw


def test_gender_and_age_win_over_voice_type():
    info = voice_info.build_voice_info("X", "ru", 1, 1, "b", voice_type="male", gender="female", age_group="elderly")
    assert (info["gender"], info["age_group"], info["voice_type"]) == ("female", "elderly", "female")
    kid = voice_info.build_voice_info("X", "ru", 1, 1, "b", gender="male", age_group="child")
    assert kid["voice_type"] == "child"
    odd = voice_info.normalize_info({"gender": "Robot", "age_group": "teen", "voice_type": "male"})
    assert (odd["gender"], odd["age_group"], odd["voice_type"]) == ("male", "", "male")   # invalid values: migrate the type


def test_new_text_fields_are_cleaned_and_keep_unicode():
    info = voice_info.normalize_info({"speaker": "  Анна\n Иванова ", "prepared_by": "Jürgen  Müller",
                                      "organization": "株式会社 " + "x" * 300, "project_url": "  https://example.org/p?q=1 "})
    assert info["speaker"] == "Анна Иванова" and info["prepared_by"] == "Jürgen Müller"
    assert info["organization"].startswith("株式会社 ") and len(info["organization"]) == voice_info.MAX_ORGANIZATION_CHARS
    assert info["project_url"] == "https://example.org/p?q=1"


@pytest.mark.parametrize("url", ["javascript:alert(1)", "ftp://example.org", "file:///etc/passwd", "example.org",
                                 "https://exa mple.org", "https://" + "x" * 600, "data:text/html,hi"])
def test_project_url_is_sanitized_like_the_license_url(url):
    assert voice_info.normalize_info({"project_url": url})["project_url"] == ""
    assert voice_info.clean_project_url(url) == voice_info.clean_license_url(url) == ""


def test_schema3_round_trip_is_stable(tmp_path):
    info = voice_info.build_voice_info("Анна", "ru-RU", 61.25, 4, "Qwen/B", voice_id="anna", speaker="Анна",
                                       prepared_by="Студия", organization="Voxprint", project_url="https://example.org",
                                       gender="female", age_group="young", license="CC-BY-4.0")
    voice_info.write_voice_json(tmp_path, info)
    raw = (tmp_path / "voice.json").read_text(encoding="utf-8")
    assert "Анна" in raw                                          # stored as UTF-8 text, not \u escapes
    back = voice_info.read_voice_json(tmp_path)
    assert back == info and voice_info.normalize_info(back) == info
    assert info["language"] == "ru-RU" and info["created"].endswith("Z")


def test_library_update_sets_and_clears_gender_and_age(tmp_path):
    from core.voice_library import VoiceLibrary
    src = tmp_path / "src"
    src.mkdir()
    (src / "adapter_model.safetensors").write_bytes(b"w")
    (src / "adapter_config.json").write_text("{}", encoding="utf-8")
    voice_info.write_voice_json(src, {**SCHEMA2, "voice_type": "male"})            # an old file
    lib = VoiceLibrary(tmp_path / "lib")
    rec = lib.add_from_adapter(src)
    assert rec.info["gender"] == "male" and rec.language == "ru"
    rec = lib.update(rec.id, gender="", age_group="elderly", speaker="Борис", project_url="javascript:x")
    assert (rec.info["gender"], rec.info["age_group"], rec.info["voice_type"]) == ("", "elderly", "")
    assert rec.info["speaker"] == "Борис" and rec.info["project_url"] == ""
    rec = lib.update(rec.id, voice_type="child")                                   # an old-style caller
    assert (rec.info["gender"], rec.info["age_group"], rec.info["voice_type"]) == ("", "child", "child")
    assert voice_info.read_voice_json(rec.path)["schema"] == 3
