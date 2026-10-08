"""Voice type in the names of trained voices (folder / package) and in voice.json; pitch suggestion; UI selector and cards."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core import i18n, voice_info
from core import voice_type as vt
from core.voice_library import VoiceLibrary
from tests.test_runner_cli import NotDueUpdater, TrueRateAligner, _inputs
from tests.test_studio import add_voice, app, lib, wait_for  # noqa: F401  (fixtures)
from tests.test_voice_info import _fake_train
from workers.pipeline_runner import KIND_LORA, TaskRequest, run_task


def test_the_naming_scheme():
    f = voice_info.with_type_suffix
    assert f("anna", "male") == "anna_male" and f("anna", "female") == "anna_female"
    assert f("anna", "") == "anna_unspecified" and f("anna", None) == "anna_unspecified" and f("anna", "robot") == "anna_unspecified"
    assert f("anna", "child") == "anna_child" and f("anna", "other") == "anna_other"
    assert f("anna_male", "female") == "anna_female" and f("anna_male", "male") == "anna_male"     # never doubled
    assert f("Anna Female", "male") == "Anna_male" and f("", "male") == "voice_male"
    assert f("male", "male") == "male_male"                                                         # a voice called "male" keeps its name
    assert voice_info.type_suffix("FEMALE") == "female" and voice_info.type_suffix("x") == "unspecified"


def test_library_ids_carry_the_type_and_stay_unique(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    assert lib.unique_id("Anna", "male") == "anna_male" and lib.unique_id("Anna") == "anna"
    (lib.root / "anna_male").mkdir(parents=True)
    assert lib.unique_id("Anna", "male") == "anna_male-2" and lib.unique_id("Anna", "female") == "anna_female"


@pytest.mark.parametrize("vtype", ["male", "female", ""])
def test_training_names_the_folders_and_voice_json_with_the_type(tmp_path, monkeypatch, vtype):
    wav, txt = _inputs(tmp_path)
    import core.lora_trainer as lt
    monkeypatch.setattr(lt, "train_lora_from_dataset", _fake_train())
    lib = VoiceLibrary(tmp_path / "lib")
    res = run_task(TaskRequest(KIND_LORA, wav, txt, out_root=tmp_path / "out", voice_type=vtype),
                   updater=NotDueUpdater(), aligner_factory=TrueRateAligner, voice_library=lib)
    suffix = vtype or "unspecified"
    info = voice_info.read_voice_json(res.adapter_path)
    assert res.adapter_path.name.endswith("_" + suffix) and info["voice_type"] == vtype
    assert info["name"] + "_" + suffix == res.adapter_path.name                 # the displayed name stays clean
    assert res.voice_id.endswith("_" + suffix) and lib.get(res.voice_id).info["voice_type"] == vtype
    assert lib.get(res.voice_id).name == info["name"]


def test_older_voices_without_a_type_still_load_and_show_as_unspecified(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    d = tmp_path / "old"
    d.mkdir()
    for n in ("adapter_model.safetensors", "adapter_config.json", "ref_sample.wav"):
        (d / n).write_bytes(b"x")
    (d / "voice.json").write_text(json.dumps({"schema": 1, "name": "Old", "language": "english"}), encoding="utf-8")
    rec = lib.import_folder(d)
    assert rec.id == "old" and rec.info["voice_type"] == ""                    # plain import keeps the old id scheme
    from ui.voices_window import meta_line
    i18n.set_language("en")
    assert "Voice type: not specified" in meta_line(rec)
    i18n.set_language("ru")
    assert "Тип голоса: не указан" in meta_line(rec)
    i18n.set_language("de")
    assert "Stimmtyp: nicht angegeben" in meta_line(rec)
    i18n.set_language("en")


def test_voice_package_zip_carries_the_type_in_its_name_and_voice_json(tmp_path):
    from tools import make_voice_package as mk
    adapter = tmp_path / "ad"
    adapter.mkdir()
    for n in ("adapter_model.safetensors", "adapter_config.json"):
        (adapter / n).write_bytes(b"x")
    spec = {"id": "anna", "name": "Anna", "license": "CC0-1.0", "voice_type": "female", "language": "English"}
    entry = mk.build(adapter, spec, tmp_path / "out", "https://example.org/anna_female.zip")
    z = tmp_path / "out" / "anna_female.zip"
    assert z.is_file() and entry["voice_type"] == "female" and (tmp_path / "out" / "anna_female.index-entry.json").is_file()
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        assert all(n.startswith("anna_female/") for n in names)
        assert json.loads(zf.read("anna_female/voice.json"))["voice_type"] == "female"
    rec = VoiceLibrary(tmp_path / "lib").import_zip(z)                          # the package imports back, type intact
    assert rec.info["voice_type"] == "female"
    plain = mk.build(adapter, spec, tmp_path / "out2", "https://example.org/anna.zip", typed_names=False)
    assert (tmp_path / "out2" / "anna.zip").is_file() and plain["voice_type"] == "female"
    mk.build(adapter, {**spec, "id": "bob", "voice_type": ""}, tmp_path / "out3", "u")
    assert (tmp_path / "out3" / "bob_unspecified.zip").is_file()


def _tone(path, f0, seconds=3, sr=16000):
    t = np.arange(int(sr * seconds)) / sr
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 5)) * 0.1
    sf.write(str(path), x.astype("float32"), sr)


def test_pitch_suggestion(tmp_path):
    assert vt.type_from_f0(110) == "male" and vt.type_from_f0(220) == "female"
    assert vt.type_from_f0(170) == "" and vt.type_from_f0(0) == "" and vt.type_from_f0(500) == ""
    _tone(tmp_path / "m.wav", 110)
    _tone(tmp_path / "f.wav", 230)
    assert vt.suggest_voice_type(tmp_path / "m.wav")[0] == "male"
    assert vt.suggest_voice_type(tmp_path / "f.wav")[0] == "female"
    assert vt.suggest_voice_type(tmp_path / "missing.wav") == ("", 0.0)


def test_train_window_suggests_but_never_overrides_the_users_choice(app, tmp_path):
    from tests.test_ui import make_window
    i18n.set_language("en")
    w = make_window(lambda *a: None)
    assert w.cmb_voice_type.currentData() == "" and "not specified" in w.cmb_voice_type.currentText()
    _tone(tmp_path / "m.wav", 110)
    w.set_audio(tmp_path / "m.wav")
    assert wait_for(lambda: w.cmb_voice_type.currentData() == "male", timeout=20)
    assert "110" in w.lbl_voice_type_hint.text() or "Hz" in w.lbl_voice_type_hint.text()
    w.cmb_voice_type.setCurrentIndex(w.cmb_voice_type.findData("female"))
    w._voice_type_user_set = True
    w._apply_voice_type_suggestion("male", 110.0)
    assert w.cmb_voice_type.currentData() == "female"                           # the user's choice wins
    w.close()


def test_selector_is_localized(app):
    from tests.test_ui import make_window
    seen = {}
    for lang, word in (("en", "Male"), ("ru", "Мужской"), ("de", "Männlich")):
        i18n.set_language(lang)
        w = make_window(lambda *a: None)
        texts = [w.cmb_voice_type.itemText(i) for i in range(w.cmb_voice_type.count())]
        seen[lang] = texts
        assert any(word.lower() in t.lower() for t in texts), (lang, texts)
        w.close()
    i18n.set_language("en")
