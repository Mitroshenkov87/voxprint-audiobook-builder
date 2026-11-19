"""Voice library: registry, import (folder/zip, hardened), update, delete."""
import json
import zipfile

import pytest

from core import voice_info
from core.errors import VoiceLibraryError
from core.voice_library import ALLOWED_FILES, VoiceLibrary, adapter_complete, slugify


def make_adapter(folder, name="Anna", license="custom/personal-only", extra=None, with_voice_json=True):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "adapter_model.safetensors").write_bytes(b"weights")
    (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
    (folder / "ref_sample.wav").write_bytes(b"RIFFxxxx")
    (folder / "training_meta.json").write_text(json.dumps({"ref_sample_text": "Hello there", "epochs": 3}), encoding="utf-8")
    if with_voice_json:
        voice_info.write_voice_json(folder, voice_info.build_voice_info(
            name, "english", 600, 3, "Qwen/Base", license=license, author="Me"))
    for n, data in (extra or {}).items():
        (folder / n).write_bytes(data)
    return folder


def test_slugify_and_unique_ids(tmp_path):
    assert slugify("  Анна / Anna!! ") == "анна-anna" and slugify("???") == "voice"
    lib = VoiceLibrary(tmp_path)
    a = lib.add_from_adapter(make_adapter(tmp_path / "src1"))
    b = lib.add_from_adapter(make_adapter(tmp_path / "src2"))
    assert (a.id, b.id) == ("anna", "anna-2")


def test_add_list_get_and_fields(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    assert lib.is_empty()
    rec = lib.add_from_adapter(make_adapter(tmp_path / "src", license="CC-BY-4.0"))
    assert rec.path.parent == tmp_path / "lib" and adapter_complete(rec.path)
    got = lib.get(rec.id)
    assert got.name == "Anna" and got.license == "CC-BY-4.0" and got.commercial_use is True
    assert got.info["id"] == rec.id and got.language == "en" and got.ref_text == "Hello there"
    assert got.preview_path == rec.path / "ref_sample.wav" and got.base_model == "Qwen/Base"
    assert [v.id for v in lib.list_voices()] == [rec.id] and not lib.is_empty()
    assert lib.get("missing") is None and lib.get("../x") is None and lib.get("") is None


def test_incomplete_adapter_is_rejected_and_junk_folders_ignored(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    (tmp_path / "bad").mkdir()
    with pytest.raises(VoiceLibraryError):
        lib.add_from_adapter(tmp_path / "bad")
    (lib.root / "not-a-voice").mkdir()
    (lib.root / ".hidden").mkdir()
    assert lib.list_voices() == []


def test_adapter_without_voice_json_gets_safe_defaults(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    rec = lib.add_from_adapter(make_adapter(tmp_path / "my_voice", with_voice_json=False))
    assert rec.name == "my_voice" and rec.license == "custom/personal-only" and not rec.commercial_use
    assert voice_info.read_voice_json(rec.path)["schema"] == voice_info.VOICE_SCHEMA


def test_import_folder_ignores_foreign_files(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    src = make_adapter(tmp_path / "src", extra={"evil.exe": b"MZ", "notes.txt": b"hi"})
    rec = lib.import_folder(src)
    names = {p.name for p in rec.path.iterdir()}
    assert names <= set(ALLOWED_FILES) and "evil.exe" not in names


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as z:
        for n, d in members.items():
            z.writestr(n, d)
    return path


def test_import_zip_flat_and_wrapped(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    base = {"adapter_model.safetensors": b"w", "adapter_config.json": b"{}", "ref_sample.wav": b"RIFF"}
    r1 = lib.import_zip(_zip(tmp_path / "a.zip", base))
    r2 = lib.import_zip(_zip(tmp_path / "b.zip", {f"anna/{k}": v for k, v in base.items()}))
    assert r1.id != r2.id and all(adapter_complete(r.path) for r in (r1, r2))


def test_import_zip_overrides_license_from_repository(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    info = voice_info.build_voice_info("Zed", "russian", 1, 1, "b", license="CC0-1.0")
    members = {"adapter_model.safetensors": b"w", "adapter_config.json": b"{}", "voice.json": json.dumps(info)}
    rec = lib.import_zip(_zip(tmp_path / "z.zip", members), overrides={"license": "CC-BY-NC-4.0", "author": "Repo"})
    assert rec.license == "CC-BY-NC-4.0" and rec.commercial_use is False and rec.info["author"] == "Repo"
    assert rec.info["license_url"] == "https://creativecommons.org/licenses/by-nc/4.0/"


@pytest.mark.parametrize("evil", ["../evil.safetensors", "/abs/adapter_model.safetensors", "..\\x\\adapter_config.json",
                                  "C:/adapter_config.json", "a/../../adapter_model.safetensors"])
def test_zip_path_traversal_is_rejected(tmp_path, evil):
    lib = VoiceLibrary(tmp_path / "lib")
    z = _zip(tmp_path / "e.zip", {"adapter_model.safetensors": b"w", "adapter_config.json": b"{}", evil: b"x"})
    with pytest.raises(VoiceLibraryError):
        lib.import_zip(z)
    assert lib.list_voices() == [] and not (tmp_path / "evil.safetensors").exists()
    assert not [p for p in lib.root.iterdir()]                       # staging folder cleaned up


def test_zip_missing_adapter_and_bad_zip(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    with pytest.raises(VoiceLibraryError):
        lib.import_zip(_zip(tmp_path / "m.zip", {"ref_sample.wav": b"x"}))
    (tmp_path / "bad.zip").write_bytes(b"not a zip")
    with pytest.raises(VoiceLibraryError):
        lib.import_zip(tmp_path / "bad.zip")


def test_zip_size_limits(tmp_path, monkeypatch):
    import core.voice_library as vl
    monkeypatch.setattr(vl, "MAX_FILE_BYTES", 10)
    lib = VoiceLibrary(tmp_path / "lib")
    z = _zip(tmp_path / "big.zip", {"adapter_model.safetensors": b"x" * 100, "adapter_config.json": b"{}"})
    with pytest.raises(VoiceLibraryError):
        lib.import_zip(z)


def test_update_changes_details_and_rederives_commercial_flag(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    rec = lib.add_from_adapter(make_adapter(tmp_path / "src"))
    upd = lib.update(rec.id, name="Anna K", author="Anna", license="CC0-1.0", voice_type="female", description=" nice\nvoice ")
    assert upd.name == "Anna K" and upd.commercial_use is True and upd.info["license_url"].endswith("zero/1.0/")
    again = lib.get(upd.id)                    # a new name renames the folder (the id follows it)
    assert again.info["description"] == "nice voice" and again.info["voice_type"] == "female" and again.id == "anna-k"
    back = lib.update(upd.id, license="CC-BY-NC-4.0")
    assert back.commercial_use is False
    with pytest.raises(VoiceLibraryError):
        lib.update("nope", name="x")


def test_delete_only_inside_the_library(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    rec = lib.add_from_adapter(make_adapter(tmp_path / "src"))
    outside = tmp_path / "precious"
    outside.mkdir()
    for bad in ("../precious", "..", ".", "", str(outside)):
        with pytest.raises(VoiceLibraryError):
            lib.delete(bad)
    assert outside.exists()
    lib.delete(rec.id)
    assert lib.is_empty() and (tmp_path / "src").exists()            # the source adapter is untouched
    with pytest.raises(VoiceLibraryError):
        lib.delete(rec.id)
