"""Typed voice names: folder follows the name; edits replace localized names."""
from core.voice_library import VoiceLibrary
from workers.pipeline_runner import TaskRequest


def _voice(lib, tmp_path):
    from tests.test_voice_library import make_adapter  # reuse the existing fixture builder
    return lib.add_from_adapter(make_adapter(tmp_path / "src"), {"name": "downloads", "names": {"ru": "Старое", "en": "Old"}})


def test_typed_name_is_folder_without_type_suffix(tmp_path):
    req = TaskRequest(kind="lora", audio=tmp_path / "downloads.wav", voice_type="female", voice_display_name="Анна К.")
    assert req.voice_name() == "Анна К."
    assert req.voice_folder() == "Анна К."
    assert TaskRequest(kind="lora", audio=tmp_path / "rec.wav", voice_type="female").voice_folder().endswith("female")


def test_rename_moves_folder_and_drops_localized_names(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    rec = _voice(lib, tmp_path)
    assert "names" in rec.info
    new = lib.update(rec.id, name="Anna")
    assert new.id != rec.id and new.path.is_dir() and not rec.path.exists()
    assert new.name == "Anna" and "names" not in new.info
    assert lib.get(new.id).info["name"] == "Anna"


def test_export_zip_roundtrip(tmp_path):
    import zipfile
    lib = VoiceLibrary(tmp_path / "lib")
    rec = _voice(lib, tmp_path)
    z = lib.export_zip(rec.id, lib.root / f"{rec.id}.zip")
    names = set(zipfile.ZipFile(z).namelist())
    assert "voice.json" in names and all("/" not in n for n in names)
    assert [v.id for v in lib.list_voices()] == [rec.id]      # the zip next to the voices is not taken for a voice
    other = VoiceLibrary(tmp_path / "lib2")
    assert other.import_zip(z).name == rec.name
