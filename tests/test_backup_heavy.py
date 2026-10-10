"""Backup / restore of every heavy file (models, llm, deepfilternet, dnsmos) with a manifest, resume, and link mode.

Uses tiny fake files only. Nothing is downloaded and no real model is loaded.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import appinfo
from core import model_locator as ml
from core.errors import BackupError
from infra import backup as bk
from infra import external_models as ex
from infra import model_downloader as md
from infra import paths
from tests.test_backup import add_voice, own_model, wipe_home
from tests.test_model_locator import make_model

REPO = "Org/Tiny-1.7B"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _heavy(models: Path) -> None:
    """Small stand-ins for the non-Hugging-Face trees under the models folder."""
    (models / "llm").mkdir(parents=True)
    (models / "llm" / "note.txt").write_bytes(b"gemma-stand-in")
    (models / "llm" / "llama.cpp-test").mkdir()
    (models / "llm" / "llama.cpp-test" / "llama-server").write_bytes(b"server")
    (models / "deepfilternet").mkdir()
    (models / "deepfilternet" / "deep-filter").write_bytes(b"df")
    (models / "dnsmos").mkdir()
    (models / "dnsmos" / "mos.onnx").write_bytes(b"mos-bytes")


def test_manifest_round_trip_records_version_build_and_per_file_hashes(tmp_path):
    src = own_model()
    _heavy(paths.models_dir())
    add_voice("anna")
    target = tmp_path / "usb"
    items = bk.collect_items(True, [])
    names = {i.name for i in items if i.kind == "model"}
    assert {"Org--Tiny-1.7B", "llm", "deepfilternet", "dnsmos"} <= names
    rep = bk.run_backup(items, target)
    assert rep.copied_files > 0
    data = json.loads((target / bk.BACKUP_DIRNAME / bk.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert data["version"] == appinfo.release_version()
    assert data["build"] == appinfo.APP_BUILD
    files = {f["path"]: f for f in data["files"]}
    rel = "models/Org--Tiny-1.7B/model.safetensors"
    assert files[rel]["size"] == (src / "model.safetensors").stat().st_size
    assert files[rel]["sha256"] == _sha(src / "model.safetensors")
    assert files["models/llm/llama.cpp-test/llama-server"]["sha256"] == hashlib.sha256(b"server").hexdigest()
    assert files["voices/anna/voice.json"]["size"] > 0
    again = bk.items_from_manifest(bk.read_manifest(target / bk.BACKUP_DIRNAME))
    assert {i.name for i in again} >= names | {"anna"}


def test_resume_skips_files_with_the_same_size_and_sha256(tmp_path):
    own_model()
    target = tmp_path / "usb"
    bk.run_backup(bk.collect_items(False, []), target)
    dst = target / bk.BACKUP_DIRNAME / "models" / "Org--Tiny-1.7B" / "model.safetensors"
    os.utime(dst, (dst.stat().st_mtime + 100, dst.stat().st_mtime + 100))   # same bytes, other timestamp
    rep = bk.run_backup(bk.collect_items(False, []), target)
    assert rep.copied_files == 0 and rep.skipped_files > 0


def test_include_models_can_be_switched_off(tmp_path):
    own_model()
    _heavy(paths.models_dir())
    add_voice("anna")
    target = tmp_path / "usb"
    bk.run_backup(bk.collect_items(True, [], include_models=False), target)
    root = target / bk.BACKUP_DIRNAME
    assert (root / "voices" / "anna").is_dir()
    assert not (root / "models").exists()


def test_manifest_paths_that_leave_the_backup_are_ignored(tmp_path):
    own_model()
    target = tmp_path / "usb"
    bk.run_backup(bk.collect_items(False, []), target)
    root = target / bk.BACKUP_DIRNAME
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"outside-the-backup")
    data = json.loads((root / bk.MANIFEST_NAME).read_text(encoding="utf-8"))
    data["files"].append({"path": "models/../../secret.bin", "size": secret.stat().st_size,
                          "sha256": hashlib.sha256(secret.read_bytes()).hexdigest()})
    (root / bk.MANIFEST_NAME).write_text(json.dumps(data), encoding="utf-8")
    assert "models/../../secret.bin" not in bk.manifest_index(root)
    problems = bk._scan_model_problems(root, {}, None, lambda *_a: None)
    assert not any("secret.bin" in p for p in problems)


def test_corrupt_or_missing_file_is_reported_and_not_installed(tmp_path):
    own_model()
    add_voice("anna")
    target = tmp_path / "usb"
    bk.run_backup(bk.collect_items(True, []), target)
    weight = target / bk.BACKUP_DIRNAME / "models" / "Org--Tiny-1.7B" / "model.safetensors"
    blob = bytearray(weight.read_bytes())
    blob[-1] ^= 0x5A
    weight.write_bytes(bytes(blob))                         # same size, other bytes
    wipe_home()
    rep = bk.run_restore(target, strict=False)
    assert any("model.safetensors" in p for p in rep.problems)
    assert not md.verify_local_model(md.local_dir_for(REPO))
    assert (paths.voices_dir() / "anna" / "voice.json").is_file()    # the good item is still restored

    missing = tmp_path / "missing"
    own_model()
    bk.run_backup(bk.collect_items(False, []), missing)
    (missing / bk.BACKUP_DIRNAME / "models" / "Org--Tiny-1.7B" / "config.json").unlink()
    wipe_home()
    rep = bk.run_restore(missing, strict=False)
    assert any("config.json" in p for p in rep.problems)
    assert not md.verify_local_model(md.local_dir_for(REPO))

    # a pinned hash that disagrees with the manifest is the same kind of failure
    good = tmp_path / "good"
    own_model()
    bk.run_backup(bk.collect_items(False, []), good)
    wipe_home()
    pins = {"models/Org--Tiny-1.7B/model.safetensors": (1, "ab" * 32)}
    rep = bk.run_restore(good, strict=False, pins=pins)
    assert any("model.safetensors" in p for p in rep.problems)
    assert not md.verify_local_model(md.local_dir_for(REPO))
    with pytest.raises(BackupError) as e:                  # strict mode still refuses the bad copy
        bk.run_restore(good, pins=pins)
    assert e.value.code == "hash"


def test_link_mode_points_the_model_locator_at_the_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_NO_EXTERNAL_MODELS", "1")            # guessing stays off; the chosen folder does not
    src = own_model()
    original = (src / "model.safetensors").read_bytes()
    _heavy(paths.models_dir())
    add_voice("anna")
    target = tmp_path / "usb"
    bk.run_backup(bk.collect_items(True, []), target)
    wipe_home()
    rep = bk.run_link(target)
    linked = target / bk.BACKUP_DIRNAME / "models"
    assert ex.configured() == linked and rep.external == str(linked)
    assert not (paths.models_dir() / "Org--Tiny-1.7B").exists()       # models were not copied
    assert (paths.voices_dir() / "anna" / "voice.json").is_file()
    found = ml.find_model(REPO, None)
    assert found is not None and found.path == linked / "Org--Tiny-1.7B" and found.kind == ex.KIND
    assert ex.verified_file("llm/note.txt").read_bytes() == b"gemma-stand-in"
    from infra import llm_tool
    assert llm_tool._dir() == linked / "llm"
    # a file whose bytes no longer match the manifest is not used; the rest of the folder still is
    bad = linked / "Org--Tiny-1.7B" / "model.safetensors"
    data = bytearray(bad.read_bytes())
    data[-1] ^= 0x11
    bad.write_bytes(bytes(data))
    assert ml.find_model(REPO, None) is None
    assert ex.verified_file("llm/note.txt") is not None
    assert bad.read_bytes() != original
    # an unplugged drive (folder gone) is not a models path
    ex.set_folder(tmp_path / "missing-drive")
    assert ex.configured() is None and ml.find_model(REPO, None) is None


def test_cli_backup_restore_json_and_link(tmp_path, capsys, monkeypatch):
    import cli as user_cli

    monkeypatch.setenv("VOXPRINT_NO_EXTERNAL_MODELS", "1")
    own_model()
    add_voice("anna")
    out = tmp_path / "usb"
    assert user_cli.cmd_backup(SimpleNamespace(out=out, no_models=False, no_voices=False, json_output=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True and payload["copied_files"] > 0 and payload["problems"] == []
    wipe_home()
    assert user_cli.cmd_restore(SimpleNamespace(src=out, link=True, json_output=True)) == 0
    linked = json.loads(capsys.readouterr().out)
    assert linked["external_models"].replace("\\", "/").endswith("Voxprint-backup/models")
    assert ex.configured() is not None
    assert ml.find_model(REPO, None) is not None
