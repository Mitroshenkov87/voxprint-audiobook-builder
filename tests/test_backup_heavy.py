"""Backup / restore of the voices and tools with a manifest, resume, and the CLI flags.

Uses tiny fake files only. The models live in the shared store (VOXPrint MDL) and are never part of a backup.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import appinfo
from core.errors import BackupError
from infra import backup as bk
from infra import paths
from tests.test_backup import add_tools, add_voice, wipe_home


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_manifest_round_trip_records_version_build_and_per_file_hashes(tmp_path):
    add_tools()
    voice = add_voice("anna")
    target = tmp_path / "usb"
    items = bk.collect_items(True)
    assert {(i.kind, i.name) for i in items} == {("tools", "tools"), ("voices", "anna")}
    rep = bk.run_backup(items, target)
    assert rep.copied_files > 0
    data = json.loads((target / bk.BACKUP_DIRNAME / bk.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert data["version"] == appinfo.release_version()
    assert data["build"] == appinfo.APP_BUILD
    files = {f["path"]: f for f in data["files"]}
    assert files["voices/anna/voice.json"]["size"] > 0
    assert files["voices/anna/voice.json"]["sha256"] == _sha(voice / "voice.json")
    from infra import assets
    assert files["tools/ffmpeg/ffmpeg.exe"]["sha256"] == _sha(assets.tools_dir() / "ffmpeg" / "ffmpeg.exe")
    again = bk.items_from_manifest(bk.read_manifest(target / bk.BACKUP_DIRNAME))
    assert {i.name for i in again} == {"tools", "anna"}


def test_resume_skips_files_with_the_same_size_and_sha256(tmp_path):
    add_voice("anna")
    target = tmp_path / "usb"
    bk.run_backup(bk.collect_items(True), target)
    dst = target / bk.BACKUP_DIRNAME / "voices" / "anna" / "adapter_model.safetensors"
    os.utime(dst, (dst.stat().st_mtime + 100, dst.stat().st_mtime + 100))   # same bytes, other timestamp
    rep = bk.run_backup(bk.collect_items(True), target)
    assert rep.copied_files == 0 and rep.skipped_files > 0


def test_the_models_folder_is_never_part_of_a_backup(tmp_path):
    add_voice("anna")
    models = paths.models_dir()
    (models / "Qwen--Something").mkdir(parents=True, exist_ok=True)
    (models / "Qwen--Something" / "model.safetensors").write_bytes(b"weights")
    target = tmp_path / "usb"
    bk.run_backup(bk.collect_items(True), target)
    root = target / bk.BACKUP_DIRNAME
    assert (root / "voices" / "anna").is_dir()
    assert not (root / "models").exists()                              # the store belongs to VOXPrint MDL


def test_manifest_paths_that_leave_the_backup_are_ignored(tmp_path):
    add_voice("anna")
    target = tmp_path / "usb"
    bk.run_backup(bk.collect_items(True), target)
    root = target / bk.BACKUP_DIRNAME
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"outside-the-backup")
    data = json.loads((root / bk.MANIFEST_NAME).read_text(encoding="utf-8"))
    data["files"].append({"path": "voices/../../secret.bin", "size": secret.stat().st_size,
                          "sha256": hashlib.sha256(secret.read_bytes()).hexdigest()})
    (root / bk.MANIFEST_NAME).write_text(json.dumps(data), encoding="utf-8")
    assert "voices/../../secret.bin" not in bk.manifest_index(root)


def test_corrupt_or_missing_file_is_reported_and_not_installed(tmp_path):
    add_tools()
    add_voice("anna")
    target = tmp_path / "usb"
    bk.run_backup(bk.collect_items(True), target)
    weight = target / bk.BACKUP_DIRNAME / "voices" / "anna" / "adapter_model.safetensors"
    blob = bytearray(weight.read_bytes())
    blob[-1] ^= 0x5A
    weight.write_bytes(bytes(blob))                         # same size, other bytes
    wipe_home()
    rep = bk.run_restore(target, strict=False)
    assert any("adapter_model.safetensors" in p for p in rep.problems)
    assert not (paths.voices_dir() / "anna" / "adapter_model.safetensors").exists()
    from infra import assets
    assert (assets.tools_dir() / "ffmpeg" / "ffmpeg.exe").is_file()    # the good item is still restored

    missing = tmp_path / "missing"
    add_tools()
    add_voice("anna")
    bk.run_backup(bk.collect_items(True), missing)
    (missing / bk.BACKUP_DIRNAME / "voices" / "anna" / "voice.json").unlink()
    wipe_home()
    rep = bk.run_restore(missing, strict=False)
    assert any("voice.json" in p for p in rep.problems)


def test_cli_backup_restore_json(tmp_path, capsys):
    import cli as user_cli

    add_tools()
    add_voice("anna")
    out = tmp_path / "usb"
    assert user_cli.cmd_backup(SimpleNamespace(out=out, no_voices=False, json_output=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True and payload["copied_files"] > 0 and payload["problems"] == []
    wipe_home()
    assert user_cli.cmd_restore(SimpleNamespace(src=out, json_output=True)) == 0
    restored = json.loads(capsys.readouterr().out)
    assert restored["ok"] is True
    assert (paths.voices_dir() / "anna" / "voice.json").is_file()
