"""Backup / restore of the voices and the tools (infra/backup.py).

The models live in the shared store, which belongs to VOXPrint MDL: they are never part of an app backup and never
restored into an app folder.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import i18n
from core.errors import BackupError, CancelledByUser
from core.events import CancelToken
from infra import backup as bk
from infra import paths

SHA = "c" * 40


def fake_usage(free):
    return lambda _p: SimpleNamespace(total=10 ** 12, used=0, free=free)


def add_voice(name="anna", text="adapter"):
    d = paths.voices_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(text.encode() * 100)
    (d / "voice.json").write_text(json.dumps({"id": name, "name": name}), encoding="utf-8")
    return d


def add_tools():
    from infra import assets

    d = assets.tools_dir() / "ffmpeg"
    d.mkdir(parents=True, exist_ok=True)
    (d / "ffmpeg.exe").write_bytes(b"MZ" + b"\0" * 500)
    return d


def snapshot(root: Path):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def target(tmp_path):
    return tmp_path / "usb"


# ------------------------------------------------------------------------------------------------ backup
def test_collect_items_covers_tools_and_voices(tmp_path):
    add_tools()
    add_voice("anna")
    items = bk.collect_items(True)
    assert {(i.kind, i.name) for i in items} == {("tools", "tools"), ("voices", "anna")}
    assert {i.kind for i in bk.collect_items(False)} == {"tools"}          # voices are optional


def test_backup_copies_everything_and_writes_a_hash_manifest(target):
    add_tools()
    voice = add_voice("anna")
    items = bk.collect_items(True)
    progress = []
    rep = bk.run_backup(items, target, lambda f, m="": progress.append((f, m)))
    root = target / bk.BACKUP_DIRNAME
    assert rep.copied_files == sum(len(i.files) for i in items) and rep.skipped_files == 0
    assert snapshot(root / "voices" / "anna") == snapshot(voice)
    data = json.loads((root / bk.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert data["schema"] == 1 and {i["kind"] for i in data["items"]} == {"tools", "voices"}
    assert all(i["complete"] for i in data["items"])
    st = next(f for i in data["items"] if i["kind"] == "voices" for f in i["files"] if f["p"] == "adapter_model.safetensors")
    assert st["sha256"] == hashlib.sha256((voice / "adapter_model.safetensors").read_bytes()).hexdigest()
    assert progress[-1][0] == 1.0
    assert not list(root.rglob("*.part"))


def test_second_run_skips_identical_files_and_copies_only_changes(target):
    voice = add_voice("anna")
    bk.run_backup(bk.collect_items(True), target)
    rep = bk.run_backup(bk.collect_items(True), target)
    assert rep.copied_files == 0 and rep.skipped_files > 0
    # a changed file (new size) is copied again, the others are still skipped
    (voice / "voice.json").write_text(json.dumps({"id": "anna", "name": "Anna-New"}), encoding="utf-8")
    rep = bk.run_backup(bk.collect_items(True), target)
    assert rep.copied_files == 1


def test_identical_content_with_another_timestamp_is_recognised_by_hash(target):
    voice = add_voice("anna")
    bk.run_backup(bk.collect_items(True), target)
    dst = target / bk.BACKUP_DIRNAME / "voices" / "anna" / "adapter_model.safetensors"
    os.utime(dst, (1000, 1000))                                     # same bytes, different time
    assert bk.run_backup(bk.collect_items(True), target).copied_files == 0
    # same size but different bytes is copied again, even when the timestamp still matches
    data = bytearray(dst.read_bytes())
    data[-1] ^= 1
    st = os.stat(voice / "adapter_model.safetensors")
    dst.write_bytes(bytes(data))
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert bk.run_backup(bk.collect_items(True), target).copied_files == 1
    assert dst.read_bytes() == (voice / "adapter_model.safetensors").read_bytes()


def test_cancel_leaves_no_part_files_and_the_next_run_continues(target):
    add_voice("anna")
    add_voice("bella")
    items = bk.collect_items(True)
    cancel = CancelToken()
    calls = []

    def prog(f, m=""):
        calls.append(f)
        if len(calls) > 1:
            cancel.cancel()

    with pytest.raises(CancelledByUser):
        bk.run_backup(items, target, prog, cancel)
    root = target / bk.BACKUP_DIRNAME
    assert not list(root.rglob("*.part"))
    rep = bk.run_backup(bk.collect_items(True), target)
    assert rep.copied_files + rep.skipped_files == sum(len(i.files) for i in items)


def test_an_interrupted_run_never_shrinks_the_existing_manifest(target):
    add_voice("anna")
    bk.run_backup(bk.collect_items(True), target)
    add_voice("bella")
    cancel = CancelToken()
    cancel.cancel()
    with pytest.raises(CancelledByUser):
        bk.run_backup(bk.collect_items(True), target, cancel=cancel)
    data = json.loads((target / bk.BACKUP_DIRNAME / bk.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert {i["name"] for i in data["items"]} >= {"anna"}


def test_not_enough_space_is_reported_clearly_before_anything_is_written(target):
    add_voice("anna")
    i18n.set_language("en")
    items = bk.collect_items(True)
    with pytest.raises(BackupError) as e:
        bk.run_backup(items, target, usage=fake_usage(1000))
    assert e.value.code == "space" and "Not enough free space" in e.value.user_message and "needed" in e.value.user_message
    assert not list((target / bk.BACKUP_DIRNAME).rglob("*.safetensors"))
    # identical files do not count: a full backup fits again on a drive that has no room for a second copy
    bk.run_backup(items, target)
    bk.run_backup(bk.collect_items(True), target, usage=fake_usage(bk.SPACE_MARGIN + 10))


def test_format_size():
    assert bk.format_size(0) == "0 B" and bk.format_size(1536) == "2 KB" and bk.format_size(3.5 * 1024 ** 3) == "3.5 GB"


# ------------------------------------------------------------------------------------------------ restore
def make_backup(target, voices=True):
    add_tools()
    if voices:
        add_voice("anna")
    bk.run_backup(bk.collect_items(voices), target)


def wipe_home():
    for sub in ("models", "voices", "tools"):
        shutil.rmtree(paths.app_home() / sub, ignore_errors=True)
    shared = paths.shared_root()
    for sub in ("models", "tools", "runtime"):
        shutil.rmtree(shared / sub, ignore_errors=True)


def test_restore_rebuilds_voices_and_tools_verified(target):
    make_backup(target)
    wipe_home()
    rep = bk.run_restore(target)
    assert (paths.voices_dir() / "anna" / "voice.json").is_file()
    from infra import assets
    assert (assets.tools_dir() / "ffmpeg" / "ffmpeg.exe").is_file()
    assert rep.items == 2 and not rep.conflicts
    assert not list(paths.voices_dir().glob("*.restoring")) and not list(paths.voices_dir().glob("*.old"))
    # a second restore finds everything in place
    rep2 = bk.run_restore(target)
    assert rep2.items == 0 and rep2.copied_files == 0


def test_restore_accepts_the_chosen_folder_or_the_backup_folder_itself(target):
    make_backup(target, voices=False)
    wipe_home()
    assert bk.find_backup(target) == target / bk.BACKUP_DIRNAME
    assert bk.find_backup(target / bk.BACKUP_DIRNAME) == target / bk.BACKUP_DIRNAME
    bk.run_restore(target / bk.BACKUP_DIRNAME)
    from infra import assets
    assert (assets.tools_dir() / "ffmpeg" / "ffmpeg.exe").is_file()


def test_restore_without_voices_and_voice_conflicts_are_not_overwritten(target):
    make_backup(target)
    wipe_home()
    bk.run_restore(target, include_voices=False)
    assert not (paths.voices_dir() / "anna").exists()
    d = add_voice("anna", text="my newer adapter")
    rep = bk.run_restore(target)
    assert rep.conflicts == ["anna"] and (d / "adapter_model.safetensors").read_bytes().startswith(b"my newer")


def test_corrupt_backup_file_is_detected_and_nothing_half_restored_is_used(target):
    make_backup(target, voices=False)
    f = target / bk.BACKUP_DIRNAME / "tools" / "ffmpeg" / "ffmpeg.exe"
    data = bytearray(f.read_bytes())
    data[-1] ^= 0xFF
    f.write_bytes(bytes(data))                                    # same size, other content
    wipe_home()
    with pytest.raises(BackupError) as e:
        bk.run_restore(target)
    assert e.value.code == "hash"


def test_restore_interrupted_continues_with_the_staged_files(target):
    make_backup(target)
    wipe_home()
    cancel = CancelToken()
    n = []

    def prog(f, m=""):
        n.append(1)
        if len(n) > 1:
            cancel.cancel()

    with pytest.raises(CancelledByUser):
        bk.run_restore(target, progress=prog, cancel=cancel)
    assert not (paths.voices_dir() / "anna" / "voice.json").exists()
    bk.run_restore(target)
    assert (paths.voices_dir() / "anna" / "voice.json").is_file()


def test_restore_errors_no_manifest_unsafe_paths_and_space(target, tmp_path):
    with pytest.raises(BackupError) as e:
        bk.run_restore(tmp_path / "empty")
    assert e.value.code == "no_manifest"
    make_backup(target, voices=False)
    mf = target / bk.BACKUP_DIRNAME / bk.MANIFEST_NAME
    data = json.loads(mf.read_text(encoding="utf-8"))
    data["items"][0]["files"][0]["p"] = "../../evil.txt"
    mf.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(BackupError):
        bk.plan_restore(target)
    data["items"][0]["files"][0]["p"] = "ffmpeg.exe"
    mf.write_text(json.dumps(data), encoding="utf-8")
    wipe_home()
    with pytest.raises(BackupError) as e:
        bk.run_restore(target, usage=fake_usage(100))
    assert e.value.code == "space"
