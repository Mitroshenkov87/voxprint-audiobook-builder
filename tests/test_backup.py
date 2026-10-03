"""Backup / restore of models and voices (infra/backup.py) and the "existing models folder" import (infra/existing_models.py)."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import i18n
from core.errors import BackupError, CancelledByUser
from core.events import CancelToken
from infra import backup as bk
from infra import existing_models as em
from infra import model_downloader as md
from infra import paths
from tests.test_model_locator import make_hf_cache, make_model

REPO = "Org/Tiny-1.7B"
OTHER = "Org/Other-0.6B"
SHA = "c" * 40


def fake_usage(free):
    return lambda _p: SimpleNamespace(total=10 ** 12, used=0, free=free)


def own_model(repo=REPO, rev=SHA, payload=64):
    d = make_model(md.local_dir_for(repo))
    (d / ".revision").write_text(rev, encoding="utf-8")
    return d


def add_voice(name="anna", text="adapter"):
    d = paths.voices_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(text.encode() * 100)
    (d / "voice.json").write_text(json.dumps({"id": name, "name": name}), encoding="utf-8")
    return d


def add_tools():
    d = paths.app_home() / "tools" / "ffmpeg"
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
def test_collect_items_models_tools_voices_and_foreign_cache(tmp_path):
    own_model()
    add_tools()
    add_voice("anna")
    hub = tmp_path / "hub"
    snap = make_hf_cache(hub, OTHER, SHA)
    items = bk.collect_items(True, [REPO, OTHER, "Org/Missing"], locate=lambda r: (snap, SHA) if r == OTHER else None)
    by = {(i.kind, i.name): i for i in items}
    assert set(by) == {("model", "Org--Tiny-1.7B"), ("model", "Org--Other-0.6B"), ("tools", "tools"), ("voices", "anna")}
    other = by[("model", "Org--Other-0.6B")]
    assert other.revision == SHA and ".revision" in {f.p for f in other.files}            # symlinks followed, revision kept
    assert all(f.size > 0 for f in other.files if f.p != ".revision")
    assert {i.kind for i in bk.collect_items(False, [REPO])} == {"model", "tools"}         # voices are optional


def test_partial_downloads_are_not_part_of_a_backup():
    own_model()
    junk = md.local_dir_for("Org/Half").with_name("Org--Half.partial")
    make_model(junk)
    (md.local_dir_for(REPO) / "x.incomplete").write_bytes(b"1")
    items = bk.collect_items(False, [])
    assert [i.name for i in items if i.kind == "model"] == ["Org--Tiny-1.7B"]
    assert not any(f.p.endswith(".incomplete") for f in items[0].files)


def test_backup_copies_everything_and_writes_a_hash_manifest(target):
    src = own_model()
    add_voice("anna")
    items = bk.collect_items(True, [])
    progress = []
    rep = bk.run_backup(items, target, lambda f, m="": progress.append((f, m)))
    root = target / bk.BACKUP_DIRNAME
    assert rep.copied_files == sum(len(i.files) for i in items) and rep.skipped_files == 0
    assert snapshot(root / "models" / "Org--Tiny-1.7B") == snapshot(src)
    assert (root / "voices" / "anna" / "voice.json").is_file()
    data = json.loads((root / bk.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert data["schema"] == 1 and {i["kind"] for i in data["items"]} == {"model", "voices"} and all(i["complete"] for i in data["items"])
    model = next(i for i in data["items"] if i["kind"] == "model")
    st = next(f for f in model["files"] if f["p"] == "model.safetensors")
    assert st["sha256"] == hashlib.sha256((src / "model.safetensors").read_bytes()).hexdigest()
    assert progress[-1][0] == 1.0 and any("Org--Tiny-1.7B" in m for _f, m in progress)
    assert not list(root.rglob("*.part"))


def test_second_run_skips_identical_files_and_copies_only_changes(target):
    src = own_model()
    add_voice("anna")
    bk.run_backup(bk.collect_items(True, []), target)
    rep = bk.run_backup(bk.collect_items(True, []), target)
    assert rep.copied_files == 0 and rep.skipped_files > 0
    # a changed file (new size) is copied again, the others are still skipped
    (src / "tokenizer.json").write_text('{"changed": true}', encoding="utf-8")
    rep = bk.run_backup(bk.collect_items(True, []), target)
    assert rep.copied_files == 1
    data = json.loads((target / bk.BACKUP_DIRNAME / bk.MANIFEST_NAME).read_text(encoding="utf-8"))
    files = {f["p"]: f for i in data["items"] if i["kind"] == "model" for f in i["files"]}
    assert files["tokenizer.json"]["sha256"] == hashlib.sha256((src / "tokenizer.json").read_bytes()).hexdigest()


def test_identical_content_with_another_timestamp_is_recognised_by_hash(target):
    src = own_model()
    bk.run_backup(bk.collect_items(False, []), target)
    dst = target / bk.BACKUP_DIRNAME / "models" / "Org--Tiny-1.7B" / "model.safetensors"
    os.utime(dst, (1000, 1000))                                     # same bytes, different time
    rep = bk.run_backup(bk.collect_items(False, []), target)
    assert rep.copied_files == 0
    # same size but different content with the same time is only caught by verify=True
    data = bytearray(dst.read_bytes())
    data[-1] ^= 1
    st = os.stat(src / "model.safetensors")
    dst.write_bytes(bytes(data))
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert bk.run_backup(bk.collect_items(False, []), target).copied_files == 0
    assert bk.run_backup(bk.collect_items(False, []), target, verify=True).copied_files == 1
    assert dst.read_bytes() == (src / "model.safetensors").read_bytes()


def test_cancel_leaves_no_part_files_and_the_next_run_continues(target):
    own_model()
    own_model(OTHER)
    items = bk.collect_items(False, [])
    cancel = CancelToken()
    calls = []

    def prog(f, m=""):
        calls.append(f)
        if len(calls) > 6:
            cancel.cancel()

    with pytest.raises(CancelledByUser):
        bk.run_backup(items, target, prog, cancel)
    root = target / bk.BACKUP_DIRNAME
    assert not list(root.rglob("*.part"))
    rep = bk.run_backup(bk.collect_items(False, []), target)
    assert rep.copied_files + rep.skipped_files == sum(len(i.files) for i in items)
    assert snapshot(root / "models") == {f"{n}/{k}": v for n in ("Org--Tiny-1.7B", "Org--Other-0.6B")
                                         for k, v in snapshot(md.local_dir_for("Org/" + n.split("--")[1])).items()}


def test_an_interrupted_run_never_shrinks_the_existing_manifest(target):
    own_model()
    add_voice("anna")
    bk.run_backup(bk.collect_items(True, []), target)
    add_voice("bella")
    cancel = CancelToken()
    cancel.cancel()
    with pytest.raises(CancelledByUser):
        bk.run_backup(bk.collect_items(True, []), target, cancel=cancel)
    data = json.loads((target / bk.BACKUP_DIRNAME / bk.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert {i["name"] for i in data["items"]} >= {"Org--Tiny-1.7B", "anna"}


def test_not_enough_space_is_reported_clearly_before_anything_is_written(target):
    own_model()
    i18n.set_language("en")
    items = bk.collect_items(False, [])
    with pytest.raises(BackupError) as e:
        bk.run_backup(items, target, usage=fake_usage(1000))
    assert e.value.code == "space" and "Not enough free space" in e.value.user_message and "needed" in e.value.user_message
    assert not list((target / bk.BACKUP_DIRNAME).rglob("*.safetensors"))
    # identical files do not count: a full backup fits again on a drive that has no room for a second copy
    bk.run_backup(items, target)
    bk.run_backup(bk.collect_items(False, []), target, usage=fake_usage(bk.SPACE_MARGIN + 10))


def test_format_size():
    assert bk.format_size(0) == "0 B" and bk.format_size(1536) == "2 KB" and bk.format_size(3.5 * 1024 ** 3) == "3.5 GB"


# ------------------------------------------------------------------------------------------------ restore
def make_backup(target, voices=True):
    src = own_model()
    add_tools()
    if voices:
        add_voice("anna")
    bk.run_backup(bk.collect_items(voices, []), target)
    return src


def wipe_home():
    import shutil

    for sub in ("models", "voices", "tools"):
        shutil.rmtree(paths.app_home() / sub, ignore_errors=True)


def test_restore_rebuilds_models_voices_and_tools_verified(target):
    src = make_backup(target)
    before = snapshot(src)
    wipe_home()
    rep = bk.run_restore(target)
    assert snapshot(md.local_dir_for(REPO)) == before and md.verify_local_model(md.local_dir_for(REPO))
    assert (paths.voices_dir() / "anna" / "voice.json").is_file()
    assert (paths.app_home() / "tools" / "ffmpeg" / "ffmpeg.exe").is_file()
    assert rep.items == 3 and not rep.conflicts
    assert not list(paths.models_dir().glob("*.restoring")) and not list(paths.models_dir().glob("*.old"))
    # a second restore finds everything in place
    rep2 = bk.run_restore(target)
    assert rep2.items == 0 and rep2.copied_files == 0


def test_restore_accepts_the_chosen_folder_or_the_backup_folder_itself(target):
    make_backup(target, voices=False)
    wipe_home()
    assert bk.find_backup(target) == target / bk.BACKUP_DIRNAME
    assert bk.find_backup(target / bk.BACKUP_DIRNAME) == target / bk.BACKUP_DIRNAME
    bk.run_restore(target / bk.BACKUP_DIRNAME)
    assert md.verify_local_model(md.local_dir_for(REPO))


def test_restore_without_voices_and_voice_conflicts_are_not_overwritten(target):
    make_backup(target)
    wipe_home()
    bk.run_restore(target, include_voices=False)
    assert not (paths.voices_dir() / "anna").exists()
    d = add_voice("anna", text="my newer adapter")
    rep = bk.run_restore(target)
    assert rep.conflicts == ["anna"] and (d / "adapter_model.safetensors").read_bytes().startswith(b"my newer")


def test_restore_replaces_a_damaged_model_only_after_the_new_copy_is_complete(target):
    src = make_backup(target, voices=False)
    good = snapshot(src)
    (src / "model.safetensors").write_bytes(b"damaged")
    rep = bk.run_restore(target)
    assert rep.items == 1 and snapshot(md.local_dir_for(REPO)) == good


def test_corrupt_backup_file_is_detected_and_nothing_half_restored_is_used(target):
    make_backup(target, voices=False)
    f = target / bk.BACKUP_DIRNAME / "models" / "Org--Tiny-1.7B" / "model.safetensors"
    data = bytearray(f.read_bytes())
    data[-1] ^= 0xFF
    f.write_bytes(bytes(data))                                    # same size, other content
    wipe_home()
    with pytest.raises(BackupError) as e:
        bk.run_restore(target)
    assert e.value.code == "hash"
    assert not md.verify_local_model(md.local_dir_for(REPO))     # never presented as a finished model


def test_restore_interrupted_continues_with_the_staged_files(target):
    make_backup(target, voices=False)
    wipe_home()
    cancel = CancelToken()
    n = []

    def prog(f, m=""):
        n.append(1)
        if len(n) > 1:
            cancel.cancel()

    with pytest.raises(CancelledByUser):
        bk.run_restore(target, progress=prog, cancel=cancel)
    assert not md.verify_local_model(md.local_dir_for(REPO))
    bk.run_restore(target)
    assert md.verify_local_model(md.local_dir_for(REPO))


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
    data["items"][0]["files"][0]["p"] = "config.json"
    mf.write_text(json.dumps(data), encoding="utf-8")
    wipe_home()
    with pytest.raises(BackupError) as e:
        bk.run_restore(target, usage=fake_usage(100))
    assert e.value.code == "space"


# ------------------------------------------------------------------------------------------------ existing models folder
@pytest.fixture
def library(tmp_path):
    """A folder with a Voxprint-style models layout (Owner--Name)."""
    lib = tmp_path / "old_install_models"
    make_model(lib / "Org--Tiny-1.7B")
    (lib / "Org--Tiny-1.7B" / ".revision").write_text(SHA, encoding="utf-8")
    return lib


def no_download(*a, **k):
    raise AssertionError("must not download")


def test_config_file_env_override_and_clear(tmp_path, monkeypatch):
    assert em.configured() is None and em.configured_text() == ""
    d = tmp_path / "models_here"
    d.mkdir()
    em.set_folder(d)
    assert em.configured() == d and em.config_file().read_text(encoding="utf-8").strip() == str(d)
    em.config_file().write_bytes(b"\xef\xbb\xbf" + str(d).encode() + b"\r\n")        # what the installer writes (UTF-8 BOM, CRLF)
    assert em.configured() == d
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv(em.ENV_VAR, str(other))
    assert em.configured() == other
    monkeypatch.delenv(em.ENV_VAR)
    em.set_folder(None)
    assert em.configured() is None
    em.set_folder(tmp_path / "gone")
    assert em.configured() is None and em.configured_text() == str(tmp_path / "gone")      # shown in Settings, but unusable


def test_ensure_model_imports_from_the_existing_folder_by_hard_link_before_downloading(library):
    em.set_folder(library)
    msgs = []
    got = md.ensure_model(REPO, lambda s, f, m="": msgs.append(m), snapshot_download=no_download, revision=SHA)
    assert got == md.local_dir_for(REPO) and md.verify_local_model(got)
    assert (got / ".revision").read_text(encoding="utf-8") == SHA
    assert os.stat(got / "model.safetensors").st_ino == os.stat(library / "Org--Tiny-1.7B" / "model.safetensors").st_ino
    assert any("existing models folder" in m or "папке с моделями" in m for m in msgs)
    assert not list(paths.models_dir().glob("*.importing"))


def test_import_copies_and_verifies_when_linking_is_not_possible(library):
    em.set_folder(library)
    found = em.find(REPO, SHA)
    assert found is not None and found.kind == em.KIND
    dest = em.import_model(found, allow_link=False)
    assert snapshot(dest).keys() >= snapshot(library / "Org--Tiny-1.7B").keys()
    assert os.stat(dest / "model.safetensors").st_ino != os.stat(library / "Org--Tiny-1.7B" / "model.safetensors").st_ino


def test_backup_folder_is_a_valid_existing_folder_and_the_manifest_hashes_are_checked(target, tmp_path):
    make_backup(target, voices=False)
    wipe_home()
    em.set_folder(target)                                             # the folder that contains Voxprint-backup
    roots = [str(p) for _k, p in em.roots()]
    assert str(target / bk.BACKUP_DIRNAME / "models") in roots
    got = md.ensure_model(REPO, snapshot_download=no_download, revision=SHA)
    assert md.verify_local_model(got)


def test_tampered_backup_file_fails_the_hash_check_and_the_download_path_runs(target):
    make_backup(target, voices=False)
    wipe_home()
    f = target / bk.BACKUP_DIRNAME / "models" / "Org--Tiny-1.7B" / "model.safetensors"
    data = bytearray(f.read_bytes())
    data[-1] ^= 0xFF
    f.write_bytes(bytes(data))
    em.set_folder(target)
    calls = []

    def fake_download(repo_id, local_dir=None, **kw):
        calls.append(repo_id)
        make_model(Path(local_dir))
        return str(local_dir)

    got = md.ensure_model(REPO, snapshot_download=fake_download, revision=SHA)
    assert calls == [REPO] and md.verify_local_model(got)
    assert (got / "model.safetensors").read_bytes() != bytes(data)       # the corrupt file was not imported
    assert not list(paths.models_dir().glob("*.importing"))


def test_import_checks_free_space_when_copying(library):
    em.set_folder(library)
    found = em.find(REPO, SHA)
    with pytest.raises(BackupError) as e:
        em.import_model(found, allow_link=False, usage=fake_usage(10))
    assert e.value.code == "space"


def test_foreign_revision_and_incomplete_models_are_not_imported(tmp_path):
    lib = tmp_path / "lib"
    make_model(lib / "Org--Tiny-1.7B")
    (lib / "Org--Tiny-1.7B" / ".revision").write_text("d" * 40, encoding="utf-8")
    em.set_folder(lib)
    assert em.find(REPO, SHA) is None                                 # other than the pinned revision
    (lib / "Org--Tiny-1.7B" / ".revision").write_text(SHA, encoding="utf-8")
    (lib / "Org--Tiny-1.7B" / "model.safetensors").write_bytes(b"short")
    assert em.find(REPO, SHA) is None                                 # truncated weights


def test_pending_state_and_model_state(library):
    assert not em.pending([REPO])
    em.set_folder(library)
    assert em.pending([REPO]) and md.model_state(REPO) == md.STATE_READY
    assert not em.pending(["Org/Not-There"])
    md.ensure_model(REPO, snapshot_download=no_download, revision=SHA)
    assert not em.pending([REPO])                                     # imported: nothing left to do on start-up


def test_import_available_reports_what_it_did(library):
    em.set_folder(library)
    prog = []
    rep = em.import_available([REPO, "Org/Not-There"], lambda f, m="": prog.append(f))
    assert rep.items == 1 and rep.copied_files >= 3 and prog[-1] == 1.0
    assert em.import_available([REPO]).items == 0                      # already there


def test_locator_lists_the_existing_folder_first_and_ignores_the_disable_switch_only_for_it(library, monkeypatch):
    from core import model_locator as ml

    em.set_folder(library)
    monkeypatch.delenv("VOXPRINT_NO_EXTERNAL_MODELS")
    roots = ml.candidate_roots()
    assert roots[0][0] == em.KIND and roots[0][1] == library
    monkeypatch.setenv("VOXPRINT_NO_EXTERNAL_MODELS", "1")
    assert ml.find_model(REPO, SHA) is None                           # guessing stays off
    assert em.find(REPO, SHA) is not None                              # the explicit folder still works
