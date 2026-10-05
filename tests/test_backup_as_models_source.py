"""A Voxprint backup picked as the "models folder" is a RESTORE SOURCE, never the live models folder.

Hybrid rule (installer page "Models folder", decided 2026-10-05): an empty / normal folder becomes the models folder (download
target); a folder with ``voxprint-backup.json`` (or ``Voxprint-backup/`` inside, or a sub-folder of a backup) is restored into the
normal folders on the first run and the live models folder stays the default.  See ``infra/paths.py: backup_root_of``,
``infra/existing_models.py: adopt_backup_choice / restore_backup`` and ``installer/Voxprint.iss: BackupRootOf``."""
import json

import pytest

from infra import backup as bk
from infra import existing_models as em
from infra import model_downloader as md
from infra import paths
from tests.test_backup import REPO, SHA, add_tools, no_download, own_model, snapshot, wipe_home
from tests.test_model_locator import make_model

VOICE = "model_voice"          # placeholder voice id (stands for [model_voice] in the docs)


def add_voice(name=VOICE):
    d = paths.voices_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(b"adapter" * 100)
    (d / "voice.json").write_text(json.dumps({"id": name, "name": name}), encoding="utf-8")
    return d


@pytest.fixture
def usb(tmp_path):
    """An external drive with a full backup (model, ffmpeg, one voice); the app's own folders are empty afterwards."""
    drive = tmp_path / "usb"
    own_model()
    add_tools()
    add_voice()
    bk.run_backup(bk.collect_items(True, []), drive)
    wipe_home()
    return drive


def test_constants_match_the_backup_module():
    assert paths.BACKUP_MANIFEST == bk.MANIFEST_NAME and paths.BACKUP_DIRNAME == bk.BACKUP_DIRNAME


def test_backup_detection_in_every_place_the_user_may_point_at(usb, tmp_path):
    root = usb / bk.BACKUP_DIRNAME
    assert paths.backup_root_of(usb) == root                       # the drive / parent folder
    assert paths.backup_root_of(root) == root                      # Voxprint-backup itself
    assert paths.backup_root_of(root / "models") == root           # a sub-folder of the backup
    assert paths.backup_root_of(root / "voices") == root
    (root / "model").mkdir()
    assert paths.backup_root_of(root / "model") == root            # "model/" as users call it
    empty = tmp_path / "empty"
    empty.mkdir()
    assert paths.backup_root_of(empty) is None
    assert paths.backup_root_of(tmp_path / "missing") is None and paths.backup_root_of(None) is None
    live = make_model(tmp_path / "live" / "Org--Tiny-1.7B").parent  # a normal models folder is no backup
    assert paths.backup_root_of(live) is None


def test_an_empty_folder_stays_the_download_target(tmp_path):
    chosen = tmp_path / "D" / "Voxprint models"
    paths.set_models_dir(chosen)
    assert em.adopt_backup_choice() is None
    assert paths.models_dir() == chosen and paths.configured_models_dir() == chosen
    assert md.local_dir_for(REPO) == chosen / "Org--Tiny-1.7B"
    assert not em.restore_pending()


def test_models_dir_never_points_at_a_backup_even_before_it_is_adopted(usb, monkeypatch):
    paths.set_models_dir(usb)                                      # what an older installer wrote
    assert paths.models_dir() == paths.default_models_dir()
    assert md.local_dir_for(REPO) == paths.default_models_dir() / "Org--Tiny-1.7B"
    paths.set_models_dir(None)
    monkeypatch.setenv(paths.MODELS_DIR_ENV, str(usb / bk.BACKUP_DIRNAME / "models"))
    assert paths.models_dir() == paths.default_models_dir()
    assert em.backup_source() == usb / bk.BACKUP_DIRNAME           # still found as a source
    assert em.adopt_backup_choice() is None                        # the environment is not rewritten


def test_adopting_turns_the_backup_into_the_restore_source(usb):
    paths.set_models_dir(usb)
    assert em.adopt_backup_choice() == usb / bk.BACKUP_DIRNAME
    assert paths.configured_models_dir() is None and paths.models_dir() == paths.default_models_dir()
    assert em.configured() == usb / bk.BACKUP_DIRNAME
    assert em.adopt_backup_choice() is None                        # idempotent


def test_first_run_restores_models_voices_and_tools_into_the_normal_folders(usb):
    paths.set_models_dir(usb / bk.BACKUP_DIRNAME)                  # the user pointed at the backup folder itself
    before = snapshot(usb)
    assert em.pending([REPO]) and em.restore_pending()
    got = md.ensure_model(REPO, snapshot_download=no_download, revision=SHA)
    assert got == paths.default_models_dir() / "Org--Tiny-1.7B" and md.verify_local_model(got)
    assert (paths.voices_dir() / VOICE / "voice.json").is_file()
    assert (paths.app_home() / "tools" / "ffmpeg" / "ffmpeg.exe").is_file()
    assert snapshot(usb) == before                                 # the backup is only read
    assert paths.configured_models_dir() is None                   # the live folder is the default
    assert not em.restore_pending() and not em.pending([REPO])
    # the next start restores nothing again
    assert md.ensure_model(REPO, snapshot_download=no_download, revision=SHA) == got


def test_prefetch_restores_before_it_decides_what_to_download(usb):
    from workers import pipeline_runner as pr

    paths.set_models_dir(usb)
    calls = []
    got = pr.prefetch_models(repos=[REPO], ensure=lambda *a, **k: calls.append(a))
    assert got == [] and calls == []                               # nothing missing after the restore: no download
    assert md.verify_local_model(paths.default_models_dir() / "Org--Tiny-1.7B")


def test_an_updated_backup_is_restored_again_and_a_missing_drive_is_ignored(usb, tmp_path):
    root = usb / bk.BACKUP_DIRNAME
    em.set_folder(root)
    assert em.restore_backup() is not None and not em.restore_pending()
    data = bk.read_manifest(root)
    data["created"] = "2099-01-01T00:00:00Z"
    (root / bk.MANIFEST_NAME).write_text(json.dumps(data), encoding="utf-8")
    assert em.restore_pending()
    em.set_folder(tmp_path / "unplugged")
    assert em.backup_source() is None and not em.restore_pending() and em.restore_backup() is None


def test_existing_folder_accepts_a_singular_model_subfolder(tmp_path):
    src = tmp_path / "old"
    make_model(src / "model" / "Org--Tiny-1.7B")
    (src / "model" / "Org--Tiny-1.7B" / ".revision").write_text(SHA, encoding="utf-8")
    em.set_folder(src)
    assert ("existing", src / "model") in em.roots()
    found = em.find(REPO, SHA)
    assert found is not None and found.path == src / "model" / "Org--Tiny-1.7B"
