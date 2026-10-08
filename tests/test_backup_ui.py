"""The backup / restore / existing-models-folder section of the Settings dialog."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from core import i18n
from infra import backup as bk
from infra import existing_models as em
from infra import model_downloader as md
from infra import paths
from tests.test_backup import REPO, SHA, add_voice, own_model
from tests.test_studio import app, lib, make_studio, wait_for  # noqa: F401


@pytest.fixture
def dlg(app, lib):
    i18n.set_language("en")
    s = make_studio(lib)
    d = s.settings_dialog()
    d.notices, d.questions = [], []
    d.notify = lambda t, m: d.notices.append(m)

    def confirm(t, m):
        d.questions.append(m)
        return d.answer

    d.answer = True
    d.confirm = confirm
    yield d
    s.shutdown()


def finished(d):
    """The worker is done and its queued signals (done / failed / cancelled) were delivered."""
    from PySide6.QtWidgets import QApplication

    ok = wait_for(lambda: not d.backing_up) and (d.backup_worker is None or d.backup_worker.wait(3000))
    for _ in range(5):
        QApplication.processEvents()
    return ok


def test_section_texts_in_every_language(dlg):
    for lang in ("en", "ru", "de"):
        i18n.set_language(lang)
        dlg.retranslate()
        for w in (dlg.lbl_backup_title, dlg.chk_models, dlg.chk_voices, dlg.chk_link, dlg.lbl_link_warn,
                  dlg.btn_backup, dlg.btn_restore, dlg.btn_repair_restore, dlg.lbl_existing_title,
                  dlg.lbl_existing_hint, dlg.btn_existing, dlg.btn_existing_clear):
            assert w.text().strip(), (lang, w)
    i18n.set_language("en")
    dlg.retranslate()
    assert dlg.btn_backup.text().startswith("Back up models") and "voices" in dlg.btn_backup.text()
    assert dlg.btn_restore.text().startswith("Restore from backup")
    assert dlg.btn_repair_restore.text() == "Restore from backup folder"
    assert dlg.chk_models.isChecked() and dlg.chk_voices.isChecked() and not dlg.chk_link.isChecked()
    assert not dlg.bar_backup.isVisibleTo(dlg) and not dlg.btn_backup_cancel.isVisibleTo(dlg)


def test_backup_asks_shows_sizes_runs_and_reports(dlg, tmp_path):
    own_model()
    add_voice("anna")
    target = tmp_path / "usb"
    target.mkdir()
    dlg.pick_folder = lambda title: str(target)
    assert dlg.start_backup()
    assert len(dlg.questions) == 1 and str(target) in dlg.questions[0] and "files" in dlg.questions[0]
    assert finished(dlg)
    assert "files copied" in dlg.lbl_backup_status.text() and "already in place" in dlg.lbl_backup_status.text()
    assert (target / bk.BACKUP_DIRNAME / bk.MANIFEST_NAME).is_file() and (target / bk.BACKUP_DIRNAME / "voices" / "anna").is_dir()
    assert not dlg.bar_backup.isVisibleTo(dlg) and dlg.btn_backup.isEnabled()
    # the second run says everything was in place
    assert dlg.start_backup() and finished(dlg)
    assert "0 files copied" in dlg.lbl_backup_status.text()


def test_voices_can_be_left_out(dlg, tmp_path):
    own_model()
    add_voice("anna")
    target = tmp_path / "usb"
    target.mkdir()
    dlg.pick_folder = lambda title: str(target)
    dlg.chk_voices.setChecked(False)
    assert dlg.start_backup() and finished(dlg)
    assert not (target / bk.BACKUP_DIRNAME / "voices").exists()


def test_declining_the_question_or_cancelling_the_picker_does_nothing(dlg, tmp_path):
    own_model()
    target = tmp_path / "usb"
    target.mkdir()
    dlg.pick_folder = lambda title: ""
    assert not dlg.start_backup() and not dlg.questions
    dlg.pick_folder = lambda title: str(target)
    dlg.answer = False
    assert not dlg.start_backup() and len(dlg.questions) == 1
    assert not (target / bk.BACKUP_DIRNAME).exists()


def test_nothing_to_back_up_is_explained(dlg, tmp_path):
    dlg.pick_folder = lambda title: str(tmp_path)
    assert not dlg.start_backup()
    assert "Nothing to back up" in dlg.notices[-1]


def test_not_enough_space_message_reaches_the_dialog(dlg, tmp_path, monkeypatch):
    own_model()
    target = tmp_path / "usb"
    target.mkdir()
    dlg.pick_folder = lambda title: str(target)
    import shutil
    from types import SimpleNamespace

    real = bk.free_bytes
    monkeypatch.setattr(bk.shutil, "disk_usage", lambda p: SimpleNamespace(total=1, used=0, free=1000))
    assert dlg.start_backup() and finished(dlg)
    assert "Not enough free space" in dlg.lbl_backup_status.text()
    assert not list(target.rglob("*.safetensors"))


def test_buttons_are_locked_while_a_task_runs(dlg, monkeypatch):
    monkeypatch.setattr(type(dlg._win), "busy", property(lambda self: True))
    dlg.refresh()
    assert not dlg.btn_backup.isEnabled() and not dlg.btn_restore.isEnabled() and not dlg.btn_existing.isEnabled()
    assert not dlg.btn_repair_restore.isEnabled() and not dlg.chk_models.isEnabled() and not dlg.chk_link.isEnabled()
    assert not dlg.start_backup() and "Wait until" in dlg.notices[-1]
    assert not dlg.start_restore()


def test_cancel_stops_the_backup_and_says_how_to_continue(dlg, tmp_path):
    own_model()
    target = tmp_path / "usb"
    target.mkdir()
    gate = threading.Event()

    def slow_job(path, include, progress, cancel, items=None):
        progress(0.1, "x")
        gate.wait(5)
        cancel.check()
        raise AssertionError("not reached")

    dlg.backup_job = slow_job
    dlg.pick_folder = lambda title: str(target)
    assert dlg.start_backup()
    assert dlg.backing_up and dlg.btn_backup_cancel.isVisibleTo(dlg) and not dlg.btn_backup.isEnabled()
    assert wait_for(lambda: dlg.bar_backup.value() == 10)
    dlg.btn_backup_cancel.click()
    gate.set()
    assert finished(dlg)
    assert "Cancelled" in dlg.lbl_backup_status.text() and dlg.btn_backup.isEnabled()


def test_restore_validates_the_folder_then_restores_and_reports_conflicts(dlg, tmp_path):
    own_model()
    add_voice("anna")
    target = tmp_path / "usb"
    target.mkdir()
    dlg.pick_folder = lambda title: str(target)
    assert dlg.start_backup() and finished(dlg)
    # a folder that is no backup
    dlg.pick_folder = lambda title: str(tmp_path)
    assert not dlg.start_restore()
    assert "No Voxprint backup" in dlg.lbl_backup_status.text() and dlg.notices
    # restore into an empty home, one voice already exists with other content
    import shutil

    shutil.rmtree(paths.models_dir())
    shutil.rmtree(paths.voices_dir())
    add_voice("anna", text="newer")
    dlg.pick_folder = lambda title: str(target)
    assert dlg.start_restore() and finished(dlg)
    assert md.verify_local_model(md.local_dir_for(REPO))
    assert "Not overwritten" in dlg.lbl_backup_status.text() and "anna" in dlg.lbl_backup_status.text()
    assert "Restore" in dlg.questions[-1] or "checksum" in dlg.questions[-1]


def test_existing_models_folder_is_saved_imported_and_cleared(dlg, tmp_path):
    from tests.test_model_locator import make_model

    old = tmp_path / "old_models"
    make_model(old / "Org--Tiny-1.7B")
    (old / "Org--Tiny-1.7B" / ".revision").write_text(SHA, encoding="utf-8")
    dlg.import_job = lambda prog, cancel: em.import_available([REPO], prog, cancel)
    dlg.pick_folder = lambda title: str(old)
    assert dlg.choose_existing_folder()
    assert str(old) in dlg.lbl_existing.text() and em.configured() == old
    assert "Import the models" in dlg.questions[-1]
    assert finished(dlg)
    assert md.verify_local_model(md.local_dir_for(REPO)) and "files copied" in dlg.lbl_backup_status.text()
    dlg.clear_existing_folder()
    assert em.configured() is None and "No folder" in dlg.lbl_existing.text() and not dlg.btn_existing_clear.isEnabled()
    # the choice can be made without importing right away
    dlg.answer = False
    assert dlg.choose_existing_folder() and em.configured() == old and not dlg.backing_up
