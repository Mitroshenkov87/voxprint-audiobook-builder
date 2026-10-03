"""UI of the online voices: remote cards in "My voices", the Narrate voice list, download on first use, the test-only
reminders and the live mini player in the Narrate window."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal  # noqa: E402

from core import i18n, voice_info  # noqa: E402
from core import narration as nr  # noqa: E402
from infra import voice_repository as repo  # noqa: E402
from tests.test_studio import (FakePreviewer, add_voice, app, fake_runner_factory, lib, make_studio, wait_for,  # noqa: E402,F401
                               write_book)
from tests.test_voice_catalog import SPEC, URL, index_for, package  # noqa: E402,F401
from tests.test_voice_library import make_adapter  # noqa: E402
from ui.mini_player import MiniPlayer  # noqa: E402
from ui.narrate_window import NarrateWindow  # noqa: E402
from ui.voices_window import RemoteCard, VoiceCard, VoicesWindow  # noqa: E402


def fake_download(tmp_path, calls=None):
    """A ``download(entry, library, progress=, cancel=)`` that adds a voice like the real one (no network)."""
    def dl(entry, library, progress=None, cancel=None, **kw):
        if calls is not None:
            calls.append(entry.id)
        if progress:
            progress(0.5, entry.name)
        rec = library.add_from_adapter(make_adapter(tmp_path / f"dl_{entry.id}", name=entry.name, license=entry.license))
        info = dict(rec.info, repo_id=entry.id, names=entry.names)
        voice_info.write_voice_json(rec.path, info)
        if progress:
            progress(1.0, entry.name)
        return library.get(rec.id)
    return dl


@pytest.fixture
def entries(tmp_path):
    _data, entry = package(tmp_path)
    return repo.parse_index(index_for(entry))


def cards(w, kind):
    return [w.cards_box.itemAt(i).widget() for i in range(w.cards_box.count()) if isinstance(w.cards_box.itemAt(i).widget(), kind)]


def test_my_voices_shows_remote_cards_with_badges_size_and_a_download_button(app, lib, tmp_path, entries):
    i18n.set_language("en")
    add_voice(lib, tmp_path, "Anna")
    w = VoicesWindow(lib, previewer=FakePreviewer(), auto_refresh=False)
    w.entries = entries
    w.refresh()
    rc = cards(w, RemoteCard)
    assert len(cards(w, VoiceCard)) == 1 and len(rc) == 1
    r = rc[0]
    assert r.lbl_name.text() == "Open Voice" and r.badge.text() and r.scope_badge.text()
    assert "MB" in r.lbl_meta.text() or "kB" in r.lbl_meta.text() or r.lbl_meta.text()
    assert "Test use only" in r.lbl_note.text() and r.btn_download.isEnabled() and r.lbl_state.text() == "Not downloaded yet"
    assert not w.empty.isVisibleTo(w)
    i18n.set_language("ru")
    w.retranslate()
    assert cards(w, RemoteCard)[0].lbl_name.text() == "Открытый голос" and "Только для тестов" in cards(w, RemoteCard)[0].lbl_note.text()
    i18n.set_language("de")
    w.retranslate()
    assert cards(w, RemoteCard)[0].btn_download.text() == "Herunterladen"


def test_download_button_fetches_and_the_card_becomes_a_local_voice(app, lib, tmp_path, entries):
    i18n.set_language("en")
    calls, changed = [], []
    w = VoicesWindow(lib, previewer=FakePreviewer(), download=fake_download(tmp_path, calls), auto_refresh=False)
    w.library_changed.connect(lambda: changed.append(1))
    w.entries = entries
    w.refresh()
    cards(w, RemoteCard)[0].btn_download.click()
    assert wait_for(lambda: not cards(w, RemoteCard) and len(cards(w, VoiceCard)) == 1)
    assert calls == ["open-voice"] and changed and "Downloaded" in w.lbl_status.text()
    local = cards(w, VoiceCard)[0]
    assert local.lbl_name.text() == "Open Voice" and local.lbl_note is not None and "Test use only" in local.lbl_note.text()


def test_list_refresh_uses_the_index_and_reports_offline_cache(app, lib, tmp_path, entries, monkeypatch):
    i18n.set_language("en")
    monkeypatch.delenv("VOXPRINT_VOICES_INDEX")                      # conftest blocks the real index; this test sets its own URL
    repo.set_index_url("https://example.org/index.json")
    res = []

    def fetch(url=None):
        res.append(1)
        return repo.IndexResult(voices=entries, error="unreachable", offline=True)
    w = VoicesWindow(lib, previewer=FakePreviewer(), fetch=fetch, auto_refresh=False)
    assert not cards(w, RemoteCard)
    w.btn_reload.click()
    assert wait_for(lambda: len(cards(w, RemoteCard)) == 1) and res
    assert "last known list" in w.lbl_status.text()
    w2 = VoicesWindow(lib, previewer=FakePreviewer(), fetch=fetch, auto_refresh=False)
    assert len(cards(w2, RemoteCard)) == 0                            # no cache file yet


def test_edit_dialog_keeps_the_stored_name_and_explains_the_test_only_licence(app, lib, tmp_path, entries):
    from ui.voices_window import VoiceEditDialog
    i18n.set_language("en")
    rec = fake_download(tmp_path)(entries[0], lib)
    dlg = VoiceEditDialog(rec)
    assert dlg.edt_name.text() == "Open Voice"                          # not the localized display name
    dlg.cmb_license.setCurrentIndex(dlg.cmb_license.findData(voice_info.LICENSE_TEST_ONLY))
    assert "Test use only" in dlg.lbl_license_note.text()


# ----------------------------------------------------------------------------- Narrate
def narrate(lib, tmp_path, entries, **kw):
    runner = kw.pop("runner", fake_runner_factory([]))
    return NarrateWindow(lib, runner=runner, out_dir=tmp_path / "out", aac_allowed=True, auto_open_folder=False,
                         model_state=lambda m: "needs_download", model_ensure=lambda m, progress: None,
                         auto_refresh=False, **kw)


def test_narrate_lists_online_voices_and_downloads_the_selected_one_on_start(app, lib, tmp_path, entries):
    i18n.set_language("en")
    add_voice(lib, tmp_path, "Anna")
    calls, jobs = [], []
    n = narrate(lib, tmp_path, entries, download=fake_download(tmp_path, calls), runner=fake_runner_factory(jobs))
    n.entries = entries
    n.refresh_voices()
    items = [n.cmb_voice.itemText(i) for i in range(n.cmb_voice.count())]
    assert items[0] == "Anna" and items[1].startswith("Open Voice (to download,")
    n.select_voice("repo:open-voice")
    assert "downloaded (and verified) when you start" in n.lbl_voice_info.text() and "Test use only" in n.lbl_voice_info.text()
    assert n.badge_box.count() == 2
    n.load_book_file(write_book(tmp_path))
    assert n.btn_start.isEnabled() and n.start()
    assert wait_for(lambda: bool(jobs) and not n.busy and n.result is not None)
    assert calls == ["open-voice"] and jobs[0].voice.info["repo_id"] == "open-voice"
    assert n.selected_voice_id() == jobs[0].voice.id and not n.selected_voice_id().startswith("repo:")
    assert "test use only" in n.lbl_status.text()                     # the reminder after the run
    assert sorted(n.cmb_voice.itemText(i) for i in range(n.cmb_voice.count())) == ["Anna", "Open Voice"]     # installed: no "to download"
    n.shutdown()


def test_failed_voice_download_does_not_start_and_keeps_the_remote_choice(app, lib, tmp_path, entries):
    from core.errors import VoiceRepositoryError
    i18n.set_language("en")
    jobs = []

    def bad(entry, library, progress=None, cancel=None, **kw):
        raise VoiceRepositoryError("The voice archive is damaged")
    n = narrate(lib, tmp_path, entries, download=bad, runner=fake_runner_factory(jobs))
    n.entries = entries
    n.refresh_voices()
    n.load_book_file(write_book(tmp_path))
    n.select_voice("repo:open-voice")
    assert n.start()
    assert wait_for(lambda: "damaged" in n.lbl_status.text() and n._dl_worker is not None and not n._dl_worker.isRunning())
    assert not jobs and n.selected_voice_id() == "repo:open-voice" and n.btn_start.isEnabled()
    n.shutdown()


def test_test_only_voice_notes_in_narrate(app, lib, tmp_path):
    i18n.set_language("en")
    add_voice(lib, tmp_path, "Orig", voice_info.LICENSE_TEST_ONLY)
    n = narrate(lib, tmp_path, [])
    n.refresh_voices()
    assert "Test use only" in n.lbl_voice_info.text() and "Personal use only" not in n.lbl_voice_info.text()
    for lang in ("ru", "de"):
        i18n.set_language(lang)
        n.refresh_voices()
        assert n.lbl_voice_info.text().count("\n") >= 1
    n.shutdown()


# ----------------------------------------------------------------------------- live player in Narrate
class FakeBackend(QObject):
    position, ended, failed = Signal(int), Signal(), Signal()

    def load(self, *a, **k):
        return True

    def play(self): ...
    def pause(self): ...
    def stop(self): ...
    def seek(self, ms): ...


def test_narrate_window_shows_the_live_player_when_the_job_announces_its_chunks(app, lib, tmp_path):
    i18n.set_language("en")
    add_voice(lib, tmp_path, "Anna")
    chunks = [tmp_path / f"k{i}.flac" for i in range(3)]
    sf.write(str(chunks[0]), np.zeros(16000, dtype="float32"), 16000)
    sf.write(str(tmp_path / "book.wav"), np.zeros(32000, dtype="float32"), 16000)

    def runner(job, progress, cancel, pause):
        job.on_plan(chunks)                                               # the real runner passes this to narrate_book
        for i in (1, 2):
            sf.write(str(chunks[i]), np.zeros(16000, dtype="float32"), 16000)
            progress(nr.NarrationProgress(i, 3, 1.0, f"Fragment {i} of 3"))
        return nr.NarrationResult(tmp_path, [tmp_path / "book.wav"], chapters=1, chunks=3)
    n = narrate(lib, tmp_path, [], runner=runner, player_backend=FakeBackend())
    n.show()
    n.refresh_voices()
    assert not n.player.isVisibleTo(n)
    n.load_book_file(write_book(tmp_path))
    assert n.start()
    assert wait_for(lambda: n.result is not None)
    assert n.player.isVisibleTo(n) and n.player.queue.planned == 1          # finished: the whole result is the playlist
    assert abs(n.player.queue.total - 2.0) < 0.01
    n.shutdown()
