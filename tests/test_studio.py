"""Studio home window, navigation, My voices, Narrate a book (offscreen Qt; fake runner/previewer/repository)."""
import json
import os
import time
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from core import audiobook_export as ex
from core import i18n, voice_info
from core import narration as nr
from core.voice_library import VoiceLibrary
from infra import text_models
from infra import voice_repository as repo
from tests.test_voice_library import make_adapter


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        QApplication.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def lib(tmp_path):
    return VoiceLibrary(tmp_path / "voices")


def add_voice(lib, tmp_path, name="Anna", license="custom/personal-only"):
    return lib.add_from_adapter(make_adapter(tmp_path / f"src_{name}", name=name, license=license))


class FakePreviewer(QObject):
    stopped = Signal()

    def __init__(self):
        super().__init__()
        self.current, self.played = None, []

    def play(self, path):
        self.current = Path(path)
        self.played.append(Path(path))

    def stop(self):
        self.current = None


def fake_runner_factory(calls, fail=False, delay=0.0):
    def runner(job, progress, cancel, pause):
        calls.append(job)
        for i in range(1, 4):
            pause.wait(cancel)
            cancel.check()
            progress(nr.NarrationProgress(i, 3, (3 - i) * 5.0, f"Fragment {i} of 3"))
            time.sleep(delay)
        if fail:
            from core.errors import NarrationError
            raise NarrationError("Synthesis failed", details="x")
        return nr.NarrationResult(Path(job.out_dir) / "B", [Path("a.opus")], chapters=2, chunks=3)
    return runner


def make_studio(lib, **kw):
    from ui.main_window import MainWindow
    from ui.narrate_window import NarrateWindow
    from ui.studio import StudioWindow
    from ui.voices_window import VoicesWindow

    previewer = kw.pop("previewer", FakePreviewer())
    runner = kw.pop("runner", fake_runner_factory([]))
    trainer = MainWindow(runner=lambda *a: None, autocheck=False, auto_open_folder=False)
    voices = VoicesWindow(lib, previewer=previewer, confirm=kw.pop("confirm", lambda t, m: True),
                          pick_folder=kw.pop("pick_folder", None), pick_zip=kw.pop("pick_zip", None),
                          repo_dialog_factory=kw.pop("repo_dialog_factory", None))
    narrate = NarrateWindow(lib, runner=runner, pick_book=kw.pop("pick_book", None),
                            pick_folder=kw.pop("pick_folder2", None), out_dir=kw.pop("out_dir", None),
                            aac_allowed=kw.pop("aac_allowed", True), auto_open_folder=False,
                            model_state=kw.pop("model_state", lambda m: "needs_download"),
                            model_ensure=kw.pop("model_ensure", lambda m, progress: None),
                            plan_builder=kw.pop("plan_builder", text_models.build_plan),
                            llm_status=kw.pop("llm_status", lambda: "needs_download"))
    return StudioWindow(library=lib, trainer=trainer, voices=voices, narrate=narrate, revoice=kw.pop("revoice", None))


# ----------------------------------------------------------------------------- Studio

def test_studio_has_three_cards_gear_and_the_product_name(app, lib):
    i18n.set_language("en")
    s = make_studio(lib)
    s.show_studio()
    assert s.windowTitle() == "Voxprint AI Audiobook Builder" and s.lbl_title.text() == "Voxprint AI Audiobook Builder"
    assert s.lbl_tagline.text() == "Train a voice, narrate books"
    assert [c.lbl_title.text() for c in (s.card_narrate, s.card_train, s.card_voices)] == [
        "Narrate a book", "Train your voice", "My voices"]
    assert s.card_narrate.property("primary") == "true" and s.card_train.property("primary") == "false"
    assert s.btn_gear is not None and s.btn_gear.isVisible() and not s.btn_back.isVisible()
    s.shutdown()


def test_narrate_card_hints_to_pick_or_train_a_voice_when_none_exist(app, lib, tmp_path):
    i18n.set_language("en")
    s = make_studio(lib)
    s.show_studio()
    assert "train your own or import one" in s.card_narrate.lbl_note.text() and s.card_narrate.lbl_note.isVisibleTo(s)
    assert not s.card_voices.lbl_note.isVisibleTo(s)
    add_voice(lib, tmp_path)
    s.refresh()
    assert not s.card_narrate.lbl_note.isVisibleTo(s)
    assert s.card_voices.lbl_note.text() == "1 voice(s) in your library"
    s.shutdown()


def test_navigation_between_all_windows_and_back(app, lib):
    s = make_studio(lib)
    s.show_studio()
    for card, page, win in ((s.card_narrate, "narrate", s.narrate_window), (s.card_train, "train", s.trainer),
                            (s.card_voices, "voices", s.voices_window)):
        card.click()
        assert s.current_page == page and win.isVisible() and not s.isVisible()
        assert [w for w in s.pages.values() if w.isVisible()] == [win]          # exactly one window at a time
        assert win.btn_back.isVisibleTo(win)
        win.btn_back.click()
        assert s.current_page == "studio" and s.isVisible() and not win.isVisible()
    s.shutdown()


def test_windows_can_navigate_to_each_other(app, lib):
    s = make_studio(lib)
    s.show_studio()
    s.navigate("narrate")
    s.narrate_window.btn_to_train.click()                  # "no voice" shortcut buttons
    assert s.current_page == "train"
    s.navigate("narrate")
    s.narrate_window.btn_to_voices.click()
    assert s.current_page == "voices"
    s.voices_window.btn_train.click()
    assert s.current_page == "train"
    s.shutdown()


def test_back_button_and_title_of_the_training_window_only_inside_the_studio(app, lib):
    from ui.main_window import MainWindow
    alone = MainWindow(runner=lambda *a: None, autocheck=False, auto_open_folder=False)
    alone.show()
    assert not alone.btn_back.isVisible() and alone.windowTitle() == "Voxprint AI Audiobook Builder"
    s = make_studio(lib)
    i18n.set_language("en")
    s.retranslate_all()
    s.trainer.retranslate()
    assert s.trainer.windowTitle() == "Voxprint AI Audiobook Builder - Train your voice"
    assert s.trainer.btn_back.text() == "\u2190 Studio"
    s.shutdown()


def test_gear_opens_settings_and_language_switches_every_window(app, lib):
    i18n.set_language("en")
    s = make_studio(lib)
    s.show_studio()
    s.btn_gear.click()
    dlg = s.settings_dialog()
    assert dlg.isVisible() and dlg.btn_update.isEnabled()
    dlg.cmb_lang.setCurrentIndex(dlg.cmb_lang.findData("de"))
    assert i18n.get_language() == "de"
    assert s.card_narrate.lbl_title.text() == "Ein Buch vertonen" and s.lbl_tagline.text() == "Stimme trainieren, Bücher vertonen"
    assert s.voices_window.lbl_title.text() == "Meine Stimmen" and s.narrate_window.btn_start.text() == "Vertonung starten"
    assert s.trainer.lbl_title.text() == "Eigene Stimme trainieren" and s.btn_back.text() == "\u2190 Studio"
    dlg.cmb_lang.setCurrentIndex(dlg.cmb_lang.findData("ru"))
    assert s.card_voices.lbl_title.text() == "Мои голоса" and s.trainer.btn_back.text() == "\u2190 Студия"
    s.shutdown()


def test_language_switch_inside_the_training_window_updates_the_studio(app, lib):
    i18n.set_language("en")
    s = make_studio(lib)
    s.trainer.set_language("ru")
    assert s.card_narrate.lbl_title.text() == "Озвучить книгу" and s.narrate_window.btn_book.text() == "Выбрать книгу…"
    s.shutdown()


def test_language_is_locked_while_a_job_runs(app, lib, tmp_path):
    i18n.set_language("en")
    calls = []
    s = make_studio(lib, runner=fake_runner_factory(calls, delay=0.15))
    add_voice(lib, tmp_path)
    nw = s.narrate_window
    nw.refresh_voices()
    (tmp_path / "b.txt").write_text("Chapter 1\n\nHello world, this is a book.", encoding="utf-8")
    assert nw.load_book_file(tmp_path / "b.txt") and nw.start()
    assert s.busy and "in progress" in (s.refresh() or s.card_narrate.lbl_note.text())
    s.set_language("de")
    assert i18n.get_language() == "en"
    assert wait_for(lambda: not nw.busy)
    s.set_language("de")
    assert i18n.get_language() == "de"
    s.shutdown()


def test_status_line_and_progress_of_the_training_window_are_mirrored(app, lib):
    s = make_studio(lib)
    s.show_studio()
    s.trainer.lbl_status.setText("Downloading models 42%")
    assert s.lbl_status.text() == "Downloading models 42%"
    s.shutdown()


def test_shutdown_is_idempotent_and_closing_a_child_shuts_down(app, lib):
    s = make_studio(lib)
    s.show_studio()
    s.navigate("voices")
    s.voices_window.close()
    assert s._shutting_down
    s.shutdown()


# ----------------------------------------------------------------------------- My voices

def test_voices_window_cards_badges_and_empty_state(app, lib, tmp_path):
    i18n.set_language("en")
    s = make_studio(lib)
    v = s.voices_window
    v.show()
    assert v.empty.isVisibleTo(v) and v.card_ids() == []
    a = add_voice(lib, tmp_path, "Anna", "custom/personal-only")
    b = add_voice(lib, tmp_path, "Boris", "CC-BY-4.0")
    v.refresh()
    assert not v.empty.isVisibleTo(v) and set(v.card_ids()) == {a.id, b.id}
    cards = {v.cards_box.itemAt(i).widget().voice_id: v.cards_box.itemAt(i).widget() for i in range(v.cards_box.count())}
    assert cards[a.id].badge.property("commercial") == "false" and "Personal use only" in cards[a.id].badge.text()
    assert cards[b.id].badge.property("commercial") == "true" and "commercial use OK" in cards[b.id].badge.text()
    assert "English" in cards[a.id].lbl_meta.text() and "by Me" in cards[a.id].lbl_meta.text()
    assert cards[b.id].badge.toolTip().startswith("https://creativecommons.org/")
    s.shutdown()


def test_preview_plays_and_stops(app, lib, tmp_path):
    i18n.set_language("en")
    prev = FakePreviewer()
    s = make_studio(lib, previewer=prev)
    rec = add_voice(lib, tmp_path)
    v = s.voices_window
    v.refresh()
    card = v.cards_box.itemAt(0).widget()
    assert card.btn_preview.isEnabled() and "Preview" in card.btn_preview.text()
    card.btn_preview.click()
    assert prev.played == [rec.path / "ref_sample.wav"]
    assert "Stop" in v.cards_box.itemAt(0).widget().btn_preview.text()
    v.cards_box.itemAt(0).widget().btn_preview.click()
    assert prev.current is None
    (rec.path / "ref_sample.wav").unlink()
    v.refresh()
    assert not v.cards_box.itemAt(0).widget().btn_preview.isEnabled()
    s.shutdown()


def test_delete_asks_first(app, lib, tmp_path):
    i18n.set_language("en")
    asked = []
    s = make_studio(lib, confirm=lambda t, m: asked.append(m) or False)
    rec = add_voice(lib, tmp_path)
    v = s.voices_window
    v.delete_voice(rec.id)
    assert lib.get(rec.id) is not None and "Anna" in asked[0]                     # declined: nothing happens
    v._confirm = lambda t, m: True
    v.delete_voice(rec.id)
    assert lib.get(rec.id) is None and v.card_ids() == [] and "Deleted" in v.lbl_status.text()
    s.shutdown()


def test_import_folder_and_zip_with_errors(app, lib, tmp_path):
    i18n.set_language("en")
    src = make_adapter(tmp_path / "ext", "Imported")
    z = tmp_path / "v.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("adapter_model.safetensors", b"w")
        zf.writestr("adapter_config.json", "{}")
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"nope")
    picks = iter([str(src), str(z), str(bad), ""])
    s = make_studio(lib, pick_folder=lambda: next(picks), pick_zip=lambda: next(picks))
    v = s.voices_window
    v.import_folder_dialog()
    assert "Imported: Imported" in v.lbl_status.text() and len(v.card_ids()) == 1
    v.import_zip_dialog()
    assert len(v.card_ids()) == 2
    v.import_zip_dialog()                                                         # damaged archive: friendly error
    assert len(v.card_ids()) == 2 and v.lbl_status.text() == i18n.tr("err.voice_zip_unsafe")
    v.import_zip_dialog()                                                         # cancelled picker
    assert len(v.card_ids()) == 2
    s.shutdown()


def test_edit_license_changes_the_badge(app, lib, tmp_path):
    from ui.voices_window import VoiceEditDialog
    i18n.set_language("en")
    s = make_studio(lib)
    rec = add_voice(lib, tmp_path)
    dlg = VoiceEditDialog(lib.get(rec.id), s.voices_window)
    assert dlg.cmb_license.currentData() == "custom/personal-only" and "Personal use only" in dlg.lbl_license_note.text()
    dlg.edt_name.setText("Anna Prime")
    dlg.cmb_license.setCurrentIndex(dlg.cmb_license.findData("CC0-1.0"))
    assert "allows commercial use" in dlg.lbl_license_note.text()
    dlg.cmb_gender.setCurrentIndex(dlg.cmb_gender.findData("female"))
    s.voices_window.apply_edit(rec.id, dlg.values())
    got = lib.get("anna-prime") or lib.get(rec.id)                                # the folder follows the new name
    assert got is not None and lib.get(rec.id) is None
    assert got.name == "Anna Prime" and got.commercial_use and got.info["voice_type"] == "female"
    card = s.voices_window.cards_box.itemAt(0).widget()
    assert card.badge.property("commercial") == "true"
    s.shutdown()


def test_narrate_with_this_voice_opens_the_narrator_with_it_selected(app, lib, tmp_path):
    s = make_studio(lib)
    s.show_studio()
    a, b = add_voice(lib, tmp_path, "Anna"), add_voice(lib, tmp_path, "Boris")
    s.navigate("voices")
    card = next(s.voices_window.cards_box.itemAt(i).widget() for i in range(2)
                if s.voices_window.cards_box.itemAt(i).widget().voice_id == b.id)
    card.btn_narrate.click()
    assert s.current_page == "narrate" and s.narrate_window.selected_voice_id() == b.id
    s.shutdown()


def test_repository_dialog_lists_downloads_and_handles_empty_and_errors(app, lib, tmp_path):
    from ui.voices_window import RepoDialog
    i18n.set_language("en")
    import hashlib, io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("adapter_model.safetensors", b"w")
        zf.writestr("adapter_config.json", "{}")
    data = buf.getvalue()
    entry = {"id": "r1", "name": "Repo Voice", "language": "english", "license": "CC-BY-NC-4.0",
             "url": "https://x.example/r1.zip", "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    results = iter([repo.IndexResult(error="not_configured"), repo.IndexResult(), repo.IndexResult(error="unreachable"),
                    repo.IndexResult(voices=repo.parse_index({"voices": [entry]}))])
    dlg = RepoDialog(lib, fetch=lambda url=None: next(results),
                     download=lambda e, library, progress=None, cancel=None: library.import_zip(
                         _write(tmp_path / "dl.zip", data), overrides={"name": e.name, "license": e.license}))
    dlg.show()
    states = []
    for _ in range(3):
        dlg.refresh()
        assert wait_for(lambda: dlg.btn_refresh.isEnabled())
        states.append(dlg.lbl_state.text())
    assert "not set up yet" in states[0] and "no voices yet" in states[1] and "Cannot reach" in states[2]
    dlg.refresh()
    assert wait_for(lambda: dlg.list.count() == 1)
    assert "Repo Voice" in dlg.list.item(0).text() and "Personal use only" in dlg.list.item(0).text()
    assert not dlg.btn_download.isEnabled()
    dlg.list.item(0).setSelected(True)
    assert dlg.btn_download.isEnabled()
    added = []
    dlg.voices_added.connect(added.extend)
    dlg.download_selected()
    assert wait_for(lambda: len(added) == 1)
    rec = lib.get(added[0])
    assert rec.name == "Repo Voice" and rec.commercial_use is False
    dlg.close()


def _write(path, data):
    Path(path).write_bytes(data)
    return path


def test_repository_button_opens_the_dialog(app, lib):
    created = []

    class Spy:
        def __init__(self, library, parent):
            created.append(library)
            self.voices_added = type("S", (), {"connect": lambda self, f: None})()
        def refresh(self): created.append("refresh")
        def show(self): created.append("show")
        def exec(self): created.append("exec")
    s = make_studio(lib, repo_dialog_factory=lambda library, parent: Spy(library, parent))
    s.voices_window.btn_repo.click()
    assert created == [lib, "refresh", "show"]
    s.shutdown()


# ----------------------------------------------------------------------------- Narrate a book

def write_book(tmp_path):
    f = tmp_path / "book.txt"
    f.write_text("Chapter 1\n\nThe first chapter has some text in it.\n\nChapter 2\n\nAnd the second one has more.", encoding="utf-8")
    return f


def test_narrate_window_without_voices_guides_the_user(app, lib, tmp_path):
    i18n.set_language("en")
    s = make_studio(lib)
    n = s.narrate_window
    n.show()
    assert n.no_voice.isVisibleTo(n) and not n.cmb_voice.isVisibleTo(n) and not n.btn_start.isEnabled()
    assert "no voices yet" in n.lbl_no_voice.text()
    s.shutdown()


def test_format_defaults_groups_and_descriptions(app, lib):
    i18n.set_language("en")
    s = make_studio(lib)
    n = s.narrate_window
    n.show()
    assert n.selected_formats() == {ex.FORMAT_OPUS_SINGLE}
    assert not n.format_checks[ex.FORMAT_M4B].isChecked() and not n.format_checks[ex.FORMAT_MP3_CHAPTERS].isChecked()
    assert "VLC" in n.lbl_players.text() and "Audiobookshelf" in n.lbl_players.text() and "varies by player" in n.lbl_players.text()
    # every format has a name and a one-line "where it plays" in every language
    for lang in ("en", "ru", "de"):
        i18n.set_language(lang)
        from ui.narrate_window import format_texts
        texts = format_texts()
        assert set(texts) == set(ex.ALL_FORMATS) and all(a and b and a != b for a, b in texts.values())
    i18n.set_language("en")
    # other formats are behind the expander (it holds formats only: bitrates live under "Advanced")
    assert not n.other_box.isVisibleTo(n) and n.btn_other.text().endswith("Other formats")
    n.btn_other.setChecked(True)
    assert n.other_box.isVisibleTo(n) and n.btn_other.text().startswith("\u25be")
    for f in (ex.FORMAT_M4B_OPUS, ex.FORMAT_OPUS_CHAPTERS, ex.FORMAT_MP3_SINGLE, ex.FORMAT_FLAC_CHAPTERS, ex.FORMAT_WAV_CHAPTERS):
        assert n.format_checks[f].isVisibleTo(n) and n.format_desc[f].text()
    assert not n.other_box.isAncestorOf(n.spn_opus) and not n.other_box.isAncestorOf(n.btn_out)
    assert (n.spn_opus.value(), n.spn_mp3.value(), n.spn_aac.value()) == (32, 96, 64)
    s.shutdown()


def test_aac_option_has_a_note_and_the_disclaimer_appears_when_selected(app, lib):
    for lang, word in (("en", "patent"), ("ru", "патент"), ("de", "Patent")):
        i18n.set_language(lang)
        s = make_studio(lib)
        n = s.narrate_window
        n.show()
        assert n.aac_box.isVisibleTo(n) and n.note.isVisibleTo(n)
        assert word.lower() in n.lbl_aac_note.text().lower() and not n.lbl_aac_disclaimer.isVisibleTo(n)
        n.format_checks[ex.FORMAT_M4B].setChecked(True)
        assert n.lbl_aac_disclaimer.isVisibleTo(n) and word.lower() in n.lbl_aac_disclaimer.text().lower()
        assert ex.FORMAT_M4B in n.selected_formats()
        n.format_checks[ex.FORMAT_M4B].setChecked(False)
        assert not n.lbl_aac_disclaimer.isVisibleTo(n)
        s.shutdown()
    i18n.set_language("en")
    en = i18n.tr("narr.aac_disclaimer")
    for phrase in ("patent-encumbered", "does not provide a patent licence", "solely responsible", "Apple Books"):
        assert phrase in en


def test_aac_can_be_hidden_by_the_flag(app, lib):
    s = make_studio(lib, aac_allowed=False)
    n = s.narrate_window
    n.show()
    assert not n.aac_box.isVisibleTo(n)
    n.format_checks[ex.FORMAT_M4B].setChecked(True)                     # even if someone checks it programmatically
    assert ex.FORMAT_M4B not in n.selected_formats() and n.options().allow_aac is False
    s.shutdown()


def test_load_book_shows_info_and_errors(app, lib, tmp_path):
    i18n.set_language("en")
    s = make_studio(lib)
    n = s.narrate_window
    assert n.load_book_file(write_book(tmp_path))
    assert "2 chapters" in n.lbl_book_info.text() and "book" in n.lbl_book_info.text()
    bad = tmp_path / "x.docx"
    bad.write_bytes(b"x")
    assert not n.load_book_file(bad) and "Unsupported" in n.lbl_book_error.text()
    assert n.book is not None                                         # the previous book stays loaded
    s.shutdown()


def test_start_needs_book_voice_and_a_format(app, lib, tmp_path):
    s = make_studio(lib)
    n = s.narrate_window
    n.show()
    assert not n.start()
    add_voice(lib, tmp_path)
    n.refresh_voices()
    assert not n.btn_start.isEnabled()                                # no book yet
    n.load_book_file(write_book(tmp_path))
    assert n.btn_start.isEnabled()
    n.format_checks[ex.FORMAT_OPUS_SINGLE].setChecked(False)
    assert not n.btn_start.isEnabled()                                # no format
    s.shutdown()


def test_full_narration_run_with_progress_eta_and_result(app, lib, tmp_path):
    i18n.set_language("en")
    calls = []
    s = make_studio(lib, runner=fake_runner_factory(calls, delay=0.05), out_dir=tmp_path / "audiobooks")
    n = s.narrate_window
    n.show()
    rec = add_voice(lib, tmp_path, license="CC-BY-4.0")
    n.refresh_voices()
    n.load_book_file(write_book(tmp_path))
    n.format_checks[ex.FORMAT_MP3_CHAPTERS].setChecked(True)
    n.spn_opus.setValue(40)                                             # Advanced: exact bitrate -> custom quality
    assert n.current_preset() == "" and not any(b.isChecked() for b in n.preset_buttons.values())
    n.chk_titles.setChecked(False)
    assert n.start()
    assert n.busy and n.btn_pause.isVisibleTo(n) and n.btn_cancel.isVisibleTo(n) and not n.btn_start.isEnabled()
    assert wait_for(lambda: "left" in n.lbl_status.text() or not n.busy)
    assert wait_for(lambda: n.result is not None)
    job = calls[0]
    assert job.voice.id == rec.id and job.out_dir == tmp_path / "audiobooks" and job.book.title == "book"
    assert job.options.formats == {ex.FORMAT_OPUS_SINGLE, ex.FORMAT_MP3_CHAPTERS} and job.options.bitrates.opus_kbps == 40
    assert job.options.speak_titles is False and job.options.allow_aac is True
    assert n.progress.value() == 100 and n.lbl_ready.isVisibleTo(n) and n.btn_open.isVisibleTo(n)
    assert "2 chapters" in n.lbl_status.text() and wait_for(lambda: not n.busy)
    s.shutdown()


def test_pause_resume_and_cancel(app, lib, tmp_path):
    i18n.set_language("en")
    s = make_studio(lib, runner=fake_runner_factory([], delay=0.15))
    n = s.narrate_window
    n.show()
    add_voice(lib, tmp_path)
    n.refresh_voices()
    n.load_book_file(write_book(tmp_path))
    n.start()
    n.btn_pause.click()
    assert n.worker.pause_token.paused and n.btn_pause.text() == "Resume" and "Paused" in n.lbl_status.text()
    n.btn_pause.click()
    assert not n.worker.pause_token.paused and n.btn_pause.text() == "Pause"
    n.btn_pause.click()                                                # pause again, then cancel while paused
    n.btn_cancel.click()
    assert wait_for(lambda: not n.busy)
    assert wait_for(lambda: "Stopped" in n.lbl_status.text())
    assert n.result is None and not n.btn_cancel.isVisibleTo(n)
    assert n.btn_start.isEnabled()
    s.shutdown()


def test_failure_is_reported_and_start_is_possible_again(app, lib, tmp_path):
    i18n.set_language("en")
    s = make_studio(lib, runner=fake_runner_factory([], fail=True))
    n = s.narrate_window
    add_voice(lib, tmp_path)
    n.refresh_voices()
    n.load_book_file(write_book(tmp_path))
    n.start()
    assert wait_for(lambda: not n.busy and "Synthesis failed" in n.lbl_status.text())
    assert "Press Start to continue" in n.lbl_status.text() and n.btn_start.isEnabled() and n.result is None
    s.shutdown()


def test_non_commercial_voice_shows_a_note_and_badge(app, lib, tmp_path):
    i18n.set_language("en")
    s = make_studio(lib)
    n = s.narrate_window
    add_voice(lib, tmp_path, "Anna", "custom/personal-only")
    n.refresh_voices()
    assert "Personal use only" in n.lbl_voice_info.text()
    assert n.badge_box.count() == 2 and n.badge_box.itemAt(0).widget().property("commercial") == "false"
    add_voice(lib, tmp_path, "Zed", "CC0-1.0")
    n.refresh_voices(select=lib.list_voices()[0].id)
    s.shutdown()
