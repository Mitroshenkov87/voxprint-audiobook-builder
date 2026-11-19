"""The "Prepare the text" block, quality presets and the collapsed sections of the Narrate window."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from core import audiobook_export as ex
from core import i18n
from core import text_prep
from infra import text_models
from tests.test_studio import app, lib, make_studio, wait_for, add_voice, fake_runner_factory  # noqa: F401
from ui import narrate_window as nw
from ui.main_window import recommended_text

RU_BOOK = ("Глава 1\n\nВ 1999 г. он купил 3 книги и пошёл домой. Это было очень хорошо.\n\n"
           "Глава 2\n\nОн вернулся к 5 часам.")
EN_BOOK = "Chapter 1\n\nIn 1999 he bought 3 books, and Dr. Smith agreed.\n\nChapter 2\n\nThe second chapter has more text."


def book_file(tmp_path, text, name="b.txt"):
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return f


def test_prepare_text_is_one_switch_on_by_default_in_every_language(app, lib):
    s = make_studio(lib)
    n = s.narrate_window
    n.show()
    for lang in i18n.LANGS:
        i18n.set_language(lang)
        n.retranslate()
        assert n.chk_prepare.isChecked() and n.chk_prepare.isEnabled()
        assert n.chk_prepare.text() == recommended_text(i18n.tr("prep.one"))
        assert n.lbl_prep_desc.text() == i18n.tr("prep.one_d") and n.lbl_prep_desc.isVisibleTo(n)
    i18n.set_language("en")
    n.retranslate()
    assert n.selected_rule_steps() == set(text_prep.STEP_KEYS)
    assert "on by default" in n.lbl_prep_hint.text() and n.lbl_prep_title.text().startswith("3.")
    assert n.lbl_format_title.text().startswith("4.")
    assert not hasattr(n, "more_prep_box")
    s.shutdown()


def test_options_carry_the_one_switch(app, lib, tmp_path):
    i18n.set_language("en")
    s = make_studio(lib)
    n = s.narrate_window
    plan = n.options().prep
    assert plan.rules.steps == frozenset(text_prep.STEP_KEYS) and plan.neural == frozenset()
    assert n.options().yo is True and text_prep.STEP_YO in n.selected_rule_steps()
    n.chk_prepare.setChecked(False)
    assert not n.options().prep.enabled and n.options().yo is False
    n.chk_prepare.setChecked(True)
    assert n.selected_rule_steps() == set(nw.RULE_STEPS)
    s.shutdown()


def test_ai_step_needs_a_download_then_becomes_available(app, lib, tmp_path):
    i18n.set_language("en")
    state = {"v": text_models.STATE_NEEDS_DOWNLOAD}
    seen = []

    def ensure(model, progress):
        seen.append(model.key)
        progress(None, 0.5, "")
        state["v"] = text_models.STATE_READY

    s = make_studio(lib, model_state=lambda m: state["v"], model_ensure=ensure)
    n = s.narrate_window
    n.show()
    n.load_book_file(book_file(tmp_path, RU_BOOK))
    assert n.chk_prepare.isChecked() and n.chk_prepare.isEnabled()
    assert "download" in n.lbl_model_state.text() and "365" in n.lbl_model_state.text()
    assert n.btn_model_download.isVisibleTo(n) and n.selected_neural_steps() == set()
    assert n.download_model()
    assert wait_for(lambda: not n.btn_model_download.isVisibleTo(n))
    n.model_worker.wait(2000)
    assert seen == ["sage-ru"] and n.options().prep.neural == frozenset({"spellfix"})
    n.chk_prepare.setChecked(False)
    assert n.options().prep.neural == frozenset()
    s.shutdown()


def test_ai_step_failure_is_shown_and_retry_is_possible(app, lib, tmp_path):
    i18n.set_language("en")

    def ensure(model, progress):
        raise OSError("disk full")

    s = make_studio(lib, model_ensure=ensure)
    n = s.narrate_window
    n.show()
    assert n.download_model()
    assert wait_for(lambda: "disk full" in n.lbl_model_state.text())
    n.model_worker.wait(2000)
    assert n.btn_model_download.isVisibleTo(n) and n.btn_model_download.isEnabled()
    s.shutdown()


def test_ai_step_is_ready_by_default_for_russian_and_greyed_for_other_languages(app, lib, tmp_path):
    i18n.set_language("en")
    s = make_studio(lib, model_state=lambda m: text_models.STATE_READY)
    n = s.narrate_window
    n.show()
    assert n.chk_prepare.isEnabled() and n.chk_prepare.isChecked() and "downloaded" in n.lbl_model_state.text()
    assert n.options().prep.neural == frozenset({"spellfix"})           # no book yet: language unknown, model ready
    n.load_book_file(book_file(tmp_path, EN_BOOK))
    assert n.chk_prepare.isEnabled() and "Russian books only" in n.lbl_model_state.text()
    assert n.options().prep.neural == frozenset()
    n.load_book_file(book_file(tmp_path, RU_BOOK, "r.txt"))
    assert n.chk_prepare.isChecked() and n.options().prep.neural == frozenset({"spellfix"})
    s.shutdown()


def test_placeholders_are_not_checkboxes(app, lib):
    i18n.set_language("en")
    s = make_studio(lib)
    n = s.narrate_window
    n.show()
    for key in (text_models.STEP_PUNCT, text_models.STEP_STRESS, text_models.STEP_ROLES):
        assert not text_models.for_step(key).integrated
    assert n.options().prep.neural == frozenset()
    s.shutdown()


def test_presets_drive_the_bitrates_and_manual_edits_switch_to_custom(app, lib):
    i18n.set_language("en")
    s = make_studio(lib)
    n = s.narrate_window
    n.show()
    assert n.current_preset() == "standard" and n.preset_buttons["standard"].isChecked()
    assert [b.text() for b in n.preset_buttons.values()] == ["Compact", "Standard", "High"]
    for name, b in ex.QUALITY_PRESETS.items():
        n.preset_buttons[name].click()
        assert (n.spn_opus.value(), n.spn_mp3.value(), n.spn_aac.value()) == (b.opus_kbps, b.mp3_kbps, b.aac_kbps)
        assert [k for k, btn in n.preset_buttons.items() if btn.isChecked()] == [name]
        assert n.options().bitrates == b
    n.spn_mp3.setValue(100)
    assert n.current_preset() == "" and not any(btn.isChecked() for btn in n.preset_buttons.values())
    assert "Custom" in n.lbl_preset_info.text()
    n.spn_mp3.setValue(ex.QUALITY_PRESETS["high"].mp3_kbps)
    assert n.current_preset() == "high"
    s.shutdown()


def test_size_hint_follows_preset_and_formats(app, lib):
    i18n.set_language("en")
    s = make_studio(lib)
    n = s.narrate_window
    n.show()
    n.apply_preset("standard")
    assert "Opus 14 MB" in n.lbl_preset_info.text() and "MP3" not in n.lbl_preset_info.text()
    n.format_checks[ex.FORMAT_MP3_CHAPTERS].setChecked(True)
    assert "MP3 43 MB" in n.lbl_preset_info.text()
    n.format_checks[ex.FORMAT_M4B].setChecked(True)
    assert "AAC 29 MB" in n.lbl_preset_info.text()
    n.apply_preset("compact")
    assert "Opus 11 MB" in n.lbl_preset_info.text() and "Smallest" in n.lbl_preset_info.text()
    for lang in ("ru", "de", "uk", "lv"):
        i18n.set_language(lang)
        n.retranslate()
        assert "MB" in n.lbl_preset_info.text() or "МБ" in n.lbl_preset_info.text()
    s.shutdown()


def test_the_window_is_uncluttered_by_default_and_advanced_holds_the_details(app, lib, tmp_path):
    i18n.set_language("en")
    s = make_studio(lib)
    n = s.narrate_window
    n.show()
    n.load_book_file(book_file(tmp_path, EN_BOOK))
    assert not n.advanced_box.isVisibleTo(n) and not n.other_box.isVisibleTo(n)
    for w in (n.spn_opus, n.spn_mp3, n.spn_aac, n.btn_out, n.lbl_out, n.chk_titles, n.lbl_sample):
        assert not w.isVisibleTo(n) and n.advanced_box.isAncestorOf(w)
    for w in (n.btn_book, n.cmb_voice if lib.list_voices() else n.lbl_no_voice, n.chk_prepare,
              n.format_checks[ex.FORMAT_OPUS_SINGLE], n.preset_buttons["standard"], n.btn_start):
        assert w.isVisibleTo(n)
    n.btn_advanced.setChecked(True)
    assert n.advanced_box.isVisibleTo(n) and n.btn_advanced.text().startswith("\u25be")
    for w in (n.spn_opus, n.btn_out, n.chk_titles, n.lbl_sample):
        assert w.isVisibleTo(n)
    s.shutdown()


def test_sample_shows_the_prepared_text_and_follows_the_checkboxes(app, lib, tmp_path):
    i18n.set_language("en")
    s = make_studio(lib)
    n = s.narrate_window
    n.show()
    assert "Choose a book" in n.lbl_sample.text()
    n.btn_advanced.setChecked(True)
    n.load_book_file(book_file(tmp_path, EN_BOOK))
    txt = n.lbl_sample.text()
    assert "1999" not in txt and "nineteen ninety nine" in txt and "Doctor" in txt
    n.chk_prepare.setChecked(False)
    assert "nothing to change" in n.lbl_sample.text()
    s.shutdown()


def test_sample_for_a_russian_book(app, lib, tmp_path):
    i18n.set_language("ru")
    s = make_studio(lib)
    n = s.narrate_window
    n.show()
    n.btn_advanced.setChecked(True)
    n.load_book_file(book_file(tmp_path, RU_BOOK))
    assert "тысяча девятьсот девяносто девятом" in n.lbl_sample.text() and "1999" not in n.lbl_sample.text()
    i18n.set_language("en")
    s.shutdown()


def test_controls_are_locked_while_a_job_runs_and_the_plan_reaches_the_runner(app, lib, tmp_path):
    i18n.set_language("en")
    calls = []
    s = make_studio(lib, runner=fake_runner_factory(calls, delay=0.1))
    n = s.narrate_window
    n.show()
    add_voice(lib, tmp_path)
    n.refresh_voices()
    n.load_book_file(book_file(tmp_path, EN_BOOK))
    assert n.start()
    assert not n.chk_prepare.isEnabled()
    assert wait_for(lambda: not n.busy and calls)
    assert calls[0].options.prep.rules.steps == frozenset(text_prep.STEP_KEYS)
    assert wait_for(lambda: n.chk_prepare.isEnabled())
    s.shutdown()


def test_status_says_the_book_and_voice_are_chosen(app, lib, tmp_path):
    i18n.set_language("en")
    s = make_studio(lib)
    n = s.narrate_window
    n.show()
    assert n.lbl_status.text() == i18n.tr("narr.idle")
    n.load_book_file(book_file(tmp_path, EN_BOOK))
    assert n.lbl_status.text() == i18n.tr("narr.idle")
    add_voice(lib, tmp_path)
    n.refresh_voices()
    assert n.lbl_status.text() == i18n.tr("narr.ready")
    assert n.lbl_status.text() != i18n.tr("narr.idle")
    s.shutdown()


def test_speaker_preview_edits_marks_without_a_modal_exec(app, lib, tmp_path):
    from core import llm_text
    from tests.test_llm_text import FakeModel

    i18n.set_language("en")
    s = make_studio(lib, llm_status=lambda: "ready")
    n = s.narrate_window
    n.show()
    add_voice(lib, tmp_path)
    n.refresh_voices()
    n.load_book_file(book_file(tmp_path, EN_BOOK))

    def answer(prompt):
        text = prompt.rsplit("TEXT:\n", 1)[-1]
        count = max(1, len([p for p in text.split("\n\n") if p.strip()]))
        return "\n".join(["FEMALE: Ann"] * count)

    n.llm_plan = lambda: llm_text.LLMPlan(lambda: FakeModel(answer), "fake")
    n._refresh_buttons()
    n.chk_speakers.setChecked(True)
    assert n.chk_speakers.isEnabled() and n.preview_speakers()
    dlg = n._speaker_preview
    assert dlg.isVisible()
    combo = dlg.table.cellWidget(0, 0)
    combo.setCurrentIndex(max(0, combo.findData("narrator")))
    dlg.btn_ok.click()
    assert n.speaker_lines and n.speaker_lines[0].role == "narrator"
    s.shutdown()


def test_disabled_checkbox_indicator_is_styled():
    from ui.main_window import build_style

    css = build_style(False)
    assert "QCheckBox::indicator:disabled" in css and "QCheckBox::indicator:checked:disabled" in css
