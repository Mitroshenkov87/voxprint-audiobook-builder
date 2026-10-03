"""Maximum quality (auto): the step registry, the model needs, the Settings button, the pre-selected recommended options and
the A/B recommendation."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from core import i18n, text_prep, voice_check
from infra import auto_steps as au
from infra import model_downloader as md
from infra import text_models
from tests.test_studio import app, lib, make_studio, wait_for  # noqa: F401
from workers import preview_runner as pr


# --------------------------------------------------------------------------- registry and model needs
def test_steps_are_the_implemented_ones():
    assert au.STEPS == text_prep.STEP_KEYS + ("spellfix", "quality_check", "compare")
    assert text_models.get("sage-ru").integrated
    # nothing that is only announced may be in the list
    for later in (text_models.STEP_PUNCT, text_models.STEP_STRESS, text_models.STEP_TRANSLATE, text_models.STEP_ROLES):
        assert later not in au.STEPS


def test_needed_models_lists_only_what_is_missing(monkeypatch):
    states = {md.ALIGNER_REPO: md.STATE_READY, md.ASR_REPO: md.STATE_MISSING}
    monkeypatch.setattr(md, "model_state", lambda r: states[r])
    monkeypatch.setattr(text_models, "state", lambda m: text_models.STATE_NEEDS_DOWNLOAD)
    keys = [n.key for n in au.needed_models()]
    assert keys == [md.ASR_REPO, "sage-ru"]
    assert [n.key for n in au.needed_models(au.RULE_STEPS)] == []        # rule steps need no model
    assert [n.key for n in au.needed_models({au.STEP_SPELLFIX})] == ["sage-ru"]
    states[md.ASR_REPO] = md.STATE_PARTIAL                                  # a partial download is resumed
    assert md.ASR_REPO in [n.key for n in au.needed_models({au.STEP_CHECK})]
    monkeypatch.setattr(text_models, "state", lambda m: text_models.STATE_READY)
    states[md.ASR_REPO] = md.STATE_READY
    assert au.needed_models() == []


def test_download_models_reports_overall_progress():
    needs = [au.ModelNeed("a/b", "hf", "A", 3.0), au.ModelNeed("sage-ru", "text", "S", 1.0)]
    seen, calls = [], []
    n = au.download_models(needs, lambda f, m="": seen.append(f),
                           ensure_hf=lambda repo, prog: (calls.append(repo), prog(None, 0.5, "x"), prog(None, 1.0, "y")),
                           ensure_text=lambda model, prog: (calls.append(model.key), prog(None, 1.0, "z")))
    assert n == 2 and calls == ["a/b", "sage-ru"]
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0) and seen[0] == pytest.approx(0.375)     # half of 3 of 4 GB


# --------------------------------------------------------------------------- A/B recommendation
def item(key, verdict="good", wer=0.05, st=0.3):
    return pr.PreviewItem(key, key, 3, 32, 64, 1e-6, 4, Path("x.wav"), 8.0, 10.0,
                          {"verdict": verdict, "wer": wer, "semitones": st}, 8)


def test_recommend():
    assert pr.recommend([]) is None
    assert pr.recommend([item("A")]) == "A"
    assert pr.recommend([item("A", voice_check.WARN), item("B", voice_check.GOOD)]) == "B"
    assert pr.recommend([item("A", voice_check.GOOD), item("B", voice_check.BAD)]) == "A"
    assert pr.recommend([item("A", wer=0.20), item("B", wer=0.05)]) == "B"
    assert pr.recommend([item("A", wer=0.05), item("B", wer=0.04)]) == "A"            # within the noise: the cheaper A
    assert pr.recommend([item("A", st=2.0), item("B", st=0.2)]) == "B"
    assert pr.recommend([item("A", st=0.3), item("B", st=0.1)]) == "A"
    assert pr.recommend([item("A", wer=None), item("B", wer=0.1)]) == "B"             # no recognition result loses


# --------------------------------------------------------------------------- UI
def test_training_window_preselects_and_marks_the_best(app, lib):
    i18n.set_language("en")
    s = make_studio(lib)
    try:
        t = s.trainer
        assert t.chk_check.isChecked() and t.chk_compare.isChecked()
        assert t.chk_compare.property("recommended") == "true"
        assert "recommended" in t.chk_compare.text()
        t.chk_compare.setChecked(False)
        t.chk_check.setChecked(False)
        s.reload_auto_steps()
        assert t.chk_check.isChecked() and t.chk_compare.isChecked()
        t.show_previews([item("A", wer=0.3), item("B", wer=0.05)])
        (rowa, _pa, usea), (rowb, _pb, useb) = t.preview_rows
        assert rowb.property("recommended") == "true" and rowa.property("recommended") != "true"
        assert useb.property("recommended") == "true" and usea.property("recommended") != "true"
        t.show_previews([item("A")])                     # a single variant is not "recommended" over anything
        assert t.preview_rows[0][0].property("recommended") != "true"
    finally:
        s.shutdown()


def test_narrate_window_preselects_and_marks_the_best(app, lib):
    i18n.set_language("en")
    s = make_studio(lib)
    try:
        n = s.narrate_window
        for key in au.RULE_STEPS:
            assert n.prep_checks[key].isChecked() and n.prep_checks[key].property("recommended") == "true"
            assert "recommended" in n.prep_checks[key].text()
        n.prep_checks[text_prep.STEP_LINKS].setChecked(False)
        s.reload_auto_steps()
        assert n.prep_checks[text_prep.STEP_LINKS].isChecked()
    finally:
        s.shutdown()


def test_settings_button_downloads_missing_models(app, lib):
    i18n.set_language("en")
    s = make_studio(lib)
    try:
        d = s.settings_dialog()
        assert d.btn_max.text() == "Maximum quality (auto)"
        s.trainer.chk_compare.setChecked(False)
        d.needed_models = lambda: []
        assert d.start_max_quality() is False                      # nothing to download, but the options are selected
        assert s.trainer.chk_compare.isChecked()
        assert "already downloaded" in d.lbl_max_status.text()

        got = []

        def job(needs, progress):
            got.append([n.key for n in needs])
            progress(0.5, "half")
            progress(1.0, "full")
            return len(needs)

        d.needed_models = lambda: [au.ModelNeed("sage-ru", "text", "SAGE", 0.35)]
        d.models_job = job
        assert d.start_max_quality() is True
        assert wait_for(lambda: not d.max_running)
        from PySide6.QtWidgets import QApplication

        for _ in range(5):
            QApplication.processEvents()
        assert got == [["sage-ru"]] and d.btn_max.isEnabled()
        assert "1 model" in d.lbl_max_status.text() and not d.bar_max.isVisible()

        def bad(needs, progress):
            raise RuntimeError("offline")

        d.models_job = bad
        assert d.start_max_quality() is True
        assert wait_for(lambda: not d.max_running)
        for _ in range(5):
            QApplication.processEvents()
        assert "offline" in d.lbl_max_status.text() and d.btn_max.isEnabled()
    finally:
        s.shutdown()


def test_texts_in_every_language(app, lib):
    s = make_studio(lib)
    try:
        d = s.settings_dialog()
        seen = set()
        for lang in ("en", "ru", "de"):
            i18n.set_language(lang)
            d.retranslate()
            assert d.btn_max.text() and d.lbl_max_hint.text() and "auto." not in d.btn_max.text()
            seen.add(d.btn_max.text())
        assert len(seen) == 3
    finally:
        s.shutdown()
