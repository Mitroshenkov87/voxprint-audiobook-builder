"""Automatic quality steps: the step registry, the pre-selected recommended options and the A/B recommendation.  (The models
these steps need come with the first-start download: tests/test_components_flow.py.)"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from core import i18n, text_prep, voice_check
from infra import auto_steps as au
from infra import text_models
from tests.test_studio import app, lib, make_studio  # noqa: F401
from workers import preview_runner as pr


# --------------------------------------------------------------------------- registry
def test_steps_are_the_implemented_ones():
    assert au.STEPS == text_prep.STEP_KEYS + ("spellfix", "quality_check", "compare")
    assert text_models.get("sage-ru").integrated
    # nothing that is only announced may be in the list
    for later in (text_models.STEP_PUNCT, text_models.STEP_STRESS, text_models.STEP_TRANSLATE, text_models.STEP_ROLES):
        assert later not in au.STEPS


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
