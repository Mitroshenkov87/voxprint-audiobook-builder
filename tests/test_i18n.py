"""Localization: the single English catalog, language detection and switching, no hard-coded user strings in code."""
import json
import re
from pathlib import Path

import pytest

from core import i18n
from core.i18n import LANGS, tr

ROOT = Path(__file__).resolve().parents[1]
LOC = ROOT / "locales"
PH = re.compile(r"\{(\w+)\}")
CYR = re.compile("[А-Яа-яЁё]")


def _cat(lang):
    return json.loads((LOC / f"{lang}.json").read_text(encoding="utf-8"))


def test_english_is_the_only_ui_language_and_catalog():
    assert LANGS == ("en",)
    assert i18n.LANG_NAMES == {"en": "English"}
    assert set(i18n.LANG_NAMES) == set(LANGS)
    # no stray catalogs of dropped/unsupported languages
    assert sorted(p.stem for p in (ROOT / "locales").glob("*.json")) == sorted(LANGS)
    base = _cat("en")
    assert len(base) > 150
    for k, v in base.items():
        assert v.strip(), k


def test_english_catalog_has_no_cyrillic():
    en = _cat("en")
    # The egg caption is the same Russian line in every catalog, including English.
    kept = {"easter_egg.caption"}
    assert not [k for k, v in en.items() if k not in kept and CYR.search(v)]


def test_every_key_used_in_code_exists():
    used = set()
    for f in list(ROOT.glob("core/*.py")) + list(ROOT.glob("infra/*.py")) + list(ROOT.glob("workers/*.py")) + \
            list(ROOT.glob("ui/*.py")):
        used |= set(re.findall(r"\btr\(\s*[\"']([a-z0-9_.]+)[\"']", f.read_text(encoding="utf-8")))
    used |= {f"about.step{i}" for i in range(1, 6)} | {f"stage.{s}" for s in
                                                      ("updates", "model", "align", "slice", "train", "save")}
    used |= {f"about.kind_{k}" for k in ("model", "library", "tool", "asset")}
    used |= {f"preset.{c}" for c in ("fast", "balanced", "maximum", "manual")} | {f"preset.desc_{c}" for c in ("fast", "balanced", "maximum", "manual")}
    used |= {f"consent.{k}_{c}" for k in ("scope", "badge", "tip") for c in ("commercial", "public_noncommercial", "private_only")}
    used |= {f"consent.mode_{c}" for c in ("auto", "manual", "none")} | {f"consent.hint_{c}" for c in ("auto", "manual", "none")}
    used |= {f"check.sugg_{c}" for c in ("no_stop", "quiet", "wer", "pitch", "sim", "mos")} | {f"check.verdict_{c}" for c in ("good", "warn", "bad")}
    used |= {f"preset.adv_{c}" for c in ("epochs", "rank", "alpha", "lr", "accum")}     # dynamic keys of the Train window presets
    from infra.install_state import ALL_CODES
    used |= {f"health.{c}" for c in ALL_CODES}          # install-check reason codes (dynamic key)
    from ui.std_buttons import KEYS as BUTTON_KEYS
    used |= set(BUTTON_KEYS.values())                     # standard dialog buttons (ui/std_buttons.py)
    en = _cat("en")
    used.discard("stage.")
    used -= {"preset.", "preset.desc_", "preset.adv_", "consent.badge_", "consent.hint_", "consent.mode_", "consent.scope_", "consent.tip_", "check.sugg_", "check.verdict_"}             # prefix of the dynamic keys stage.<stage>
    assert used and not (used - set(en)), used - set(en)
    unused = set(en) - used
    assert not unused, unused            # no unused keys in the catalog


def test_no_cyrillic_user_literals_left_in_code():
    """All user messages go through tr(): no Russian string literals are left in the code
    (except the normalizer's abbreviation tables and the text of the update smoke-test script)."""
    import ast
    bad = []
    for f in list(ROOT.glob("core/*.py")) + list(ROOT.glob("infra/*.py")) + list(ROOT.glob("workers/*.py")) + \
            list(ROOT.glob("ui/*.py")):
        # exceptions: abbreviation tables, language names, the bilingual USAGE.txt, the update smoke-test script,
        # service comments of the requirements-file generator
        if f.name in ("normalizer.py", "text_utils.py", "i18n.py", "model_export.py", "updater.py",
                      "verified_manifest.py", "book_parsers.py", "num_words.py", "text_prep.py", "text_cleanup.py", "asr_dataset.py", "asr.py", "consent.py", "script_match.py", "spoken_date.py", "selftest_narrate.py", "selftest_text.py", "preview_runner.py", "voice_check.py",
                      "ai_disclosure.py", "languages.py", "llm_text.py",
                      "chunker.py", "pace.py", "ordinals.py", "ordinal_rules.py"):   # Russian word tables (llm_text: refusal patterns; chunker: conjunctions; pace: solemn words; ordinals: trigger nouns, endings, months) and rules (data); the disclosure is spoken in the BOOK language
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        doc = {id(n.body[0].value) for n in ast.walk(tree)
               if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
               and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and CYR.search(n.value) \
                    and id(n) not in doc:
                bad.append(f"{f.name}:{n.lineno}")
    assert bad == [], bad


def test_tr_fallbacks(monkeypatch):
    i18n.set_language("en")
    assert tr("ui.btn_dataset") == _cat("en")["ui.btn_dataset"]
    assert tr("no.such.key") == "no.such.key"
    assert tr("progress.pieces") == _cat("en")["progress.pieces"]            # missing parameter - no exception
    assert tr("progress.pieces", n=3) == _cat("en")["progress.pieces"].format(n=3)
    # a dropped UI language falls back to English
    monkeypatch.setenv("VOXPRINT_LANG", "de")
    i18n.reset()
    assert tr("ui.btn_dataset") == _cat("en")["ui.btn_dataset"]
    assert i18n.tr_lang("ui.ready", "ru") == _cat("en")["ui.ready"]
    monkeypatch.delenv("VOXPRINT_LANG", raising=False)
    i18n.reset()


def test_normalize_code():
    n = i18n.normalize_code
    assert n("EN") == "en" and n("en-US") == "en"
    # the UI is English-only: every other code (incl. the former UI languages) is not offered
    assert n("ru-RU") is None and n("de_AT.UTF-8") is None and n("de-AT") is None
    assert n("uk_UA.UTF-8") is None and n("lv_LV.UTF-8") is None
    assert n("be-BY") is None                                          # Belarusian was never offered
    assert n("fr-FR") is None and n("") is None and n(None) is None and n("C") is None


def test_detection_order(monkeypatch, tmp_path):
    for v in ("LC_ALL", "LC_MESSAGES", "LANG", "VOXPRINT_LANG"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(i18n.locale, "getlocale", lambda *a: (None, None))
    monkeypatch.setattr(i18n.sys, "platform", "linux")      # on Windows the OS language comes from GetUserDefaultLocaleName, not from LANG
    assert i18n.detect_language() == "en"                       # the default is English
    i18n.reset()
    assert tr("ui.btn_lora") == "Train the voice"
    monkeypatch.setenv("LANG", "fr_FR.UTF-8")
    assert i18n.detect_language() == "en"                       # unsupported system language -> English
    monkeypatch.setenv("LANG", "uk_UA.UTF-8")
    assert i18n.detect_language() == "en"                       # a dropped UI language -> English
    monkeypatch.setenv("LANG", "lv_LV.UTF-8")
    assert i18n.detect_language() == "en"
    monkeypatch.setenv("LANG", "ru_RU.UTF-8")
    assert i18n.detect_language() == "en"                       # Russian stays a content language, not a UI language
    i18n.set_language("en", persist=True)
    assert i18n.saved_language() == "en" and i18n.detect_language() == "en"
    monkeypatch.setenv("VOXPRINT_LANG", "en")
    assert i18n.detect_language() == "en"                       # the environment variable beats everything
    monkeypatch.setenv("VOXPRINT_LANG", "ru")
    assert i18n.detect_language() == "en"                       # a dropped code is ignored
    monkeypatch.setenv("VOXPRINT_LANG", "xx")
    assert i18n.detect_language() == "en"                       # an unknown code is ignored


def test_windows_locale_name(monkeypatch):
    import ctypes
    import sys

    class K32:
        @staticmethod
        def GetUserDefaultLocaleName(buf, n):
            buf.value = "de-DE"
            return 5

    class Win:
        kernel32 = K32()

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", Win(), raising=False)
    assert i18n.system_language() is None               # German is a dropped UI language -> not offered


def test_stage_labels_follow_language():
    from core.events import Stage
    i18n.set_language("en")
    assert Stage.ALIGN.label == "Alignment"
    assert Stage.TRAIN.label == "Training the voice"


def test_errors_use_current_language():
    from core.errors import CancelledByUser, OutOfMemoryError_
    i18n.set_language("en")
    assert CancelledByUser().user_message == "The operation was cancelled."
    assert OutOfMemoryError_().user_message == "Not enough video memory (VRAM)."


# ------------------------------------------------------------------ UI
@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_ui_language_switcher_retranslates_and_persists(app):
    from ui.main_window import MainWindow
    w = MainWindow(runner=lambda *a: None, autocheck=False, auto_open_folder=False)
    w.show()
    cmb = w.settings_dialog().cmb_lang
    assert cmb.count() == len(i18n.LANGS) == 1 and cmb.currentData() == "en"
    assert [cmb.itemText(i) for i in range(cmb.count())] == ["English"]
    w.set_language("en")
    assert w.btn_lora.text() == "Train the voice" and w.btn_merge.text() == "Build a voice for other apps"
    assert w.lbl_status.text() == "Choose the audio and the text - I will do the rest."
    assert w.lbl_audio.text().startswith("No file chosen")
    assert w._chips[next(iter(w._chips))].text() == "Updates"
    # one language: the switcher cannot change anything, but the persistence machinery still stores the choice
    assert i18n.set_language("en", persist=True) == "en"
    assert i18n.saved_language() == "en"
    w.set_language("ru")                                         # dropped language: ignored, English stays
    assert i18n.get_language() == "en" and w.settings_dialog().cmb_lang.currentData() == "en"
    w.set_language("be")                                         # not offered: ignored
    assert i18n.get_language() == "en" and w.settings_dialog().cmb_lang.currentData() == "en"
    # the chosen file is kept when the language changes
    w.set_audio(Path("/tmp/voice.wav"))
    w.set_language("en")
    assert w.lbl_audio.text() == "voice.wav"
    w.close()


def test_ui_language_switch_disabled_while_busy(app, monkeypatch):
    from ui.main_window import MainWindow
    w = MainWindow(runner=lambda *a: None, autocheck=False, auto_open_folder=False)
    monkeypatch.setattr(MainWindow, "busy", property(lambda self: True))
    w._refresh_buttons()
    dlg = w.settings_dialog()
    assert not dlg.cmb_lang.isEnabled() and not dlg.btn_update.isEnabled()
    before = i18n.get_language()
    dlg.cmb_lang.setCurrentIndex(0)
    assert i18n.get_language() == before                       # the language does not change while a task is running
    w.close()


def test_ui_defaults_to_english_without_settings(app, monkeypatch):
    for v in ("LC_ALL", "LC_MESSAGES", "LANG", "VOXPRINT_LANG"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(i18n.locale, "getlocale", lambda *a: (None, None))
    i18n.reset()
    from ui.main_window import MainWindow
    w = MainWindow(runner=lambda *a: None, autocheck=False, auto_open_folder=False)
    assert w.btn_audio.text() == "Choose audio" and w.settings_dialog().cmb_lang.currentData() == "en"
    w.close()
