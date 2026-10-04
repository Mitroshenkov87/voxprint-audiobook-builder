"""Localization: key/placeholder parity of en/ru/de, language detection and switching, no hard-coded user strings in code."""
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


def test_three_languages_and_identical_keys_and_placeholders():
    assert LANGS == ("en", "de", "ru")
    assert set(i18n.LANG_NAMES) == set(LANGS)
    # no stray catalogs of dropped/unsupported languages
    assert sorted(p.stem for p in (ROOT / "locales").glob("*.json")) == sorted(LANGS)
    cats = {l: _cat(l) for l in LANGS}
    base = cats["en"]
    assert len(base) > 150
    for l in LANGS:
        assert set(cats[l]) == set(base), (l, set(cats[l]) ^ set(base))
        for k, v in cats[l].items():
            assert v.strip(), (l, k)
            assert set(PH.findall(v)) == set(PH.findall(base[k])), (l, k)


def test_scripts_match_languages():
    en, de = _cat("en"), _cat("de")
    assert not [k for k, v in {**{("en", k): v for k, v in en.items()}, **{("de", k): v for k, v in de.items()}}.items()
                if CYR.search(v)]
    c = _cat("ru")
    assert CYR.search(c["ui.btn_lora"]) and CYR.search(c["about.what"])
    assert c["ui.language"] == "Язык"


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
    used |= {f"check.sugg_{c}" for c in ("no_stop", "quiet", "wer", "pitch")} | {f"check.verdict_{c}" for c in ("good", "warn", "bad")}
    used |= {f"preset.adv_{c}" for c in ("epochs", "rank", "alpha", "lr", "accum")}     # dynamic keys of the Train window presets
    from infra.install_state import ALL_CODES
    used |= {f"health.{c}" for c in ALL_CODES}          # install-check reason codes (dynamic key)
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
                      "verified_manifest.py", "book_parsers.py", "num_words.py", "text_prep.py", "text_cleanup.py", "asr_dataset.py", "asr.py", "consent.py", "script_match.py", "spoken_date.py", "selftest_narrate.py", "selftest_text.py", "preview_runner.py", "voice_check.py"):   # Russian word tables and rules (data)
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
    i18n.set_language("de")
    assert tr("ui.btn_dataset") == "Datensatz erstellen"
    assert tr("no.such.key") == "no.such.key"
    assert tr("progress.pieces") == _cat("de")["progress.pieces"]            # missing parameter - no exception
    assert tr("progress.pieces", n=3) == "Erhaltene Fragmente: 3"
    # no translation in that language -> English
    monkeypatch.setitem(i18n.load_catalog("de"), "ui.cancel", "")
    del i18n.load_catalog("de")["ui.cancel"]
    i18n.set_language("de")
    assert tr("ui.cancel") == _cat("en")["ui.cancel"]
    assert i18n.tr_lang("ui.ready", "ru") == _cat("ru")["ui.ready"]


def test_normalize_code():
    n = i18n.normalize_code
    assert n("ru-RU") == "ru" and n("de_AT.UTF-8") == "de" and n("EN") == "en" and n("de-AT") == "de"
    assert n("uk_UA.UTF-8") is None and n("be-BY") is None             # Ukrainian/Belarusian are no longer offered
    assert n("fr-FR") is None and n("") is None and n(None) is None and n("C") is None


def test_detection_order(monkeypatch, tmp_path):
    for v in ("LC_ALL", "LC_MESSAGES", "LANG", "VOXPRINT_LANG"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(i18n.locale, "getlocale", lambda *a: (None, None))
    monkeypatch.setattr(i18n.sys, "platform", "linux")      # on Windows the OS language comes from GetUserDefaultLocaleName, not from LANG
    assert i18n.detect_language() == "en"                       # the default is English
    i18n.reset()
    assert tr("ui.btn_lora") == "Create voice (LoRA)"
    monkeypatch.setenv("LANG", "uk_UA.UTF-8")
    assert i18n.detect_language() == "en"                       # unsupported system language -> English
    monkeypatch.setenv("LANG", "ru_RU.UTF-8")
    assert i18n.detect_language() == "ru"                       # the system language
    i18n.set_language("de", persist=True)
    assert i18n.saved_language() == "de" and i18n.detect_language() == "de"   # a saved choice beats the system language
    monkeypatch.setenv("VOXPRINT_LANG", "de")
    assert i18n.detect_language() == "de"                       # the environment variable beats everything
    monkeypatch.setenv("VOXPRINT_LANG", "xx")
    assert i18n.detect_language() == "de"                       # an unknown code is ignored


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
    assert i18n.system_language() == "de"


def test_stage_labels_follow_language():
    from core.events import Stage
    i18n.set_language("ru")
    assert Stage.ALIGN.label == "Выравнивание"
    i18n.set_language("en")
    assert Stage.ALIGN.label == "Alignment"
    i18n.set_language("de")
    assert Stage.TRAIN.label == "LoRA-Training"


def test_errors_use_current_language():
    from core.errors import CancelledByUser, OutOfMemoryError_
    i18n.set_language("en")
    assert CancelledByUser().user_message == "The operation was cancelled."
    assert OutOfMemoryError_().user_message == "Not enough video memory (VRAM)."
    i18n.set_language("de")
    assert "Grafikspeicher" in OutOfMemoryError_().user_message


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
    assert cmb.count() == 3 and cmb.currentData() == "ru"
    assert [cmb.itemText(i) for i in range(3)] == ["English", "Deutsch", "Русский"]
    w.set_language("en")
    assert w.btn_lora.text() == "Create voice (LoRA)" and w.btn_merge.text() == "Build universal model (~4 GB)"
    assert w.lbl_status.text() == "Choose the audio and the text - I will do the rest."
    assert w.lbl_audio.text().startswith("No file chosen")
    assert w._chips[next(iter(w._chips))].text() == "Updates"
    assert i18n.saved_language() == "en"
    w.set_language("de")
    assert w.btn_dataset.text() == "Datensatz erstellen"
    w.set_language("ru")
    assert w.settings_dialog().btn_update.text() == _cat("ru")["ui.btn_update"]
    w.set_language("uk")                                         # not offered any more: ignored
    assert i18n.get_language() == "ru" and w.settings_dialog().cmb_lang.currentData() == "ru"
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
    dlg.cmb_lang.setCurrentIndex(dlg.cmb_lang.findData("en") if before != "en" else 1)
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
