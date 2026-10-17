"""The shared audiobook screen: one settings state, plain labels, and the gaps in this pin."""
import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QApplication, QComboBox, QFrame, QLabel, QPushButton, QTabBar
from voxprint_common.i18n import add_catalogs, set_language
from voxprint_common.i18n import tr as shell_tr
from voxprint_common.mdl import load_manifest
from voxprint_common.ui.presets import PresetMatch
from voxprint_common.ui.simple_image import Phase
from voxprint_common.ui.tokens import load as load_tokens

from core.audiobook_export import PRESET_HIGH, PRESET_STANDARD, QUALITY_PRESETS
from core.consent import TEMPLATES
from core.pauses import DEFAULT_LENGTHS_MS
from infra.mdl_gate import manifest_path
from ui.model_slots import choices
from ui.narrate_state import SURFACE_ID, narrate_presets, narrate_settings
from ui.shell_window import MISSING_SHELL_API, AudiobookShell

ROOT = Path(__file__).resolve().parents[1]
SHELL_LOCALES = ROOT / "locales" / "shell"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _open(language: str = "en") -> AudiobookShell:
    set_language(language)
    window = AudiobookShell()
    window.show()
    QApplication.processEvents()
    return window


def test_narration_groups_share_the_shell_pace_state():
    settings = narrate_settings()
    groups = [group.group_id for group in narrate_presets(settings).groups]
    assert SURFACE_ID == "audiobook-builder"
    assert groups[0] == "pace"
    assert groups[1:] == [
        "voices", "language", "prepare", "file", "quality", "style", "sound", "disclosure",
    ]
    pauses = settings.get("narr.pause.sentence")
    assert pauses is not None and pauses.default == DEFAULT_LENGTHS_MS["sentence"]
    opus = settings.get("narr.opus_kbps")
    assert opus is not None
    assert opus.default == QUALITY_PRESETS[PRESET_STANDARD].opus_kbps
    high = narrate_presets(settings).group("quality").option(PRESET_HIGH)
    assert high is not None and dict(high.values)["narr.check_chunks"] is True


def test_model_choices_come_from_the_manifest_and_every_catalog_names_them():
    found = choices()
    seen: set[str] = set()
    available: list[str] = []
    for model in load_manifest(manifest_path()).models:
        assert model.unavailable is False
        if model.id in seen:
            continue
        seen.add(model.id)
        available.append(model.id)
    assert [choice.value for choice in found] == available
    catalogs = {
        name: json.loads((SHELL_LOCALES / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("en", "de", "ru", "uk", "lv")
    }
    keys = set(catalogs["en"])
    assert all(set(data) == keys for data in catalogs.values())
    for choice in found:
        assert choice.label_key == f"model.{choice.value}"
        assert choice.label_key in keys


def test_every_narration_label_translates():
    set_language("en")
    add_catalogs(SHELL_LOCALES)
    settings = narrate_settings()
    for spec in settings.visible(expert=True):
        assert shell_tr(spec.label_key) != spec.label_key
        assert shell_tr(spec.help_key) != spec.help_key
        for choice in spec.choices:
            assert shell_tr(choice.label_key) != choice.label_key


def test_consent_english_stays_in_plain_words():
    catalog = json.loads((ROOT / "locales" / "en.json").read_text(encoding="utf-8"))
    assert catalog["consent.title"] == "Voice owner's consent"
    assert "not legal advice" in catalog["consent.hint_auto"]
    assert catalog["consent.scope_private_only"] == "Private only: results stay on this computer"
    spoken = TEMPLATES["en"]["private_only"]
    assert spoken.startswith("I, {name}, today, {date}, give permission")


def test_first_launch_is_the_audio_surface(app):
    window = _open("en")
    assert window.widget.windowTitle() == "VOXPrint AI Audiobook Builder"
    assert window.widget.width() == 1200 and window.widget.height() == 900
    assert window.widget.minimumWidth() == 960 and window.widget.minimumHeight() == 640
    assert window.surface.min_width == 960 and window.surface.min_height == 640
    assert window.layout.phase is Phase.EMPTY
    buttons = [button.text() for button in window.widget.findChildren(QPushButton)]
    assert "Create audiobook" in buttons
    drop = window.widget.findChild(QLabel, "fileDrop")
    assert drop is not None and "Choose a book" in drop.text()
    voice = window.widget.findChild(QComboBox, "voicePicker")
    assert voice is not None and voice.count() == 3
    assert [voice.itemText(index) for index in range(voice.count())] == ["Warm", "Clear", "Low"]
    assert window.widget.findChildren(QTabBar) == []
    assert load_tokens().name == "voxprint-dark"


def test_a_close_edit_reads_natural_plus_changes_and_reset_restores_only_pace(app):
    window = _open("en")
    window.layout.choose("voices", "each")
    window.layout.set_value("audio.loudness", -14.0)
    status = window.layout.preset_status("pace")
    assert status is not None
    assert status.kind is PresetMatch.MODIFIED and status.option_id == "natural"
    window.screen.refresh()
    QApplication.processEvents()
    texts = [button.text() for button in window.widget.findChildren(QPushButton)]
    assert "Natural + changes" in texts
    window.layout.set_value("audio.pace", "calm")
    window.layout.set_value("audio.sample_rate", "22")
    window.layout.set_value("audio.format", "wav")
    status = window.layout.preset_status("pace")
    assert status is not None and status.kind is PresetMatch.CUSTOM
    window.screen.refresh()
    assert shell_tr("simple.preset.custom") in [
        button.text() for button in window.widget.findChildren(QPushButton)
    ]
    window.layout.reset_group("pace")
    status = window.layout.preset_status("pace")
    assert status is not None and status.kind is PresetMatch.EXACT
    assert window.layout.get("audio.loudness") == -16.0
    assert window.layout.get("audio.pace") == "natural"
    assert window.layout.get("narr.multi_voice") is True


def test_expert_mode_keeps_the_same_values(app):
    window = _open("en")
    window.layout.set_value("narr.speed", 0.9)
    window.screen.set_expert(True)
    QApplication.processEvents()
    panel = window.widget.findChild(QLabel, "expertTitle")
    assert panel is not None and panel.isVisible()
    frame = window.widget.findChild(QFrame, "expertPanel")
    assert frame is not None and frame.isVisible()
    assert 220 <= frame.width() <= 420
    window.screen.set_expert(False)
    assert window.layout.get("narr.speed") == 0.9
    assert window.layout.get("audio.loudness") == -16.0


def test_progress_cancel_and_a_painted_player(app):
    window = _open("en")
    window.show_progress(2, 8, "Reading chapter 2", eta_seconds=40)
    QApplication.processEvents()
    assert window.layout.phase is Phase.BUSY
    label = window.widget.findChild(QLabel, "progressText")
    assert label is not None and label.text() == "Reading chapter 2"
    names = {button.objectName() for button in window.widget.findChildren(QPushButton)}
    assert "cancel" in names and "pause" not in names
    window.layout.finish("chapter-2")
    window.screen.refresh()
    QApplication.processEvents()
    player = window.widget.findChild(QLabel, "player")
    assert player is not None
    assert type(player) is QLabel


def test_create_without_a_book_asks_for_one_and_settings_opens_the_existing_dialog(app, monkeypatch):
    window = _open("en")
    window._on_generate()
    assert window.layout.phase is Phase.BLOCKED
    assert window.layout.reason == "Choose a book."

    class Host:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.opened = False

        def hide(self):
            return None

        def open_settings(self):
            self.opened = True

    monkeypatch.setattr("ui.shell_window.MainWindow", Host)
    window._on_command("view.settings")
    assert isinstance(window._settings_host, Host)
    assert window._settings_host.opened is True
    assert window._settings_host.kwargs["autocheck"] is False
    assert window._settings_host.kwargs["prefetch"] is False
    window.set_book("/tmp/story.txt")
    assert window.book_path() == "/tmp/story.txt"


def test_this_pin_has_no_section_row_or_step_wizard(app):
    window = _open("en")
    for name in MISSING_SHELL_API:
        assert not callable(getattr(window.screen, name, None)), name
    for name in ("set_section", "add_section", "set_steps", "open_wizard"):
        assert not callable(getattr(window.screen, name, None)), name


def test_chip_text_fits_in_english_and_russian(app):
    for language in ("en", "ru", "de"):
        window = _open(language)
        window.widget.resize(1200, 900)
        window.layout.set_value("audio.loudness", -14.0)
        window.screen.refresh()
        window.screen.set_expert(True)
        QApplication.processEvents()
        for button in window.widget.findChildren(QPushButton):
            if not button.isVisible() or not button.text():
                continue
            advance = QFontMetrics(button.font()).horizontalAdvance(button.text())
            assert advance <= button.contentsRect().width(), (language, button.text())
        window.widget.close()
