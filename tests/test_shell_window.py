"""The shared audiobook screen: one settings state, plain labels, and the gaps in this pin."""
import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication, QComboBox, QFrame, QLabel, QPushButton, QScrollArea, QTabBar, QWidget
from voxprint_common.i18n import add_catalogs, set_language
from voxprint_common.i18n import tr as shell_tr
from voxprint_common.mdl import load_manifest
from voxprint_common.ui.presets import PresetMatch
from voxprint_common.ui.simple_image import Phase
from voxprint_common.ui.tokens import load as load_tokens

from core.audiobook_export import (
    FORMAT_FLAC_CHAPTERS,
    FORMAT_M4B,
    FORMAT_MP3_CHAPTERS,
    FORMAT_MP3_SINGLE,
    FORMAT_OPUS_CHAPTERS,
    FORMAT_OPUS_SINGLE,
    PRESET_STANDARD,
    QUALITY_PRESETS,
)
from core.pace import AUTO, FICTION, SCRIPTURE
from core.consent import TEMPLATES
from core.pauses import DEFAULT_LENGTHS_MS
from infra.mdl_gate import manifest_path
from ui.model_slots import choices
from ui.narrate_state import SURFACE_ID, export_format, narrate_presets, narrate_settings
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
    assert groups[1:] == ["voices", "format", "chapters", "music", "manner"]
    pauses = settings.get("narr.pause.sentence")
    assert pauses is not None and pauses.default == DEFAULT_LENGTHS_MS["sentence"]
    opus = settings.get("narr.opus_kbps")
    assert opus is not None
    assert opus.default == QUALITY_PRESETS[PRESET_STANDARD].opus_kbps
    assert settings.get("narr.language").default == "keep"
    assert settings.get("narr.prepare").default is True
    assert settings.get("narr.file").default == FORMAT_OPUS_SINGLE
    assert settings.get("narr.check_chunks").default is False
    assert settings.get("narr.style").default == FICTION
    assert any(choice.value == AUTO for choice in settings.get("narr.style").choices)
    assert settings.get("narr.container").default == "opus"
    assert settings.get("narr.split_chapters").default is False
    assert settings.get("narr.sound").default == "off"
    assert settings.get("narr.disclosure").default is False


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


def test_moved_choices_stay_as_live_expert_values(app):
    window = _open("en")
    window.screen.set_expert(True)
    QApplication.processEvents()
    language = window.widget.findChild(QComboBox, "narr.language")
    style = window.widget.findChild(QComboBox, "narr.style")
    saved = window.widget.findChild(QComboBox, "narr.file")
    assert language is not None and language.currentText() == "Keep the book's language"
    assert style is not None and style.currentText() == "Fiction"
    assert saved is not None and saved.currentText() == "One Opus file"
    assert window.layout.get("narr.prepare") is True
    assert window.layout.get("narr.sound") == "off"
    assert window.layout.get("narr.disclosure") is False
    assert window.layout.get("narr.opus_kbps") == QUALITY_PRESETS[PRESET_STANDARD].opus_kbps
    labels = [row.text() for row in window.widget.findChildren(QLabel) if row.objectName().startswith("preset-")]
    assert labels == ["Pace", "Voices", "Format", "Split into chapters", "Background music", "Reading manner"]
    assert "Language" not in labels and "Prepare text" not in labels
    window.widget.close()


def test_russian_reading_style_says_avto():
    set_language("ru")
    add_catalogs(SHELL_LOCALES)
    catalog = json.loads((SHELL_LOCALES / "ru.json").read_text(encoding="utf-8"))
    assert shell_tr("abb.preset.style.auto") == "Авто"
    assert shell_tr("setting.narr.vram.auto") == "Авто"
    assert "Само" not in catalog.values()


def test_empty_history_strip_stays_hidden_until_a_job_finishes(app):
    window = _open("en")
    strip = window.widget.findChild(QScrollArea, "historyStrip")
    assert strip is not None and not strip.isVisible()
    window.show_progress(1, 2, "Reading", eta_seconds=10)
    QApplication.processEvents()
    assert not strip.isVisible()
    window.layout.finish("chapter-1")
    window.screen.refresh()
    QApplication.processEvents()
    assert strip.isVisible() and strip.height() == 104
    window.widget.close()


def test_chip_text_fits_in_english_russian_and_german(app):
    density = load_tokens().density
    pad = 14 * 2 + density.border_px * 2
    for language in ("en", "ru", "de"):
        window = _open(language)
        window.widget.resize(1200, 900)
        window.layout.set_value("audio.loudness", -14.0)
        window.screen.refresh()
        QApplication.processEvents()
        seen = []
        for button in window.widget.findChildren(QPushButton):
            if button.property("chip") != "true" or not button.isVisible() or not button.text():
                continue
            seen.append(button.text())
            font = QFont(button.font())
            font.setPixelSize(density.body_px)
            font.setWeight(QFont.Weight.DemiBold)
            advance = QFontMetrics(font).horizontalAdvance(button.text())
            assert advance + pad <= button.width(), (language, button.text(), advance, button.width())
        assert any("voice" in text.lower() or "голос" in text.lower() or "stimme" in text.lower() for text in seen)
        window.widget.close()


def test_export_chips_follow_the_audiobook_file():
    assert export_format("opus", False) == FORMAT_OPUS_SINGLE
    assert export_format("opus", True) == FORMAT_OPUS_CHAPTERS
    assert export_format("mp3", False) == FORMAT_MP3_SINGLE
    assert export_format("mp3", True) == FORMAT_MP3_CHAPTERS
    assert export_format("m4b", False) == FORMAT_M4B
    assert export_format("m4b", True) == FORMAT_M4B


def test_format_chips_and_expert_edits_share_one_file_value(app):
    window = _open("ru")
    window.screen.set_expert(True)
    QApplication.processEvents()
    assert window.layout.get("narr.file") == FORMAT_OPUS_SINGLE
    window.layout.choose("format", "m4b")
    window.layout.choose("chapters", "on")
    assert window.layout.get("narr.file") == FORMAT_M4B
    assert window.layout.get("narr.split_chapters") is True
    window.layout.choose("format", "mp3")
    assert window.layout.get("narr.file") == FORMAT_MP3_CHAPTERS
    window.layout.choose("chapters", "off")
    assert window.layout.get("narr.file") == FORMAT_MP3_SINGLE
    window.layout.choose("music", "on")
    window.layout.choose("manner", "calm")
    assert window.layout.get("narr.sound") == "when"
    assert window.layout.get("narr.style") == SCRIPTURE
    window.screen.refresh()
    QApplication.processEvents()
    assert window.widget.findChild(QComboBox, "narr.style").currentText() == "Писание"
    assert window.widget.findChild(QComboBox, "narr.file").currentText() == "Один файл MP3"
    assert window.widget.findChild(QComboBox, "narr.sound").currentText() == "Когда книга просит"
    window.layout.set_value("narr.file", FORMAT_FLAC_CHAPTERS)
    window.layout.choose("manner", "expressive")
    assert window.layout.get("narr.file") == FORMAT_FLAC_CHAPTERS
    window.layout.choose("format", "opus")
    assert window.layout.get("narr.file") == FORMAT_OPUS_SINGLE
    window.layout.set_value("narr.file", FORMAT_OPUS_CHAPTERS)
    assert window.layout.get("narr.container") == "opus"
    assert window.layout.get("narr.split_chapters") is True
    voice = window.widget.findChild(QComboBox, "voicePicker")
    assert voice is not None and voice.width() >= 400
    window.widget.close()


def _sibling_overlaps(root: QWidget) -> list[tuple[str, str, QRect]]:
    hits: list[tuple[str, str, QRect]] = []
    seen: set[int] = set()
    for parent in [root, *root.findChildren(QWidget)]:
        if id(parent) in seen:
            continue
        seen.add(id(parent))
        boxes: list[tuple[QWidget, QRect]] = []
        for child in parent.findChildren(QWidget):
            if child.parentWidget() is not parent or not child.isVisible():
                continue
            rect = child.geometry()
            if rect.width() <= 2 or rect.height() <= 2:
                continue
            boxes.append((child, rect))
        for index, (left, left_rect) in enumerate(boxes):
            for right, right_rect in boxes[index + 1:]:
                shared = left_rect.intersected(right_rect)
                if shared.width() > 2 and shared.height() > 2:
                    hits.append((left.objectName() or left.__class__.__name__, right.objectName() or right.__class__.__name__, shared))
    return hits


def _assert_labels_fit(root: QWidget, language: str, width: int, mode: str) -> None:
    density = load_tokens().density
    pad = 14 * 2 + density.border_px * 2 + 8
    for button in root.findChildren(QPushButton):
        name = button.objectName()
        chip = button.property("chip") == "true" or name.startswith("reset-") or name == "expertToggle"
        if not chip or not button.isVisible() or not button.text():
            continue
        assert button.width() + 1 >= button.sizeHint().width(), (
            language, width, mode, name, button.text(), button.width(), button.sizeHint().width(),
        )
        font = QFont(button.font())
        font.setPixelSize(density.body_px)
        font.setWeight(QFont.Weight.DemiBold)
        needed = QFontMetrics(font).horizontalAdvance(button.text()) + pad
        assert button.width() >= needed, (language, width, mode, name, button.text(), button.width(), needed)


def _assert_source_row(window: AudiobookShell, language: str, width: int, mode: str) -> None:
    drop = window.widget.findChild(QLabel, "fileDrop")
    assert drop is not None and drop.isVisible()
    assert drop.wordWrap() is False
    assert drop.width() >= int(width * 0.35), (language, width, mode, drop.width())
    font = QFont(drop.font())
    font.setPixelSize(15)
    advance = QFontMetrics(font).horizontalAdvance(drop.text())
    assert drop.width() >= advance + 32, (language, width, mode, drop.text(), drop.width(), advance)
    voice = window.widget.findChild(QComboBox, "voicePicker")
    assert voice is not None and voice.width() >= int(width * 0.55), (
        language, width, mode, None if voice is None else voice.width(),
    )


def test_simple_and_expert_rows_do_not_overlap(app):
    for language in ("en", "ru", "de"):
        for width in (960, 1200):
            window = _open(language)
            window.widget.resize(width, 900)
            QApplication.processEvents()
            hits = _sibling_overlaps(window.widget)
            assert hits == [], (language, width, "simple", hits[:4])
            _assert_labels_fit(window.widget, language, width, "simple")
            _assert_source_row(window, language, width, "simple")
            window.screen.set_expert(True)
            window.layout.set_value("audio.loudness", -14.0)
            window.screen.refresh()
            window.widget.resize(width, 900)
            QApplication.processEvents()
            hits = _sibling_overlaps(window.widget)
            assert hits == [], (language, width, "expert", hits[:4])
            _assert_labels_fit(window.widget, language, width, "expert")
            _assert_source_row(window, language, width, "expert")
            form = window.widget.findChild(QWidget, "expertForm")
            assert form is not None and form.layout() is not None
            assert form.layout().contentsMargins().right() >= 16
            rightmost = 0
            for child in form.findChildren(QWidget):
                if child.parentWidget() is not form or not child.isVisible() or child.width() <= 2:
                    continue
                rightmost = max(rightmost, child.geometry().right())
            assert rightmost <= form.width() - 12, (language, width, rightmost, form.width())
            expert_voice = form.findChild(QComboBox, "narr.tts")
            assert expert_voice is not None
            assert expert_voice.width() >= form.width() - 40, (
                language, width, expert_voice.width(), form.width(),
            )
            panel_resets = [
                button for button in window.widget.findChildren(QPushButton, "expertReset") if button.isVisible()
            ]
            assert panel_resets == [], (language, width, [button.text() for button in panel_resets])
            pace = window.widget.findChild(QPushButton, "reset-pace")
            assert pace is not None and pace.isVisible()
            window.widget.close()
