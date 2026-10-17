"""Narration presets on the shared audiobook screen.

The shell's pace group stays. The groups added here write the same settings
the expert panel shows. Choosing a chip, editing a value, and Reset all go
through that one state.

    from ui.narrate_state import narrate_presets, narrate_settings
    settings = narrate_settings()
    presets = narrate_presets(settings)
"""
from __future__ import annotations

from voxprint_common.ui.presets import PresetFile, parse_presets
from voxprint_common.ui.settings import Choice, SettingSpec, Settings, SettingType
from voxprint_common.ui.variants import settings_for, surface_for, variant_presets

from core.audiobook_export import (
    FORMAT_FLAC_CHAPTERS,
    FORMAT_M4B,
    FORMAT_MP3_CHAPTERS,
    FORMAT_OPUS_CHAPTERS,
    FORMAT_OPUS_SINGLE,
    FORMAT_WAV_CHAPTERS,
    PRESET_COMPACT,
    PRESET_HIGH,
    PRESET_STANDARD,
    QUALITY_PRESETS,
)
from core.pace import AUTO, DIALOGUE, FICTION, SCRIPTURE
from core.pauses import DEFAULT_LENGTHS_MS
from core.text_prep import STEP_KEYS
from infra.mdl_gate import APP_ID
from ui.model_slots import choices as model_choices

SURFACE_ID = APP_ID

_PAUSE_FIELDS = tuple(
    (f"narr.pause.{kind}", kind) for kind in ("comma", "mid", "sentence", "paragraph", "chapter")
)

_FILE_CHOICES = (
    Choice(FORMAT_OPUS_SINGLE, "abb.preset.file.opus"),
    Choice(FORMAT_MP3_CHAPTERS, "abb.preset.file.mp3"),
    Choice(FORMAT_M4B, "abb.preset.file.m4b"),
    Choice(FORMAT_OPUS_CHAPTERS, "abb.file.opus_chapters"),
    Choice(FORMAT_FLAC_CHAPTERS, "abb.file.flac"),
    Choice(FORMAT_WAV_CHAPTERS, "abb.file.wav"),
)
_LANGUAGE_CHOICES = (
    Choice("keep", "abb.preset.language.keep"),
    Choice("en", "abb.preset.language.en"),
    Choice("ru", "abb.preset.language.ru"),
    Choice("de", "abb.preset.language.de"),
)
_STYLE_CHOICES = (
    Choice(AUTO, "abb.preset.style.auto"),
    Choice(SCRIPTURE, "abb.preset.style.scripture"),
    Choice(FICTION, "abb.preset.style.fiction"),
    Choice(DIALOGUE, "abb.preset.style.dialogue"),
)
_SOUND_CHOICES = (
    Choice("off", "abb.preset.sound.off"),
    Choice("when", "abb.preset.sound.when"),
)
_VRAM_CHOICES = (
    Choice("auto", "setting.narr.vram.auto"),
    Choice("8gb", "setting.narr.vram.8gb"),
    Choice("16gb", "setting.narr.vram.16gb"),
    Choice("24gb", "setting.narr.vram.24gb"),
)


def narrate_settings() -> Settings:
    """Shell audio settings, then the narration values the expert panel shows."""
    surface = surface_for(SURFACE_ID)
    settings = Settings(settings_for(surface).visible(expert=True))
    for spec in _narration_specs():
        settings.add(spec)
    return settings


def narrate_presets(settings: Settings) -> PresetFile:
    """Pace from the shell, then the audiobook chips, in one preset file."""
    surface = surface_for(SURFACE_ID)
    pace = variant_presets(surface, settings)
    extra = parse_presets(_preset_document(), settings)
    return PresetFile(pace.groups + extra.groups)


def _narration_specs() -> tuple[SettingSpec, ...]:
    specs: list[SettingSpec] = []
    models = model_choices()
    if models:
        specs.append(_enum("narr.tts", models[0].value, models, "narration"))
    specs.extend((
        _int("narr.batch", 1, 1, 12, "narration"),
        _enum("narr.vram", "auto", _VRAM_CHOICES, "narration"),
        _bool("narr.thermal", False, "narration"),
        _path("narr.work_dir", "narration"),
        _enum("narr.language", "keep", _LANGUAGE_CHOICES, "translation"),
        _bool("narr.literary", False, "translation"),
        _bool("narr.prepare", True, "prepare"),
        _bool("narr.llm_prepare", False, "prepare"),
    ))
    specs.extend(_bool(f"narr.step.{step}", True, "prepare") for step in STEP_KEYS)
    specs.extend((
        _bool("narr.ordinals", True, "prepare"),
        _bool("narr.multi_voice", False, "speakers"),
        _enum("narr.file", FORMAT_OPUS_SINGLE, _FILE_CHOICES, "output"),
        _int("narr.opus_kbps", QUALITY_PRESETS[PRESET_STANDARD].opus_kbps, 12, 128, "output"),
        _int("narr.mp3_kbps", QUALITY_PRESETS[PRESET_STANDARD].mp3_kbps, 32, 320, "output"),
        _int("narr.aac_kbps", QUALITY_PRESETS[PRESET_STANDARD].aac_kbps, 24, 256, "output"),
        _bool("narr.check_chunks", False, "output"),
        _enum("narr.style", AUTO, _STYLE_CHOICES, "reading"),
        _float("narr.speed", 1.0, 0.7, 1.3, "reading"),
    ))
    specs.extend(
        _int(key, int(DEFAULT_LENGTHS_MS[kind]), 0, 6000, "reading") for key, kind in _PAUSE_FIELDS
    )
    specs.extend((
        _enum("narr.sound", "off", _SOUND_CHOICES, "reading"),
        _bool("narr.disclosure", False, "reading"),
    ))
    return tuple(specs)


def _preset_document() -> dict[str, object]:
    return {
        "schema": "voxprint-presets/1",
        "groups": [
            _group("voices", "abb.preset.voices", "one", (
                ("one", "abb.preset.voices.one", {"narr.multi_voice": False}),
                ("each", "abb.preset.voices.each", {"narr.multi_voice": True}),
            )),
            _group("language", "abb.preset.language", "keep", tuple(
                (code, f"abb.preset.language.{code}", {"narr.language": code, "narr.literary": False})
                for code in ("keep", "en", "ru", "de")
            )),
            _group("prepare", "abb.preset.prepare", "on", (
                ("on", "abb.preset.prepare.on", _prepare(True)),
                ("off", "abb.preset.prepare.off", _prepare(False)),
            )),
            _group("file", "abb.preset.file", FORMAT_OPUS_SINGLE, (
                (FORMAT_OPUS_SINGLE, "abb.preset.file.opus", {"narr.file": FORMAT_OPUS_SINGLE}),
                (FORMAT_MP3_CHAPTERS, "abb.preset.file.mp3", {"narr.file": FORMAT_MP3_CHAPTERS}),
                (FORMAT_M4B, "abb.preset.file.m4b", {"narr.file": FORMAT_M4B}),
            )),
            _group("quality", "abb.preset.quality", PRESET_STANDARD, (
                (PRESET_COMPACT, "abb.preset.quality.compact", _quality(PRESET_COMPACT, False)),
                (PRESET_STANDARD, "abb.preset.quality.standard", _quality(PRESET_STANDARD, False)),
                (PRESET_HIGH, "abb.preset.quality.high", _quality(PRESET_HIGH, True)),
            )),
            _group("style", "abb.preset.style", AUTO, tuple(
                (style, f"abb.preset.style.{style}", _style(style))
                for style in (AUTO, SCRIPTURE, FICTION, DIALOGUE)
            )),
            _group("sound", "abb.preset.sound", "off", (
                ("off", "abb.preset.sound.off", {"narr.sound": "off"}),
                ("when", "abb.preset.sound.when", {"narr.sound": "when"}),
            )),
            _group("disclosure", "abb.preset.disclosure", "off", (
                ("off", "abb.preset.disclosure.off", {"narr.disclosure": False}),
                ("on", "abb.preset.disclosure.on", {"narr.disclosure": True}),
            )),
        ],
    }


def _group(group_id: str, label_key: str, default: str, options: tuple) -> dict[str, object]:
    return {
        "id": group_id,
        "label_key": label_key,
        "default": default,
        "options": [
            {"id": option_id, "label_key": label_key, "values": values}
            for option_id, label_key, values in options
        ],
    }


def _prepare(enabled: bool) -> dict[str, bool]:
    values = {"narr.prepare": enabled}
    for step in STEP_KEYS:
        values[f"narr.step.{step}"] = enabled
    return values


def _quality(name: str, check_chunks: bool) -> dict[str, object]:
    rates = QUALITY_PRESETS[name]
    return {
        "narr.opus_kbps": rates.opus_kbps,
        "narr.mp3_kbps": rates.mp3_kbps,
        "narr.aac_kbps": rates.aac_kbps,
        "narr.check_chunks": check_chunks,
    }


def _style(style: str) -> dict[str, object]:
    values: dict[str, object] = {"narr.style": style, "narr.speed": 1.0}
    for key, kind in _PAUSE_FIELDS:
        values[key] = int(DEFAULT_LENGTHS_MS[kind])
    return values


def _enum(setting_id: str, default: str, items: tuple[Choice, ...], group: str) -> SettingSpec:
    return SettingSpec(
        setting_id, f"setting.{setting_id}", f"setting.{setting_id}.help",
        SettingType.ENUM, default, expert=True, group=group, choices=items,
    )


def _bool(setting_id: str, default: bool, group: str) -> SettingSpec:
    return SettingSpec(
        setting_id, f"setting.{setting_id}", f"setting.{setting_id}.help",
        SettingType.BOOL, default, expert=True, group=group,
    )


def _int(setting_id: str, default: int, minimum: int, maximum: int, group: str) -> SettingSpec:
    return SettingSpec(
        setting_id, f"setting.{setting_id}", f"setting.{setting_id}.help",
        SettingType.INT, default, expert=True, group=group, minimum=minimum, maximum=maximum,
    )


def _float(setting_id: str, default: float, minimum: float, maximum: float, group: str) -> SettingSpec:
    return SettingSpec(
        setting_id, f"setting.{setting_id}", f"setting.{setting_id}.help",
        SettingType.FLOAT, default, expert=True, group=group, minimum=minimum, maximum=maximum,
    )


def _path(setting_id: str, group: str) -> SettingSpec:
    return SettingSpec(
        setting_id, f"setting.{setting_id}", f"setting.{setting_id}.help",
        SettingType.PATH, "", expert=True, group=group,
    )
