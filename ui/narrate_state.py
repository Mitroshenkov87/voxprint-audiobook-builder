"""Narration presets on the shared audiobook screen.

The shell's pace group stays. The groups added here write the same settings
the expert panel shows: voices, format, chapter split, background music, and
reading manner. Choosing a chip, editing a value, and Reset all go through
that one state. Language, text preparation, bitrates, and the fine reading
controls stay expert fields.

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
    FORMAT_MP3_SINGLE,
    FORMAT_OPUS_CHAPTERS,
    FORMAT_OPUS_SINGLE,
    FORMAT_WAV_CHAPTERS,
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
    Choice(FORMAT_OPUS_CHAPTERS, "abb.file.opus_chapters"),
    Choice(FORMAT_MP3_SINGLE, "abb.file.mp3_single"),
    Choice(FORMAT_MP3_CHAPTERS, "abb.preset.file.mp3"),
    Choice(FORMAT_M4B, "abb.preset.format.m4b"),
    Choice(FORMAT_FLAC_CHAPTERS, "abb.file.flac"),
    Choice(FORMAT_WAV_CHAPTERS, "abb.file.wav"),
)
_CONTAINER_CHOICES = (
    Choice("mp3", "abb.preset.format.mp3"),
    Choice("m4b", "abb.preset.format.m4b"),
    Choice("opus", "abb.preset.format.opus"),
)
# M4B is always one AAC file with chapter markers. Split applies to MP3 and Opus.
_EXPORT_BY_CHIPS = {
    ("opus", False): FORMAT_OPUS_SINGLE,
    ("opus", True): FORMAT_OPUS_CHAPTERS,
    ("mp3", False): FORMAT_MP3_SINGLE,
    ("mp3", True): FORMAT_MP3_CHAPTERS,
    ("m4b", False): FORMAT_M4B,
    ("m4b", True): FORMAT_M4B,
}
_CHIPS_BY_FILE = {
    FORMAT_OPUS_SINGLE: ("opus", False),
    FORMAT_OPUS_CHAPTERS: ("opus", True),
    FORMAT_MP3_SINGLE: ("mp3", False),
    FORMAT_MP3_CHAPTERS: ("mp3", True),
    FORMAT_M4B: ("m4b", None),
}
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
    """Pace from the shell, then voices, format, chapters, music, and manner."""
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
        _enum("narr.container", "opus", _CONTAINER_CHOICES, "output"),
        _bool("narr.split_chapters", False, "output"),
        _enum("narr.file", FORMAT_OPUS_SINGLE, _FILE_CHOICES, "output"),
        _int("narr.opus_kbps", QUALITY_PRESETS[PRESET_STANDARD].opus_kbps, 12, 128, "output"),
        _int("narr.mp3_kbps", QUALITY_PRESETS[PRESET_STANDARD].mp3_kbps, 32, 320, "output"),
        _int("narr.aac_kbps", QUALITY_PRESETS[PRESET_STANDARD].aac_kbps, 24, 256, "output"),
        _bool("narr.check_chunks", False, "output"),
        _enum("narr.style", FICTION, _STYLE_CHOICES, "reading"),
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
            _group("format", "abb.preset.format", "opus", (
                ("mp3", "abb.preset.format.mp3", {"narr.container": "mp3"}),
                ("m4b", "abb.preset.format.m4b", {"narr.container": "m4b"}),
                ("opus", "abb.preset.format.opus", {"narr.container": "opus"}),
            )),
            _group("chapters", "abb.preset.chapters", "off", (
                ("off", "abb.preset.chapters.off", {"narr.split_chapters": False}),
                ("on", "abb.preset.chapters.on", {"narr.split_chapters": True}),
            )),
            _group("music", "abb.preset.music", "off", (
                ("off", "abb.preset.sound.off", {"narr.sound": "off"}),
                ("on", "abb.preset.music.on", {"narr.sound": "when"}),
            )),
            _group("manner", "abb.preset.manner", "expressive", (
                ("calm", "abb.preset.manner.calm", {"narr.style": SCRIPTURE}),
                ("expressive", "abb.preset.manner.expressive", {"narr.style": FICTION}),
            )),
        ],
    }


def export_format(container: str, split: bool) -> str:
    """Return the audiobook file key for a format chip and the chapter chip.

    Args:
        container: ``mp3``, ``m4b``, or ``opus``.
        split: True saves MP3 or Opus as one file per chapter. M4B stays one file.
    """
    return _EXPORT_BY_CHIPS[(container, bool(split))]


def chips_for_export(file_format: object) -> tuple[str, bool | None] | None:
    """Return the chips that produce ``file_format``, or None for an expert-only file.

    Args:
        file_format: A value of ``narr.file``.
    """
    if not isinstance(file_format, str):
        return None
    return _CHIPS_BY_FILE.get(file_format)


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
