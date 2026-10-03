# Changelog

All notable changes to Voxprint. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).
The project is not released yet; everything below is the history of the 0.1.0 development line.

## [Unreleased]

### Added
- **Voice-owner consent**: spoken consent statement at the end of the recording (script block 18, ru/en/de templates) is read automatically into a usage scope (commercial / public non-commercial / private only), confirmed with one click, stored in `voice.json` (`consent` block, optional clip), mapped to the licence fields and shown as badges and reminders in My voices and Narrate. `core/consent.py`, `workers/consent_runner.py`.
- **Training presets** (Fast / Balanced / Maximum / Manual) in the Train window with a GPU-calibrated time estimate and a collapsed Advanced panel (`core/train_presets.py`); Balanced = the automatic plan, the learning rate is never raised by a preset.
- **No-transcript mode** in the Train window: audio only (many files / a folder / drag and drop) -> Qwen3-ASR per file -> plausibility, duration, duplicate and signal-quality gates -> one merged, loudness-normalised dataset, with totals (files, kept clips, minutes kept), a localized accuracy warning and an explicit opt-in. `core/asr.py`, `core/asr_dataset.py`.
- **Studio** - new home window with three cards (*Narrate a book*, *Train your voice*, *My voices*) and the gear; every sub-window has a *← Studio* button; language changes apply to all windows.
- **Voice library** (`%LOCALAPPDATA%\Voxprint\voices\<id>\`): trained voices are registered automatically; *My voices* with licence badge, preview, details, delete, import from folder / zip (hardened), and
  *Download voices from repository* (configurable index URL, placeholder by default, SHA-256 verified).
- **`voice.json` schema 2**: id, name, author, licence (+ URL), derived `commercial_use`; schema-1 files are migrated. Licence logic in `core/voice_info.py`.
- **Narrate a book**: TXT / FB2 / EPUB parsers (standard library only), chapters, sentence chunks, resumable chunk-by-chunk synthesis with a Qwen3-TTS voice adapter (`core/tts_engine.py`, not yet run on a GPU),
  progress with ETA, pause / cancel, exports through ffmpeg: Opus with chapters (default), MP3 per chapter, M4B (AAC) opt-in, plus M4B-Opus, Opus / FLAC / WAV per chapter, one MP3 with chapter marks, bitrate settings.
- **AAC / M4B patent notice** (UI in en / ru / de, README, `THIRD_PARTY_NOTICES.md`) and a feature flag to hide / disable the option (`VOXPRINT_ENABLE_AAC`, `state/features.json`, `infra/features.py`).
- **Book preparation** (fully automatic, no editor): rule-based steps for Russian and English - layout, footnotes / page numbers, quotes and dashes, links, chapter headings, numbers in words (own `core/num_words.py`
  with Russian declension), abbreviations - as check boxes (all on by default) in the Narrate window; prepared text and a report are saved to `.debug/` next to the job cache. Tests for every rule step in both languages.
- **AI clean-up for Russian (optional, on demand)**: SAGE `sage-fredt5-distilled-95m` (MIT, ~365 MB, pinned revision) downloaded on request; its proposals pass a strict rule-based validator
  (close spelling fixes, `ё`, commas only), cached per paragraph. Engine written but **not run on real hardware**; tests use fakes.
- **Registry of text models** (`infra/text_models.py`) with placeholders (greyed out under *More preparation (coming later)*): RUPunct punctuation, stress / `ё`, en / de spelling, translation, speaker roles.
- **Quality presets** *Compact / Standard / High* with a size-per-hour hint; exact bitrates moved to a collapsed *Advanced* section (also: output folder, chapter titles, sample of the prepared text); *Other formats* holds only formats.
- Extension point `NarrationOptions.preprocessors` (translation / roles will plug in there).
- **Backup and restore of models and voices** (Settings -> *Models and voices*): copy models (own folder + complete copies from other programs' caches), the ffmpeg tool and optionally the voice library to any folder or drive
  (`infra/backup.py`): progress and cancel, resumable (`.part` files, manifest `voxprint-backup.json` with SHA-256), identical files skipped, free-space check before writing, hash-verified restore through staging folders, voices never overwritten.
- **Existing models folder** (`infra/existing_models.py`): models from a previous install / backup are imported (hard link on the same drive, else copy; verified by hash) before any download; set in Settings or by the installer.
- **Installer**: new optional wizard page *Existing models* (writes `state\existing_models_dir.txt`, copies nothing; silent `/ModelsDir=`); wizard texts are now English / Russian / German via `[CustomMessages]`.
- New docs screenshots: Studio, My voices, Narrate (incl. Russian and German renders).
- **Settings dialog** behind a gear button: language, *Check for updates*, open models folder, open data & log folder, *Repair the installation*, *About*.
  The main screen keeps only the core workflow.
- **Scroll area** around the whole main UI: the window is usable when the work area is shorter than the layout (e.g. 1366x768 at 150% scaling).
- **`voice.json`** written next to the trained adapter: voice name, language, creation date (UTC), speech duration, epochs, base model, plus optional
  `voice_type` (male / female / child / other) and `description` set under the main button. New module `core/voice_info.py`.
- Optional voice-type and description fields on the main screen.
- Contrast test for the theme (WCAG AA, 4.5:1) and parity tests for the new locale keys (en / ru / de).
- Repository documentation in English: rewritten `README.md`, `CONTRIBUTING.md`, `docs/ARCHITECTURE.md`, this changelog, and English screenshots in `docs/screenshots/`.

### Changed
- **Product name**: *Voxprint AI Audiobook Builder* (tagline "Train a voice, narrate books") in window titles, About, installer display name and docs. Technical names (`voxprint`, `Voxprint.exe`, `%LOCALAPPDATA%\Voxprint`) are unchanged.
- The previous main window became the *Train your voice* window of the Studio (same workflow).
- **Darker, more opaque theme**: strong dark tint over the Acrylic backdrop, solid panels and buttons, opaque text colours; the plain fallback uses the same palette.
- All code comments and docstrings are English (UI strings stay localized in en / ru / de); comments in `build.bat`, the installer script and the requirements files are English too.
- Repo docs no longer carry a Russian section; Russian-only development notes were translated or dropped.

## [0.1.0] - development history (not released)

### Added
- Forced alignment with `Qwen3-ForcedAligner-0.6B` (long audio chunked at pauses), CTC backup aligner, Russian text normalization, slicing at pauses, automatic quality filter,
  dataset in the Alexandria `train_lora.py` format.
- LoRA voice training (Alexandria recipe) with VRAM-based planning; optional universal (merged, `custom_voice`) model export.
- Reuse of models downloaded by other apps (Hugging Face cache, Pinokio/Alexandria, ModelScope), read-only, with completeness and revision checks; ModelScope download mirror.
- Environment probe and "verified by Voxprint" manifest; update flow with staging, smoke test and rollback; one-click upgrade offers for components in the user's own environment;
  install completion manifest, `--verify-install`, `--repair`, Repair banner, model state line.
- Localization (English, German, Russian), About dialog with credits and generated third-party notices, build script, Inno Setup installer, icons, `--selftest`, `--selftest-imports`.

### Fixed (found while testing on Windows Server 2025 + RTX 4090)
- `CERTIFICATE_VERIFY_FAILED` on a fresh Windows: HTTPS requests retry with certifi (`infra/net.py`); `certifi` and `hf_xet` added to the requirements.
- New `nvidia-smi` prints "CUDA UMD Version": it is parsed now (CPU torch was chosen before).
- The repair venv defaults to Python 3.11 so `--verify-install` passes after `--repair`.
- Unpunctuated long texts are split into clauses of at most 14 words, so long recordings are chunked instead of being truncated at ~180 s.
- m4a/aac input: decoded directly with ffmpeg (imageio-ffmpeg ships no ffprobe).
- Frozen exe missing `six` (found by `--selftest-imports`).
- Window default size adapts to the screen work area; the minimum is the layout's real minimum.
- Tests isolate `APPDATA` / `LOCALAPPDATA` and the Windows locale (suite green on Windows: 236 passed, 1 skipped).
