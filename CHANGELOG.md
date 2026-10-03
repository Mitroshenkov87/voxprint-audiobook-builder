# Changelog

All notable changes to Voxprint. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).
The project is not released yet; everything below is the history of the 0.1.0 development line.

## [Unreleased]

### Added
- **Settings dialog** behind a gear button: language, *Check for updates*, open models folder, open data & log folder, *Repair the installation*, *About*.
  The main screen keeps only the core workflow.
- **Scroll area** around the whole main UI: the window is usable when the work area is shorter than the layout (e.g. 1366x768 at 150% scaling).
- **`voice.json`** written next to the trained adapter: voice name, language, creation date (UTC), speech duration, epochs, base model, plus optional
  `voice_type` (male / female / child / other) and `description` set under the main button. New module `core/voice_info.py`.
- Optional voice-type and description fields on the main screen.
- Contrast test for the theme (WCAG AA, 4.5:1) and parity tests for the new locale keys (en / ru / de).
- Repository documentation in English: rewritten `README.md`, `CONTRIBUTING.md`, `docs/ARCHITECTURE.md`, this changelog, and English screenshots in `docs/screenshots/`.

### Changed
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
