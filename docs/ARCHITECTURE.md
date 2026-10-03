# Voxprint architecture

This document is the map of the code base: which module does what, how data flows through the application, how installation / update / model reuse work and how the UI
is wired. Read it together with the module docstrings (every module starts with one).

## 1. Layers

```
main.py                 entry point: CLI switches (--selftest, --selftest-imports, --verify-install, --repair), logging, start of the UI
ui/                     PySide6 windows and dialogs - no heavy work, only signals/slots
workers/                glue between UI and core: Qt threads (QThread) and Qt-free scenario runners
core/                   the pipeline itself: audio/text processing, alignment, dataset, LoRA training, export. No Qt, no installer logic
infra/                  everything around the pipeline: folders, environment probe, install state, downloads, updates, Windows glue
locales/                en.json / de.json / ru.json - the only place with user-visible text
credits.json, licenses/ third-party list (single source for the About dialog and THIRD_PARTY_NOTICES.md)
installer/, build.bat   PyInstaller + Inno Setup packaging
tools/                  maintenance scripts (fetch licence texts, generate notices)
tests/                  pytest suite (runs without GPU, network or models)
```
Dependencies point downwards: `ui -> workers -> core / infra`; `core` never imports `ui` or `workers`.

## 2. Module map

### `core/`
| Module | Purpose |
|---|---|
| `types.py`, `errors.py`, `events.py` | shared dataclasses; `DatasetMakerError` hierarchy with localized messages; pipeline stages, progress callback protocol, cancellation token |
| `appinfo.py`, `i18n.py` | metadata from `credits.json`; JSON-catalog localization (`tr("key", **params)`, language detection: saved choice -> Windows locale -> English) |
| `cli.py` | developer command line for the dataset/training stages (`--fake-aligner` = dry run without a network) |
| `audio_utils.py` | reading any format (soundfile, else ffmpeg directly), resampling, WAV writing, energy / pause detection |
| `text_utils.py`, `normalizer.py` | reading the text, sentence splitting, mapping aligner words back to the text; Russian number/abbreviation normalization |
| `aligner.py` | forced alignment (`Qwen3-ForcedAligner-0.6B`, CTC backup, fake/injected aligners for tests); long audio is chunked at pauses (`align_long`, chunks <= 150 s) |
| `slicer.py`, `quality.py` | slicing on word timestamps at pauses; automatic quality filter |
| `dataset_builder.py` | writes the Alexandria-format dataset (`segment_XXX.wav`, `ref.wav`, `ref_text.txt`, `metadata.jsonl`, `report.json`) |
| `teacher_forcing.py`, `lora_trainer.py` | teacher-forced input construction; LoRA training of Qwen3-TTS-12Hz-Base (adapter, `training_meta.json`) |
| `model_export.py` | merged "universal" model (`merge_and_unload`) |
| `model_locator.py` | finds models downloaded by other apps (Hugging Face cache, Pinokio/Alexandria, ModelScope) - read-only |
| `voice_info.py` | `voice.json`: schema, voice types, description cleanup, read/write |

### `infra/`
| Module | Purpose |
|---|---|
| `paths.py` | application folders (`%LOCALAPPDATA%\Voxprint`, override `VOXPRINT_HOME`): `models/`, `packages/`, `state/`, `logs/`, `.staging/` |
| `net.py` | HTTPS through the stdlib; retries with certifi roots on `CERTIFICATE_VERIFY_FAILED` |
| `platform_win.py` | OS check, dark title bar, Acrylic backdrop (all guarded by `sys.platform`) |
| `assets.py` | pinned non-pip assets (ffmpeg): download -> sha256 -> staging -> smoke test -> atomic swap -> rollback; ownership marker `.voxprint-owned` |
| `env_probe.py` | read-only probe of Python/torch/CUDA/packages/models and the reuse / upgrade / offer / install decision per component |
| `install_state.py` | completion manifest, health check with stable reason codes (`--verify-install`), uv-based venv plan for `--repair` |
| `model_downloader.py`, `modelscope_mirror.py` | model download from Hugging Face with a ModelScope fallback |
| `version_manager.py`, `verified_manifest.py` | PyPI/HF version queries and comparison; the "verified by Voxprint" manifest (`verified_manifest.json`) with update channels |
| `updater.py` | check -> install into `.staging/` -> smoke test -> swap, rollback on failure |
| `vram_optimizer.py` | chooses LoRA training parameters (batch, accumulation, 8-bit optimizer, checkpointing) from the available VRAM |

### `workers/`
`pipeline_runner.py` holds the Qt-free scenarios ("dataset", "voice (LoRA)", "universal model", first-run prefetch) with injectable collaborators;
`process_worker.py` wraps them in `QThread` workers whose only interface to the UI is signals (`progress`, `finished`, `failed`, `cancelled`).

### `ui/`
`main_window.py` (core workflow + theme), `settings_dialog.py` (gear), `about_dialog.py`, `upgrade_dialog.py`.

## 3. Data flow of voice training
```
recording + text
   |  audio_utils.read_audio            any format -> mono float32
   |  text_utils / normalizer           clean text, Russian numbers & abbreviations "as pronounced"
   v
aligner.align_long                      forced alignment, chunks <= 150 s at pauses; text without punctuation is split into clauses <= 14 words
   v
slicer -> quality                       2-20 s clips cut at pauses; clips with problems are dropped
   v
dataset_builder                         dataset/ (Alexandria train_lora.py format) + report.json
   v
vram_optimizer -> lora_trainer          LoRA adapter (safetensors), training_meta.json, voice.json (core/voice_info.py)
   v (optional)
model_export                            merged universal model with the voice as `custom_voice`
```
Every stage reports `(stage, fraction, message)` through the progress callback and polls the cancellation token between units of work; a cancelled run leaves no half-written
result in the output folder.

## 4. Installation, environment and updates
* **Installer** (Inno Setup, per user) installs the PyInstaller `onedir` build and runs the first-start setup: environment probe, optional Python venv with the right torch build
  (`uv`), model download. Everything is recorded in the **completion manifest** (`state/`); `main.py --verify-install` re-checks it and prints stable reason codes; `--repair` rebuilds the venv.
* **Environment probe** (`env_probe`): for torch / transformers / peft / ffmpeg / models decides `reuse` (suitable and verified), `upgrade` (offer one click), `offer` or `install`.
  User environments are never modified silently - only after the one-click prompt (`ui/upgrade_dialog.py`).
* **Models**: `model_locator` finds snapshots from other apps and uses them read-only (completeness and revision checked); otherwise `model_downloader` fetches into `models/`
  (Hugging Face, falling back to ModelScope).
* **Updates** (`updater`): versions come from PyPI / the HF Hub; candidates are restricted by the verified manifest and its channel; the new package is installed into `.staging/`,
  smoke-tested (`SMOKE_CODE` in a child process) and then swapped into `packages/` (put on `sys.path` at start-up) with a backup kept for rollback.

## 5. Threading
The UI thread never blocks. A `QThread` worker runs a runner scenario; progress/finish/failure/cancel arrive as signals. Cancellation is cooperative (a token, see `core/events.py`);
the main window disables the controls that must not change while a worker runs (including the UI language switch).

## 6. UI structure and theme
* The main window contains the core workflow only: choose audio + text, optional voice name / type / description, one button, progress and result. Everything else lives in the
  **Settings dialog** behind the gear button: language, updates, model and data folders, repair, About. The whole content sits in a scroll area, so the window fits small screens.
* Windows 11 gets an Acrylic backdrop (`platform_win`) under a **strong dark tint**; panels and controls are solid and all text colours are opaque. The palette constants
  (`TEXT`, `TEXT_MUTED`, `CARD_GLASS`, ...) and `CONTRAST_PAIRS` live at the top of `ui/main_window.py`; `tests/test_ui.py::test_theme_contrast` checks WCAG AA (>= 4.5:1),
  also for the glass tint over a white backdrop. Without Acrylic the plain palette (`ROOT_PLAIN`, `CARD_PLAIN`) is used.
* All text goes through `tr()`; changing the language retranslates live.

## 7. File formats
**Adapter folder** (output of training): `adapter_model.safetensors`, `adapter_config.json`, `training_meta.json`, `voice.json`.

**`voice.json`** (`core/voice_info.py`, `VOICE_SCHEMA = 1`):
```json
{
  "schema": 1, "voice_name": "...", "language": "Russian", "created": "2026-10-03T12:00:00Z",
  "speech_seconds": 1543.2, "epochs": 5, "base_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "voice_type": "", "description": ""
}
```
`voice_type` is `male | female | child | other` or empty; `description` has collapsed whitespace and at most 500 characters.

## 8. Testing strategy
* No test needs a GPU, a network connection or a model: aligners are replaced by `TrueRateAligner` (`tests/synth.py` synthesizes readings with a known ground truth),
  and network/process access is injected (`opener`, `run`, `which`, `runner`, `updater`).
* `tests/conftest.py` isolates app-data folders and the language for every test.
* `tests/test_i18n.py` guards the localization rules; `tests/test_credits.py` keeps `credits.json`, the notices and the installer in sync.
* Windows-only behaviour (Acrylic, real GPU training) is verified manually - see the "Tested on Windows" section of the README.
