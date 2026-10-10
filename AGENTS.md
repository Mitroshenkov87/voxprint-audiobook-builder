# AGENTS.md - notes for AI coding agents and new contributors

**Project:** Voxprint AI Audiobook Builder - an offline desktop app (PySide6) that trains a voice (LoRA on Qwen3-TTS) from a short recording and narrates TXT/FB2/EPUB books in it; optional offline translation (Opus-MT). Beta. Apache-2.0.

Driving the installed program (commands, JSON, exit codes, where the executable lives): [docs/AGENTS.md](docs/AGENTS.md). This file is for agents that change the source.

## Layout
* `core/` - pipeline logic, no Qt (alignment, dataset, narration, translation, consent, export) · `workers/` - background threads · `ui/` - PySide6 windows (no heavy work)
* `infra/` - downloads, mirrors, paths, updater, environment, platform code · `tools/` - build/maintenance scripts · `locales/` - `en.json`, `ru.json`, `de.json`
* `installer/`, `.github/workflows/` - Windows installers and CI (touch only when the task says so; releases are built by CI on `v*` tags)
* `docs/` - all long documentation (map: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)); `docs/legal/` - End User Agreements; `docs/manual/`, `docs/recording-scripts/` - PDFs/texts built by scripts

## Run tests
`pip install -r requirements.txt -r requirements-verified.txt -r requirements-dev.txt && pip install --no-deps -r requirements-nodeps.txt`, then `QT_QPA_PLATFORM=offscreen python -m pytest` (no GPU, no network; tests are isolated by `tests/conftest.py`). Add a test with every change. Setup details: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Build installers
Windows (Inno Setup, online/full/thin) and the Linux package: see [docs/BUILDING.md](docs/BUILDING.md) and [docs/DOWNLOADS.md](docs/DOWNLOADS.md). Do not rebuild or re-tag releases unless asked.

## Invariants - do not break
* **Windows and Linux must both work**: Windows-only code stays in `infra/platform_win.py` behind `sys.platform`; everything else imports cleanly on Linux (Linux is experimental: [docs/LINUX.md](docs/LINUX.md)).
* **UI languages en / de / ru / uk / lv**: every user-visible string goes through `tr("key")`; all catalogs keep identical keys and `{placeholders}` (tests enforce it).
* **No desktop shortcut** is created by the installer (Start menu entry only).
* **Voices need consent**: every voice carries a licence and a consent scope (`voice.json`); never add or commit recordings/voices of other people; the output-use limits shown in the UI must stay accurate. Terms: [docs/legal/](docs/legal/EULA-audiobook-builder.md). Not legal advice.
* **No secrets** in the repo, logs or commits (tokens, keys, personal paths); downloads are hash-checked (SHA-256) - keep it that way.
* **Offline and private**: no telemetry, no account; network only as listed in [docs/PRIVACY.md](docs/PRIVACY.md).
* **Keep the README short** (about one screen per section); put details into `docs/` and link them. English in code, comments and docs.
* **Reusable parts** are only a few large pieces (the yo normaliser, speaker marking, the resumable model downloader; the exe-patch applier when that file exists). Each stays the one file it already lives in. The module docstring is short: what it does, a usage example, dependencies, licence, and `Credit: Voxprint AI Audiobook Builder https://github.com/Mitroshenkov87/voxprint-audiobook-builder`. Do not add a folder, a wrapper, or a behaviour change to make a piece reusable. Yo data lives in `core/data/` only. List the files in the README. Never do this for a small helper.
* Git: `git pull --rebase` before every push; commit only your own files; never rewrite published history.

## Cursor Cloud specific instructions
* Python is 3.11 in `/opt/voxprint-venv` (the version the tests and the Windows build target). `python`, `python3`, `pip`, and `pytest` on `PATH` are wrappers for that venv. PyTorch is the CPU build 2.11.0; this machine has no NVIDIA GPU.
* Run tests with `QT_QPA_PLATFORM=offscreen python -m pytest`. The same variable is required for `python main.py --selftest` (opens the studio and quits). `python main.py --selftest-text` checks text preparation, chunking, stub translation, and ffmpeg with no models.
* A model-free dataset build: `python -m core.cli audio.wav text.txt --out dataset --fake-aligner` (synthetic audio comes from `tests/synth.py`).
* `tests/test_download_lingering.py::test_a_stall_after_data_is_marked_as_progressed` loses a race on this CPU: the helper writes the `.incomplete` file before the watcher records the baseline, so `progressed` stays false. Treat the rest of the suite as the environment check.
* After a full `pytest` run the interpreter can segfault while native libraries unload (Qt, torch). The summary line is the result; a small run exits 0.
