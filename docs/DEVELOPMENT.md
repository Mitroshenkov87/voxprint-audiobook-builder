# Development guide (setup, tests, code style, localization, pull requests)

Short version: [CONTRIBUTING.md](../CONTRIBUTING.md). Notes for AI coding agents: [AGENTS.md](../AGENTS.md).

Thanks for your interest! Voxprint is a young project; small, focused contributions are the easiest to review.
Please read [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) first - it explains the module map and the data flow.

## Ground rules
* **Voices need consent.** Use your own voice, or the voice of someone who has explicitly given permission, in examples, issues, test data and screenshots. Never commit recordings of other people without their permission.
* Be kind and constructive. Assume good intent.
* Mind licences: do not add GPL/AGPL Python dependencies to the shipped program (e.g. `soynlp` is deliberately not installed - a test guards this); [docs/LICENSES.md](LICENSES.md)
  explains the existing exceptions. Everything third-party must be listed in `credits.json` (see "Third-party components" below).
* **No behaviour changes in pure refactor / documentation PRs.**

## Development setup
The test suite runs on Linux, macOS or Windows **without an RTX 40-series GPU and without network access**.
A CPU build of PyTorch is **internal**: it exists so this suite and the CI jobs can import the libraries. It is not a way to run Voxprint, and there is no CPU-only mode for users.

```bash
git clone <your fork>
cd voxprint-audiobook-builder
python -m venv .venv && . .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu    # internal, tests only
pip install -r requirements.txt -r requirements-verified.txt -r requirements-dev.txt
pip install --no-deps -r requirements-nodeps.txt          # qwen-asr / qwen-tts: their transformers pins conflict
QT_QPA_PLATFORM=offscreen python -m pytest                # ~250 tests; Windows: set QT_QPA_PLATFORM=offscreen
```

`tests/conftest.py` sets `VOXPRINT_ALLOW_NO_GPU=1`, and the CI workflows set the same variable. It lets a process start when no RTX 40-series GPU is present. It is internal. Do not document it for users and do not set it on a machine where the real check should run. The Windows installer smoke jobs pass the hidden `/SKIPGPUCHECK` switch for the same reason; the Linux CI install passes `--skip-gpu-check`.

Running the GUI and the full pipeline needs Windows 11 24H2+ or a Linux release from 2025, and an NVIDIA RTX 40-series or newer GPU - see "Run from source" in [docs/BUILDING.md](BUILDING.md).
A dry run of the pipeline without any model: `python -m core.cli audio.wav text.txt --out dataset --fake-aligner`.

On Windows, never run `uv run` without `--no-sync` (it replaces CUDA torch with the CPU build).

## Tests
* `pytest` must stay green before you open a PR. `pytest.ini` points at `tests/`.
* `tests/conftest.py` isolates every test (own app-data folder, fixed language `ru` by default, no network, no foreign model caches) - do not undo this.
  Tests that need another language set `VOXPRINT_LANG` themselves.
* `tests/synth.py` generates synthetic readings with a known ground truth and a `TrueRateAligner`, so pipeline tests need no models.
* Anything that talks to the network, subprocesses or other programs' folders is **injectable** (`opener`, `run`, `which`, `fetch_json`, `runner` ...). Follow that pattern in new code.
* UI tests run under `QT_QPA_PLATFORM=offscreen`; blocking dialogs are skipped there.
* Add a test with every behaviour change or bug fix.

## Code style
* Python 3.14, type hints where they help, `from __future__ import annotations`.
* **Documentation is part of the code.** A module docstring says what the module is for. A public function or class gets a one-line summary; a longer note belongs only on logic that is genuinely hard to follow. Do not pad a trivial function with an Args or Returns section. The check on `core/` (including `dry_run.py`), `infra/`, `cli.py`, `main.py`, the main windows, `ui/suite_icons.py` and `tools/codenames.py` is interrogate at 70% of public code, not a docstring on every object. Add comments for *why* something non-obvious is done (not for what the next line does). Code, comments and docstrings are **English**.
* Keep `core/` free of Qt and of installer logic, and keep `ui/` free of heavy work (it belongs in `workers/` or `core/`).
* Windows-only code lives in `infra/platform_win.py` (guarded by `sys.platform`); other modules must import cleanly on Linux.
* Errors the user can see are `DatasetMakerError` subclasses with a localized message (`tr(...)`) - never show a raw traceback in the UI.
* Never touch other programs' folders: model reuse is **read-only**, and only folders carrying Voxprint's ownership marker may be replaced or deleted.

## Localization
All user-visible text goes through `tr("some.key")` and lives in `locales/en.json`, `de.json`, `ru.json`, `uk.json`, `lv.json` (identical keys and `{placeholders}`, and every key used in code must exist - all enforced by `tests/test_i18n.py`, which also rejects Cyrillic string literals in code). New UI text therefore means one catalog entry per language. See "Adding a language" in [docs/BUILDING.md](BUILDING.md).
Russian *data* (abbreviation tables of the text normalizer, the number words of `core/num_words.py`, the preparation rules of `core/text_prep.py` and the validator of `core/text_cleanup.py`, the Russian heading words of `core/book_parsers.py`, test fixtures) legitimately contains Cyrillic; such modules are listed in the exception list of `tests/test_i18n.py`.

## Third-party components
`credits.json` is the single source for the About dialog, `THIRD_PARTY_NOTICES.md` and the `licenses/` folder. After adding a dependency: add its entry (a purpose in every UI language,
licence, URL), run `python tools/fetch_licenses.py --only <id>` to fetch the licence text, then `python tools/gen_notices.py` (a test fails if the notices are out of date).
Pin new runtime packages in `infra/verified_manifest.json` only after testing them (then regenerate the requirements files with
`python -m infra.verified_manifest > requirements-verified.txt` and `python -m infra.verified_manifest --nodeps > requirements-nodeps.txt`).

## Commits and pull requests
* Small commits with a clear first line in the imperative ("Fix ...", "Add ..."). Explain *why* in the body when it is not obvious.
* One topic per PR; describe what you changed, how you tested it and (for UI changes) attach an English screenshot - `docs/screenshots/` shows how they are made
  (offscreen Qt `widget.grab()`).
* Update `CHANGELOG.md` (the "Unreleased" section) for user-visible changes.
* If you change the output formats (`dataset/`, adapter folder, `voice.json`), update `docs/HOW-IT-WORKS.md` and `docs/ARCHITECTURE.md`; `voice.json` carries a `schema` number for exactly this reason.

## Reporting bugs
Open an issue with: Windows build, GPU/driver, what you did, the message shown (it has a stable wording per error kind) and the relevant lines of
`%LOCALAPPDATA%\Voxprint\logs\voxprint.log`. `main.py --verify-install` prints stable reason codes that help a lot. Do not attach recordings of other people.

## Licence of contributions
Voxprint's source code is licensed under the **Apache License 2.0** (see [`LICENSE`](../LICENSE) and [`NOTICE`](../NOTICE)). Unless you state otherwise, any contribution you
submit for inclusion is licensed under the same terms (Apache-2.0, section 5) - no extra agreement is needed. Only contribute code you wrote or that you may license this way.
Do not commit trained voices, recordings, model weights or other people's data: voices have their own licences and are distributed through the voices repository, not through this repository.
