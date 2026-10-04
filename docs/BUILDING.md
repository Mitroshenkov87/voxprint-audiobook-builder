# Building, running from source and contributing

## Build the installer locally
`build.bat` produces `installer\Output\Voxprint-Setup.exe` (Inno Setup 6, per-machine install, Windows 11 x64; about 1.8 GB because PyTorch is inside).
The installer does not contain the models: on first start Voxprint downloads about 7 GB once (internet needed).
The wizard is available in **English, Russian and German** and has an optional page **"Existing models"** (right after the install folder): *Do you already have downloaded models from a previous install?* - choose the
folder or leave the field empty to skip. The installer **copies nothing**; it only writes the chosen path to `%LOCALAPPDATA%\Voxprint\state\existing_models_dir.txt` (UTF-8), and on the first start the program imports the
models from there instead of downloading them (see [Backup, restore and existing models](MODELS.md#backup-restore-and-existing-models)). Silent install: `Voxprint-Setup.exe /VERYSILENT /ModelsDir="D:\old\models"`.
Caveat: the installer runs elevated (per-machine); if the administrator account differs from the account that uses the program, `%LOCALAPPDATA%` is the administrator's - then set the folder in *Settings* instead.

## Run from source (Windows)
```bat
py -3.11 -m venv .venv && .venv\Scripts\activate
pip install uv
uv pip install torch torchaudio --torch-backend=auto       :: fallback: pip install -r requirements-torch.txt
uv pip install -r requirements.txt -r requirements-verified.txt
uv pip install --no-deps -r requirements-nodeps.txt        :: qwen-asr/qwen-tts: their transformers pins conflict
python -m bitsandbytes                                     :: optional check of the 8-bit optimizer (plain AdamW otherwise - fine)
python main.py
```
**Never run `uv run` without `--no-sync`** - it re-syncs the environment and replaces CUDA torch with the CPU build. flash-attn is not needed on Windows.

## Build options
```bat
build.bat              :: venv + torch (uv, CUDA auto-detect) + dependencies + notices + tests + dist\Voxprint.exe (--onefile) + installer if Inno Setup 6 exists
build.bat onedir       :: dist\Voxprint\ folder (preferred: faster start, and Qt/PySide6 stay replaceable - see Licences)
:: quick rebuild:  set VOX_SKIP_TESTS=1 & set VOX_SKIP_INSTALLER=1 & build.bat onedir
:: verify a frozen build contains every library:  dist\Voxprint\Voxprint.exe --selftest-imports   (result in <app home>\logs\selftest_imports.txt)
```
Experimental thin installer (shell only, modules downloaded by the app): `build_thin.bat`, then `installer\build_online.ps1 -Thin` - see [THIN-INSTALLER.md](THIN-INSTALLER.md).
Command-line maintenance flags of `main.py`: `--prefetch` (download models now), `--selftest`, `--selftest-imports`, `--selftest-narrate [voice]` (headless: narrates two sentences with the first voice, result in `<app home>\logs\selftest_narrate.txt`, exit code 0 / 1 / 2 = no voice), `--verify-install`, `--repair`.
App data lives in `%LOCALAPPDATA%\Voxprint` (`models\`, `logs\`, `state\`, ...; override with `VOXPRINT_HOME`).
Interface language override: `VOXPRINT_LANG=en|de|ru`.

## Contributing
Contributions are welcome - see [`CONTRIBUTING.md`](../CONTRIBUTING.md) and [`DEVELOPMENT.md`](DEVELOPMENT.md) (setup, tests, code style, localization, how to propose changes) and
[`docs/ARCHITECTURE.md`](ARCHITECTURE.md) (module map, data flow). Quick start for developers:
```bash
python -m venv .venv && . .venv/bin/activate        # Linux: enough to run the tests
pip install -r requirements.txt -r requirements-verified.txt -r requirements-dev.txt
pip install --no-deps -r requirements-nodeps.txt
QT_QPA_PLATFORM=offscreen python -m pytest          # ~350 tests, no GPU, no network
python -m core.cli audio.wav text.txt --out dataset --fake-aligner     # dry run of the pipeline without a neural network
```
CLI for stage-by-stage checks on a GPU machine: `python -m core.cli audio.wav text.txt --out dataset [--language Russian] [--device cuda] [--train --output-dir output]`.
Regenerate the notices after editing `credits.json`: `python tools/gen_notices.py` (a test checks they are in sync).

### Localization
Supported UI languages: **English (default), German, Russian**. Catalogs are flat JSON files `locales/{en,de,ru}.json` (`"ui.start": "...{name}..."`);
`core/i18n.py` provides `tr(key, **params)`. Language order: `VOXPRINT_LANG`, saved choice (`state\language`), Windows user locale, English. A system language that
is not supported (e.g. Ukrainian) falls back to English. Tests check that all catalogs have identical keys and placeholders and that no user-facing literals are left in the code.

#### Adding a language
Ukrainian and Belarusian were dropped on purpose (fewer languages to keep in sync); their last complete catalogs are in git history
(`git show cdea827:locales/uk.json`, `...be.json`; component texts in `git show cdea827:credits.json`). To add a language `xx`:
1. `core/i18n.py`: append `"xx"` to `LANGS` and its own-language name to `LANG_NAMES` (the language switcher and the system-locale mapping `normalize_code` are driven by these two).
2. `locales/xx.json`: copy `en.json` and translate all values; keep every key and every `{placeholder}` (a test enforces both).
3. `credits.json`: add an `"xx"` text to every `purpose` (and `note`) entry - the credits test requires all `LANGS`.
4. Run `python -m pytest` - the parity tests (`tests/test_i18n.py`, `tests/test_credits.py`) and the per-language UI tests iterate over `i18n.LANGS`; update the
   places that count languages (the switcher test in `tests/test_i18n.py` and the `len(seen) == 3` checks in `tests/test_env_install.py` / `tests/test_model_locator.py`).
5. Mention it in the README.

### Repository link
The GitHub URL lives in one place: `"repo_url"` in `credits.json` (read as `core.appinfo.REPO_URL`). A placeholder containing `OWNER` hides the link in **About**.

### Versions: "verified by Voxprint"
`infra/verified_manifest.json` pins the package versions and model revisions (HF commit shas) Voxprint was tested with. The updater installs exactly those
(rolling back if needed: `restore_verified()`) and merely logs newer PyPI releases as "not verified yet". Channel "latest": `VOXPRINT_CHANNEL=latest`
(or `{"channel":"latest"}` in `state\updater_state.json`). `requirements-verified.txt` / `requirements-nodeps.txt` are generated from the manifest
(`python -m infra.verified_manifest [--nodeps]`; a test keeps them in sync).
