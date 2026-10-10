# Building, running from source and contributing

## Build the installer locally
`build.bat` produces `installer\Output\Voxprint-Setup.exe` (Inno Setup 6, per-machine install, Windows 11 x64; about 1.8 GB because PyTorch is inside).
The installer does not contain the models: on first start Voxprint downloads about 7 GB once (internet needed).
The wizard is available in **English, Russian and German** and has a page **"Models folder"** (right after the install folder): default = the current location `%LOCALAPPDATA%\Voxprint\models`, or any other folder / drive (Program Files is refused: the app runs without admin rights).
The installer **copies nothing**; it only writes the chosen path to `%LOCALAPPDATA%\Voxprint\state\models_dir.txt` (UTF-8; removed when the default is kept). The app then keeps using models already complete in the default folder, picks up models the chosen folder already holds
(Voxprint layout, Hugging Face cache) and downloads the rest into the chosen folder ([details](MODELS.md#models-folder)). A folder with a **Voxprint backup** (`voxprint-backup.json`) is not a models folder: it is written to `state\existing_models_dir.txt` instead, the default folder stays the live store, and the app restores models and voices from the backup on its first start. Voices and settings stay in `%LOCALAPPDATA%\Voxprint`. Silent install: `Voxprint-Setup.exe /VERYSILENT /ModelsFolder="D:\Voxprint models"`; the older `/ModelsDir="D:\old\models"` (a folder to import models from) still works.
Caveat: the installer runs elevated (per-machine); if the administrator account differs from the account that uses the program, `%LOCALAPPDATA%` is the administrator's - then set the folder in *Settings* instead.

## Run from source (Windows)
```bat
py -3.14 -m venv .venv && .venv\Scripts\activate
pip install uv
uv pip install torch==2.11.0 torchaudio==2.11.0 --torch-backend=cu130       :: fallback: pip install -r requirements-torch.txt
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
Command-line maintenance flags of `main.py`: `--prefetch` (download models now), `--selftest`, `--selftest-imports`, `--selftest-speech` (model-free TTS / ASR smoke), `--selftest-narrate [voice]` (headless: narrates two sentences with the first voice, result in `<app home>\logs\selftest_narrate.txt`, exit code 0 / 1 / 2 = no voice), `--verify-install`, `--repair`.
App data lives in `%LOCALAPPDATA%\Voxprint` (`models\`, `logs\`, `state\`, ...; override with `VOXPRINT_HOME`).
Interface language override: `VOXPRINT_LANG=en|de|ru|uk|lv`.

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
Supported UI languages: **English (default), German, Russian, Ukrainian, Latvian**. Catalogs are flat JSON files `locales/{en,de,ru,uk,lv}.json` (`"ui.start": "...{name}..."`);
`core/i18n.py` provides `tr(key, **params)`. Language order: `VOXPRINT_LANG`, saved choice (`state\language`), Windows user locale, English. A system language that
is not supported (e.g. French) falls back to English. Tests check that all catalogs have identical keys and placeholders and that no user-facing literals are left in the code.

#### Adding a language
Belarusian was dropped on purpose (fewer languages to keep in sync); its last complete catalog is in git history
(`git show cdea827:locales/be.json`; component texts in `git show cdea827:credits.json`). To add a language `xx`:
1. `core/i18n.py`: append `"xx"` to `LANGS` and its own-language name to `LANG_NAMES` (the language switcher and the system-locale mapping `normalize_code` are driven by these two).
2. `locales/xx.json`: copy `en.json` and translate all values; keep every key and every `{placeholder}` (a test enforces both).
3. `credits.json`: add an `"xx"` text to every `purpose` (and `note`) entry - the credits test requires all `LANGS`.
4. Run `python -m pytest` - the parity tests (`tests/test_i18n.py`, `tests/test_credits.py`) and the per-language UI tests iterate over `i18n.LANGS`.
5. Mention it in the README.

### Repository link
The GitHub URL lives in one place: `"repo_url"` in `credits.json` (read as `core.appinfo.REPO_URL`). A placeholder containing `OWNER` hides the link in **About**.

### Build numbers and codenames
Every CI installer build (workflow *build-installer*, job *build-thin*) gets a build number: `GITHUB_RUN_NUMBER` + `offset` from
`BUILD.json` (build 665 = v0.1.1-beta, then 666, 667 ...). `BUILD.json` also holds the **codename**: one Biblical Hebrew word in
Latin transliteration (ASCII letters only) that names the build's changes - edit it for each release. `tools/build_number.py --stamp`
writes number, codename and commit into `credits.json` before PyInstaller runs; the app shows `0.1.1-beta · build 665 "Tikkun"`
(About, splash, Settings, diagnostic report), the installer gets `VersionInfoVersion` `0.1.1.665` and a numbered copy
`Voxprint-Setup-online-<version>-build<N>.exe` (the plain `Voxprint-Setup-online.exe` stays for stable links), and the components
manifest records `build`, so the Components window offers a same-version build only when its number is higher. Local builds are build 0.
To keep the numbering after failed runs, lower `offset` so the next successful build gets the intended number.
The offset was **628** for 0.1.4: run 39 stamped 668 with offset 629, but that installer was built with an empty release tag (manifest URL `v0.0.0-dev`, setup HTTP 404) and cannot be installed. Run 40 + 628 = 668, codename **Kolot**. Runs 41-44 were test or failed runs and published nothing public. The offset was **655** for **0.2.0-beta**, build **700**, codename **Kaporet**: run 45 + 655 = 700. Run 46 failed in the Windows unit tests before it published anything, so the offset is **654** for **0.2.1-beta**, build **701**, codename **Shalem**: run 47 + 654 = 701. The offset stays **654** for **0.2.2-beta**, build **702**, codename **Shelishi**: run 48 + 654 = 702, for **0.2.3-beta**, build **703**, codename **Toledot**: run 49 + 654 = 703. Run 50 failed in the Windows tests before it published anything, so the offset is **653** for **0.2.4-beta**, build **704**, codename **Achim**: run 51 + 653 = 704. Run 52 failed in the Windows tests before it published anything, so the offset is **946** for **1.0.0-rc**, build **999**, codename **Nachon** (Genesis 41:32, established, ready): run 53 + 946 = 999, release tag `v1.0.0-rc`.

### Versions: "verified by Voxprint"
`infra/verified_manifest.json` pins the package versions and model revisions (HF commit shas) Voxprint was tested with. The updater installs exactly those
(rolling back if needed: `restore_verified()`) and merely logs newer PyPI releases as "not verified yet". Channel "latest": `VOXPRINT_CHANNEL=latest`
(or `{"channel":"latest"}` in `state\updater_state.json`). `requirements-verified.txt` / `requirements-nodeps.txt` are generated from the manifest
(`python -m infra.verified_manifest [--nodeps]`; a test keeps them in sync).

### Closing the app and model downloads (runtime behaviour)
- **X on the main window = hard exit.** `ui/studio.py` calls `infra.hard_exit.fire()` from `closeEvent`: every descendant process is killed and the process ends with `os._exit` (no waiting for threads that sit in a socket read). `main.py` arms it only for a real run; tests and `--selftest*` are unaffected. On Windows `arm()` also puts the process into a job object with KILL_ON_JOB_CLOSE. Interrupted downloads are resumable (`.partial` folders), so nothing is lost.
- **One download per model.** `infra.model_downloader.ensure_model` takes an OS file lock `models/.<name>.lock`; a second process/thread asking for the same model waits and then returns the finished folder. The lock is released by the OS when its owner dies.
- **Finishing a download** (`finalize_download`): `.partial` that is already complete is not downloaded again; the rename is retried (10 attempts, ~30 s), the hub's `.cache` is removed first, and a copy is the last resort. The log names the processes that hold files open.
