@echo off
rem ==========================================================================
rem  Voxprint - build Voxprint.exe (PyInstaller) and the installer (Inno Setup).
rem  Usage:    build.bat            -> a single file dist\Voxprint.exe  (--onefile)
rem            build.bat onedir     -> folder dist\Voxprint\          (--onedir, starts faster; recommended
rem                                    if --onefile with PyTorch gets too big/slow)
rem  Requires Windows 11 x64 and Python 3.11 (py launcher). The models are NOT part of the exe: they are downloaded on first start.
rem
rem  Install rules (from our research, see README):
rem   * PyTorch only through "uv pip install torch --torch-backend=auto" (it picks the CUDA wheels for the driver);
rem     fallback: the cu128 index from requirements-torch.txt.
rem   * NEVER run "uv run" without --no-sync: it re-syncs the environment and replaces CUDA torch with the CPU build.
rem   * flash-attn is not installed on Windows (training uses eager/SDPA attention).
rem   * qwen-asr and qwen-tts pin different transformers versions -> they are installed with --no-deps.
rem   * soynlp (GPLv3) is neither installed nor packaged: only qwen-asr's Korean alignment needs it, which is unavailable.
rem   * 'build.bat onedir' is preferred so that the Qt/PySide6 (LGPL) libraries can be replaced.
rem   * Versions come from the "verified by Voxprint" set (requirements-verified.txt <- infra\verified_manifest.json).
rem ==========================================================================
setlocal enableextensions
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv || (echo [ERROR] Python 3.11 not found ^(py -3.11^). & exit /b 1)
)
call ".venv\Scripts\activate.bat"
python -m pip install -U pip wheel uv || exit /b 1

echo === PyTorch (CUDA, automatic selection) ===
uv pip install torch torchaudio --torch-backend=auto
if errorlevel 1 (
    echo [i] uv could not select wheels - trying the cu128 index
    python -m pip install -r requirements-torch.txt || exit /b 1
)

echo === Dependencies (verified versions) ===
uv pip install -r requirements.txt -r requirements-verified.txt || exit /b 1
uv pip install --no-deps -r requirements-nodeps.txt || exit /b 1
uv pip install pyinstaller pytest || exit /b 1

echo === Checking bitsandbytes (8-bit optimizer) ===
python -m bitsandbytes >nul 2>&1
if errorlevel 1 (
    echo [i] bitsandbytes does not work - training will use plain AdamW ^(this is fine^).
) else (
    echo bitsandbytes works.
)

echo === Third-party licences ===
rem  credits.json -> THIRD_PARTY_NOTICES.md (+ the licences of the packages actually installed, including dependencies)
if not exist "build\notices" mkdir "build\notices"
python tools\gen_notices.py --with-installed --out "build\notices\THIRD_PARTY_NOTICES.md" || (echo [ERROR] THIRD_PARTY_NOTICES.md was not created. & exit /b 1)

echo === Tests ===
if not defined VOX_SKIP_TESTS ( python -m pytest || (echo [ERROR] Tests failed. & exit /b 1) )

set MODE=--onefile
if /I "%~1"=="onedir" set MODE=--onedir

echo === Build (%MODE%) ===
pyinstaller %MODE% --windowed --noconfirm --clean --name Voxprint ^
  --icon "assets\voxprint.ico" --add-data "assets\voxprint.ico;assets" --add-data "assets\check.png;assets" ^
  --paths . ^
  --add-data "infra\verified_manifest.json;infra" --add-data "infra\assets_manifest.json;infra" ^
  --add-data "locales;locales" --add-data "credits.json;." --add-data "licenses;licenses" ^
  --add-data "build\notices\THIRD_PARTY_NOTICES.md;." ^
  --hidden-import core.i18n --hidden-import core.appinfo --hidden-import core.model_export ^
  --hidden-import core.aligner --hidden-import core.slicer --hidden-import core.dataset_builder ^
  --hidden-import core.audio_utils --hidden-import core.lora_trainer --hidden-import core.teacher_forcing ^
  --hidden-import core.normalizer --hidden-import core.quality ^
  --hidden-import core.text_utils --hidden-import core.events --hidden-import core.errors --hidden-import core.types ^
  --hidden-import infra.version_manager --hidden-import infra.vram_optimizer --hidden-import infra.model_downloader ^
  --hidden-import infra.updater --hidden-import infra.paths --hidden-import infra.platform_win ^
  --hidden-import infra.verified_manifest --hidden-import core.model_locator --hidden-import infra.modelscope_mirror ^
  --hidden-import infra.assets --hidden-import infra.net --hidden-import infra.env_probe --hidden-import infra.install_state ^
  --hidden-import workers.process_worker --hidden-import workers.pipeline_runner --hidden-import ui.main_window ^
  --hidden-import qwen_asr --hidden-import qwen_asr.inference.qwen3_forced_aligner ^
  --hidden-import qwen_asr.core.transformers_backend --hidden-import qwen_tts --hidden-import qwen_tts.inference.qwen3_tts_model ^
  --hidden-import qwen_tts.inference.qwen3_tts_tokenizer ^
  --hidden-import peft --hidden-import bitsandbytes --hidden-import accelerate --hidden-import safetensors.torch ^
  --hidden-import scipy.signal --hidden-import soundfile --hidden-import pydub --hidden-import imageio_ffmpeg ^
  --hidden-import onnxruntime --hidden-import sox --hidden-import nagisa --hidden-import six ^
  --hidden-import huggingface_hub --hidden-import librosa ^
  --collect-all qwen_asr --collect-all qwen_tts --collect-all nagisa --collect-all imageio_ffmpeg ^
  --collect-all bitsandbytes --collect-data librosa --collect-all hf_xet --collect-all certifi ^
  --collect-all ru_normalizr --collect-all rutextnorm --collect-all pymorphy3 --collect-all pymorphy3_dicts_ru ^
  --collect-all num2words --collect-all eng_to_ipa ^
  --copy-metadata transformers --copy-metadata tokenizers --copy-metadata huggingface_hub --copy-metadata safetensors ^
  --copy-metadata accelerate --copy-metadata peft --copy-metadata torch --copy-metadata numpy --copy-metadata tqdm ^
  --copy-metadata regex --copy-metadata requests --copy-metadata packaging --copy-metadata filelock ^
  --copy-metadata qwen-asr --copy-metadata qwen-tts ^
  --exclude-module gradio --exclude-module flask --exclude-module soynlp --exclude-module tkinter --exclude-module matplotlib ^
  main.py
if errorlevel 1 (echo [ERROR] PyInstaller failed. & exit /b 1)

rem --- installer (if Inno Setup 6 is installed); VOX_SKIP_INSTALLER=1 - exe only (quick rebuild), VOX_SKIP_TESTS=1 - skip the tests
if defined VOX_SKIP_INSTALLER goto :finish
set ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe
if not exist "%ISCC%" set ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe
if exist "%ISCC%" (
    if /I "%~1"=="onedir" ( "%ISCC%" /DONEDIR installer\Voxprint.iss ) else ( "%ISCC%" installer\Voxprint.iss )
    echo Installer: installer\Output\Voxprint-Setup.exe
) else (
    echo [i] Inno Setup 6 not found - the installer was not built. Exe: dist\Voxprint.exe
)
:finish
endlocal
