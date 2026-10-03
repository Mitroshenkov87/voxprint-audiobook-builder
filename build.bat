@echo off
rem ==========================================================================
rem  Voxprint - сборка Voxprint.exe (PyInstaller) и установщика (Inno Setup).
rem  Запуск:   build.bat            -> один файл dist\Voxprint.exe  (--onefile)
rem            build.bat onedir     -> папка dist\Voxprint\          (--onedir, быстрее стартует; рекомендуется,
rem                                    если --onefile с PyTorch получается слишком большим/медленным)
rem  Требуется Windows 11 x64, Python 3.11 (py launcher). Модели в exe НЕ входят: скачиваются при первом запуске.
rem
rem  Правила установки (из исследования, см. README):
rem   * PyTorch - только через "uv pip install torch --torch-backend=auto" (сам выбирает колёса CUDA под драйвер);
rem     запасной вариант - индекс cu128 из requirements-torch.txt.
rem   * НИКОГДА не запускайте "uv run" без --no-sync: он пересинхронизирует окружение и заменит CUDA-torch на CPU.
rem   * flash-attn на Windows не ставим (обучение идёт на eager/SDPA).
rem   * qwen-asr и qwen-tts закрепляют разные версии transformers -> ставятся с --no-deps.
rem   * soynlp (GPLv3) не ставится и не упаковывается: нужен только корейскому выравниванию qwen-asr, оно недоступно.
rem   * Для замены библиотек Qt/PySide6 (LGPL) предпочтительна сборка 'build.bat onedir'.
rem   * Версии берутся из набора "проверено Voxprint" (requirements-verified.txt <- infra\verified_manifest.json).
rem ==========================================================================
setlocal enableextensions
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv || (echo [ОШИБКА] Не найден Python 3.11 ^(py -3.11^). & exit /b 1)
)
call ".venv\Scripts\activate.bat"
python -m pip install -U pip wheel uv || exit /b 1

echo === PyTorch (CUDA, автоподбор) ===
uv pip install torch torchaudio --torch-backend=auto
if errorlevel 1 (
    echo [i] uv не смог подобрать колёса - пробую индекс cu128
    python -m pip install -r requirements-torch.txt || exit /b 1
)

echo === Зависимости (проверенные версии) ===
uv pip install -r requirements.txt -r requirements-verified.txt || exit /b 1
uv pip install --no-deps -r requirements-nodeps.txt || exit /b 1
uv pip install pyinstaller pytest || exit /b 1

echo === Проверка bitsandbytes (8-битный оптимизатор) ===
python -m bitsandbytes >nul 2>&1
if errorlevel 1 (
    echo [i] bitsandbytes не работает - обучение будет использовать обычный AdamW ^(это нормально^).
) else (
    echo bitsandbytes работает.
)

echo === Лицензии третьих сторон ===
rem  credits.json -> THIRD_PARTY_NOTICES.md (+ список лицензий реально установленных пакетов, включая зависимости)
if not exist "build\notices" mkdir "build\notices"
python tools\gen_notices.py --with-installed --out "build\notices\THIRD_PARTY_NOTICES.md" || (echo [ОШИБКА] Не создан THIRD_PARTY_NOTICES.md. & exit /b 1)

echo === Тесты ===
python -m pytest || (echo [ОШИБКА] Тесты не прошли. & exit /b 1)

set MODE=--onefile
if /I "%~1"=="onedir" set MODE=--onedir

echo === Сборка (%MODE%) ===
pyinstaller %MODE% --windowed --noconfirm --clean --name Voxprint ^
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
  --hidden-import infra.assets --hidden-import infra.env_probe --hidden-import infra.install_state ^
  --hidden-import workers.process_worker --hidden-import workers.pipeline_runner --hidden-import ui.main_window ^
  --hidden-import qwen_asr --hidden-import qwen_asr.inference.qwen3_forced_aligner ^
  --hidden-import qwen_asr.core.transformers_backend --hidden-import qwen_tts --hidden-import qwen_tts.inference.qwen3_tts_model ^
  --hidden-import qwen_tts.inference.qwen3_tts_tokenizer ^
  --hidden-import peft --hidden-import bitsandbytes --hidden-import accelerate --hidden-import safetensors.torch ^
  --hidden-import scipy.signal --hidden-import soundfile --hidden-import pydub --hidden-import imageio_ffmpeg ^
  --hidden-import onnxruntime --hidden-import sox --hidden-import nagisa ^
  --hidden-import huggingface_hub --hidden-import librosa ^
  --collect-all qwen_asr --collect-all qwen_tts --collect-all nagisa --collect-all imageio_ffmpeg ^
  --collect-all bitsandbytes --collect-data librosa ^
  --collect-all ru_normalizr --collect-all rutextnorm --collect-all pymorphy3 --collect-all pymorphy3_dicts_ru ^
  --collect-all num2words --collect-all eng_to_ipa ^
  --copy-metadata transformers --copy-metadata tokenizers --copy-metadata huggingface_hub --copy-metadata safetensors ^
  --copy-metadata accelerate --copy-metadata peft --copy-metadata torch --copy-metadata numpy --copy-metadata tqdm ^
  --copy-metadata regex --copy-metadata requests --copy-metadata packaging --copy-metadata filelock ^
  --copy-metadata qwen-asr --copy-metadata qwen-tts ^
  --exclude-module gradio --exclude-module flask --exclude-module soynlp --exclude-module tkinter --exclude-module matplotlib ^
  main.py
if errorlevel 1 (echo [ОШИБКА] PyInstaller завершился с ошибкой. & exit /b 1)

rem --- установщик (если установлен Inno Setup 6)
set ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe
if not exist "%ISCC%" set ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe
if exist "%ISCC%" (
    if /I "%~1"=="onedir" ( "%ISCC%" /DONEDIR installer\Voxprint.iss ) else ( "%ISCC%" installer\Voxprint.iss )
    echo Установщик: installer\Output\Voxprint-Setup.exe
) else (
    echo [i] Inno Setup 6 не найден - установщик не собран. Exe: dist\Voxprint.exe
)
endlocal
