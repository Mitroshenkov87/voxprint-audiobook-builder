@echo off
rem  THIN shell of Voxprint (see docs\THIN-INSTALLER.md).  Same environment as build.bat (run build.bat once, or at least its venv
rem  and "uv pip install" steps, first): this script only runs PyInstaller with the heavy libraries EXCLUDED, so the folder
rem  dist\thin\Voxprint contains Qt, numpy, soundfile and the program (about 150-250 MB) - nothing else.  The libraries it leaves
rem  out are packed as runtime modules by installer\build_online.ps1 -Thin and downloaded by the app (infra\modules.py).
rem  Usage:  build_thin.bat   then   powershell installer\build_online.ps1 -Dist dist\thin\Voxprint -Tag v0.1.0-beta -Thin
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" ( echo [ERROR] Run build.bat first ^(it creates .venv^). & exit /b 1 )
call .venv\Scripts\activate.bat
if not exist "build\notices" mkdir "build\notices"
python tools\gen_notices.py --with-installed --out "build\notices\THIRD_PARTY_NOTICES.md" || (echo [ERROR] THIRD_PARTY_NOTICES.md was not created. & exit /b 1)

set HEAVY=torch torchaudio transformers tokenizers safetensors huggingface_hub hf_xet accelerate peft bitsandbytes sentencepiece ^
 scipy librosa numba llvmlite sklearn joblib onnxruntime av nagisa qwen_asr qwen_tts einops sox pydub imageio_ffmpeg PIL ^
 pymorphy3 pymorphy3_dicts_ru ru_normalizr rutextnorm num2words eng_to_ipa dynet cython yaml regex requests urllib3 tqdm
set EXCL=
for %%m in (%HEAVY%) do call set EXCL=%%EXCL%% --exclude-module %%m

rem  The runtime (PyTorch & co.) is downloaded later and imports the WHOLE standard library (timeit, unittest ...): bundle all of it.
python tools\gen_stdlib_bundle.py build\stdlib_bundle || (echo [ERROR] stdlib list. & exit /b 1)

rem  PyInstaller drops the *.dist-info of bundled packages; the downloaded libraries look them up (transformers: "No package metadata was
rem  found for packaging").  Copy the metadata of every shell package listed in infra\runtime_lock.json "shell".
set META=
for /f "delims=" %%a in ('python tools\make_runtime_lock.py --pyinstaller-metadata-args') do set META=%%a
if "%META%"=="" (echo [ERROR] shell metadata list. & exit /b 1)

echo === Build thin shell ===
pyinstaller --onedir --windowed --noconfirm --clean --name Voxprint --distpath dist\thin --workpath build\thin-work --specpath build\thin-work ^
  --icon "%CD%\assets\voxprint.ico" --add-data "%CD%\assets\voxprint.ico;assets" --add-data "%CD%\assets\check.png;assets" ^
  --paths "%CD%" --paths "%CD%\infra" --paths "%CD%\build\stdlib_bundle" --hidden-import _vx_stdlib ^
  --add-data "%CD%\infra\verified_manifest.json;infra" --add-data "%CD%\infra\assets_manifest.json;infra" --add-data "%CD%\infra\model_mirrors.json;infra" --add-data "%CD%\infra\model_release.json;infra" --add-data "%CD%\infra\runtime_lock.json;infra" ^
  --add-data "%CD%\locales;locales" --add-data "%CD%\credits.json;." --add-data "%CD%\licenses;licenses" ^
  --add-data "%CD%\build\notices\THIRD_PARTY_NOTICES.md;." ^
  --collect-submodules core --collect-submodules infra --collect-submodules ui --collect-submodules workers --collect-submodules tools ^
  --hidden-import netroute --hidden-import soundfile --collect-all certifi ^
  --exclude-module gradio --exclude-module flask --exclude-module soynlp --exclude-module tkinter --exclude-module matplotlib ^
  %META% %EXCL% main.py
if errorlevel 1 (echo [ERROR] PyInstaller failed. & exit /b 1)
echo Thin shell: dist\thin\Voxprint  ^(the packaging step adds _internal\modules.json^)
endlocal
