# Voxprint

**Requires Windows 11 26H2, NVIDIA GPU (16 GB VRAM recommended)**

A native Windows 11 window (PySide6): give it a 5-15 minute recording of your own voice and the text you read.
Voxprint aligns the text to the recording with a neural forced aligner (`Qwen/Qwen3-ForcedAligner-0.6B`), cuts it into an
**Alexandria (Qwen3-TTS) dataset** and, if you want, trains your voice as a LoRA adapter. Optionally it merges the
adapter into a standalone ~4 GB model that works in any app that runs Qwen3-TTS. No browser, no Gradio, no WSL, no web server.

> **Privacy.** Use only your own voice (or the voice of someone who explicitly agreed). Recordings, text and the finished
> voice stay on your computer; the program sends nothing to the internet except downloading models and checking for updates.
> A notice is shown on first start and a reminder sits at the bottom of the window.

## For users
1. Install `Voxprint-Setup.exe` (on first start the program downloads the models, ~7 GB, once; internet needed).
2. Press **Choose audio** and **Choose text** (or drag the files into the window). The text is a UTF-8 `.txt`.
3. Press **Create voice (LoRA)** - everything else is automatic. When it finishes the folder opens:
   `...\<recording name>_Voxprint\dataset` (dataset) and `...\output\<voice name>` (the small adapter, tens of MB).
   **Create dataset** builds only the dataset.
4. Optional: **Build universal model (~4 GB)** merges the trained adapter into a standalone model in
   `...\output\<voice name>\merged_model` (Qwen3-TTS `custom_voice` format; the voice is `speaker=<voice name>`).
   The small adapter works only in a few apps (e.g. Alexandria); the ~4 GB model is universal - it works in any app that runs
   Qwen3-TTS. Before starting, Voxprint checks the free disk space and asks for confirmation; the big model is never built
   automatically. The button is enabled once a trained adapter exists.
5. Language: English, Deutsch, Русский, Українська, Беларуская - the list in the top right corner (default: your system
   language, otherwise English; disabled while a task runs). **About** shows the help, the authors and the open-source
   components with their licences, the GitHub link (shown only after the repository URL is set) and the third-party notices.

Bad fragments (clipping, silence, noise) are dropped automatically; numbers and abbreviations in Russian text are spelled out.
Updates are checked weekly (**Check for updates** runs it manually).

## Result format (Alexandria `train_lora.py` contract)
```
dataset\  metadata.jsonl  {"audio":"segment_001.wav","text":"...","ref_audio":"ref.wav"}  (UTF-8, relative paths)
          segment_NNN.wav (24 kHz mono, 3-12 s of speech + ~1 s trailing silence, cuts at pauses, never mid-word)
          ref.wav (24 kHz, 5-10 s, the cleanest fragment)   ref_text.txt (exact transcript of ref.wav)
          report.json (incl. text_raw - the original text before normalization, quality drops, training language)
output\<voice name>\  adapter_model.safetensors, adapter_config.json, ref_sample.wav, training_meta.json
          checkpoints\epoch_NN\ (adapter copy after every epoch; may be deleted)
          merged_model\ (only after "Build universal model": model.safetensors bf16 ~4 GB, config.json with
          tts_model_type=custom_voice + talker_config.spk_id, speech_tokenizer\, ref_sample.wav, ref_text.txt,
          speaker_embedding.safetensors, voxprint_voice.json, USAGE.txt)
```
Use of the merged model: `Qwen3TTSModel.from_pretrained(folder).generate_custom_voice(text, language="Russian", speaker="<voice name>")`.
Hyperparameters are chosen automatically (Alexandria `lora.md`): LoRA r=32, alpha=128 on q/k/v/o_proj of the **talker**, batch 1 with
accumulation 4-8, lr 1e-6 (< 90 fragments) or 2e-6, epochs ~ 320 / fragments, eager attention, bf16, gradient checkpointing; a warning
if the final loss is < 3.5 (the threshold is not confirmed for Russian).

**Not implemented (future step):** converting the merged model to GGUF / LiteRT for phones. It is only documented here; the
merged folder is a plain Hugging Face directory, so a converter can start from it.

## For developers (Windows)
```bat
:: build (venv + torch via uv with CUDA auto-detect + dependencies + notices + tests + exe + installer if Inno Setup 6 exists)
build.bat            :: dist\Voxprint.exe (--onefile)     |  build.bat onedir -> dist\Voxprint\   (onedir is preferred, see Licences)
:: run from source
py -3.11 -m venv .venv && .venv\Scripts\activate
pip install uv
uv pip install torch torchaudio --torch-backend=auto       :: fallback: pip install -r requirements-torch.txt
uv pip install -r requirements.txt -r requirements-verified.txt
uv pip install --no-deps -r requirements-nodeps.txt        :: qwen-asr/qwen-tts: their transformers pins conflict
python -m bitsandbytes                                     :: check the 8-bit optimizer (plain AdamW otherwise - fine)
python main.py
:: tests
pip install pytest && python -m pytest
:: CLI (developer tool, no GUI)
python -m core.cli audio.wav text.txt --out dataset [--language Russian] [--device cuda] [--train --output-dir output]
python -m core.cli audio.wav text.txt --out dataset --fake-aligner     :: dry run without a neural network
:: regenerate the notices after editing credits.json
python tools\gen_notices.py            :: THIRD_PARTY_NOTICES.md (a test checks it is in sync)
```
**Never run `uv run` without `--no-sync`** (it re-syncs the environment and replaces CUDA torch with CPU torch). No flash-attn on Windows.
Versions: `infra/verified_manifest.json` pins the package versions and model revisions Voxprint was tested with; the updater installs
those by default (`VOXPRINT_CHANNEL=latest` for newest). `requirements-verified.txt` / `requirements-nodeps.txt` are generated from it.
App data: `%LOCALAPPDATA%\Voxprint` (`models\`, `logs\`, `state\`, ...; override with `VOXPRINT_HOME`). Interface language override: `VOXPRINT_LANG=en|de|ru|uk|be`.

### Localization
Catalogs are flat JSON files `locales/{en,de,ru,uk,be}.json` (`"ui.start": "...{name}..."`); `core/i18n.py` provides `tr(key, **params)`.
Order: `VOXPRINT_LANG`, saved choice (`state\language`), Windows user locale (`GetUserDefaultLocaleName`), English. A test checks that all
five files have identical keys and placeholders and that no user-facing literals are left in the code. The component descriptions
are localized inside `credits.json`.

### Repository link
The GitHub URL lives in one place: `"repo_url"` in `credits.json` (read as `core.appinfo.REPO_URL`). While it still contains the
placeholder `OWNER` (`https://github.com/OWNER/voxprint`) the link is hidden in **About**; replace it with the real URL and the button
appears and opens the default browser.

## Licences and third-party components
Voxprint stands on open-source software; the single source of truth is `credits.json`. From it come the **About** list,
`THIRD_PARTY_NOTICES.md` (shipped by the installer together with the `licenses\` folder of full licence texts) and the build-time
appendix with the licences of all installed packages (`tools\gen_notices.py --with-installed`, called by `build.bat`).
Important points:
* **Qt / PySide6 - LGPL-3.0.** Used unmodified and dynamically linked. To let users replace the Qt/PySide6 libraries, prefer
  `build.bat onedir` (the libraries are ordinary files); `--onefile` makes this harder. The LGPL/GPL texts and the source pointers
  (<https://code.qt.io>, <https://pyside.org>) are included.
* **FFmpeg is a GPL-3.0 build.** The ffmpeg binary inside the `imageio-ffmpeg` wheel was built with `--enable-gpl --enable-version3`
  (verified in the Windows 7.1 binary of imageio-ffmpeg 0.6.0). It is a separate executable started as a subprocess, but you are
  distributing a GPL binary: keep `licenses\gpl-3.0.txt` and the source link, or ship an LGPL ffmpeg build / require ffmpeg on `PATH`.
* **soynlp (GPLv3) is excluded.** `qwen-asr` imports it lazily, only for Korean. It is not in `requirements.txt` and the build excludes it
  (`--exclude-module soynlp`), so **Korean alignment is unavailable**.
* **CC-BY-NC-4.0 model.** The optional backup aligner model `MahmoudAshraf/mms-300m-1130-forced-aligner` (used only if you install the
  optional `ctc-forced-aligner`) is non-commercial. The default Qwen models and libraries are Apache-2.0 / MIT / BSD-style.
* PyInstaller is GPL-2.0-or-later **with a bootloader exception** (apps may use any licence); Inno Setup has its own permissive licence.

## What was verified against the primary sources (Linux box, no GPU)
* **qwen-asr 0.0.6** (PyPI sources): `Qwen3ForcedAligner.from_pretrained(path, dtype, device_map)`, `.align(audio=(np, sr), text, language="Russian")` ->
  `ForcedAlignItem(text, start_time, end_time)` in seconds; the unit is a space-separated word, punctuation is dropped, so the text is normalized first.
  The package cuts audio at 180 s, the model handles ~5 min: recordings are cut at pauses into chunks <= 150 s (`align_long`).
* **Alexandria** (`train_lora.py`, `tts.py`, `lora.md`): dataset and adapter formats, the teacher-forcing input (ported to `core/teacher_forcing.py`),
  peft on `hf_model.talker`, `PeftModel.from_pretrained(talker, adapter_path)`. Verified on CPU with real qwen-tts classes and a tiny model, incl. reload (peft 0.18.1 and 0.21.2).
* **Merged model** (`core/model_export.py`, mirrors the official `finetuning/sft_12hz.py`): on the tiny model the merged talker output equals the adapter model output (atol 1e-4),
  the folder has no `lora_`/`speaker_encoder` keys, `codec_embedding[spk_id]` holds the speaker embedding, and the weights reload.
* **Qwen3-TTS-12Hz-1.7B-Base on HF**: the audio tokenizer is inside the repo (`speech_tokenizer/`).
* **DWM** (Microsoft Learn): `DWMWA_USE_IMMERSIVE_DARK_MODE=20`, `DWMWA_WINDOW_CORNER_PREFERENCE=33`, `DWMWA_SYSTEMBACKDROP_TYPE=38`;
  backdrop types AUTO 0, NONE 1, MAINWINDOW 2, **TRANSIENTWINDOW 3 (Acrylic)**, TABBEDWINDOW 4.
* Licences in `credits.json` were read from PyPI metadata, GitHub and Hugging Face cards on 2026-10-03.

## Still to verify on Windows / GPU (TODO-needs-GPU-test)
* Real weights, alignment quality/speed, `align_long` and quality-filter thresholds on real recordings.
* Training on an RTX 4090 Mobile 16 GB (bf16 + eager + checkpointing + peft, 8-bit AdamW on Windows, loss threshold 3.5 and lr 1e-6...2e-6 for Russian).
* **Building the universal model from real 1.7B weights** (memory, time, ~4 GB size, loading with `Qwen3TTSModel` and `generate_custom_voice`) and the real adapter size.
* Reading the adapter in Alexandria with its pinned `peft==0.18.1`; the optional `ctc-forced-aligner` (never run here).
* Acrylic look, PyInstaller build size, the Inno Setup script, the CUDA index of the fallback install.
* Whether the Windows ffmpeg binary you ship is the GPL build (it was in imageio-ffmpeg 0.6.0).

---

## Русский

Нативное окно Windows 11 (PySide6): вы даёте запись своего голоса (5–15 минут) и текст, который читали, — программа
сама размечает запись, нарезает датасет в формате Alexandria (Qwen3-TTS) и при желании обучает ваш голосовой
LoRA-адаптер (папка с `adapter_model.safetensors`, `adapter_config.json`, `ref_sample.wav`, `training_meta.json`).

**Требуется: Windows 11 26H2 (сборка 26100 и новее), видеокарта NVIDIA (рекомендуется 16 ГБ видеопамяти).**
Без NVIDIA датасет создаётся, но обучение голоса пойдёт на процессоре (очень долго, с предупреждением).

> **Конфиденциальность.** Используйте только свой собственный голос (или голос человека, давшего явное согласие).
> Записи, текст и готовый голос хранятся только на вашем компьютере; программа ничего не отправляет в интернет,
> кроме скачивания моделей и проверки обновлений. При первом запуске показывается памятка, внизу окна — напоминание.

### Для пользователя
1. Установите `Voxprint-Setup.exe` (при первом запуске программа сама скачает модели ~7 ГБ — один раз, нужен интернет).
2. Нажмите «Выбрать аудио» и «Выбрать текст» (или перетащите файлы в окно). Текст — `.txt` в UTF-8.
3. Нажмите **«Создать голос (LoRA)»** — всё остальное автоматически. По окончании откроется папка:
   `…\<имя записи>_Voxprint\dataset` (датасет) и `…\output\<имя голоса>` (маленький адаптер: десятки МБ).
   «Создать датасет» — только датасет.
4. По желанию — **«Собрать универсальную модель (~4 ГБ)»**: вливает обученный адаптер в самостоятельную модель
   `…\output\<имя голоса>\merged_model` (формат Qwen3-TTS `custom_voice`, голос доступен как `speaker=<имя>`).
   Маленький адаптер работает лишь в нескольких приложениях (например, Alexandria), большая модель универсальна:
   подходит любому приложению, которое запускает Qwen3-TTS. Перед запуском программа проверяет свободное место и просит подтверждение;
   автоматически модель никогда не собирается. Кнопка активна, когда есть обученный адаптер.
5. Язык интерфейса (English, Deutsch, Русский, Українська, Беларуская) выбирается списком в правом верхнем углу
   (по умолчанию — язык системы, иначе английский). Кнопка «О программе» — справка, авторы и список компонентов с лицензиями.
Плохие фрагменты записи (искажения, тишина, шум) отбрасываются автоматически, числа и сокращения в русском тексте
раскрываются сами. Обновления проверяются раз в неделю (кнопка «Проверить обновления» — вручную; лог `logs\updater.log`).

### Формат результата (контракт Alexandria `train_lora.py`)
```
dataset\  metadata.jsonl  {"audio":"segment_001.wav","text":"…","ref_audio":"ref.wav"}  (UTF-8, пути относительные)
          segment_NNN.wav (24 кГц моно, 3–12 с речи + ~1 с тишины в конце, разрезы по паузам, не посреди слова)
          ref.wav (24 кГц, 5–10 с, самый чистый фрагмент)   ref_text.txt (точная транскрипция ref.wav)
          report.json (в т.ч. text_raw — исходный текст до нормализации, отброшенные по качеству, язык обучения)
output\<имя голоса>\  adapter_model.safetensors, adapter_config.json, ref_sample.wav, training_meta.json
          checkpoints\epoch_NN\ (копия адаптера после каждой эпохи; можно удалять)
          merged_model\ (только если нажата «Собрать универсальную модель»: model.safetensors bf16 ~4 ГБ, config.json,
          speech_tokenizer\, ref_sample.wav, ref_text.txt, voxprint_voice.json, USAGE.txt)
```
`text` в датасете — «произносимая» форма (русский: «5 км» → «пять километров»); язык обучения — `russian` (у Alexandria по умолчанию `english`).
Автоматические гиперпараметры (по `lora.md` Alexandria): LoRA r=32, α=128 на q/k/v/o_proj **talker** (не всей модели),
batch 1 + накопление 4–8, lr 1e-6 (до ~90 фрагментов) или 2e-6, эпох ≈ 320/число фрагментов (правило «примеров × эпох ≈ 250–400»),
`attn_implementation="eager"`, bf16, gradient checkpointing. Предупреждение, если итоговый loss < 3,5 (для русского порог не подтверждён).

### Версии: «проверено Voxprint»
`infra/verified_manifest.json` — набор версий пакетов и ревизий моделей (HF commit sha), с которыми приложение
тестировалось. Обновлятор по умолчанию ставит **именно их** (при необходимости откатывает вниз: `restore_verified()`),
а более новые версии PyPI только фиксирует как «ещё не проверены» (в журнал). Канал «последняя стабильная»:
`VOXPRINT_CHANNEL=latest` (или `{"channel":"latest"}` в `state\updater_state.json`). Необязательный удалённый
манифест — `REMOTE_MANIFEST_URL` / `VOXPRINT_MANIFEST_URL` (принимаются только известные пакеты, версии PEP 440 и 40-значные sha).
Откат пакетов к предыдущему состоянию — `Updater.rollback()`. Проверенные pins: transformers 4.57.6, peft 0.18.1 (как у Alexandria;
тесты проходят и на 0.21.2), accelerate 1.15.0, huggingface_hub 0.36.2, qwen-asr 0.0.6, qwen-tts 0.1.1.
`requirements-verified.txt` и `requirements-nodeps.txt` генерируются из манифеста (`python -m infra.verified_manifest [--nodeps]`, тест следит за совпадением).

### Для разработчика / бота на Windows
```bat
:: сборка (venv + torch(uv, автоподбор CUDA) + зависимости + тесты + exe + установщик, если есть Inno Setup 6)
build.bat            :: dist\Voxprint.exe (--onefile)     |  build.bat onedir -> dist\Voxprint\
:: запуск из исходников
py -3.11 -m venv .venv && .venv\Scripts\activate
pip install uv
uv pip install torch torchaudio --torch-backend=auto       :: запасной путь: pip install -r requirements-torch.txt
uv pip install -r requirements.txt -r requirements-verified.txt
uv pip install --no-deps -r requirements-nodeps.txt        :: qwen-asr/qwen-tts: их pins transformers конфликтуют
python -m bitsandbytes                                     :: проверка 8-bit оптимизатора (иначе обычный AdamW - нормально)
python main.py
:: тесты
pip install pytest && python -m pytest
:: CLI (только для проверки этапов на GPU; без GUI)
python -m core.cli audio.wav text.txt --out dataset [--language Russian] [--device cuda] [--train --output-dir output]
python -m core.cli audio.wav text.txt --out dataset --fake-aligner     :: сухой прогон без нейросети
```
**Не запускайте `uv run` без `--no-sync`** (он пересинхронизирует окружение и заменит CUDA-torch на CPU). flash-attn на Windows не используется (eager/SDPA).
Запасной выравниватель `ctc-forced-aligner` — необязательный (`requirements-optional.txt`), включается сам, если установлен и Qwen не сработал.
Данные приложения: `%LOCALAPPDATA%\Voxprint` (`models\`, `logs\`, `.staging\`, `packages\`, `state\`; переопределяется `VOXPRINT_HOME`).
Установщик: `installer\Voxprint.iss` (Inno Setup 6; для `onedir` — `/DONEDIR`).
Структура: `locales/` (en, de, ru, uk, be), `credits.json`, `licenses/`, `tools/` (gen_notices, fetch_licenses), `core/` (i18n, appinfo, model_export, aligner, slicer, dataset_builder, normalizer, quality, audio_utils, lora_trainer, teacher_forcing, text_utils, cli),
`infra/` (verified_manifest, version_manager, vram_optimizer, model_downloader, updater, paths, platform_win), `workers/`, `ui/`, `tests/`.
Окно: нативный DWM через ctypes (`DwmSetWindowAttribute`); GPL-библиотеки (PyQt-Frameless-Window, PySide6-Fluent-Widgets) не используются.

### Что проверено по первоисточникам (на Linux-боксе, без GPU)
* **qwen-asr 0.0.6** (PyPI, исходники): `Qwen3ForcedAligner.from_pretrained(path, dtype, device_map)`;
  `.align(audio=(np, sr), text, language="Russian")` → `[ForcedAlignResult]` с `ForcedAlignItem(text, start_time, end_time)` (секунды).
  Единица — слово по пробелам, пунктуация отбрасывается (`тест-кейс`→`тесткейс`); поэтому текст нормализуется до выравнивания.
  Пакет режет аудио по 180 c, модель — до ~5 мин: записи режутся по паузам на куски ≤150 c (`align_long`), текст — по предложениям/клаузам.
* **Alexandria** (`train_lora.py`, `tts.py`, `lora.md`, исходники): формат датасета и адаптера, teacher-forcing вход (портирован в
  `core/teacher_forcing.py`), peft на `hf_model.talker`, загрузка потребителем `PeftModel.from_pretrained(talker, adapter_path)`,
  `ref_sample.wav` + `ref_sample_text` из `training_meta.json`. Проверено на CPU на реальных классах qwen-tts с «крошечной» моделью:
  ключи адаптера `base_model.model.model.layers.N.self_attn.q_proj.lora_*` (относительно talker), перезагрузка на свежую модель, peft 0.18.1 и 0.21.2.
* **Qwen3-TTS-12Hz-1.7B-Base на HF**: аудио-токенайзер лежит внутри репозитория (`speech_tokenizer/`), отдельная модель токенайзера не нужна.
* **DWM** (Microsoft Learn): `DWMWA_USE_IMMERSIVE_DARK_MODE=20`, `DWMWA_WINDOW_CORNER_PREFERENCE=33`, `DWMWA_SYSTEMBACKDROP_TYPE=38` (Win11 22621+);
  `DWM_SYSTEMBACKDROP_TYPE`: AUTO 0, NONE 1, MAINWINDOW 2 (Mica), **TRANSIENTWINDOW 3 (Acrylic)**, TABBEDWINDOW 4 (Mica Alt).
* Нормализация русского текста: `ru-normalizr` 0.3.0 и `rutextnorm` 2.1.0 запущены на примерах (числа с падежами, т.д./т.е./т.ч., даты); встроенный запасной вариант покрыт тестами.

### Что ещё нужно проверить на Windows/GPU (TODO-needs-GPU-test)
* Загрузка реальных весов, качество и скорость выравнивания, пороги `align_long` и фильтра качества (клиппинг, −42 dBFS, SNR 8 дБ) на реальных записях.
* Обучение на RTX 4090 Mobile 16 ГБ: bf16 + eager + gradient checkpointing + peft, реальная VRAM, 8-bit AdamW (bitsandbytes на Windows),
  пригодность порога loss 3,5 и lr 1e-6…2e-6 для **русского** (рецепт Alexandria описан для английского).
* Чтение адаптера самой Alexandria на установленной у неё версии peft (она закрепляет `peft==0.18.1`).
* `ctc-forced-aligner` (запасной) не запускался: API прочитан по README, формат результата обрабатывается защитно.
* Acrylic и вид окна; сборка PyInstaller (`--onefile` с torch огромен и стартует долго — запасной вариант `build.bat onedir`); обновление
  пакетов из exe требует установленного Python (иначе пропускается, модели обновляются всегда); Inno Setup скрипт не запускался.
* Индекс CUDA при запасной установке (`requirements-torch.txt`, сейчас cu128); `uv --torch-backend=auto` на Windows.

### Лицензии и ссылка на репозиторий
Список компонентов и лицензий — в `credits.json` (источник для окна «О программе», `THIRD_PARTY_NOTICES.md` и папки `licenses\`;
генерация: `python tools\gen_notices.py`). Qt/PySide6 (LGPL-3.0) используется без изменений и с динамической компоновкой — для замены
библиотек собирайте `build.bat onedir`. Бинарник ffmpeg из `imageio-ffmpeg` — сборка под GPL-3.0 (отдельная программа; текст лицензии и ссылка на
исходники приложены). `soynlp` (GPLv3) исключён → корейское выравнивание недоступно. Необязательная запасная модель MMS-aligner — CC-BY-NC-4.0
(некоммерческая). Адрес репозитория задаётся одним полем `"repo_url"` в `credits.json`; пока там заглушка `OWNER`, ссылка в «О программе» скрыта.
Конвертация универсальной модели в GGUF / LiteRT для телефонов — только планируется (не реализовано).
