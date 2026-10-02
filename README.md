# Voxprint

Нативное окно Windows 11 (PySide6): вы даёте запись своего голоса (5–15 минут) и текст, который читали, — программа
сама размечает запись, нарезает датасет в формате Alexandria (Qwen3-TTS) и при желании обучает ваш голосовой
LoRA-адаптер (папка с `adapter_model.safetensors`, `adapter_config.json`, `ref_sample.wav`, `training_meta.json`).

**Требуется: Windows 11 26H2 (сборка 26100 и новее), видеокарта NVIDIA (рекомендуется 16 ГБ видеопамяти).**
Без NVIDIA датасет создаётся, но обучение голоса пойдёт на процессоре (очень долго, с предупреждением).

> **Конфиденциальность.** Используйте только свой собственный голос (или голос человека, давшего явное согласие).
> Записи, текст и готовый голос хранятся только на вашем компьютере; программа ничего не отправляет в интернет,
> кроме скачивания моделей и проверки обновлений. При первом запуске показывается памятка, внизу окна — напоминание.

## Для пользователя
1. Установите `Voxprint-Setup.exe` (при первом запуске программа сама скачает модели ~7 ГБ — один раз, нужен интернет).
2. Нажмите «Выбрать аудио» и «Выбрать текст» (или перетащите файлы в окно). Текст — `.txt` в UTF-8.
3. Нажмите **«Создать голос (LoRA)»** — всё остальное автоматически. По окончании откроется папка:
   `…\<имя записи>_Voxprint\dataset` (датасет) и `…\output` (адаптер). «Создать датасет» — только датасет.
Плохие фрагменты записи (искажения, тишина, шум) отбрасываются автоматически, числа и сокращения в русском тексте
раскрываются сами. Обновления проверяются раз в неделю (кнопка «Проверить обновления» — вручную; лог `logs\updater.log`).

## Формат результата (контракт Alexandria `train_lora.py`)
```
dataset\  metadata.jsonl  {"audio":"segment_001.wav","text":"…","ref_audio":"ref.wav"}  (UTF-8, пути относительные)
          segment_NNN.wav (24 кГц моно, 3–12 с речи + ~1 с тишины в конце, разрезы по паузам, не посреди слова)
          ref.wav (24 кГц, 5–10 с, самый чистый фрагмент)   ref_text.txt (точная транскрипция ref.wav)
          report.json (в т.ч. text_raw — исходный текст до нормализации, отброшенные по качеству, язык обучения)
output\   adapter_model.safetensors, adapter_config.json, ref_sample.wav, training_meta.json
          checkpoints\epoch_NN\ (копия адаптера после каждой эпохи; можно удалять)
```
`text` в датасете — «произносимая» форма (русский: «5 км» → «пять километров»); язык обучения — `russian` (у Alexandria по умолчанию `english`).
Автоматические гиперпараметры (по `lora.md` Alexandria): LoRA r=32, α=128 на q/k/v/o_proj **talker** (не всей модели),
batch 1 + накопление 4–8, lr 1e-6 (до ~90 фрагментов) или 2e-6, эпох ≈ 320/число фрагментов (правило «примеров × эпох ≈ 250–400»),
`attn_implementation="eager"`, bf16, gradient checkpointing. Предупреждение, если итоговый loss < 3,5 (для русского порог не подтверждён).

## Версии: «проверено Voxprint»
`infra/verified_manifest.json` — набор версий пакетов и ревизий моделей (HF commit sha), с которыми приложение
тестировалось. Обновлятор по умолчанию ставит **именно их** (при необходимости откатывает вниз: `restore_verified()`),
а более новые версии PyPI только фиксирует как «ещё не проверены» (в журнал). Канал «последняя стабильная»:
`VOXPRINT_CHANNEL=latest` (или `{"channel":"latest"}` в `state\updater_state.json`). Необязательный удалённый
манифест — `REMOTE_MANIFEST_URL` / `VOXPRINT_MANIFEST_URL` (принимаются только известные пакеты, версии PEP 440 и 40-значные sha).
Откат пакетов к предыдущему состоянию — `Updater.rollback()`. Проверенные pins: transformers 4.57.6, peft 0.18.1 (как у Alexandria;
тесты проходят и на 0.21.2), accelerate 1.15.0, huggingface_hub 0.36.2, qwen-asr 0.0.6, qwen-tts 0.1.1.
`requirements-verified.txt` и `requirements-nodeps.txt` генерируются из манифеста (`python -m infra.verified_manifest [--nodeps]`, тест следит за совпадением).

## Для разработчика / бота на Windows
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
Структура: `core/` (aligner, slicer, dataset_builder, normalizer, quality, audio_utils, lora_trainer, teacher_forcing, text_utils, cli),
`infra/` (verified_manifest, version_manager, vram_optimizer, model_downloader, updater, paths, platform_win), `workers/`, `ui/`, `tests/`.
Окно: нативный DWM через ctypes (`DwmSetWindowAttribute`); GPL-библиотеки (PyQt-Frameless-Window, PySide6-Fluent-Widgets) не используются.

## Что проверено по первоисточникам (на Linux-боксе, без GPU)
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

## Что ещё нужно проверить на Windows/GPU (TODO-needs-GPU-test)
* Загрузка реальных весов, качество и скорость выравнивания, пороги `align_long` и фильтра качества (клиппинг, −42 dBFS, SNR 8 дБ) на реальных записях.
* Обучение на RTX 4090 Mobile 16 ГБ: bf16 + eager + gradient checkpointing + peft, реальная VRAM, 8-bit AdamW (bitsandbytes на Windows),
  пригодность порога loss 3,5 и lr 1e-6…2e-6 для **русского** (рецепт Alexandria описан для английского).
* Чтение адаптера самой Alexandria на установленной у неё версии peft (она закрепляет `peft==0.18.1`).
* `ctc-forced-aligner` (запасной) не запускался: API прочитан по README, формат результата обрабатывается защитно.
* Acrylic и вид окна; сборка PyInstaller (`--onefile` с torch огромен и стартует долго — запасной вариант `build.bat onedir`); обновление
  пакетов из exe требует установленного Python (иначе пропускается, модели обновляются всегда); Inno Setup скрипт не запускался.
* Индекс CUDA при запасной установке (`requirements-torch.txt`, сейчас cu128); `uv --torch-backend=auto` на Windows.
