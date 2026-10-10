# Tests and verification

## Tested on Windows
Real run on **Windows Server 2025 + NVIDIA RTX 4090** (driver 610.88 / CUDA 13.3, Python 3.11.9, torch 2.14.1+cu130), 2026-10-03.
The 4090 has 22.5 GiB; the app was tested against a 16 GiB cap to emulate a 16 GB laptop GPU.

| Area | Result |
|---|---|
| Test-suite on Windows | **236 passed, 1 skipped** (2 Windows-only failures found and fixed) |
| Install (Python, Inno Setup, venv, torch cu130, dependencies) | passed, about 2:50 with `uv`; silent installer run 88 s |
| Model download (Hugging Face) | aligner 34 s, TTS-1.7B 58 s; reuse from another app's HF cache: 0 files copied |
| Pipeline A - SAPI voice, English, 170.6 s | 20 segments, 154.9 s kept, 0 dropped; LoRA r=32, 15 epochs: **112 s** (~7.5 s/epoch); peak VRAM **5.18 GiB** (torch) / **6.13 GiB** (whole GPU); adapter **58 MB**; merged export **13 s, 4.21 GiB** |
| Pipeline B - LibriSpeech, 293 s, unpunctuated text | 30 segments (5.4-10.8 s), 721 of 721 words used; cut-edge error mean 0.33 s (start) / 0.24 s (end); 11 epochs in 130 s |
| ASR round trip (Qwen3-ASR-0.6B) on generated speech | 2 of 2 sentences transcribed exactly (WER 0) |
| Adapter loading | `PeftModel.from_pretrained` + voice-clone generation works; merged model generation works |
| Build | PyInstaller onedir 3.59 GiB; installer 1.83 GiB; full `build.bat` about 16 min |
| Installer | silent install, installed exe `--selftest-imports` and `--selftest` OK, uninstall in 4 s |
| `--verify-install` / `--repair` | passed after two fixes (idempotent) |
| Input formats | mp3 passed; m4a passed after a fix; Russian text handling (UTF-8/cp1251, normalizer) passed |
| GUI as a normal window | verified at 1920x1080 at 100/125/150% and 1366x768 at 125% |

**Second round (2026-10-03, same machine, after the Studio was built):**

| Area | Result |
|---|---|
| Test-suite | 600+ tests pass on Linux (offscreen Qt); the Windows-only parts were exercised by the runs below |
| Narration on the GPU (real Qwen3-TTS + adapter, Russian and English, technical text) | works: ~20 s samples are synthesised in about a minute, round-trip ASR error (WER) 3-20 % (Latin terms inside Russian text are the main source) |
| Real Russian recording (10 min, phone) | 66-70 clips, 10 epochs at lr 1e-6 in about 4 min; the voice is recognisably the speaker, but the **pitch drifts up by about +2...+3 semitones** compared with the recording (a known LoRA effect; the automatic check warns above +4) |
| Quick preview / compare, automatic voice check | single 57 s, compare 116 s wall; estimate within 10 %; peak VRAM 6.6 GiB |
| Voice-owner consent (ASR) | scope read correctly in 9 of 9 synthesised statements (ru/en/de); a bug with dates written in words was found and fixed; **a real human recording of the statement was not tested** |
| No-transcript mode | 609 s recording: 52 clips, 49 kept, 113 s, WER 15.7 % against the script |
| Installer end to end (silent install, first start, self-tests, `--verify-install`, `--repair`, uninstall) | passed after fixes; models imported by hard link; `--selftest-narrate` produced an MP3 from the installed exe |
| Backup / restore on NTFS | 9 GB backup in 18 s, restore in 29 s, trees identical |
| ffmpeg | the pinned LGPL build downloads and works (`libopus`, `libmp3lame`, `aac`, FLAC) |

**Not verified (honest caveats):** clicking through the GUI of the *installed* app (only process-level smoke tests were done), other GPUs (the training-time table for 4080 / 4070 ... is
derived from specifications, only the 4090 was measured), other speakers and languages, a real human consent recording, the full `lzma2/max` installer compile (a fast-compression build was tested),
loading the adapter in Alexandria with its own pinned peft, the 8-bit optimizer, the ModelScope fallback.
Earlier (first round) caveats: real **Russian** audio through alignment and training was not run in round one (done in round two), voice similarity had not been judged by ear.
Cosmetic: the `sox` package prints a "SoX could not be found" notice on import (hidden from the log); the exe has no version info yet.

The Linux test-suite (`QT_QPA_PLATFORM=offscreen python -m pytest`) covers the pipeline with synthetic audio and a tiny randomly initialised model.

## What was verified against primary sources
* **qwen-asr 0.0.6** (PyPI sources): `Qwen3ForcedAligner.from_pretrained(path, dtype, device_map)`, `.align(audio=(np, sr), text, language="Russian")` ->
  `ForcedAlignItem(text, start_time, end_time)` in seconds; the unit is a space-separated word and punctuation is dropped, so the text is normalized first.
  The package cuts audio at 180 s and the model handles ~5 min: recordings are cut at pauses into chunks <= 150 s (`align_long`).
* **Alexandria** (`train_lora.py`, `tts.py`, `lora.md`): dataset and adapter formats, the teacher-forcing input (ported to `core/teacher_forcing.py`), peft on `hf_model.talker`,
  `PeftModel.from_pretrained(talker, adapter_path)`. Verified on CPU with real qwen-tts classes and a tiny model, including reload (peft 0.18.1 and 0.21.2).
* **Merged model** (`core/model_export.py`, mirrors the official `finetuning/sft_12hz.py`): on the tiny model the merged talker output equals the adapter model output (atol 1e-4),
  the folder has no `lora_`/`speaker_encoder` keys, `codec_embedding[spk_id]` holds the speaker embedding, and the weights reload.
* **Qwen3-TTS-12Hz-1.7B-Base on HF**: the audio tokenizer is inside the repo (`speech_tokenizer/`).
* **DWM** (Microsoft Learn): `DWMWA_USE_IMMERSIVE_DARK_MODE=20`, `DWMWA_WINDOW_CORNER_PREFERENCE=33`, `DWMWA_SYSTEMBACKDROP_TYPE=38`; backdrop types AUTO 0, NONE 1,
  MAINWINDOW 2, **TRANSIENTWINDOW 3 (Acrylic)**, TABBEDWINDOW 4. Only the native DWM through `ctypes` is used (no GPL frameless-window libraries).
* Licences in `credits.json` were read from PyPI metadata, GitHub and Hugging Face cards on 2026-10-03.
* **Model reuse / mirror / environment probe** (tests with fake directory trees and fake machines): HF cache layouts incl. symlinks, refs/snapshots, truncated and sharded weights,
  Pinokio/Alexandria-style tree, ModelScope layout, pinned-revision mismatch, read-only guarantee, resumable ModelScope download (fake server), reuse/upgrade/install decisions,
  torch flavor mapping, completion manifest + reason codes, pinned-asset installer (sha256, staging, smoke test, rollback).

## Still to verify
* Other GPUs (16 GB laptop cards, 3000/5000 series), other speakers, languages and microphones; a **female** voice (the pitch drift above was seen on a male voice).
* A real human recording of the consent statement; real USB/NAS backup targets, exFAT timestamps.
* Text preparation on real FB2/EPUB corpora and the optional AI clean-up (`SageEngine`) on real hardware; reuse against a real Alexandria/Pinokio install; the ModelScope fallback.
* GUI click-through of the installed app on a clean machine, full `lzma2/max` installer compile and Windows SmartScreen / signing behaviour.
* The optional `ctc-forced-aligner`; reading the adapter in Alexandria with its pinned `peft==0.18.1`.

## Code audit

`tools/audit.py` is the release gate. It runs on every push and pull request to `main`, weekly, and as the first job of the installer workflow. What it blocks on, and which findings are accepted, is [AUDIT.md](AUDIT.md).

    python -m pip install -r requirements-audit.txt
    python tools/audit.py --python <app venv python>

CodeQL runs weekly and on demand (`codeql.yml`). Dependabot (`.github/dependabot.yml`) opens a few update PRs (the model-critical pins are ignored). Reviewed false positives carry `# nosec <id> - reason` / `# noqa: <code> - reason`.
