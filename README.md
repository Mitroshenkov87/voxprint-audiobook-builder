# Voxprint AI Audiobook Builder

**Train a voice from a short recording, then narrate whole books in it - offline, on your own computer.**

![status: beta](https://img.shields.io/badge/status-beta%20%2F%20experimental-orange) ![licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue) ![platform: Windows 11 + NVIDIA](https://img.shields.io/badge/platform-Windows%2011%20%2B%20NVIDIA-lightgrey)

> **Beta / experimental (v0.1.0).** The whole pipeline has run end to end on one machine (RTX 4090, Windows), but only one speaker was tested and settings may still change. Back up your recordings and voices, and report problems as issues. Needs **Windows 11 (24H2+) and an NVIDIA GPU** (16 GB VRAM recommended). Details: [tests and caveats](docs/TESTING.md).

<p align="center"><img src="docs/screenshots/studio_home.png" alt="Voxprint Studio" width="420"> <img src="docs/screenshots/train_notranscript.png" alt="Train your voice" width="460"></p>

## 🎙 Train your own voice: read one of these scripts
> **Recording scripts (EN / RU / DE)** - ready-made texts to read aloud when you record your voice for Voxprint. Read for **about 10-13 minutes** (plus 2-3 optional minutes), save the recording, give the app the TXT as the text - done. The **consent sentence** is built in at the end, no swear words.
>
> | English | Русский | Deutsch |
> |---|---|---|
> | [TXT](docs/recording-scripts/Voxprint-RecordingScript-v4-en.txt) · [PDF](docs/recording-scripts/Voxprint-RecordingScript-v4-en.pdf) · [print](docs/recording-scripts/Voxprint-RecordingScript-v4-en-print.pdf) | [TXT](docs/recording-scripts/Voxprint-RecordingScript-v4-ru.txt) · [PDF](docs/recording-scripts/Voxprint-RecordingScript-v4-ru.pdf) · [print](docs/recording-scripts/Voxprint-RecordingScript-v4-ru-print.pdf) | [TXT](docs/recording-scripts/Voxprint-RecordingScript-v4-de.txt) · [PDF](docs/recording-scripts/Voxprint-RecordingScript-v4-de.pdf) · [print](docs/recording-scripts/Voxprint-RecordingScript-v4-de-print.pdf) |
>
> Only record your own voice, or one whose owner agreed. More: [about the scripts](docs/recording-scripts/README.md).

## What it does
* **Narrate books** - TXT, FB2 (also `.fb2.zip`) and EPUB with chapters; output as one Opus file, MP3 per chapter or M4B; pause, cancel and resume at any time.
* **Train your voice** - give it a 5-15 minute recording (yours, or of someone who agreed) and the text you read; everything else is automatic ([Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) LoRA).
* **Offline translation** before narrating: Russian, English, German ([details](docs/TRANSLATION.md)).
* **Per-voice licences and consent** - every voice carries a licence and a usage scope; the UI shows whether commercial use is allowed ([voices](docs/VOICES.md)).
* **Resumable and safe** - downloads, narration and backups continue where they stopped; SHA-256 checks everywhere.
* **Local and private** - no account, no telemetry; your recordings never leave the computer ([privacy](docs/PRIVACY.md)).
* UI in English, Deutsch and Русский; automatic text clean-up (numbers in words, abbreviations, footnotes) for Russian and English.

More: [all features and screenshots](docs/FEATURES.md) · [how it works](docs/HOW-IT-WORKS.md).

## Download
| Build | File | Size |
|---|---|---|
| **Recommended - online installer** | [`Voxprint-Setup-online.exe`](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/v0.1.0-beta/Voxprint-Setup-online.exe) | about 33 MB (+ about 3 GB downloaded during setup) |
| Full installer (built by GitHub Actions) | [`Voxprint-Setup-github-build.exe`](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/v0.1.0-beta/Voxprint-Setup-github-build.exe) | about 1.96 GiB |
| Linux (**experimental**, Ubuntu 24.04 / Debian 12-13, x86-64) | [`install-voxprint-linux.sh`](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/v0.1.0-beta/install-voxprint-linux.sh) | script; sets up a user-only venv (3-8 GB with PyTorch) |

All files are on the [release page](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/tag/v0.1.0-beta) (pre-release). The installer is **not code-signed**: Windows SmartScreen will warn - choose *More info -> Run anyway* only after the SHA-256 matches. Hashes and verification commands: [docs/DOWNLOADS.md](docs/DOWNLOADS.md). Models (about 7 GB) are downloaded once on the first start.
**Linux, one line** (check the hash against [docs/DOWNLOADS.md](docs/DOWNLOADS.md) first; untested on real desktops, please report problems with the [checklist](docs/LINUX-TEST-CHECKLIST.md)):
```bash
curl -fsSLO https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/v0.1.0-beta/install-voxprint-linux.sh && sha256sum install-voxprint-linux.sh && bash install-voxprint-linux.sh --install-deps
```
Options, paths, removal and the one-file tarball: [docs/LINUX.md](docs/LINUX.md).

## Quick start
1. Install and start Voxprint - the **Studio** opens. Under *Train your voice* choose your recording and the text you read, press **Create voice (LoRA)**.
2. Open *Narrate a book*, choose a TXT / FB2 / EPUB file and your voice (optionally tick *Translate the book*).
3. Press **Start narration** and listen while the rest is being made. The audiobook lands in `Documents\Voxprint\Audiobooks`.

## Documentation
* **User manual (PDF):** [English](docs/manual/Voxprint-Manual-en.pdf) · [Русский](docs/manual/Voxprint-Manual-ru.pdf) · [Deutsch](docs/manual/Voxprint-Manual-de.pdf) (print versions and sources: [`docs/manual/`](docs/manual/))
* **Recording scripts** to read when you record a voice (ru / en / de, TXT + PDF): [`docs/recording-scripts/`](docs/recording-scripts/)
* [User guide](docs/USER-GUIDE.md) · [FAQ](docs/FAQ.md) · [Translation](docs/TRANSLATION.md) · [Voices and licences](docs/VOICES.md) · [AAC / M4B notice](docs/AAC-M4B.md)
* [Models, mirrors and backups](docs/MODELS.md) · [Downloads and hashes](docs/DOWNLOADS.md) · [Linux](docs/LINUX.md) · [Building and contributing](docs/BUILDING.md)
* [How it works](docs/HOW-IT-WORKS.md) · [Architecture](docs/ARCHITECTURE.md) · [Tests and verification](docs/TESTING.md) · [Roadmap](docs/ROADMAP.md) · [Changelog](CHANGELOG.md)

## Licence and credits
Source code: **Apache License 2.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). Models and voices keep their own licences - see [docs/LICENSES.md](docs/LICENSES.md). Terms of use (End User Agreement, not legal advice): [docs/legal/](docs/legal/EULA-audiobook-builder.md). Open-source components and credits: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

*Independent, non-commercial project by Aleksandr Mitroshenkov; not affiliated with any product or company with a similar name ("VoxPrint"/"Voxprint").*
Use voices only with the owner's permission.
