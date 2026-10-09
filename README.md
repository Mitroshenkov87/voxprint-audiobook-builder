# Voxprint AI Audiobook Builder

**Train a voice from a short recording, then narrate whole books in it - offline, on your own computer.**

![status: beta](https://img.shields.io/badge/status-beta%20%2F%20experimental-orange) ![licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue) ![platform: Windows 11 + NVIDIA](https://img.shields.io/badge/platform-Windows%2011%20%2B%20NVIDIA-lightgrey)

> **Beta / experimental (v0.1.3, build 667 "Menuchah").** The whole pipeline has run end to end on one machine (RTX 4090, Windows), but only one speaker was tested and settings may still change. Back up your recordings and voices, and report problems as issues. Needs **Windows 11 (24H2+) and an NVIDIA GPU** (16 GB VRAM recommended). Details: [tests and caveats](docs/TESTING.md).

<p align="center"><img src="docs/screenshots/en-667/01-main-window.png" alt="Voxprint Studio" width="420"> <img src="docs/screenshots/en-667/04-voice-library-boaz-tirzah.png" alt="My voices: build 667 screenshot. Tirzah and Gideon are the bundled voices; the picture also shows Boaz, which is retired" width="420"></p>
<p align="center"><img src="docs/screenshots/en-667/02-narrate-book-top.png" alt="Narrate a book" width="420"> <img src="docs/screenshots/en-667/07-settings-pauses-speed-repair.png" alt="Settings: pauses and reading speed" width="300"></p>

## 🎙 Train your own voice: read one of these scripts
> **Recording scripts (EN / RU / DE)** - ready-made texts to read aloud when you record your voice for Voxprint. Read for **about 15 minutes** (plus 2-3 optional minutes), save the recording, give the app the TXT as the text - done. The **consent sentence** is built in at the end, no swear words.
>
> | English | Русский | Deutsch |
> |---|---|---|
> | [TXT](docs/recording-scripts/Voxprint-RecordingScript-v6-en.txt) · [PDF](docs/recording-scripts/Voxprint-RecordingScript-v6-en.pdf) · [print](docs/recording-scripts/Voxprint-RecordingScript-v6-en-print.pdf) | [TXT](docs/recording-scripts/Voxprint-RecordingScript-v6-ru.txt) · [PDF](docs/recording-scripts/Voxprint-RecordingScript-v6-ru.pdf) · [print](docs/recording-scripts/Voxprint-RecordingScript-v6-ru-print.pdf) | [TXT](docs/recording-scripts/Voxprint-RecordingScript-v6-de.txt) · [PDF](docs/recording-scripts/Voxprint-RecordingScript-v6-de.pdf) · [print](docs/recording-scripts/Voxprint-RecordingScript-v6-de-print.pdf) |
>
> Only record your own voice, or one whose owner agreed. More: [about the scripts](docs/recording-scripts/README.md).

## What it does
* **Narrate books** - TXT, FB2 (also `.fb2.zip`) and EPUB with chapters; output as one Opus file, MP3 per chapter or M4B; pause, cancel and resume at any time.
* **Pauses that follow the text** (new in 0.1.3) - the text is cut per sentence and at strong breaks, each spoken piece is trimmed of its own silence and measured pauses are inserted: comma 0.25 s, strong break 0.4 s, sentence 0.6 s, paragraph or verse line 1.0 s, chapter or scene break 2.0 s. Adjustable in *Settings -> Narration: pauses and speed* or with `voxprint narrate --pause-...`.
* **Reading speed that adapts to the text** (new in 0.1.3) - long, descriptive and scripture-like sentences are read a little slower, dialogue at the voice's own pace; a reading style per book (*Automatic*, *Solemn / scripture*, *Fiction*, *Dialogue-heavy*) and a global speed of 70-130 %. Applied after synthesis with the pitch kept, so changing pauses or speed never re-synthesizes finished parts.
* **Train your voice** - give it a 5-15 minute recording (yours, or of someone who agreed) and the text you read; everything else is automatic ([Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) LoRA).
* **Offline translation** before narrating: Russian, English, German ([details](docs/TRANSLATION.md)).
* **Re-voice** a recording: turn it into editable text, or convert it directly into one of your voices ([details](docs/REVOICE.md)).
* **Open voices included** - *Tirzah* (female) and *Gideon* (male), Russian, **CC0-1.0** (free for any use, also commercial), trained only from public-domain LibriVox recordings; part of the standard model download and read-only in the library ([details](voices/BUNDLED.md)). Listen: [Genesis 1:1-2:3 with Tirzah](samples/genesis-tirzah.mp3) ([samples](samples/README.md)). *Boaz* shipped with 0.1.3 and is retired; Gideon replaces it, used as recorded. *Asher* and *Noa* are optional catalog voices on the same `voices-v1` release, not part of that download ([voices](voices/README.md)).
* **Per-voice licences and consent** - every voice carries a licence and a usage scope; the UI shows whether commercial use is allowed ([voices](docs/VOICES.md)).
* **Resumable and safe** - downloads, narration and backups continue where they stopped; SHA-256 checks everywhere; *Check & repair* verifies every component and model file and re-downloads only the damaged ones.
* **Local and private** - no account, no telemetry; your recordings never leave the computer ([privacy](docs/PRIVACY.md)).
* UI in English, Deutsch, Русский, Українська and Latviešu; automatic text clean-up (numbers in words, abbreviations, footnotes) for Russian and English.

More: [all features and screenshots](docs/FEATURES.md) · [how it works](docs/HOW-IT-WORKS.md).

## Download
| Build | File | Size |
|---|---|---|
| **Online installer (Windows 11 x64, NVIDIA GPU)** | [`Voxprint-Setup-online.exe`](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/v0.1.3-beta/Voxprint-Setup-online.exe) | 35,235,971 bytes (about 34 MB, build 667 "Menuchah"; downloads the libraries from PyTorch/PyPI during setup) |

SHA-256: `c145a9fc84c736d655fbbe9bfd5c1cc94794b1be7fb8e9a5eec293e6f1578e42` - release page: [v0.1.3-beta](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/tag/v0.1.3-beta) (pre-release). Setup type **Full** (default) downloads all models (about 28 GB with the components) on the first start; **Quick** installs only the program and offers the same download in the Components window.
A full offline installer may return later; the Linux version is temporarily unavailable.

### Windows security warnings
The installer is not code-signed yet, so a new download has no reputation. Download it only from the [GitHub releases page](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases). Check the SHA-256 against the `.sha256` file next to the installer, and only then keep or run the file ([details](docs/DOWNLOADS.md)):

```powershell
Get-FileHash .\Voxprint-Setup-online.exe -Algorithm SHA256
```

Edge is the strictest. Its download bar offers **Delete** for a file that is rarely downloaded, and the file can be removed before you run it. Do not click Delete. Hover the download, then **... -> Keep -> Show more -> Keep anyway**.

Chrome and other browsers may show a similar download warning. When you start the installer, from any browser, Microsoft Defender SmartScreen may say **Windows protected your PC**. Choose **More info -> Run anyway**. That step is usually milder than Edge. An antivirus program may also flag the unsigned file.

Code signing is being applied for (SignPath Foundation). The warning will go away later.

**Option 1.** Download `Voxprint-Setup-online.exe` from the table above and run it.

**Option 2: one command in PowerShell (as Administrator).** Read [install.ps1](install.ps1) first. It downloads the installer and its `.sha256` file only from the official GitHub releases, checks the hash, then starts setup. `-Silent` passes `/VERYSILENT`. `-Version v0.1.3-beta` picks that tag; otherwise the latest pre-release is used.

```powershell
irm https://raw.githubusercontent.com/Mitroshenkov87/voxprint-audiobook-builder/main/install.ps1 | iex
```

## Quick start
1. Install and start Voxprint - the **Studio** opens. Under *Train your voice* choose your recording and the text you read, press **Create voice (LoRA)**.
2. Open *Narrate a book*, choose a TXT / FB2 / EPUB file and your voice (optionally tick *Translate the book*).
3. Press **Start narration** and listen while the rest is being made. The audiobook lands in the projects folder (`%LOCALAPPDATA%\Voxprint\Projects\Audiobooks`, shortcut *Voxprint Projects* in Documents).

## Documentation
* **User manual (PDF, 0.1.3 build 667):** [English](docs/manual/Voxprint-Manual-en.pdf) · [Русский](docs/manual/Voxprint-Manual-ru.pdf) · [Deutsch](docs/manual/Voxprint-Manual-de.pdf) (print versions and sources: [`docs/manual/`](docs/manual/); screenshots: [`docs/screenshots/en-667/`](docs/screenshots/en-667/))
* **Recording scripts** to read when you record a voice (ru / en / de, TXT + PDF): [`docs/recording-scripts/`](docs/recording-scripts/)
* [User guide](docs/USER-GUIDE.md) · [Command-line interface](docs/CLI.md) · [Driving the app (for agents)](docs/AGENTS.md) · [FAQ](docs/FAQ.md) · [Translation](docs/TRANSLATION.md) · [Re-voice](docs/REVOICE.md) · [Voices and licences](docs/VOICES.md) · [AAC / M4B notice](docs/AAC-M4B.md)
* [Models, mirrors and backups](docs/MODELS.md) · [Downloads and hashes](docs/DOWNLOADS.md) · [Linux](docs/LINUX.md) · [Building and contributing](docs/BUILDING.md)
* [How it works](docs/HOW-IT-WORKS.md) · [Voice quality](docs/VOICE-QUALITY.md) · [Architecture](docs/ARCHITECTURE.md) · [Tests and verification](docs/TESTING.md) · [Roadmap](docs/ROADMAP.md) · [Changelog](CHANGELOG.md)

## Licence and credits
Source code: **Apache License 2.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). Models and voices keep their own licences - see [docs/LICENSES.md](docs/LICENSES.md). Terms of use (End User Agreement, not legal advice): [docs/legal/](docs/legal/EULA-audiobook-builder.md). Open-source components and credits: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

*Independent, non-commercial project by Aleksandr Mitroshenkov; not affiliated with any product or company with a similar name ("VoxPrint"/"Voxprint").*
Use voices only with the owner's permission.

Contributing: [CONTRIBUTING.md](CONTRIBUTING.md) · [AGENTS.md](AGENTS.md) (for AI coding agents).

## Reusable parts
A few large pieces can be copied on their own. Each file's docstring says what it does, how to call it, and how to credit Voxprint. Small helpers are not listed.

* [`core/yo.py`](core/yo.py) - Russian letter yo, only where the dictionary is sure, plus "все" / "всё" by context. Data: [`core/data/`](core/data/YO_DATASET.md) (the context rules are `core/data/yo_context.json`).
* [`core/speakers.py`](core/speakers.py) - paragraph speaker marks (narrator, male, female), the reply parser, and the voice cast (a second voice per role, characters pinned by name).
* [`infra/model_downloader.py`](infra/model_downloader.py) - resumable, hash-checked model downloads.

## ☕ Support the Project

If you find this project useful and would like to support its development, you can buy me a coffee!

- **Network:** TRON (TRC-20)
- **Accepted:** USDT or TRX
- **Address:** `TYveZBXaSpM4FEHa6iGrc7A6zuLfS4z3x6`

> ⚠️ **Important:** Please ensure you are sending funds **only** via the TRON (TRC-20) network. 
> Sending assets from other networks (such as Ethereum ERC-20, BSC BEP-20, etc.) to this address will result in **permanent loss of funds**.
