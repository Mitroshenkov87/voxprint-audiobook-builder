# FAQ

## What is Voxprint?
*Train a voice, narrate books.* (Short name and technical identifier: **Voxprint** - the package, the executable and the data folder keep that name.)

**Your voice from a recording in one click - and then whole books in that voice.** Give Voxprint a 5-15 minute recording of a voice (yours, or of a person who has given permission) and
the text you read: it aligns the text to the audio, cuts a training dataset and trains your voice as a LoRA adapter for
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (optionally merged into a standalone model for any app that runs Qwen3-TTS).
The voice goes into your **voice library**; then pick a book (TXT, FB2, EPUB) and a voice and get a finished audiobook with chapters
(one Opus file, per-chapter MP3, ...). A native Windows 11 application - no command line, no browser, no Gradio, no WSL.

## Which computer do I need?
> **Requires an NVIDIA GeForce RTX 40-series or newer GPU, and Windows 11 (24H2 / build 26100 or newer) or a Linux release from 2025.** 16 GB of VRAM is recommended.
> There is no CPU-only mode. The program and the installers exit if that GPU is not found.
> Linux is experimental. A macOS release is not a goal.

Full table: [Downloads and requirements](DOWNLOADS.md#hardware-requirements).

## Is it finished?
> **Beta / experimental software (0.1.0)**
> Voxprint is **beta**: the whole pipeline (alignment, training, narration, installer) has been run end to end on a real RTX 4090 / Windows Server 2025 machine
> (see [Tested on Windows](TESTING.md#tested-on-windows)), but the only release so far is the pre-release [`v0.1.0-beta`](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/tag/v0.1.0-beta), only one machine and one speaker were tested, voice *quality* is judged by ear of one person,
> and many settings are defaults that may change. Expect rough edges; back up your recordings and voices ([Backup](MODELS.md#backup-restore-and-existing-models)); report problems as issues.
> Voices are personal data - read [Voice owner's consent](USER-GUIDE.md#voice-owners-consent-and-usage-scope) and the licence notes before sharing anything.

## Who is behind it? Is it connected to other "Voxprint" products?
> **Independent project.** Voxprint AI Audiobook Builder is an independent, non-commercial open-source project by Aleksandr Mitroshenkov. It is **not affiliated with, endorsed by or connected to** any other product, service, company or project that has a similar name (for example web transcription services or voice-identity tools called "VoxPrint"/"Voxprint", or printing companies). All product names mentioned belong to their owners.

## Is my data sent anywhere?
No telemetry, no account; recordings, text and voices stay on your computer. The complete list of network connections is in [PRIVACY.md](PRIVACY.md).

## Can I use any voice?
Only with the voice owner's permission and within the voice's licence (CC0, CC-BY ... or personal-only); every voice carries a licence and a consent record. See [Voices and licences](VOICES.md) and the consent section of the [User guide](USER-GUIDE.md#voice-owners-consent-and-usage-scope).

## Windows warns about the installer. Is it safe?
The installer is not code-signed yet, so a new download has no reputation. Download it only from the GitHub releases page. Check the SHA-256 against [DOWNLOADS.md](DOWNLOADS.md) (or the `.sha256` file) with `Get-FileHash .\Voxprint-Setup-online.exe -Algorithm SHA256`, and only then keep or run it. Edge is the strictest: do not click **Delete** on the download bar. Hover the download, then **... -> Keep -> Show more -> Keep anyway**. The file can be removed before you run it. SmartScreen (*Windows protected your PC* -> *More info -> Run anyway*) is usually milder, and an antivirus program may also flag the file. Code signing is being applied for (SignPath Foundation); the warning will go away later. The source is in this repository and the installer is built by a public GitHub Actions workflow; you can also [build it yourself](BUILDING.md).

## Does it work on Linux or without an NVIDIA GPU?
Linux is experimental ([LINUX.md](LINUX.md)). The same GPU is required there: an NVIDIA GeForce RTX 40-series or newer. There is no CPU-only mode.

## Downloads fail with a VPN or an unusual network adapter
Voxprint tries the normal connection first (8 s connect timeout). If it cannot connect, it lists your local network interfaces (IPv4/IPv6, loopback and link-local skipped) and retries from each address in turn, also without the system proxy; the first one that works is remembered (`state/net_route.json`), used for the rest of the download - resumed ranges and the local -> original -> mirror order included - and re-tested if it stops working. This covers the online installer (`voxprint-fetch`), model, translation-model, voice and update downloads that Voxprint makes itself (`pip` runs of the updater only honour the system proxy). The log line is short, e.g. `network route for huggingface.co: tun0 (10.8.0.2)`.
Choose manually in *Settings -> Network interface* (Automatic / System default only / a specific adapter) or with the environment variable `VOXPRINT_NET_IFACE` (`auto`, `default`, an adapter name or a local IP; the installer downloader also has `--iface`). A named adapter is used alone, with no fallback. Hugging Face downloads through a specific address use the plain Python downloader (the Rust accelerator `hf_xet` cannot bind to an address). Nothing here detects particular VPN programs.

## Where do I find the manual?
PDF manuals in English, Russian and German are in [`manual/`](manual/); recording scripts in [`recording-scripts/`](recording-scripts/).

## Something went wrong. How do I send a diagnostic report?
Open **Settings** (gear button) -> **Save diagnostic report...** and choose where to save the zip (headless: `voxprint diag --out report.zip`,
or `python main.py diag`). It contains the program's log files (`logs/` in the data folder, `%LOCALAPPDATA%\Voxprint\logs` on
Windows; the main log `voxprint.log` rotates at 5 MB and keeps 6 files, about 30 MB at most), `system_info.json` (version, OS,
Python, CPU, RAM, GPU, driver, CUDA, VRAM) and `settings.json` (the settings, with paths cut to the last folder or file name and
anything that looks like a token or password left out). No recordings, books or voices are included. Attach the zip to a
[GitHub issue](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/issues) and describe what you did.

