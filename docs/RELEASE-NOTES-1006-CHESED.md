**VOXPrint 1.0.0 RC8 · build 1006 "Chesed"** (tag `v1.0.0-rc.8`): the suite library pin moves to voxprint-common 0.7.2 — the Manage models window gains This app / Shared model tabs and a `16 GB` badge, and the shared RU strings are gone.

Install: download `voxprint-audiobook-builder-setup-win-x64.exe` from this release (the same file is also named `voxprint-audiobook-builder-setup-1.0.0-rc.8-win-x64.exe`). Check its SHA-256 against the `.sha256` file, then run it. Or read [install.ps1](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/install.ps1) and, in an Administrator PowerShell:

```powershell
irm https://raw.githubusercontent.com/Mitroshenkov87/voxprint-audiobook-builder/main/install.ps1 | iex
```

**Upgrade from 1005 / rc.7.** Run this installer over the installed build. Models, voices and settings in your VOXPrint home stay. The program lives in `C:\Program Files\VOXPrint AI Suite\Audiobook Builder`.

**Requirements.** Python 3.14. An NVIDIA GeForce RTX 40-series or newer GPU (compute capability 8.9 or higher) with driver 600 or newer, and Windows 11 24H2+ or a Linux release from 2025+. There is no CPU-only mode: the app and the installer stop with a message naming the card they found.

## Windows security warnings

The installer is not code-signed yet, so a new download has no reputation. Download it only from this GitHub releases page. Check the hash first:

```powershell
Get-FileHash .\voxprint-audiobook-builder-setup-win-x64.exe -Algorithm SHA256
```

Edge is the strictest. Its download bar offers **Delete** for a file that is rarely downloaded, and the file can be removed before you run it. Do not click Delete. Hover the download, then **... -> Keep -> Show more -> Keep anyway**.

Chrome and other browsers may show a similar download warning. When you start the installer, from any browser, Microsoft Defender SmartScreen may say **Windows protected your PC**. Choose **More info -> Run anyway**. An antivirus program may also flag the unsigned file.

Code signing is being applied for (SignPath Foundation). The warning will go away later.

Changed:

- **voxprint-common 0.7.2.** The suite library pin moves to the released 0.7.2 wheel (the thin installer bundles it; the hash-checked wheel stays in the runtime lock for an installer without a git pin). The pin is `voxprint-common[ui] @ git+https://github.com/Mitroshenkov87/voxprint-common@v0.7.2`; the matching wheel `voxprint_common-0.7.2-py3-none-any.whl` (SHA-256 `91783f7409bfc357baf4f54ccad35c2897b8fbb323f886672d92b626b16e8d2a`) is recorded in the lock's `suite` object.
- **Manage models: This app / Shared tabs.** Models already shared with another VOXPrint app move to their own tab, so this app's rows and the shared rows no longer mix.
- **`16 GB` badge.** The Manage models window labels the 16 GB VRAM tier with its own badge.
- **Shared RU strings removed.** The suite library no longer ships a Russian UI catalog; the interface stays English-only. Russian remains a fully supported content language for narration, TTS, the yo/stress normaliser, and voice training. No new `mdl.json` fields are needed.

Full list: [CHANGELOG](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/CHANGELOG.md).
