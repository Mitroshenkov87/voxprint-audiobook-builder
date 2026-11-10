**VOXPrint 1.0.0 RC7 · build 1005 "Yachad"** (tag `v1.0.0-rc.7`): the shared UI shell lands, the suite library moves to voxprint-common 0.7.1, the interface and the installer wizard become English-only, and pushing a `v*` tag now attaches the Windows installer to the release automatically.

Install: download `voxprint-audiobook-builder-setup-win-x64.exe` from this release (the same file is also named `voxprint-audiobook-builder-setup-1.0.0-rc.7-win-x64.exe`). Check its SHA-256 against the `.sha256` file, then run it. Or read [install.ps1](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/install.ps1) and, in an Administrator PowerShell:

```powershell
irm https://raw.githubusercontent.com/Mitroshenkov87/voxprint-audiobook-builder/main/install.ps1 | iex
```

**Upgrade from 1004.** Run this installer over the installed build. Models, voices and settings in your VOXPrint home stay. The program lives in `C:\Program Files\VOXPrint AI Suite\Audiobook Builder`.

**Requirements.** Python 3.14. An NVIDIA GeForce RTX 40-series or newer GPU (compute capability 8.9 or higher) with driver 600 or newer, and Windows 11 24H2+ or a Linux release from 2025+. There is no CPU-only mode: the app and the installer stop with a message naming the card they found.

## Windows security warnings

The installer is not code-signed yet, so a new download has no reputation. Download it only from this GitHub releases page. Check the hash first:

```powershell
Get-FileHash .\voxprint-audiobook-builder-setup-win-x64.exe -Algorithm SHA256
```

Edge is the strictest. Its download bar offers **Delete** for a file that is rarely downloaded, and the file can be removed before you run it. Do not click Delete. Hover the download, then **... -> Keep -> Show more -> Keep anyway**.

Chrome and other browsers may show a similar download warning. When you start the installer, from any browser, Microsoft Defender SmartScreen may say **Windows protected your PC**. Choose **More info -> Run anyway**. An antivirus program may also flag the unsigned file.

Code signing is being applied for (SignPath Foundation). The warning will go away later.

New:

- **The shared UI shell.** The main window is the shared voxprint-common screen for Audiobook Builder: Choose a book, a full-width voice, and chips for pace, voices, format, split into chapters, background music, and reading manner, then Create audiobook. The same settings drive the chips and the expert panel.
- **voxprint-common 0.7.1.** The suite library pin moves to the released 0.7.1 wheel (the thin installer bundles it; the hash-checked wheel stays in the runtime lock for an installer without a git pin).
- **MDL display fields.** Every model in the catalogue now carries a short label, a one-line purpose, and a task group for the Manage models page. Qwen3-TTS 12Hz Base (the 1.7B model) stays the default narrator; no models were added or removed.
- **Windows installer attaches on a tag.** Pushing a `v*` tag now builds the thin installer and attaches it to that GitHub release; a manual release-dispatch run still works. The removed heavy full-payload build stays gone.

Changed:

- **English-only interface.** The UI and the installer wizard are English only. Russian, German, Ukrainian and Latvian remain fully supported content languages for narration, TTS, the yo/stress normaliser, and voice training; books and voices in those languages are unaffected.

Fixed:

- **Chips keep their text.** The simple screen grows each chip to fit its label even after a style or font change.
- **Audit dependencies.** `filelock` 4.2.1 or newer is required (CVE-2026-112940 is fixed there).

Full list: [CHANGELOG](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/CHANGELOG.md).
