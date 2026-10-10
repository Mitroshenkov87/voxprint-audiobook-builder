# Release notes: 1.0.0-rc.2, build 1000 "Chazak"

`BUILD.json` codename is Chazak. Build-installer run 53 published 999 Nachon. Run 54 failed in the Windows tests that need no PyTorch (all tests passed, then the process crashed on exit) and published nothing, so the offset is 945 and build-installer run 55 stamps build 1000 (55 + 945 = 1000). The release tag is `v1.0.0-rc.2`, published as a pre-release (not latest). The Windows file version is `1.0.0.1000`. The body below is the GitHub release text. Keep the Windows section.

**Voxprint 1.0.0 RC2 · build 1000 "Chazak"** (Biblical Hebrew *chazak*, חֲזַק, be strong; Deuteronomy 31:6, *chazak ve'ematz*, "be strong and of good courage"): a build that holds under load. The runtime moves to Python 3.14 and PyTorch 2.11 for CUDA 13, an RTX 40-series GPU is now required, and the app reads Voxprint books (`.vxbook`) with an optional soundscape.

Install: download `Voxprint-Setup-online.exe` from this release only. Check its SHA-256 against the `.sha256` file, then run it. Or read [install.ps1](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/install.ps1) and, in an Administrator PowerShell:

```powershell
irm https://raw.githubusercontent.com/Mitroshenkov87/voxprint-audiobook-builder/main/install.ps1 | iex
```

**Upgrade from 999.** Run this installer over the installed build. The runtime changes (Python 3.14, PyTorch 2.11.0 `cu130`), so setup downloads the new runtime. Models, voices and settings in `%LOCALAPPDATA%\Voxprint` stay.

**Requirements.** An NVIDIA GeForce RTX 40-series or newer GPU (compute capability 8.9 or higher) with driver 600 or newer, and Windows 11 24H2+ or a Linux release from 2025+. There is no CPU-only mode: the app and the installer stop with a message naming the card they found.

## Windows security warnings

The installer is not code-signed yet, so a new download has no reputation. Download it only from this GitHub releases page. Check the hash first:

```powershell
Get-FileHash .\Voxprint-Setup-online.exe -Algorithm SHA256
```

Edge is the strictest. Its download bar offers **Delete** for a file that is rarely downloaded, and the file can be removed before you run it. Do not click Delete. Hover the download, then **... -> Keep -> Show more -> Keep anyway**.

Chrome and other browsers may show a similar download warning. When you start the installer, from any browser, Microsoft Defender SmartScreen may say **Windows protected your PC**. Choose **More info -> Run anyway**. An antivirus program may also flag the unsigned file.

Code signing is being applied for (SignPath Foundation). The warning will go away later.

New:

- **Voxprint books (`.vxbook`).** `narrate book.vxbook --voice ID` reads the roles and cast stored in the book; the letter yo and U+0301 stress marks are kept as the author wrote them. A `.vxbook` is not sent to Gemma for speaker marks.
- **Optional soundscape** on ACE-Step 1.5 (code and weights MIT; generated music may be used commercially). Quiet background music and transition accents are ducked under the narration, only for `.vxbook` books that carry Plotweaver's sound markup (`sound/1`), and only after you turn it on in Settings. The model (about 10 GB) is downloaded only then. It has not yet been run on a real GPU in CI.
- **Plotweaver books on the command line** (`narrate book.md --speaker-marks marks.txt`).
- **[PHILOSOPHY.md](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/PHILOSOPHY.md)**: what the Voxprint AI Media Suite stands for.

Changed:

- **Python 3.14 and PyTorch 2.11.0 / TorchAudio 2.11.0 for CUDA 13 (`cu130`) only**, the same runtime as Voxprint AI Movie Dubber. The CUDA 12.6 and 12.8 builds are gone.
- **RTX 40-series or newer required.** The CPU-only mode and the Linux `--cpu` option are removed.
- **English <-> German translation** uses the Opus-MT tc-bible-big pair that Movie Dubber already stores, so the shared models folder keeps one copy. The 2020 en<->de models are dropped.

Fixed:

- **Components window:** after a failed download, Download is enabled again reliably (it could stay off for a moment while the failure was already shown).

Full list: [CHANGELOG](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/CHANGELOG.md).
