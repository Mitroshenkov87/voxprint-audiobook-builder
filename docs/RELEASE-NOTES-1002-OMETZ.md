**Voxprint 1.0.0 RC4 · build 1002 "Ometz"** (courage and resolve; tag `v1.0.0-rc.4`): the fixes from a real GPU test. Setup installs the shared programs into one folder, a script can follow narration as JSON only, and a book file whose chapter id starts with a digit is accepted.

Install: download `Voxprint-Setup-online.exe` from this release only. Check its SHA-256 against the `.sha256` file, then run it. Or read [install.ps1](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/install.ps1) and, in an Administrator PowerShell:

```powershell
irm https://raw.githubusercontent.com/Mitroshenkov87/voxprint-audiobook-builder/main/install.ps1 | iex
```

**Upgrade from 1001.** Run this installer over the installed build. Models, voices and settings in `%LOCALAPPDATA%\Voxprint` stay.

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

- **Shared suite install.** Setup, including a silent install, installs the Python runtime, PyTorch, ffmpeg and the required models into the shared Voxprint folder. A compatible PyTorch already installed by another Voxprint app is reused.
- **Console program.** `voxprint-cli.exe` is the program scripts should call. `narrate --json` prints only JSON on stdout and ends with one result line.
- **Components window.** Free disk space is checked before a download. Essential models are downloaded first; the optional models are not all started at once.
- **Build number.** This build is 1002, taken from the project file. A failed build job does not skip a number.
- **GPU check scripts.** `scripts/vm_test.sh` and `scripts/vm_test.ps1` run the smoke and full checks on a machine with a GPU.
- **README screenshots.** The README shows the Narrate window and the first-run Components window.
- **Library updates.** huggingface-hub 1.32.0 and transformers 5.16.1 replace the previous pins, which had high security advisories. This set has not been run on a GPU.

Fixed:

- **GPU-busy message.** While this program waits, the message names the other program and the step it is on now.
- **Book chapter ids.** A `.vxbook` chapter or speaker id may start with a digit and may include a hyphen.
- **Windows installer job.** When a Windows test hangs, the job stops instead of waiting until it is cancelled.

Full list: [CHANGELOG](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/CHANGELOG.md).
