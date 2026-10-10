# Release notes: 1.0.0-rc.3, build 1001 "Tamim"

`BUILD.json` codename is Tamim. The offset stays 945 and build-installer run 56 stamps build 1001 (56 + 945 = 1001). The release tag is `v1.0.0-rc.3`, published as a pre-release (not latest). The Windows file version is `1.0.0.1001`. The body below is the GitHub release text. Keep the Windows section.

**Voxprint 1.0.0 RC3 · build 1001 "Tamim"** (whole and complete): a build ready to be tested end to end. The Save button and the related actions use new icons, other programs can check a command without a GPU, and the documentation adds a public roadmap and a generated site.

Install: download `Voxprint-Setup-online.exe` from this release only. Check its SHA-256 against the `.sha256` file, then run it. Or read [install.ps1](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/install.ps1) and, in an Administrator PowerShell:

```powershell
irm https://raw.githubusercontent.com/Mitroshenkov87/voxprint-audiobook-builder/main/install.ps1 | iex
```

**Upgrade from 1000.** Run this installer over the installed build. Models, voices and settings in `%LOCALAPPDATA%\Voxprint` stay.

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

- **Suite icons.** Save, Save as, Save a copy, Export and Download are stroke icons. They replace the floppy-disk save picture.
- **Dry-run for other programs.** `voxprint --dry-run --json <command>` checks the arguments, the files and the output folder, prints JSON and exits with the documented codes. It does not load a model, use a GPU or download anything, so it runs on a machine that has no NVIDIA GPU.
- **Codename check.** A release codename cannot repeat an earlier build in this repository (`python tools/codenames.py`).
- **Roadmap and documentation site.** [ROADMAP.md](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/ROADMAP.md) is public. The [documentation site](https://mitroshenkov87.github.io/voxprint-audiobook-builder/) includes a command reference generated from the program, an architecture map, and pages built from the code's docstrings.

Fixed:

- **Installer job after a green test run.** When the Windows tests pass and the test process then crashes while shutting down, the installer job uses the status recorded before shutdown and continues. A crash during the tests still stops the job.

Full list: [CHANGELOG](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/CHANGELOG.md).
