**Voxprint 1.0.0 RC5 · build 1003 "Amud"** (pillar; tag `v1.0.0-rc.5`): the app now stands on the shared suite library. Paths, settings, the GPU lock, pauses when the card runs hot, the graphics-card check, downloads, build info, file checks, the theme and the start-up window come from that one library.

Install: download `Voxprint-Setup-online.exe` from this release only. Check its SHA-256 against the `.sha256` file, then run it. Or read [install.ps1](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/install.ps1) and, in an Administrator PowerShell:

```powershell
irm https://raw.githubusercontent.com/Mitroshenkov87/voxprint-audiobook-builder/main/install.ps1 | iex
```

**Upgrade from 1002.** Run this installer over the installed build. Models, voices and settings in `%LOCALAPPDATA%\Voxprint` stay. The Python runtime this app already installed is moved into the shared runtime. A compatible PyTorch already on the computer is reused.

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

- **Shared suite library.** Version 0.1.1 of the shared library now supplies paths, settings, the GPU lock, thermal pauses, the graphics-card check, downloads that try a mirror when the first host fails, build info, file hashes, the theme and the splash.
- **One shared runtime.** PyTorch lives in one place, with a manifest that counts the programs using it. Upgrading moves the runtime this app already had. A compatible PyTorch is reused instead of downloaded again. Removing this app drops its claim and deletes that runtime only when no other Voxprint app still lists it.
- **Suite models folder.** Required models are stored in the suite models folder. Copies already in the older shared models folder are still used when the suite folder does not have them yet.
- **Suite splash screen.** The start-up window is the shared splash (960×540). Six clicks within two seconds still show the extra picture.

Fixed:

- **Release checks.** The checks that follow a silent install now look at the shared runtime and the suite models folder. They cover reuse of a compatible PyTorch, a bad manifest, and uninstall dropping this app's claim. These checks run on every pull request.

Full list: [CHANGELOG](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/CHANGELOG.md).
