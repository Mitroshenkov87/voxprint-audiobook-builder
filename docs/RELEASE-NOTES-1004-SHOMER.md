**VOXPrint 1.0.0 RC6 · build 1004 "Shomer"** (tag `v1.0.0-rc.6`): this build keeps shared model, runtime and voice files in place when one app is removed, and it downloads models through VOXPrint MDL.

Install: download `voxprint-audiobook-builder-setup-win-x64.exe` from this release (the same file is also named `voxprint-audiobook-builder-setup-1.0.0-rc.6-win-x64.exe`). Check its SHA-256 against the `.sha256` file, then run it. Or read [install.ps1](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/install.ps1) and, in an Administrator PowerShell:

```powershell
irm https://raw.githubusercontent.com/Mitroshenkov87/voxprint-audiobook-builder/main/install.ps1 | iex
```

**Upgrade from 1003.** Run this installer over the installed build. Models, voices and settings in `%LOCALAPPDATA%\Voxprint` stay. An older `C:\Program Files\Voxprint` folder is removed. A folder spelled `Voxprint AI Suite` is renamed to `VOXPrint AI Suite`. The program then lives in `C:\Program Files\VOXPrint AI Suite\Audiobook Builder`.

**Requirements.** An NVIDIA GeForce RTX 40-series or newer GPU (compute capability 8.9 or higher) with driver 600 or newer, and Windows 11 24H2+ or a Linux release from 2025+. There is no CPU-only mode: the app and the installer stop with a message naming the card they found.

## Windows security warnings

The installer is not code-signed yet, so a new download has no reputation. Download it only from this GitHub releases page. Check the hash first:

```powershell
Get-FileHash .\voxprint-audiobook-builder-setup-win-x64.exe -Algorithm SHA256
```

Edge is the strictest. Its download bar offers **Delete** for a file that is rarely downloaded, and the file can be removed before you run it. Do not click Delete. Hover the download, then **... -> Keep -> Show more -> Keep anyway**.

Chrome and other browsers may show a similar download warning. When you start the installer, from any browser, Microsoft Defender SmartScreen may say **Windows protected your PC**. Choose **More info -> Run anyway**. An antivirus program may also flag the unsigned file.

Code signing is being applied for (SignPath Foundation). The warning will go away later.

New:

- **Install folder and program names.** Windows setup installs into `C:\Program Files\VOXPrint AI Suite\Audiobook Builder`. The window is `voxprint-audiobook-builder.exe`. Scripts use `voxprint-audiobook-cli.exe`. The setup file is `voxprint-audiobook-builder-setup-<version>-win-x64.exe`, and the same bytes are also published as `voxprint-audiobook-builder-setup-win-x64.exe`. An older install is moved or removed on upgrade. Linux uses `/opt/voxprint-ai-suite/audiobook-builder` and the launcher `voxprint-audiobook-cli`.
- **VOXPrint MDL downloads the models.** One shared downloader fetches every model, resumes a partial file, and switches to a faster mirror. Setup runs that download and stops with a clear message when the models cannot be fetched, including a silent install. In the app, **Manage models** opens the model window.
- **UTF-8 JSON.** Commands that print `--json` write UTF-8, so names in other languages stay readable.
- **Shared voices.** Voices live in the shared folder under your VOXPrint home. Voices you already had are moved there.

Fixed:

- **Shared folders are not deleted with one app.** Removing this app, cancelling a failed setup, or unregistering the shared runtime drops only this app's claim. The shared models folder, the shared runtime folder and the shared voices folder stay, even when no app still lists them. A folder that belongs only to this app can still be removed.

Full list: [CHANGELOG](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/CHANGELOG.md).
