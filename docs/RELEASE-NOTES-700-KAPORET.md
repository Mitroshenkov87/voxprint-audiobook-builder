# Release notes draft: 0.2.0-beta, build 700 "Kaporet"

Not released. `BUILD.json` codename is Kaporet. The offset is still 628; change it only for the run that must stamp 700. Paste the body below into the GitHub release when that build exists. Keep the Windows section.

**Voxprint 0.2.0-beta · build 700 "Kaporet"** (Biblical Hebrew *kaporet*, כַּפֹּרֶת, the Ark cover; gematria 700; Numbers 7:89, the Voice spoke from above the kaporet). Everything since build 668 "Kolot".

Install: download `Voxprint-Setup-online.exe` from this release only. Check its SHA-256 against the `.sha256` file, then run it. Or read [install.ps1](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/install.ps1) and, in an Administrator PowerShell:

```powershell
irm https://raw.githubusercontent.com/Mitroshenkov87/voxprint-audiobook-builder/main/install.ps1 | iex
```

`-Silent` passes `/VERYSILENT`. `-Version` picks a tag; the default is the latest pre-release.

**Upgrade from 668.** Uninstall, then install this build. On 668's question "Also delete the downloaded models and logs?", choose **No** (the default; a silent uninstall already keeps the folder). Yes deletes `%LOCALAPPDATA%\Voxprint`, including models, voices and settings. This build's uninstaller does not delete that folder. `models`, `voices` and `state` stay, and a models folder on another drive was never removed.

## Windows security warnings

The installer is not code-signed yet, so a new download has no reputation. Download it only from this GitHub releases page. Check the hash first:

```powershell
Get-FileHash .\Voxprint-Setup-online.exe -Algorithm SHA256
```

Edge is the strictest. Its download bar offers **Delete** for a file that is rarely downloaded, and the file can be removed before you run it. Do not click Delete. Hover the download, then **... -> Keep -> Show more -> Keep anyway**.

Chrome and other browsers may show a similar download warning. When you start the installer, from any browser, Microsoft Defender SmartScreen may say **Windows protected your PC**. Choose **More info -> Run anyway**. That step is usually milder than Edge. An antivirus program may also flag the unsigned file.

Code signing is being applied for (SignPath Foundation). The warning will go away later.

New:

- **Command line for speaker marks and Check & repair.** `voxprint narrate --speakers`, `--male-voice`, `--female-voice`, and `--speaker-marks`. `voxprint speakers` writes marks and does not narrate. `voxprint check` (alias `repair`) is Settings -> Check & repair.
- **Russian letter yo.** Before synthesis, yo is restored only where the dictionary is sure. `текст`, `все` and `берег` stay as written. Stress marks are not sent to the stock Qwen3-TTS base model. No extra model download.
- **Catalog voices Asher and Noa** (optional, not installed automatically). Hidden until `url`, `sha256` and `size_bytes` replace the placeholders.
- **PowerShell install.** `install.ps1`, as above.
- **Exe-only patch artifact.** A separate `Voxprint-patch` upload so the program exe can be replaced without the whole installer.

Fixed:

- **Speaker marks.** Gemma replies with a thinking channel, a preamble, or numbered lines are read as marks. A dialogue that comes back entirely as the narrator is a warning, not a silent success.
- **Build stamp.** Status, About and diagnostics read the build number baked into `Voxprint.exe`, so an exe-only swap no longer stays on the old number. The thin-shell build can import that stamp step.
- **Windowed command line.** `Voxprint.exe status --json` on an invalid stdout handle (Windows error 22) exits with the command's code and does not open a traceback window.
- **Uninstall.** This build does not delete `%LOCALAPPDATA%\Voxprint`.

Changed:

- **Bundled voices.** Boaz is no longer installed. Tirzah stays. Gideon (LibriVox reader Kazbek, *Vekhi*, CC0) is the male voice that replaces Boaz, used as recorded, with no retraining. The Gideon package is not pinned in this draft (no url, SHA-256 or size yet).
- **Windows install warnings.** Unsigned installer, Edge first, then SmartScreen. See the section above.
- **Icon and splash.** New artwork. The already-built 668 installer does not contain it.

Full list: [CHANGELOG](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/CHANGELOG.md).
