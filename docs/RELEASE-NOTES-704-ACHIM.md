# Release notes: 0.2.4-beta, build 704 "Achim"

`BUILD.json` codename is Achim. Run 50 failed in CI before it published anything, so the offset is 653 and build-installer run 51 stamps build 704. The body below is the GitHub release text. Keep the Windows section.

**Voxprint 0.2.4-beta · build 704 "Achim"** (Biblical Hebrew *achim*, אַחִים, brothers; Genesis 13:8, *ki anashim achim anachnu*, "for we are brothers"). Voxprint AI Audiobook Builder and Voxprint AI Movie Dubber are now sibling programs: one models folder, one settings file, one Start menu folder, and neither one's uninstaller touches the other.

Install: download `Voxprint-Setup-online.exe` from this release only. Check its SHA-256 against the `.sha256` file, then run it. Or read [install.ps1](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/install.ps1) and, in an Administrator PowerShell:

```powershell
irm https://raw.githubusercontent.com/Mitroshenkov87/voxprint-audiobook-builder/main/install.ps1 | iex
```

**Upgrade from 703 (or 700-702).** Run this installer over the installed build. Models, voices and settings in `%LOCALAPPDATA%\Voxprint` stay; nothing large is downloaded.

## Windows security warnings

The installer is not code-signed yet, so a new download has no reputation. Download it only from this GitHub releases page. Check the hash first:

```powershell
Get-FileHash .\Voxprint-Setup-online.exe -Algorithm SHA256
```

Edge is the strictest. Its download bar offers **Delete** for a file that is rarely downloaded, and the file can be removed before you run it. Do not click Delete. Hover the download, then **... -> Keep -> Show more -> Keep anyway**.

Chrome and other browsers may show a similar download warning. When you start the installer, from any browser, Microsoft Defender SmartScreen may say **Windows protected your PC**. Choose **More info -> Run anyway**. An antivirus program may also flag the unsigned file.

Code signing is being applied for (SignPath Foundation). The warning will go away later.

New:

- **Shared settings with Voxprint AI Movie Dubber.** `%LOCALAPPDATA%\Voxprint\state\suite.json` holds what both programs share: interface language, theme, models folder and GPU. Your earlier choices are copied into it at the first start. When the Movie Dubber is installed, setup says so and proposes the same models folder, so every model is downloaded only once.
- **One Start menu folder, *Voxprint*,** for every Voxprint program. The old *Voxprint AI Audiobook Builder* folder is moved there.
- **Video memory rule.** Narration plans its batch from the memory that is free at that moment and never plans above 75 % of the card (setting `gpu.vram_fraction`, 70-80 %), so another program on the same GPU keeps room. On a 16 GB card the batch is 7 instead of 10.
- **`voxprint bench`** measures the realtime factor, batch size and peak video memory of the batched path and of the optional CUDA Graphs decode (`--install-graphs` fetches faster-qwen3-tts 0.3.2, MIT; not part of the installer). The fast decode stays off unless you turn it on (`voxprint settings gpu.fast_decode graphs`); anything it cannot handle falls back to the batched path.

Changed:

- **Uninstall never touches a sibling program.** Each program removes only its own files and Start menu entry. Models are offered for deletion only when no other Voxprint program uses them (default: keep); voices, settings and `suite.json` are never deleted.

Full list: [CHANGELOG](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/CHANGELOG.md).
