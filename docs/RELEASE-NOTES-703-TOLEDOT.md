# Release notes: 0.2.3-beta, build 703 "Toledot"

`BUILD.json` codename is Toledot. The offset stays 654, so build-installer run 49 stamps build 703. The body below is the GitHub release text. Keep the Windows section.

**Voxprint 0.2.3-beta · build 703 "Toledot"** (Biblical Hebrew *toledot*, תּוֹלְדֹת, generations; Genesis 5:1, *zeh sefer toledot adam*, "this is the book of the generations of man"). A new generation of voices: Levi and Miriam now come with Voxprint, and multi-voice narration starts with a cast chosen for calm, low, unhurried reading.

Install: download `Voxprint-Setup-online.exe` from this release only. Check its SHA-256 against the `.sha256` file, then run it. Or read [install.ps1](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/install.ps1) and, in an Administrator PowerShell:

```powershell
irm https://raw.githubusercontent.com/Mitroshenkov87/voxprint-audiobook-builder/main/install.ps1 | iex
```

**Upgrade from 700, 701 or 702.** Run this installer over the installed build. Models, voices and settings in `%LOCALAPPDATA%\Voxprint` stay; only the two new bundled voices (about 106 MB) are downloaded. Voices you already have, including Gideon and Tirzah, stay in your library.

## Windows security warnings

The installer is not code-signed yet, so a new download has no reputation. Download it only from this GitHub releases page. Check the hash first:

```powershell
Get-FileHash .\Voxprint-Setup-online.exe -Algorithm SHA256
```

Edge is the strictest. Its download bar offers **Delete** for a file that is rarely downloaded, and the file can be removed before you run it. Do not click Delete. Hover the download, then **... -> Keep -> Show more -> Keep anyway**.

Chrome and other browsers may show a similar download warning. When you start the installer, from any browser, Microsoft Defender SmartScreen may say **Windows protected your PC**. Choose **More info -> Run anyway**. An antivirus program may also flag the unsigned file.

Code signing is being applied for (SignPath Foundation). The warning will go away later.

New:

- **Levi and Miriam are the bundled voices.** Levi (male, the default narrator; LibriVox reader Виталий, Chekhov's short stories) and Miriam (female; LibriVox reader Maya S, Ehrenburg's *Portraits of Russian Poets*) are part of the standard download, replacing Gideon and Tirzah. CC0-1.0, free for any use, also commercial. The names are Voxprint's own and do not imply the readers' endorsement.
- **Default cast for multi-voice narration.** Natan and Shimon read the men, Miriam the women, when they are installed (Natan, Shimon and Rivka are one click away: Voices -> Download voices, or `voxprint voices download natan`). A missing voice falls back to the first library voice of that gender. `voxprint narrate --speakers` without voice flags uses the same cast.
- **Shared models folder with Voxprint AI Movie Dubber.** Both programs keep their models in one folder and note themselves in `models\.users.json`. Uninstalling Audiobook Builder removes only its own entry; the models are offered for deletion only when no other Voxprint program uses them, and the default answer is to keep them.

Fixed:

- **Multi-voice marks lost a whole block.** When the text model merged a title with the next paragraph, the reply had one mark too few and the whole block fell back to the narrator (16 of 26 marks on the 702 test dialogue). The block is now split and asked again, down to single paragraphs.
- **Letter yo: "признаёт" and "твёрдо" were left as "е"** (19/21 on the 702 test dialogue, now 21/21). Added as project words, with the present tense of признавать, распознавать and осознавать.
- **`voxprint revoice --language ru` failed.** ISO codes and names in any case are accepted (`ru`, `EN`, `de-DE`, `Russian`).

Full list: [CHANGELOG](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/CHANGELOG.md).
