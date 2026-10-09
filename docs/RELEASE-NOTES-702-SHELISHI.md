# Release notes: 0.2.2-beta, build 702 "Shelishi"

`BUILD.json` codename is Shelishi. The offset stays 654, so build-installer run 48 stamps build 702. The body below is the GitHub release text. Keep the Windows section.

**Voxprint 0.2.2-beta · build 702 "Shelishi"** (Biblical Hebrew *shelishi*, שְׁלִישִׁי, third; Genesis 1:13, *yom shelishi*, the third day). A third male voice for multi-voice narration and fixes from the test of build 701 "Shalem" on a real PC.

Install: download `Voxprint-Setup-online.exe` from this release only. Check its SHA-256 against the `.sha256` file, then run it. Or read [install.ps1](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/install.ps1) and, in an Administrator PowerShell:

```powershell
irm https://raw.githubusercontent.com/Mitroshenkov87/voxprint-audiobook-builder/main/install.ps1 | iex
```

**Upgrade from 700 or 701.** Run this installer over the installed build. Models, voices and settings in `%LOCALAPPDATA%\Voxprint` stay, and nothing is downloaded again.

## Windows security warnings

The installer is not code-signed yet, so a new download has no reputation. Download it only from this GitHub releases page. Check the hash first:

```powershell
Get-FileHash .\Voxprint-Setup-online.exe -Algorithm SHA256
```

Edge is the strictest. Its download bar offers **Delete** for a file that is rarely downloaded, and the file can be removed before you run it. Do not click Delete. Hover the download, then **... -> Keep -> Show more -> Keep anyway**.

Chrome and other browsers may show a similar download warning. When you start the installer, from any browser, Microsoft Defender SmartScreen may say **Windows protected your PC**. Choose **More info -> Run anyway**. An antivirus program may also flag the unsigned file.

Code signing is being applied for (SignPath Foundation). The warning will go away later.

New:

- **Eitan, a third male voice** (optional, in the voice catalog: Voices -> Download voices, or `voxprint voices download eitan`). Trained only from the public-domain LibriVox recording of Ivan Turgenev's *Zapiski Okhotnika* (A Sportsman's Sketches) read by tovarisch. CC0-1.0, free for any use, also commercial. With Gideon as the narrator, Asher and Eitan give two men in a dialogue two different voices (`--male-voice asher --male2-voice eitan`). The name Eitan is Voxprint's own and does not imply the reader's endorsement.
- **`voxprint voices catalog` / `voxprint voices download VOICE`.** The voice catalog from the command line.
- **Better default cast in the Narrate window.** The male and female lists start with a voice that is not the narrator, and *Second male voice* starts with a third male voice when the library has one. The retired voice Boaz is no longer offered for the roles (an old copy in the library still works as a narrator).

Fixed:

- **`voxprint prepare --steps yo` ran the typo model.** With `--steps`, the Russian typo model now runs only when `--typos` is given, so a yo-only run changes nothing but the letter yo (22/22 on the test dialogue). A full preparation without `--steps` still uses the model when it is downloaded.
- **The typo model could add a second yo.** After the dictionary wrote "стерёг", the model proposed "стёрёг" and it was accepted. Its proposals are now refused when the word already has a yo, when a yo would move or disappear, when a word would get two, and when a spelling fix would add one.
- **"Setting `pad_token_id` to `eos_token_id`" in the log.** transformers wrote this once per narration chunk (14 lines on the 701 test). It is dropped.

Full list: [CHANGELOG](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/CHANGELOG.md).
