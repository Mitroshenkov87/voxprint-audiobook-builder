# Release notes: 0.2.1-beta, build 701 "Shalem"

`BUILD.json` codename is Shalem. Run 46 failed in the Windows unit tests before it published anything, so the offset is 654 and build-installer run 47 stamps build 701. The body below is the GitHub release text. Keep the Windows section.

**Voxprint 0.2.1-beta · build 701 "Shalem"** (Biblical Hebrew *shalem*, שָׁלֵם, whole, complete; Genesis 33:18, Jacob came *shalem* to the city of Shechem). Fixes from the test of build 700 "Kaporet" on a real PC.

Install: download `Voxprint-Setup-online.exe` from this release only. Check its SHA-256 against the `.sha256` file, then run it. Or read [install.ps1](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/install.ps1) and, in an Administrator PowerShell:

```powershell
irm https://raw.githubusercontent.com/Mitroshenkov87/voxprint-audiobook-builder/main/install.ps1 | iex
```

**Upgrade from 700.** Uninstall 700, then install this build. The 700 uninstaller deletes only the program folder; `%LOCALAPPDATA%\Voxprint` (models, voices, settings) stays, and nothing is downloaded again.

## Windows security warnings

The installer is not code-signed yet, so a new download has no reputation. Download it only from this GitHub releases page. Check the hash first:

```powershell
Get-FileHash .\Voxprint-Setup-online.exe -Algorithm SHA256
```

Edge is the strictest. Its download bar offers **Delete** for a file that is rarely downloaded, and the file can be removed before you run it. Do not click Delete. Hover the download, then **... -> Keep -> Show more -> Keep anyway**.

Chrome and other browsers may show a similar download warning. When you start the installer, from any browser, Microsoft Defender SmartScreen may say **Windows protected your PC**. Choose **More info -> Run anyway**. An antivirus program may also flag the unsigned file.

Code signing is being applied for (SignPath Foundation). The warning will go away later.

New:

- **"все" / "всё" by context.** The Russian letter yo step now decides the most frequent ambiguous word from its neighbours: "всё" before a singular verb or a neuter adjective, at the end of a clause, before ", что", in "всё равно", "всё-таки", "всё ещё" and before a comparative; "все" before a plural word or pronoun and after a plural subject or verb. On the 700 test dialogue the step now restores 22/22 yo with none wrong (700: 16/22, every miss was "всё"). The rules are data (`core/data/yo_context.json`).
- **A second voice per role in multi-voice narration.** `--male2-voice` (and `--female2-voice`): different characters alternate between the two voices in order of first appearance, so two men in a dialogue no longer share one voice. `--character NAME=VOICE` pins one character. The Narrate window has a *Second male voice* list (None by default). Without these, narration works as before.
- **Command line for every step.** `voxprint prepare` (Prepare text without narration: rules, letter yo, Russian typo model, optional text-model rewrite; plain text and a JSON report), `voxprint translate` (offline Opus-MT, `--literary` with Gemma) and `voxprint settings list|get|set`. See [docs/CLI.md](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/docs/CLI.md).

Fixed:

- **"'sox' is not recognized" on the console.** The SoX wrapper that the speech package imports no longer starts a shell to look for SoX when SoX is not installed. Voxprint never uses SoX.
- **"triton not found" in the log.** PyTorch's and bitsandbytes' notices about the optional Triton compiler are dropped.

Full list: [CHANGELOG](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/CHANGELOG.md).
