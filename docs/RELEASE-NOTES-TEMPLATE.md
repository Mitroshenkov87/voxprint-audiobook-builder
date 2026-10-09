# Release notes template

Copy this into the GitHub release. Replace the version line and the New / Fixed lists. Keep the Windows section as written. The same wording is in the README under Download.

**Voxprint VERSION · build N "CODENAME"** - one sentence on what changed.

Install: download `Voxprint-Setup-online.exe` from this release only. Check its SHA-256 against the `.sha256` file, then run it.

## Windows security warnings

The installer is not code-signed yet, so a new download has no reputation. Download it only from this GitHub releases page. Check the hash first:

```powershell
Get-FileHash .\Voxprint-Setup-online.exe -Algorithm SHA256
```

Edge is the strictest. Its download bar offers **Delete** for a file that is rarely downloaded, and the file can be removed before you run it. Do not click Delete. Hover the download, then **... -> Keep -> Show more -> Keep anyway**.

Chrome and other browsers may show a similar download warning. When you start the installer, from any browser, Microsoft Defender SmartScreen may say **Windows protected your PC**. Choose **More info -> Run anyway**. That step is usually milder than Edge. An antivirus program may also flag the unsigned file.

Code signing is being applied for (SignPath Foundation). The warning will go away later.

New:

- 

Fixed:

- 

Full list: [CHANGELOG](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/blob/main/CHANGELOG.md).
