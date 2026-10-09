# Downloads and verification

The current pre-release is **[v0.1.3-beta](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/tag/v0.1.3-beta)** (build 667 "Menuchah", marked *pre-release*). Planned installer channels are listed in the [Roadmap](ROADMAP.md).

**Right now only the small online installer is offered** (`Voxprint-Setup-online.exe`, 35,235,971 bytes). A full offline installer may return later. The Linux version is temporarily unavailable (see the end).

## Current download: SHA-256 (v0.1.3-beta, build 667 "Menuchah", 2026-10-08)

| File | SHA-256 |
|---|---|
| `Voxprint-Setup-online.exe` (recommended, about 34 MB; 35,235,971 bytes; v0.1.3-beta build 667, version info 0.1.3.667, same file as `Voxprint-Setup-online-0.1.3-build667.exe`) | `c145a9fc84c736d655fbbe9bfd5c1cc94794b1be7fb8e9a5eec293e6f1578e42` |
| `Voxprint-shell-01.zip` (our program shell, 78,665,942 bytes; fetched by the installer) | `a889eba72f037a779e5cc4eac37885162f99d46123744ef8c4fc5ce20da51b97` |
| `manifest-thin-beta.json` (the list of everything the installer downloads, 40,910 bytes) | `d05132e1f9ab9d1b7459635e54db4a015fccfee79a2df7133bc7527c4c35baa0` |
| `docopt-0.6.2-...whl`, `eng_to_ipa-0.0.2-...whl`, `sox-1.5.0-...whl` (tiny pure-Python packages that have no wheel upstream; built in CI) | `e9023069...f8948`, `2c0b45d9...5c41`, `18b5ef2d...420b` |

The installer downloads the PyTorch libraries from **download.pytorch.org**, the other ~70 Python libraries from **PyPI**, the Visual C++ runtime from **Microsoft** (skipped if already present) and ffmpeg from its official build host - each file is verified against a SHA-256 pinned in the manifest. Only our own small files come from this release. Details: [THIN-INSTALLER.md](THIN-INSTALLER.md). You only need to check the `.exe` you start. If the numbers here and in the release differ, the release (its `.sha256` file and the digest shown by GitHub) is authoritative and this page is out of date - please open an issue.

**Verify a download** - compare the output with the table above or with the `.sha256` file next to the asset:

```powershell
Get-FileHash .\Voxprint-Setup-online.exe -Algorithm SHA256        # PowerShell
certutil -hashfile Voxprint-Setup-online.exe SHA256              # cmd
```
```bash
sha256sum Voxprint-Setup-online.exe                              # Linux / macOS: shasum -a 256 <file>
sha256sum -c Voxprint-Setup-online.exe.sha256                    # with the .sha256 file in the same folder
```

## Online installer
`Voxprint-Setup-online.exe` installs only our program shell and then downloads the rest from the developers' own sites, every file verified by a pinned SHA-256 (see the table above and [THIN-INSTALLER.md](THIN-INSTALLER.md)). A broken or interrupted download is resumed, components already on your PC (a compatible PyTorch, ffmpeg, the Visual C++ runtime) are reused after a check, and your data (models, voices, logs) stays in `%LOCALAPPDATA%\Voxprint`. It asks for administrator rights **once** (UAC). Silent install: `Voxprint-Setup-online.exe /VERYSILENT /DIR="C:\Voxprint"`; `/Manifest=<url or file>` uses another manifest (a mirror, a test). The wizard offers two setup types (`/Mode=full` or `/Mode=quick` when silent): **Full** (default) downloads everything on the first start without another click - the components and ALL models including the optional ones, about 28 GB from the pinned sizes, after a free-space check; **Quick** installs only the program and the first start offers the same complete download in the Components window. Both end with the same program. Source: `tools/online_fetch.py`, `tools/make_online_payload.py`, `installer/Voxprint.iss /DONLINE`, `installer/build_online.ps1`.

**Shortcuts.** The installers create no desktop shortcut (on purpose): Voxprint is in the Start menu (folder *Voxprint AI Audiobook Builder*) and in *Settings -> Apps*, where it is uninstalled. An upgrade over an older install removes a desktop shortcut that an earlier version created.

**The installer is not code-signed yet**, so a new download has no reputation. Download it only from the GitHub releases page, check the SHA-256 above, and only then keep or run the file. Edge is the strictest: its download bar offers **Delete**, and the file can be removed before you run it. Do not click Delete. Hover the download, then **... -> Keep -> Show more -> Keep anyway**. Chrome and other browsers may show a similar download warning. When you start the installer, SmartScreen may say *"Windows protected your PC"*: **More info -> Run anyway**. That step is usually milder than Edge. An antivirus program may also flag the unsigned file. Code signing is being applied for (SignPath Foundation); the warning will go away later. The installer asks for administrator rights once (per-machine install). The source is in this repository and you can build the installer yourself (below).

## Hardware requirements
| | Minimum | Recommended |
|---|---|---|
| OS | Windows 11 24H2 (build 26100) x64 (Linux: experimental, Ubuntu 24.04 or newer, x86-64) | Windows 11 26H2 |
| GPU | NVIDIA with ~6 GB VRAM (0.6B model, 8-bit Adam) | NVIDIA with 16 GB VRAM (1.7B model); a 4090 peaked at 6.1 GB whole-GPU usage for a 170 s recording |
| Without NVIDIA | dataset only; training on the CPU is possible but very slow | - |
| Disk | ~32 GB (all models ~25 GB, components ~3 GB, program) + ~4.2 GB for the optional universal model | SSD |
| Network | needed once for the model download (Hugging Face, or the ModelScope mirror) | - |
| Driver | NVIDIA driver with CUDA 11.8+ (PyTorch flavor cu118-cu130 is picked from `nvidia-smi`) | current driver |

The VRAM tiers used by the planner: >= 14 GB -> 1.7B; >= 10 GB -> 1.7B + 8-bit Adam; >= 6 GB -> 0.6B + 8-bit Adam; otherwise CPU. Speech recognition: Qwen3-ASR-1.7B from ~8 GB of VRAM, else 0.6B (Settings can override it or keep both).

## Temporarily unavailable

* **Full offline installer** (everything inside, no download during setup): may return later.
* **Linux (experimental):** the install script and program archive stay attached to the release but are being rebuilt, so their files may not match older hashes; the `SHA256SUMS-linux.txt` and `.sha256` files of the release are authoritative. Options and checklist: [LINUX.md](LINUX.md).
