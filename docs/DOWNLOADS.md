# Downloads and verification

The first pre-release is **[v0.1.0-beta](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/tag/v0.1.0-beta)** (marked *pre-release*). Planned installer channels are listed in the [Roadmap](ROADMAP.md).

**Right now only the small online installer is offered** (`Voxprint-Setup-online.exe`, 34,653,757 bytes). A full offline installer may return later. The Linux version is temporarily unavailable (see the end).

## Current download: SHA-256 (v0.1.0-beta, rebuilt 2026-10-04)

| File | SHA-256 |
|---|---|
| `Voxprint-Setup-online.exe` (recommended, about 33 MB; 34,653,757 bytes) | `f323ffd7c07a7a18433fc31014d991d91cf1cf27e1b19b241dadf69c529e6760` |
| `Voxprint-shell-01.zip` (our program shell, 77,461,877 bytes; fetched by the installer) | `8fb79cbe626e8fe6b37b631e84beb9b082cf1223d2f3bdaa3076358e7c66cf63` |
| `manifest-thin-beta.json` (the list of everything the installer downloads, 40,844 bytes) | `d59cd9951db698d14d12a9c4f75ddabc2a8486cb3fe6efa66e29b61aae528b00` |
| `docopt-0.6.2-...whl`, `eng_to_ipa-0.0.2-...whl`, `sox-1.5.0-...whl` (tiny pure-Python packages that have no wheel upstream; built in CI) | `aa013a4e...0139`, `a287de76...fa8b`, `44e0d93c...971` |

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
`Voxprint-Setup-online.exe` installs only our program shell and then downloads the rest from the developers' own sites, every file verified by a pinned SHA-256 (see the table above and [THIN-INSTALLER.md](THIN-INSTALLER.md)). A broken or interrupted download is resumed, components already on your PC (a compatible PyTorch, ffmpeg, the Visual C++ runtime) are reused after a check, and your data (models, voices, logs) stays in `%LOCALAPPDATA%\Voxprint`. It asks for administrator rights **once** (UAC). Silent install: `Voxprint-Setup-online.exe /VERYSILENT /DIR="C:\Voxprint"`; `/Manifest=<url or file>` uses another manifest (a mirror, a test). It needs the internet during setup (about 3 GB of libraries, then up to about 12 GB of models on the first start). Source: `tools/online_fetch.py`, `tools/make_online_payload.py`, `installer/Voxprint.iss /DONLINE`, `installer/build_online.ps1`.

**Shortcuts.** The installers create no desktop shortcut (on purpose): Voxprint is in the Start menu (folder *Voxprint AI Audiobook Builder*) and in *Settings -> Apps*, where it is uninstalled. An upgrade over an older install removes a desktop shortcut that an earlier version created.

**The installer is not code-signed.** Windows SmartScreen will show *"Windows protected your PC - Unknown publisher"*: click **More info -> Run anyway** (only after the SHA-256 above matches). The installer asks for administrator rights once (per-machine install). Antivirus programs may warn about large unsigned PyInstaller programs; the source is in this repository and you can build the installer yourself (below).

## Hardware requirements
| | Minimum | Recommended |
|---|---|---|
| OS | Windows 11 24H2 (build 26100) x64 (Linux: experimental, Ubuntu 24.04 or newer, x86-64) | Windows 11 26H2 |
| GPU | NVIDIA with ~6 GB VRAM (0.6B model, 8-bit Adam) | NVIDIA with 16 GB VRAM (1.7B model); a 4090 peaked at 6.1 GB whole-GPU usage for a 170 s recording |
| Without NVIDIA | dataset only; training on the CPU is possible but very slow | - |
| Disk | ~16 GB (models up to ~12 GB, program 3.6 GB installed) + ~4.2 GB for the optional universal model | SSD |
| Network | needed once for the model download (Hugging Face, or the ModelScope mirror) | - |
| Driver | NVIDIA driver with CUDA 11.8+ (PyTorch flavor cu118-cu130 is picked from `nvidia-smi`) | current driver |

The VRAM tiers used by the planner: >= 14 GB -> 1.7B; >= 10 GB -> 1.7B + 8-bit Adam; >= 6 GB -> 0.6B + 8-bit Adam; otherwise CPU. Speech recognition: Qwen3-ASR-1.7B from ~8 GB of VRAM, else 0.6B (Settings can override it or keep both).

## Temporarily unavailable

* **Full offline installer** (everything inside, no download during setup): may return later.
* **Linux (experimental):** the install script and program archive stay attached to the release but are being rebuilt, so their files may not match older hashes; the `SHA256SUMS-linux.txt` and `.sha256` files of the release are authoritative. Options and checklist: [LINUX.md](LINUX.md).
