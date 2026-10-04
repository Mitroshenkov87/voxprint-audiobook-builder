# Downloads and verification

The first pre-release is **[v0.1.0-beta](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/tag/v0.1.0-beta)** (marked *pre-release*). Planned installer channels are listed in the [Roadmap](ROADMAP.md).

**Right now only the small online installer is offered** (`Voxprint-Setup-online.exe`, 34,653,757 bytes). The standalone full installer and the Linux version are temporarily unavailable and being rebuilt - see [the section at the end](#temporarily-unavailable--outdated).

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

## Online installer (small download)
> The paragraph below describes the earlier *classic* online installer (`Voxprint-payload-NN.zip`, `manifest-beta.json`; these files are still in the release, unchanged, and are legacy). The current installer works as described in [THIN-INSTALLER.md](THIN-INSTALLER.md).

`Voxprint-Setup-online.exe` (a few MB, from the same release) is the **smart installer**: it downloads the program in parts (`Voxprint-payload-NN.zip`, each below the 2 GiB GitHub limit) listed in `manifest-beta.json` / `manifest-stable.json` (the channel follows the release: a pre-release such as `v0.1.0-beta` is *beta*). Every part is **verified by its SHA-256** from the manifest, a broken or interrupted download is **resumed** (HTTP `Range`; a corrupt part is discarded), and parts that are **already installed** (same SHA-256 recorded in `voxprint-components.json` in the install folder) are skipped on a re-run or repair. It asks for administrator rights **once** (UAC); your data (models, voices, logs) stays in `%LOCALAPPDATA%\Voxprint`. Silent install: `Voxprint-Setup-online.exe /VERYSILENT /DIR="C:\Voxprint"`; `/Manifest=<url or file>` uses another manifest (a mirror, a test). Unlike the full installer it needs the internet during the setup (about 2 GB of parts, then about 7 GB of models on the first start). Source: `tools/online_fetch.py` (downloader), `tools/make_online_payload.py` (splitter), `installer/Voxprint.iss /DONLINE`, `installer/build_online.ps1`.

**Shortcuts.** The installers create no desktop shortcut (on purpose): Voxprint is in the Start menu (folder *Voxprint AI Audiobook Builder*) and in *Settings -> Apps*, where it is uninstalled. An upgrade over an older install removes a desktop shortcut that an earlier version created.

**The installer is not code-signed.** Windows SmartScreen will show *"Windows protected your PC - Unknown publisher"*: click **More info -> Run anyway** (only after the SHA-256 above matches). The installer asks for administrator rights once (per-machine install). Antivirus programs may warn about large unsigned PyInstaller programs; the source is in this repository and you can build the installer yourself (below).

## Hardware requirements
| | Minimum | Recommended |
|---|---|---|
| OS | Windows 11 24H2 (build 26100) x64 (Linux: experimental, Ubuntu 24.04 or newer, x86-64) | Windows 11 26H2 |
| GPU | NVIDIA with ~6 GB VRAM (0.6B model, 8-bit Adam) | NVIDIA with 16 GB VRAM (1.7B model); a 4090 peaked at 6.1 GB whole-GPU usage for a 170 s recording |
| Without NVIDIA | dataset only; training on the CPU is possible but very slow | - |
| Disk | ~12 GB (models ~7 GB, program 3.6 GB installed) + ~4.2 GB for the optional universal model | SSD |
| Network | needed once for the model download (Hugging Face, or the ModelScope mirror) | - |
| Driver | NVIDIA driver with CUDA 11.8+ (PyTorch flavor cu118-cu130 is picked from `nvidia-smi`) | current driver |

The VRAM tiers used by the planner: >= 14 GB -> 1.7B; >= 10 GB -> 1.7B + 8-bit Adam; >= 6 GB -> 0.6B + 8-bit Adam; otherwise CPU.

## Temporarily unavailable / outdated

> **Not recommended right now.** The files below are still attached to the release (nothing was deleted), but they are **stale** (built before the latest changes) or being rebuilt. The hashes are kept so that the files can still be verified. Use the online installer above.

### Hashes of the old assets

| File | SHA-256 |
|---|---|
| `Voxprint-Setup-github-build.exe` (full, about 1.96 GiB) | `582cd36a4e26861841014e864c7b1c00a3a759dc978d4cb2c528087e36a6ca1c` |
| `Voxprint-Setup-grokbot-build.exe` (full, earlier build made before the translation feature; prefer the others) | `c2f8f01dc02568a55b60a1c5ed1f456c2f0c03fa4468f27fdc54a93cb53ebc24` |
| `install-voxprint-linux.sh` (Linux, experimental) | `a408b15ac59d351a2ce6dfef665dd21cdf22725162bcf349227f913e8c15dec7` |
| `Voxprint-linux-experimental.tar.gz` (Linux, script + program tree) | `e7858f8c0942ba48f91b52b950658ce4306b6e2c3aa0c9302de3197527d55c90` |
| `Voxprint-linux-app.zip` (Linux program, fetched by the script) | `3a486ec3c23ec40ed61806e8193281faa4662349bb1d128347e15996d2fb76d0` |
| `manifest-linux.json` | `db5b162d44a5e36752f28b0f12f97363e7b8fa2cc97a072cd62507085bddd3b4` |
| `voxprint-fetch.py` (downloader used by the Linux script) | `a914947c8861fe8425b3dd6111cb1eabee76a4a79b1a708698b9356d9050ff0b` |

The Linux hashes are also in `SHA256SUMS-linux.txt` of the release (`sha256sum -c SHA256SUMS-linux.txt`). These files are **stale**: the files attached to the release may no longer match the hashes listed here until they are rebuilt (the `.sha256` file next to each asset is authoritative).

### Standalone full installers (Windows 11 x64, NVIDIA GPU)
Full installers with everything inside (kept for reference; currently stale, being rebuilt):

| File | Built by | Size |
|---|---|---|
| `Voxprint-Setup-github-build.exe` (+ `.sha256`) | the public **GitHub Actions** workflow `.github/workflows/build-installer.yml` from the tagged source - the build log is public, so you can see how it was made | about 1.96 GiB |
| `Voxprint-Setup-grokbot-build.exe` (+ `.sha256`) | built before the repository was renamed (its old URLs redirect), by the maintainers' AI assistant on a Windows GPU server with the same `build.bat` and `installer\Voxprint.iss` (maximum LZMA2 compression); it is the build that was installed and tested end to end | about 1.86 GiB |
| `Voxprint-Setup-online.exe` (+ `.sha256`, `manifest-beta.json`, `Voxprint-payload-NN.zip`) | the small online installer (about 33 MB; see [Online installer](#online-installer-small-download) above); downloads the two payload parts (about 3.1 GB) and verifies them with the manifest | about 33 MB + 3.1 GB |

Both are full installers (PyTorch with CUDA is inside, about 3.8 GB installed); the models (about 7 GB) are downloaded once on the first start.

**Verify the download** (SHA-256). Each file has a `.sha256` file next to it with the expected hash. In PowerShell:
```powershell
Get-FileHash .\Voxprint-Setup-github-build.exe -Algorithm SHA256     # compare with the text in Voxprint-Setup-github-build.exe.sha256
# or:  certutil -hashfile Voxprint-Setup-github-build.exe SHA256
```

### Linux (experimental)

Linux is **experimental**: install script, options and the test checklist are in [LINUX.md](LINUX.md) (one-line install command and hashes: see the Linux notes there).
