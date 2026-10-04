# Downloads and verification

The first pre-release is **[v0.1.0-beta](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/tag/v0.1.0-beta)** (marked *pre-release*). Planned installer channels are listed in the [Roadmap](ROADMAP.md).

## SHA-256 of the current release assets (v0.1.0-beta, rebuilt 2026-10-04)

| File | SHA-256 |
|---|---|
| `Voxprint-Setup-online.exe` (recommended, about 33 MB) | `701e3584f97ade9943e96be435b8bee8ef0dc58beac13f445ea3b25294725ecd` |
| `Voxprint-payload-01.zip` | `a94c596674fba4bc2793ba61a9e6c02b734ea8169edc87cc006a41e7f08b9c3f` |
| `Voxprint-payload-02.zip` | `a321bc045d1b80858e5ba98818429920a657c11301c9a76f5e5880c3be6e641d` |
| `manifest-beta.json` | `a34f8a87abfdf7cb16bde8e975a7278e3ec43115e8854583b7b5917ad4dd5e51` |
| `Voxprint-Setup-github-build.exe` (full, about 1.96 GiB) | `92c86d6229a877a204db0f7db25a0909a48736a6800ac9570b7b26b59fc02bfa` |
| `Voxprint-Setup-grokbot-build.exe` (full, earlier build made before the translation feature; prefer the others) | `c2f8f01dc02568a55b60a1c5ed1f456c2f0c03fa4468f27fdc54a93cb53ebc24` |
| `install-voxprint-linux.sh` (Linux, experimental) | `a408b15ac59d351a2ce6dfef665dd21cdf22725162bcf349227f913e8c15dec7` |
| `Voxprint-linux-experimental.tar.gz` (Linux, script + program tree) | `3f062016ae4e300453baef09c9609a8d3c2bd67fe2b3a29f7b50ceee838fb614` |
| `Voxprint-linux-app.zip` (Linux program, fetched by the script) | `d21e0713fc67f5cec015daa527e4357fce3943eb69be44e25145991814c42701` |
| `manifest-linux.json` | `e4063abf1ee1840b857189488126841fff0eb58bdf2fa30ba290ab0e4f4f077a` |
| `voxprint-fetch.py` (downloader used by the Linux script) | `ba6fc0a8db2b3915e261e61f897e73446647d2c823a58efb87a95c30e8f3b272` |

The Linux hashes are also in `SHA256SUMS-linux.txt` of the release (`sha256sum -c SHA256SUMS-linux.txt`). The online installer verifies the payload parts itself with the SHA-256 values in `manifest-beta.json`; you only need to check the `.exe` you start.
If the numbers here and in the release differ, the release (its `.sha256` files and the digest shown by GitHub) is authoritative and this page is out of date - please open an issue.

**Verify a download** - compare the output with the table above or with the `.sha256` file next to the asset:

```powershell
Get-FileHash .\Voxprint-Setup-online.exe -Algorithm SHA256        # PowerShell
certutil -hashfile Voxprint-Setup-online.exe SHA256              # cmd
```
```bash
sha256sum Voxprint-Setup-online.exe                              # Linux / macOS: shasum -a 256 <file>
sha256sum -c Voxprint-Setup-online.exe.sha256                    # with the .sha256 file in the same folder
```

## Download the installer (Windows 11 x64, NVIDIA GPU)
The release has two builds of the **same program**; take either one:

| File | Built by | Size |
|---|---|---|
| `Voxprint-Setup-github-build.exe` (+ `.sha256`) | the public **GitHub Actions** workflow `.github/workflows/build-installer.yml` from the tagged source - the build log is public, so you can see how it was made | about 1.96 GiB |
| `Voxprint-Setup-grokbot-build.exe` (+ `.sha256`) | built before the repository was renamed (its old URLs redirect), by the maintainers' AI assistant on a Windows GPU server with the same `build.bat` and `installer\Voxprint.iss` (maximum LZMA2 compression); it is the build that was installed and tested end to end | about 1.86 GiB |
| `Voxprint-Setup-online.exe` (+ `.sha256`, `manifest-beta.json`, `Voxprint-payload-NN.zip`) | the small online installer (about 33 MB; see "Online installer" below); downloads the two payload parts (about 3.1 GB) and verifies them with the manifest | about 33 MB + 3.1 GB |

Both are full installers (PyTorch with CUDA is inside, about 3.8 GB installed); the models (about 7 GB) are downloaded once on the first start. If in doubt, use the **github-build**.

**Verify the download** (SHA-256). Each file has a `.sha256` file next to it with the expected hash. In PowerShell:
```powershell
Get-FileHash .\Voxprint-Setup-github-build.exe -Algorithm SHA256     # compare with the text in Voxprint-Setup-github-build.exe.sha256
# or:  certutil -hashfile Voxprint-Setup-github-build.exe SHA256
```
## Online installer (small download)
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

## Linux

Linux is **experimental**: install script, options and the test checklist are in [LINUX.md](LINUX.md) (one-line install: see the [README](../README.md#download)).
