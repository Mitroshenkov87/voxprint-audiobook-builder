# Linux (experimental)

> **Temporarily unavailable:** the Linux package is being rebuilt and is not advertised right now (the release files are kept but may be stale; see [DOWNLOADS.md](DOWNLOADS.md#temporarily-unavailable)). The notes below describe the intended installation.

**Status: experimental.** The Linux package is built and checked by the public workflow `.github/workflows/build-linux.yml` on a clean Ubuntu 24.04 runner (install, the whole unit-test suite, text/translation/ffmpeg pipeline step, headless start of the window). What no CI can check - a real desktop session, sound output, NVIDIA/CUDA - is still **untested**; please test with [`docs/LINUX-TEST-CHECKLIST.md`](LINUX-TEST-CHECKLIST.md) and report problems. Hashes of the Linux assets of `v0.1.0-beta`: `install-voxprint-linux.sh` `a408b15ac59d351a2ce6dfef665dd21cdf22725162bcf349227f913e8c15dec7`, `Voxprint-linux-experimental.tar.gz` `3f062016ae4e300453baef09c9609a8d3c2bd67fe2b3a29f7b50ceee838fb614`, `Voxprint-linux-app.zip` `d21e0713fc67f5cec015daa527e4357fce3943eb69be44e25145991814c42701` (all in `SHA256SUMS-linux.txt`). Supported systems: current and previous year OS releases (Linux distributions released from 2025). Older systems are not a goal: the installer prints a warning and continues. x86-64, Python 3.10-3.13. Other distributions: install the equivalent packages yourself (`--no-system-check`).

There is no frozen binary: the installer script creates a Python environment (venv) for your user only - nothing is installed system-wide except the optional `apt` libraries.
```bash
# download the installer script and check its SHA-256 (see the release notes / SHA256SUMS-linux.txt)
curl -fsSLO https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/v0.1.0-beta/install-voxprint-linux.sh
sha256sum install-voxprint-linux.sh      # v0.1.0-beta: a408b15ac59d351a2ce6dfef665dd21cdf22725162bcf349227f913e8c15dec7
bash install-voxprint-linux.sh --check            # lists missing system packages (Qt xcb libraries, ffmpeg, python3-venv ...)
bash install-voxprint-linux.sh --install-deps     # installs them with sudo apt-get, then Voxprint (about 3-8 GB with PyTorch)
voxprint                                          # or start "Voxprint AI Audiobook Builder" from the application menu
```
* **Where things are:** program `~/.local/share/voxprint/app`, environment `~/.local/share/voxprint/venv`, models / voices / settings / logs `~/.local/share/voxprint/` (`$XDG_DATA_HOME`, or `VOXPRINT_HOME`), launcher `~/.local/bin/voxprint`, menu entry `~/.local/share/applications/voxprint.desktop`. Projects (audiobooks, trainings, Re-voice) go to `~/.local/share/voxprint/Projects` (changeable in Settings) with a *Voxprint Projects* link in `~/Documents`.
* **PyTorch:** the CUDA build matching your NVIDIA driver is installed (`uv pip install torch --torch-backend=auto`). Voxprint requires an NVIDIA GeForce RTX 40-series or newer GPU (compute capability 8.9 or higher). The installer stops, with a clear message and a non-zero exit, when that GPU is missing. There is no CPU-only install. `ffmpeg` comes from apt; the bundled `imageio-ffmpeg` is the fallback.
* **Update:** run the script again (the environment is reused). **Remove:** `bash install-voxprint-linux.sh --uninstall` (models and voices stay; `--purge` deletes them too).
* **One-file alternative:** `Voxprint-linux-experimental.tar.gz` contains the same script and the program tree (`tar xzf ...; ./voxprint-linux/install-voxprint-linux.sh`); Python packages are still downloaded from PyPI. The script downloads `Voxprint-linux-app.zip` through the manifest `manifest-linux.json` with the same SHA-256-verified, resumable downloader as the Windows online installer (`voxprint-fetch.py`).
* **Wayland / Qt problems:** `QT_QPA_PLATFORM=xcb voxprint` forces X11 (XWayland). A missing `xcb` plugin message means a missing `libxcb-cursor0`.
* **Not available on Linux:** the Windows-only window effects (Mica); an AppImage / Flatpak / Snap is not provided (PyTorch + CUDA make them several GB; the venv installer is simpler and easier to repair).
* Self-tests: `voxprint --selftest-imports`, `--selftest-text` (no GPU or model needed), `--selftest`; logs in `~/.local/share/voxprint/logs`.
