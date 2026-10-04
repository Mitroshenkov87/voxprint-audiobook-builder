# Thin installer: design and status

Goal: the online installer puts only a small program shell on the disk, starts it, and the shell downloads the heavy parts as **modules**
with progress in the UI. Status: groundwork is in the repository (opt-in, the current full and online installers are unchanged);
what is untested is listed at the end.

## Where the 5 GB go (release v0.1.0-beta, measured)

| Part | Unpacked | Note |
|---|---|---|
| `_internal/torch/lib` (CUDA/cuDNN/cuBLAS DLLs) | ~4.0 GB | about 80 % of everything |
| the rest of the PyInstaller folder | ~0.95 GB | bitsandbytes 117 MB, llvmlite 114, imageio-ffmpeg 83, `Voxprint.exe` 74, av 62, scipy 52, nagisa 46, transformers 44, onnxruntime 35, PySide6 ... |
| total installed | ~5.0 GB | 2 payload zips, 1.9 + 1.0 GB to download |
| models (downloaded by the app, not by the installer) | ~7 GB + 4 x ~0.3 GB translation | `infra/model_downloader.py`, mirrors, resumable |

The program itself imports only numpy and soundfile at start; torch, transformers, scipy, librosa ... are imported inside functions.
Checked: the Studio window opens with every heavy package blocked from importing (the GPU line then says "no GPU" until PyTorch is there).

## Modules

`tools/make_runtime_modules.py` splits the installed distributions of the build environment (files from each `RECORD`) into modules. Zips
are < 2 GiB each (several parts if needed), listed in the same manifest (`thin: true`, `modules: [...]`, `role: runtime`).

| Module | Contents | Size (zip, approx.) | Required |
|---|---|---|---|
| shell (`role: core`, in the installer) | Qt (PySide6), numpy, soundfile, certifi, psutil, the program, `voxprint-fetch` | 150-250 MB unpacked | yes |
| `torch` | torch, torchaudio, nvidia-* CUDA libraries | ~2.5-3 GB (CUDA build) | yes |
| `audio` | scipy, librosa, numba, llvmlite, sklearn, onnxruntime, av, PIL | ~0.45 GB | yes |
| `text` | nagisa, pymorphy3, ru_normalizr, num2words, eng_to_ipa ... | ~0.1 GB | yes |
| `ffmpeg` | imageio-ffmpeg | ~0.08 GB | yes |
| `ml` | everything else (transformers, qwen-tts/asr, peft, bitsandbytes, accelerate, safetensors ...) | ~0.2 GB | yes |
| models | Qwen3-TTS, ASR, aligner, translation | ~7 GB | on demand, existing downloader |

(`python tools/make_runtime_modules.py --site <site-packages> --out x --base-url x --list` prints the real split of an environment.) Phase 2: a CPU-only
`torch` variant (~0.2 GB) for users without an NVIDIA GPU, per-module feature gating.

## Start-up flow

1. `Voxprint-Setup-thin.exe` (Inno, `/DONLINE /DTHIN`) runs `voxprint-fetch --role core`: downloads the shell only (admin rights as before, `Program Files`), writes `_internal\modules.json` (manifest address). vc_redist is installed silently; exit codes 0, 1638 (newer version present), 3010, 1641 are fine and never shown.
2. The app starts, sees `modules.json` (`infra/modules.is_thin()`), activates what is already in `<app home>\runtime` (`modules.activate()` puts it first on `sys.path`; no restart is needed after a download) and opens the **Components** window (also: Settings -> Components).
3. The window reads the manifest (last good copy cached in the state folder, so it also works offline), lists the modules and downloads the missing ones into `<app home>\runtime` (user-writable, no admin) with the same `tools/online_fetch.py` and the network interface hopper.
4. When all required modules are present: the GPU is detected again, the model prefetch starts (as before), the features work. Until then the Studio window is usable for everything that needs no heavy library (opening the manual, settings, voice library browsing).
5. CLI for scripts and CI: `Voxprint.exe --modules-status`, `--install-modules [ids]`. A full build prints "This is a full build".

## Failure and resume

* Every part: HTTP Range resume, up to 8 retries, size + SHA-256 check, safe unzip; finished parts are recorded in `voxprint-components.json` (in the runtime folder) and skipped next time. Closing the window or the PC going to sleep just stops; "Download" continues.
* Network problems: short connect timeout, then the interface hopper (`infra/netroute.py`, Settings -> Network interface), remembered route.
* Cancel is checked between blocks. A broken part is deleted and downloaded again. The runtime folder can be deleted at any time (the window offers the download again).
* Updates: a new release has a new manifest; changed parts have a new hash and are replaced; unchanged modules stay.

## What changes

* Installer: `Voxprint.iss` `THIN` flavour (own output name, `--role core`, less disk space); `installer/build_online.ps1 -Thin -RuntimeSite <site-packages>`; the old `[Run]` vc_redist entry became `InstallVcRedist()` in `[Code]`.
* Build: `build_thin.bat` (PyInstaller with the heavy libraries excluded - they must not be baked into the shell's PYZ); `tools/make_online_payload.py --runtime-site` writes roles, `modules`, `modules.json`.
* App: `infra/modules.py`, `ui/modules_dialog.py`, `main.py` (activation, thin start flow, CLI), Settings button.

## Tested / not tested

Checked on the GitHub Windows runner (manual run with `smoke_only`, no release assets touched; jobs `online-smoke` and `thin-smoke`): the changed `Voxprint.iss` compiles for the online and the thin flavour; the online installer still installs, reuses, fails and uninstalls correctly; the thin installer installs only the shell (+ `modules.json`), `Voxprint --modules-status / --install-modules` downloads and unpacks the modules into the runtime folder; Windows adapter enumeration of `netroute` (ctypes) lists the real adapters. Covered on Linux by `tests/test_thin_modules.py` and `tests/test_netroute.py`.

Still untested: a real `build_thin.bat` run (PyInstaller with the exclusions), importing torch from the runtime folder inside a frozen exe (DLL search path of `torch/lib`, `os.add_dll_directory`), the `vc_redist` step with a real redistributable, the Components window on Windows, real VPN setups.
