# Runtime

Voxprint runs on **Python 3.14** (Windows x64 and Linux x86-64).

| Piece | Version |
|---|---|
| Python | 3.14 |
| PyTorch | 2.11.0, flavor `cu130` |
| TorchAudio | 2.11.0, flavor `cu130` (same release as torch; its native ops are built against that torch) |
| nvidia-cublas-cu12 | 12.9.2.10 |
| nvidia-cudnn-cu12 | 9.27.0.42 |
| PySide6 | 6.11 or newer |

TorchAudio has no wheel for torch 2.12–2.14 on `cu130` and Python 3.14, so both packages stay on 2.11.0. `torchcodec` is not installed. CI installs the CPU build of the same torch and torchaudio.

The pinned Windows wheels are in `infra/runtime_lock.json` (`python tools/make_runtime_lock.py`). A few libraries publish an abi3 wheel tagged for an older 3.x (`audioop-lts` cp313, `av` and `soxr` cp312). Those wheels install on Python 3.14. Every package that publishes a cp314 wheel is pinned to that wheel.

`voxprint-common` 0.2.1 is pinned by the released wheel in `requirements.txt` and in the lock's `suite` object (`voxprint_common-0.2.1-py3-none-any.whl`, SHA-256 `7a5a41284a74fc8eb2ad724647c05575d16c42881d30dba461fddf15b96a3d2f`). pip checks that hash when it installs the wheel. The thin shell bundles it. It is not one of the PyPI runtime downloads. VOXPrint MDL (`voxprint-mdl`) from that wheel downloads the models in `voxprint-app-models.json`.

The shared runtime is one `shared/manifest.json` (schema 1, `components`). Torch 2.11.0+cu130 lives at `shared/runtime/py314-torch2.11-cu130` and is reused when another Voxprint app already refs that component, instead of being downloaded again. This app's other wheels, including a CPU `onnxruntime`, stay in `shared/wheels/audiobook-builder`. An upgrade moves an existing `<app home>/runtime` (or a flat `shared/runtime` that still has torch at the top) into those two folders. Uninstall drops this app's ref and deletes a directory only when this app was the last ref and the path is not a shared root (the models root, `shared/runtime`, the voices root, a cache root, or the suite home). Model files stay with VOXPrint MDL. There is no separate setup page for that.

## Why cu130 only

The wheels we ship are CUDA 13.0 (`cu130`). They need an NVIDIA driver that supports CUDA 13, which starts at driver **600**. Older drivers are not offered a wheel: `cu126` and `cu128` are not in the lock. CI installs the CPU wheels (`ci_flavors` in the lock); that flavor is not offered to users. `VOXPRINT_TORCH_FLAVOR=cpu` still selects those wheels for tests.

## CTranslate2 and CUDA 12

CTranslate2's published wheels are CUDA 12 builds. Torch `cu130` does not ship those libraries, so the runtime also installs `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`. `infra/cuda12_libs.py` is the one place that puts their `bin` (Windows, `os.add_dll_directory`) or `lib` (Linux, `LD_LIBRARY_PATH`, and a preload of the SONAMEs when CTranslate2 itself is imported) on the library path. Import CTranslate2 with `import_ctranslate2()` from that module.
