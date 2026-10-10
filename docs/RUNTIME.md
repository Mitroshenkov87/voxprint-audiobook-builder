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

## Why cu130 only

The wheels we ship are CUDA 13.0 (`cu130`). They need an NVIDIA driver that supports CUDA 13, which starts at driver **600**. Older drivers are not offered a wheel: `cu126` and `cu128` are not in the lock. CI installs the CPU wheels (`ci_flavors` in the lock); that flavor is not offered to users. `VOXPRINT_TORCH_FLAVOR=cpu` still selects those wheels for tests.

## CTranslate2 and CUDA 12

CTranslate2's published wheels are CUDA 12 builds. Torch `cu130` does not ship those libraries, so the runtime also installs `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`. `infra/cuda12_libs.py` is the one place that puts their `bin` (Windows, `os.add_dll_directory`) or `lib` (Linux, `LD_LIBRARY_PATH`, and a preload of the SONAMEs when CTranslate2 itself is imported) on the library path. Import CTranslate2 with `import_ctranslate2()` from that module.
