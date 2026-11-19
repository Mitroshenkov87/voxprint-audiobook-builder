"""Import the native libraries the runtime ships, and exit 1 on the first failure.

CI runs this on Python 3.14 after the CPU PyTorch wheels and ``requirements.txt`` are installed.
CTranslate2 is imported through :func:`infra.cuda12_libs.import_ctranslate2` so the CUDA 12
libraries are on the path first.

Usage: ``python tools/check_native_imports.py``
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _ensure_root_on_path() -> None:
    """``python tools/check_native_imports.py`` puts ``tools/`` on ``sys.path``, not the repo root."""
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_root_on_path()

from infra.cuda12_libs import import_ctranslate2  # noqa: E402 - repo root must be on the path first

#: Import names (not distribution names). ``audioop`` is the module provided by ``audioop-lts``.
NATIVE = (
    "numpy",
    "soundfile",
    "av",
    "onnxruntime",
    "tokenizers",
    "sentencepiece",
    "nagisa",
    "scipy",
    "PySide6",
    "audioop",
    "torch",
    "torchaudio",
)


def check_torchaudio_ops() -> None:
    """Load a TorchAudio op. Importing the package is not enough: a mismatched torch fails here.

    ``importlib`` stays inside this function so ``import tools.check_native_imports`` can list
    ``NATIVE`` on a machine that has not installed torch yet. The checker reports that failure
    the same way as the other native imports.
    """
    torch = importlib.import_module("torch")
    torchaudio = importlib.import_module("torchaudio")
    out = torchaudio.functional.resample(torch.zeros(16), 16000, 8000)
    if out.numel() != 8 or not bool(torch.isfinite(out).all()):
        raise RuntimeError(f"torchaudio.functional.resample returned shape {tuple(out.shape)}")


def main() -> int:
    bad = 0
    for name in NATIVE:
        try:
            importlib.import_module(name)
            print(f"OK    {name}")
        except Exception as exc:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    try:
        check_torchaudio_ops()
        print("OK    torchaudio.functional.resample")
    except Exception as exc:  # noqa: BLE001
        bad += 1
        print(f"FAIL  torchaudio.functional.resample: {type(exc).__name__}: {exc}")
    try:
        import_ctranslate2()
        print("OK    ctranslate2")
    except Exception as exc:  # noqa: BLE001
        bad += 1
        print(f"FAIL  ctranslate2: {type(exc).__name__}: {exc}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
