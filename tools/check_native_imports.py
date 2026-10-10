"""Import the native libraries the runtime ships, and exit 1 on the first failure.

CI runs this on Python 3.14 after the CPU PyTorch wheels and ``requirements.txt`` are installed.
CTranslate2 is imported through :func:`infra.cuda12_libs.import_ctranslate2` so the CUDA 12
libraries are on the path first.

Usage: ``python tools/check_native_imports.py``
"""
from __future__ import annotations

import importlib
import sys

from infra.cuda12_libs import import_ctranslate2

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
        import_ctranslate2()
        print("OK    ctranslate2")
    except Exception as exc:  # noqa: BLE001
        bad += 1
        print(f"FAIL  ctranslate2: {type(exc).__name__}: {exc}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
