"""The CI native-import list names every extension the runtime must be able to load."""
from tools.check_native_imports import NATIVE


def test_native_import_list_covers_the_locked_extensions():
    for name in ("torch", "torchaudio", "numpy", "soundfile", "av", "onnxruntime", "tokenizers",
                 "sentencepiece", "nagisa", "scipy", "PySide6", "audioop"):
        assert name in NATIVE
