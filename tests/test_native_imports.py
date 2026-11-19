"""The CI native-import list names every extension the runtime must be able to load."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_native_imports import NATIVE, check_torchaudio_ops

ROOT = Path(__file__).resolve().parents[1]


def test_native_import_list_covers_the_locked_extensions():
    for name in ("torch", "torchaudio", "numpy", "soundfile", "av", "onnxruntime", "tokenizers",
                 "sentencepiece", "nagisa", "scipy", "PySide6", "audioop"):
        assert name in NATIVE


def test_running_the_file_finds_infra_without_pythonpath(tmp_path):
    """CI launches ``python tools/check_native_imports.py``. That puts ``tools/`` on the path."""
    script = ROOT / "tools" / "check_native_imports.py"
    code = "import runpy; runpy.run_path(%r, run_name='not_main'); import infra" % str(script)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_torchaudio_resample_runs_when_torch_is_installed():
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    check_torchaudio_ops()
