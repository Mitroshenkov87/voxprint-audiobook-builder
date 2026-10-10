"""MkDocs hook: regenerate the command reference, the roadmap copy and the API pages before each build."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VOXPRINT_ALLOW_NO_GPU", "1")
os.environ.setdefault("VOXPRINT_NO_ENV_PROBE", "1")
os.environ.setdefault("VOXPRINT_HOME", tempfile.mkdtemp(prefix="voxprint-docs-"))

from tools.gen_docs import write_all  # noqa: E402  path and environment are set just above


def on_pre_build(config: object) -> None:
    """Refresh generated pages so the site matches the code being built."""
    del config
    write_all()
