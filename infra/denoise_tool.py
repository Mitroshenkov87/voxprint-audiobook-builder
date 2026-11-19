"""The optional noise-cleanup tool: DeepFilterNet3 through the pre-compiled ``deep-filter`` program (Rust, CPU, no Python deps).

Chosen after checking the candidates (docs/VOICE-QUALITY.md, B5): DeepFilterNet is dual-licensed MIT OR Apache-2.0
(code and the bundled models), and the release program of v0.5.6 embeds the DeepFilterNet3 model, so one ~26 MB file is the
whole download (real-time factor ~0.2 on one CPU core).  MossFormer2 (ClearerVoice-Studio, Apache-2.0) would need a PyTorch
pipeline and a larger model; Sidon (MIT) needs w2v-BERT 2.0 (~600M parameters) and a GPU.

The pinned program lives in the shared model store (manifest row ``deepfilternet``,
``deepfilternet/deep-filter-0.5.6-x86_64-unknown-linux-musl``) and arrives through VOXPrint MDL
(:mod:`infra.mdl_models`).  This module copies it from the read-only view into the app's derived folder
(:func:`infra.mdl_models.derived_dir`) and makes the copy executable - the store blob itself is never modified.
It is a separate optional download, started only by the user from the Train window; without it the clean-up is simply
not offered.  There is no row for Windows (the platform is parked): the tool reports unavailable there.
"""
from __future__ import annotations

import logging
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast

from infra import mdl_models

log = logging.getLogger("voxprint.denoise")

VERSION = "0.5.6"
LABEL = "DeepFilterNet3"
MODEL_ID = "deepfilternet"
_BASE = f"https://github.com/Rikorose/DeepFilterNet/releases/download/v{VERSION}/"
#: platform -> release asset (checked 2026-10-08; the assets of v0.5.6 are unchanged since 2023-08-31).
#: Only the linux-x64 row exists in the app manifest; the copy stays as reference data.
ASSETS: Dict[str, Dict[str, object]] = {
    "win-x64": {"name": f"deep-filter-{VERSION}-x86_64-pc-windows-msvc.exe", "size": 26912256,
                "sha256": "75e11fa16445f560cb6b021521ddb89e89270d13b83089705d98776f58fd7915"},
    "linux-x64": {"name": f"deep-filter-{VERSION}-x86_64-unknown-linux-musl", "size": 36417296,
                  "sha256": "70775e251eee44c0f2451a1e833326cf8bcbbe304d3e7cd12851e6fce72ef7da"},
}


def platform_key(platform: Optional[str] = None, machine: Optional[str] = None) -> Optional[str]:
    """``win-x64`` / ``linux-x64``, or None where no pinned program exists (the clean-up is then not offered)."""
    import platform as _pf

    plat = platform if platform is not None else sys.platform
    mach = (machine if machine is not None else _pf.machine()).lower()
    if mach not in ("amd64", "x86_64", "x64"):
        return None
    if plat.startswith("win"):
        return "win-x64"
    if plat.startswith("linux"):
        return "linux-x64"
    return None


def asset(key: Optional[str] = None) -> Optional[Dict[str, object]]:
    """The pinned asset of this (or the given) platform; None where the manifest has no row (Windows is parked)."""
    key = key or platform_key()
    if key != "linux-x64":
        return None
    return ASSETS["linux-x64"]


def _derived() -> Path:
    """The app-owned folder holding the executable copy of the program."""
    return mdl_models.derived_dir(MODEL_ID)


def tool_path(models_dir: Optional[Path] = None, key: Optional[str] = None) -> Optional[Path]:
    """The executable copy of the program in the derived folder (None on an unsupported platform)."""
    a = asset(key)
    if a is None:
        return None
    if models_dir is not None:
        return Path(models_dir) / "deepfilternet" / str(a["name"])
    return _derived() / str(a["name"])


def ready(models_dir: Optional[Path] = None, key: Optional[str] = None) -> Optional[Path]:
    """The program's path when it is installed (the copy exists; the store checked the hash), else None."""
    p = tool_path(models_dir, key)
    a = asset(key)
    try:
        return p if p is not None and a is not None and p.is_file() and p.stat().st_size == int(cast(Any, a["size"])) else None
    except OSError:
        return None


def download_size(key: Optional[str] = None) -> int:
    """Bytes of the download on this platform (0 when unsupported)."""
    a = asset(key)
    return int(cast(Any, a["size"])) if a else 0


def _silent_progress(f: float, m: str = "") -> None:
    pass


def ensure(progress: Callable[[float, str], None] = _silent_progress, models_dir: Optional[Path] = None,
           opener=None, timeout: float = 30.0, key: Optional[str] = None) -> Path:
    """Fetch the program through VOXPrint MDL and return the executable copy.  Only call this on the user's explicit request.

    The bytes are copied from the read-only store view into :func:`infra.mdl_models.derived_dir` and chmod +x there -
    a store blob is never modified.  Raises :class:`core.errors.ModelDownloadError` (also on an unsupported platform)."""
    from core.errors import ModelDownloadError
    from core.i18n import tr

    del opener, timeout
    a = asset(key)
    if a is None:
        raise ModelDownloadError(tr("err.mdl_not_ready", model=LABEL), details="noise clean-up is not available for this platform")
    if models_dir is not None:
        explicit = tool_path(models_dir, key)  # tests pass an explicit root: nothing to fetch
        if explicit is None:
            raise ModelDownloadError(tr("err.mdl_not_ready", model=LABEL), details="noise clean-up is not available for this platform")
        return explicit
    have = ready(key=key)
    if have is not None:
        return have
    view = mdl_models.ensure(MODEL_ID, progress=progress)
    src = view / "deepfilternet" / str(a["name"])
    dest_dir = _derived()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / str(a["name"])
    shutil.copy2(src, dest)
    if os.name != "nt":                                   # the store blob stays untouched: the copy becomes executable
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log.info("noise clean-up program ready: %s", dest)
    return dest
