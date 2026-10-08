"""The optional noise-cleanup tool: DeepFilterNet3 through the pre-compiled ``deep-filter`` program (Rust, CPU, no Python deps).

Chosen after checking the candidates (docs/VOICE-QUALITY.md, B5): DeepFilterNet is dual-licensed MIT OR Apache-2.0
(code and the bundled models), and the release program of v0.5.6 embeds the DeepFilterNet3 model, so one ~26 MB file is the
whole download (real-time factor ~0.2 on one CPU core).  MossFormer2 (ClearerVoice-Studio, Apache-2.0) would need a PyTorch
pipeline and a larger model; Sidon (MIT) needs w2v-BERT 2.0 (~600M parameters) and a GPU.

The program is pinned by URL, size and SHA-256 per platform.  It is a separate optional download, started only by the user
from the Train window (never by the "download all" step, never silently); without it the clean-up is simply not offered.
"""
from __future__ import annotations

import logging
import os
import stat
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

from core.i18n import tr
from infra import model_release

log = logging.getLogger("voxprint.denoise")

VERSION = "0.5.6"
LABEL = "DeepFilterNet3"
_BASE = f"https://github.com/Rikorose/DeepFilterNet/releases/download/v{VERSION}/"
#: platform -> release asset (checked 2026-10-08; the assets of v0.5.6 are unchanged since 2023-08-31)
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
    """The pinned asset of this (or the given) platform."""
    key = key or platform_key()
    return ASSETS.get(key) if key else None


def tool_path(models_dir: Optional[Path] = None, key: Optional[str] = None) -> Optional[Path]:
    """``<models folder>/deepfilternet/<asset name>`` (None on an unsupported platform)."""
    a = asset(key)
    if a is None:
        return None
    if models_dir is None:
        from infra import paths

        models_dir = paths.models_dir()
    return Path(models_dir) / "deepfilternet" / str(a["name"])


def ready(models_dir: Optional[Path] = None, key: Optional[str] = None) -> Optional[Path]:
    """The program's path when it is downloaded (pinned size; the hash was checked on download), else None."""
    p = tool_path(models_dir, key)
    a = asset(key)
    try:
        return p if p is not None and p.is_file() and p.stat().st_size == int(a["size"]) else None  # type: ignore[index]
    except OSError:
        return None


def download_size(key: Optional[str] = None) -> int:
    """Bytes of the download on this platform (0 when unsupported)."""
    a = asset(key)
    return int(a["size"]) if a else 0  # type: ignore[arg-type]


def ensure(progress: Callable[[float, str], None] = lambda f, m="": None, models_dir: Optional[Path] = None,
           opener=None, timeout: float = 30.0, key: Optional[str] = None) -> Path:
    """Download the program if missing (size + SHA-256 checked, resumable) and return its path.  Only call this on the user's
    explicit request.  Raises :class:`infra.model_release.ReleaseError` (also on an unsupported platform)."""
    a = asset(key)
    target = tool_path(models_dir, key)
    if a is None or target is None:
        raise model_release.ReleaseError("noise clean-up is not available for this platform")
    have = ready(models_dir, key)
    if have is not None:
        return have
    size = int(a["size"])  # type: ignore[arg-type]
    done = [0]

    def on_bytes(n: int) -> None:
        done[0] += n
        frac = min(1.0, done[0] / size)
        progress(min(0.99, frac), tr("progress.downloading", short=LABEL, pct=int(100 * frac)))

    model_release.fetch_file(_BASE + str(a["name"]), target, {"size": size, "sha256": a["sha256"]}, on_bytes,
                             opener or model_release._open, timeout)
    if os.name != "nt":                                   # a downloaded file is not executable on Linux
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    progress(1.0, tr("progress.model_verifying", short=LABEL))
    log.info("noise clean-up program downloaded: %s", target)
    return target
