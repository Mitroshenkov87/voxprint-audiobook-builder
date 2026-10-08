"""The optional AI text model ("Literary translation" / "Prepare text for narration"): Gemma 4 12B (GGUF) run by llama.cpp.

Chosen per the research notes (docs/TRANSLATION.md): Gemma 4 12B Instruct is Apache-2.0 (weights included) and ranks well for
translation into Russian; the Q4_K_M quantization (7.1 GB) fits GPUs with ~10 GB of VRAM.  It runs in the pre-built
``llama-server`` program of llama.cpp (MIT), **Vulkan** build: 32 MB instead of ~650 MB for the CUDA build plus its runtime,
and it works on any NVIDIA driver.  A separate process also means "unload before TTS" is simply ending the process: the
whole VRAM is given back before the voice model loads.

Both files are pinned (URL, size, SHA-256) and form one optional download that only the user starts (Narrate window); the
"download all" step never fetches them.  Without them, or on a GPU below :data:`MIN_VRAM_GB`, the options stay greyed out.
"""
from __future__ import annotations

import json
import logging
import os
import re
import socket
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Optional

from core.i18n import tr
from infra import model_release

log = logging.getLogger("voxprint.llm")

LABEL = "Gemma 4 12B"
KEY = "llm-gemma4-12b"                       # ``.key`` for the shared download worker (workers/narrate_worker.py)
#: An "10/12 GB" card reports a bit less through CUDA; 7.1 GB weights + KV cache (8k context) + Vulkan overhead.
MIN_VRAM_GB = 9.5
LLAMA_BUILD = "b11476"                       # llama.cpp release of 2026-10-07 (supports Gemma 4)
_LLAMA_BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/"
SERVER_ASSETS: Dict[str, Dict[str, object]] = {
    "win-x64": {"name": f"llama-{LLAMA_BUILD}-bin-win-vulkan-x64.zip", "size": 33380424,
                "sha256": "5c71e7b749697da4a8d46e9ee55486845cbba27c9dfbecb4007f31ba6610d523"},
    "linux-x64": {"name": f"llama-{LLAMA_BUILD}-bin-ubuntu-vulkan-x64.tar.gz", "size": 31684207,
                  "sha256": "5bb4306d7917f33e81efda02e6f791ae6a82e86bee121227a3ab2b4e8e40427f"},
}
MODEL: Dict[str, object] = {
    "repo": "unsloth/gemma-4-12b-it-GGUF", "revision": "fc034cfff751157913579611efad8462ac1be606",
    "file": "gemma-4-12b-it-Q4_K_M.gguf", "size": 7121861440,
    "sha256": "0a270ec9fe6b34f4a0d33992b6135117b484ebc4766ab76b51d4ae8c457e4c42",
}
MODEL_TAG = f"gemma-4-12b-it-Q4_K_M@{str(MODEL['revision'])[:8]}"


def platform_key() -> Optional[str]:
    """``win-x64`` / ``linux-x64`` (same rule as the noise clean-up tool), None elsewhere."""
    from infra import denoise_tool

    return denoise_tool.platform_key()


def _dir(models_dir: Optional[Path] = None) -> Path:
    if models_dir is None:
        from infra import paths

        models_dir = paths.models_dir()
    return Path(models_dir) / "llm"


def model_path(models_dir: Optional[Path] = None) -> Path:
    return _dir(models_dir) / str(MODEL["file"])


def server_dir(models_dir: Optional[Path] = None) -> Path:
    return _dir(models_dir) / f"llama.cpp-{LLAMA_BUILD}"


def server_exe(models_dir: Optional[Path] = None) -> Optional[Path]:
    """``llama-server(.exe)`` of the unpacked build (the archives nest it differently), or None."""
    d = server_dir(models_dir)
    if not (d / ".complete").is_file():
        return None
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    return next(iter(sorted(d.rglob(name))), None)


def model_ready(models_dir: Optional[Path] = None) -> bool:
    p = model_path(models_dir)              # the hash was checked when it was downloaded; the size catches a damaged copy
    try:
        return p.is_file() and p.stat().st_size == int(MODEL["size"])  # type: ignore[arg-type]
    except OSError:
        return False


def download_mb() -> int:
    a = SERVER_ASSETS.get(platform_key() or "")
    return int((int(MODEL["size"]) + (int(a["size"]) if a else 0)) / 1e6)  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def _vram_gb() -> float:
    from infra.vram_optimizer import detect_gpu

    g = detect_gpu()
    return float(g.total_gb) if g.available else 0.0


def status(models_dir: Optional[Path] = None, vram_gb: Optional[float] = None) -> str:
    """``unsupported`` (platform) | ``low_vram`` | ``needs_download`` | ``ready``."""
    if platform_key() is None:
        return "unsupported"
    if (_vram_gb() if vram_gb is None else vram_gb) < MIN_VRAM_GB:
        return "low_vram"
    if server_exe(models_dir) is None or not model_ready(models_dir):
        return "needs_download"
    return "ready"


def _unpack(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")
    if os.name != "nt":
        for p in dest.rglob("llama-server"):
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (dest / ".complete").write_text(LLAMA_BUILD, encoding="utf-8")


def ensure(progress: Callable[[float, str], None] = lambda f, m="": None, models_dir: Optional[Path] = None,
           opener=None, timeout: float = 30.0) -> Path:
    """Download llama-server and the GGUF model if missing (size + SHA-256 checked, resumable); returns the model path.
    Only on the user's explicit request.  Raises :class:`infra.model_release.ReleaseError`."""
    a = SERVER_ASSETS.get(platform_key() or "")
    if a is None:
        raise model_release.ReleaseError("the AI text model is not available for this platform")
    opener = opener or model_release._open
    total = float(download_mb() * 1e6) or 1.0
    done = [0]

    def on_bytes(n: int) -> None:
        done[0] += n
        frac = min(1.0, done[0] / total)
        progress(min(0.99, frac), tr("progress.downloading", short=LABEL, pct=int(100 * frac)))

    if server_exe(models_dir) is None:
        archive = _dir(models_dir) / str(a["name"])
        model_release.fetch_file(_LLAMA_BASE + str(a["name"]), archive, {"size": a["size"], "sha256": a["sha256"]},
                                 on_bytes, opener, timeout)
        _unpack(archive, server_dir(models_dir))
        archive.unlink(missing_ok=True)
    else:
        on_bytes(int(a["size"]))  # type: ignore[arg-type]
    if not model_ready(models_dir):
        url = f"https://huggingface.co/{MODEL['repo']}/resolve/{MODEL['revision']}/{MODEL['file']}"
        model_release.fetch_file(url, model_path(models_dir), {"size": MODEL["size"], "sha256": MODEL["sha256"]},
                                 on_bytes, opener, timeout)
    progress(1.0, tr("progress.model_verifying", short=LABEL))
    log.info("AI text model ready: %s + llama.cpp %s", model_path(models_dir), LLAMA_BUILD)
    return model_path(models_dir)


_DEVICE_LINE = re.compile(r"^\s*(Vulkan\d+)\s*:\s*(.+?)\s*(?:\(|$)", re.M)
_NVIDIA = re.compile(r"NVIDIA|GeForce|RTX|Quadro|Tesla", re.I)


def pick_device(exe: Path, run=subprocess.run) -> Optional[str]:
    """The Vulkan device of the NVIDIA GPU (``Vulkan1`` ...) from ``llama-server --list-devices``, or None = llama.cpp's
    default.  Hybrid laptops list the integrated GPU too, and the model must not land there."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0
    try:
        r = run([str(exe), "--list-devices"], capture_output=True, text=True, timeout=60, creationflags=flags)
        listing = (r.stdout or "") + "\n" + (r.stderr or "")
    except (OSError, subprocess.SubprocessError):
        log.warning("llama-server --list-devices failed; using its default device", exc_info=True)
        return None
    devices = _DEVICE_LINE.findall(listing)
    chosen = next((d for d, name in devices if _NVIDIA.search(name)), None)
    log.info("AI text model devices: %s -> %s", "; ".join(f"{d}: {n}" for d, n in devices) or "none listed",
             chosen or "default")
    return chosen


class LlamaServer:
    """``llama-server`` on a free localhost port; :meth:`complete` sends one chat request, :meth:`close` ends the process
    (and frees all of its VRAM)."""

    def __init__(self, exe: Path, model: Path, ctx: int = 8192, popen=subprocess.Popen, log_dir: Optional[Path] = None,
                 start_timeout: float = 300.0, run=subprocess.run) -> None:
        self.exe, self.model, self.ctx, self._popen, self._run = Path(exe), Path(model), ctx, popen, run
        self.log_dir, self.start_timeout = log_dir, start_timeout
        self.proc = None
        self.port = 0
        # localhost only: never through a system proxy
        self._http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def start(self) -> "LlamaServer":
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        cmd = [str(self.exe), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port), "-c", str(self.ctx),
               "-ngl", "999", "-np", "1", "--jinja"]
        device = pick_device(self.exe, self._run)
        if device:
            cmd += ["--device", device]
        if self.log_dir is None:
            from infra import paths

            self.log_dir = paths.logs_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        out = open(self.log_dir / "llama-server.log", "w", encoding="utf-8", errors="replace")  # noqa: SIM115 - owned by the process
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0
        t0 = time.monotonic()
        self.proc = self._popen(cmd, stdout=out, stderr=subprocess.STDOUT, creationflags=flags)
        while time.monotonic() - t0 < self.start_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode} (see llama-server.log)")
            try:
                with self._http.open(f"http://127.0.0.1:{self.port}/health", timeout=5) as r:
                    if r.status == 200:
                        log.info("AI text model loaded in %.0f s (%s)", time.monotonic() - t0, MODEL_TAG)
                        return self
            except OSError:
                pass
            time.sleep(1.0)
        self.close()
        raise RuntimeError("llama-server did not become ready in time")

    def complete(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.2) -> str:
        body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": temperature, "top_p": 0.9,
                           "max_tokens": max_tokens, "stream": False}).encode("utf-8")
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with self._http.open(req, timeout=900) as r:
            data = json.loads(r.read().decode("utf-8"))
        return str(data["choices"][0]["message"].get("content") or "")

    def close(self) -> None:
        p, self.proc = self.proc, None
        if p is None or p.poll() is not None:
            return
        p.terminate()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=15)
        log.info("AI text model unloaded")


def make_plan(models_dir: Optional[Path] = None):
    """:class:`core.llm_text.LLMPlan` with the downloaded model, or None if it is not ready."""
    from core.llm_text import LLMPlan

    exe = server_exe(models_dir)
    if exe is None or not model_ready(models_dir):
        return None
    return LLMPlan(lambda: LlamaServer(exe, model_path(models_dir)).start(), MODEL_TAG)
