"""How many CPU threads narration may use besides the GPU (Qt-free).

During narration the GPU thread generates speech while CPU work runs next to it: writing the chunk cache (FLAC), joining
chapters, ffmpeg encodes.  :func:`plan` sizes that pool from the *physical* cores: one core stays with the thread that
feeds the GPU (the Python side of ``generate`` needs it, and a starved feeder slows the GPU down), one or two more stay
free for the UI and the operating system, the rest is the pool (capped: more parallel ffmpeg processes than this only
fight over the disk).  torch's own CPU threads are limited too, so they do not compete with the pool; without CUDA torch
synthesizes on the CPU and gets nearly all cores, and the pool shrinks to one worker.

``VOXPRINT_CPU_WORKERS`` overrides the pool size (benchmarks, troubleshooting).
"""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: More parallel encodes than this give nothing (per-chapter ffmpeg jobs are disk-bound by then).
MAX_WORKERS = 8
ENV_WORKERS = "VOXPRINT_CPU_WORKERS"


@dataclass(frozen=True)
class Budget:
    """``workers``: size of the narration CPU pool; ``torch_threads``: ``torch.set_num_threads`` while synthesizing."""
    cores: int
    workers: int
    torch_threads: int


def plan(use_cuda: bool = True, cores: Optional[int] = None) -> Budget:
    """The CPU budget for ``cores`` physical cores (default: this computer)."""
    if cores is None:
        from infra.sysinfo import physical_cores

        cores = physical_cores()
    cores = max(1, int(cores))
    reserve = 2 if cores >= 6 else 1                 # UI + operating system
    if use_cuda:
        workers = max(1, min(MAX_WORKERS, cores - reserve - 1))       # - 1: the thread that feeds the GPU
        torch_threads = max(min(2, cores), min(4, cores - reserve - workers))
    else:
        workers = 1
        torch_threads = max(1, cores - reserve)      # torch synthesizes on the CPU: it needs the cores, not the pool
    override = os.environ.get(ENV_WORKERS, "").strip()
    if override.isdigit() and int(override) > 0:
        workers = int(override)
    return Budget(cores, workers, torch_threads)


def cuda_likely() -> bool:
    """Will the speech model run on an NVIDIA GPU?  torch's answer when torch is loaded already; otherwise whether the
    NVIDIA driver is installed (``nvidia-smi``), because importing torch only to ask takes seconds - and the pool is sized
    before the engine (which imports torch) has loaded."""
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            return False
    return shutil.which("nvidia-smi") is not None or Path(r"C:\Windows\System32\nvidia-smi.exe").is_file()
