"""Which speech recognition model (Qwen3-ASR) this PC uses: 1.7B on NVIDIA GPUs with about 8 GB of VRAM or more, else 0.6B.

Qwen3-ASR-1.7B makes clearly fewer errors on our languages (research notes: ru Fleurs 9.9 -> 6.0 % WER, de 6.5 -> 3.9 %), but it
needs ~4.7 GB of download and several GB of video memory; the 0.6B model stays for weaker GPUs and the CPU.  Settings has an
override (``state/asr_model.json``): automatic, always 0.6B, always 1.7B, or keep both downloaded (the automatic choice is used).

Only the first-run / Components "download all" step downloads the models (:func:`download_repos` feeds
:func:`workers.pipeline_runner.required_model_repos`).  At run time :func:`ready` never downloads: it returns the preferred model
if it is installed, else the other variant (an existing 0.6B install keeps working until the 1.7B download has finished).
"""
from __future__ import annotations

import json
import logging
from typing import Callable, List, Optional, Tuple

from infra import paths

log = logging.getLogger("voxprint.asr")

SMALL = "Qwen/Qwen3-ASR-0.6B"
LARGE = "Qwen/Qwen3-ASR-1.7B"
#: An "8 GB" card reports a little less than 8 GiB through CUDA (e.g. 7.6-8.0 GiB), so the cut is a bit below 8.
LARGE_MIN_VRAM_GB = 7.5
SETTINGS_FILE = "asr_model.json"
AUTO, USE_SMALL, USE_LARGE, BOTH = "auto", "small", "large", "both"
CHOICES = (AUTO, USE_SMALL, USE_LARGE, BOTH)


def preference() -> str:
    """The saved choice (``auto`` when nothing or something unknown is saved)."""
    try:
        data = json.loads((paths.state_dir() / SETTINGS_FILE).read_text(encoding="utf-8"))
        choice = data.get("model") if isinstance(data, dict) else None
        return choice if choice in CHOICES else AUTO
    except (OSError, ValueError):
        return AUTO


def set_preference(choice: str) -> None:
    """Remember the choice (one of :data:`CHOICES`)."""
    if choice not in CHOICES:
        raise ValueError(f"unknown ASR choice {choice!r}")
    f = paths.state_dir() / SETTINGS_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"model": choice}), encoding="utf-8")


def _gpu(gpu=None):
    if gpu is not None:
        return gpu
    from infra.vram_optimizer import detect_gpu

    return detect_gpu()


def auto_repo(gpu=None) -> str:
    """1.7B with a CUDA GPU of at least :data:`LARGE_MIN_VRAM_GB`, else 0.6B (``gpu`` is a ``vram_optimizer.GpuInfo``)."""
    g = _gpu(gpu)
    return LARGE if getattr(g, "available", False) and float(getattr(g, "total_gb", 0.0)) >= LARGE_MIN_VRAM_GB else SMALL


def preferred_repo(choice: Optional[str] = None, gpu=None, force_cpu: bool = False) -> str:
    """The model to use.  ``force_cpu`` (the "CPU only" option) means 0.6B unless the user explicitly chose 1.7B."""
    choice = choice or preference()
    if choice == USE_LARGE:
        return LARGE
    if choice == USE_SMALL or force_cpu:
        return SMALL
    return auto_repo(gpu)


def download_repos(choice: Optional[str] = None, gpu=None) -> List[str]:
    """What the "download all" step fetches for speech recognition: the preferred model, or both."""
    choice = choice or preference()
    if choice == BOTH:
        first = auto_repo(gpu)
        return [first, SMALL if first == LARGE else LARGE]
    return [preferred_repo(choice, gpu)]


def other(repo: str) -> str:
    return SMALL if repo == LARGE else LARGE


def ready(ready_path: Optional[Callable[[str], object]] = None, choice: Optional[str] = None, gpu=None,
          force_cpu: bool = False) -> Optional[Tuple[str, object]]:
    """``(repo, path)`` of the installed model to use: the preferred one, else the other variant; None if neither is installed.

    Never downloads (``ready_path`` defaults to :func:`infra.model_downloader.ready_model_path`)."""
    if ready_path is None:
        from infra import model_downloader as md

        ready_path = md.ready_model_path
    want = preferred_repo(choice, gpu, force_cpu)
    for repo in (want, other(want)):
        path = ready_path(repo)
        if path is not None:
            if repo != want:
                log.info("speech recognition: %s is not installed yet, using %s", want, repo)
            return repo, path
    return None
