"""Training presets of the Train window: Fast / Balanced / Maximum / Manual, plus a time estimate calibrated on real runs.

A preset changes only *how long and how big* the LoRA training is (passes over the data = epochs, adapter rank); the
learning rate stays at the conservative automatic value for every preset.  Real tests (RTX 4090, 66-70 clips of a
10-minute Russian recording) showed why: 15 epochs at 3e-6 and 8e-6 gave a lower training loss but a voice that babbles
and never stops, while the default (1e-6, 5 epochs) stayed usable.  The training loss alone is therefore not a quality
signal; the post-training check (``core/voice_check.py``) judges the result.

Time model: ``seconds = LOAD_SEC + clips * epochs * SEC_PER_CLIP_EPOCH * speed_factor(gpu)``, with
``SEC_PER_CLIP_EPOCH = 0.34`` measured on the RTX 4090 (24.0 s per epoch for 70 clips, 1.7B model, bf16, gradient
checkpointing) and ``LOAD_SEC = 12`` (model load and audio-code preparation).  Other GPUs are scaled by a table; unknown
GPUs by VRAM tier.  It is an estimate (shown with "about"), not a promise.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

from infra.vram_optimizer import (GpuInfo, LORA_ALPHA, LORA_R, TrainPlan, compute_epochs, plan_training)

FAST, BALANCED, MAXIMUM, MANUAL = "fast", "balanced", "maximum", "manual"
PRESETS = (FAST, BALANCED, MAXIMUM, MANUAL)
DEFAULT_PRESET = BALANCED

SEC_PER_CLIP_EPOCH_4090 = 0.34     # measured: 24.0 s / epoch / 70 clips
LOAD_SEC = 12.0                    # measured: model load + audio-code preparation
CLIPS_PER_SPEECH_SEC = 70 / 535.0  # measured: 70 clips from 535 s of speech (mean clip 7.6 s)
SPEECH_FRACTION = 0.88             # measured: 88 % of a read recording is speech that ends up in clips
CPU_FACTOR = 40.0                  # rough: a CPU is ~40x slower than a 4090 for this job (not measured)

#: (substring of the GPU name, speed factor relative to the RTX 4090 = 1.0); first match wins.  Estimates from the
#: relative training throughput of the cards (geometric mean of the dense bf16 tensor-core and memory-bandwidth ratios to the
#: 4090; checked against the measured 4090 only), not measured on the other cards - rounded UP, an overestimate beats a surprise.
GPU_FACTORS = (("5090", 0.7), ("4090", 1.0), ("4080", 1.5), ("3090", 1.7), ("4070 ti", 1.9), ("4070", 2.4), ("3080", 2.0),
               ("4060 ti", 2.6), ("3070", 2.8), ("4060", 3.2), ("3060", 3.6), ("2080", 3.2), ("2070", 3.8), ("2060", 4.5),
               ("1080", 5.0), ("1070", 6.0), ("1060", 8.0))


@dataclass(frozen=True)
class Manual:
    """Hand-picked values of the Manual preset (the advanced panel)."""
    epochs: int = 5
    lora_r: int = LORA_R
    lora_alpha: int = LORA_ALPHA
    lr: float = 1e-6
    grad_accum: int = 4


def speed_factor(gpu: GpuInfo) -> float:
    """How many times slower than an RTX 4090 this machine trains (>= ~0.7)."""
    if not gpu.available:
        return CPU_FACTOR
    name = gpu.name.lower()
    for key, f in GPU_FACTORS:
        if key in name:
            return f
    t = gpu.total_gb
    return 1.0 if t >= 22 else 1.8 if t >= 15 else 3.0 if t >= 11 else 5.0 if t >= 7 else 8.0


def clips_from_audio(audio_seconds: float) -> int:
    """Expected number of training clips for a recording of ``audio_seconds`` (at least 1)."""
    return max(1, round(audio_seconds * SPEECH_FRACTION * CLIPS_PER_SPEECH_SEC))


def build_plan(preset: str, gpu: GpuInfo, n_items: int, *, force_cpu: bool = False, language: str = "russian",
               manual: Optional[Manual] = None) -> TrainPlan:
    """The :class:`TrainPlan` for ``preset``; Balanced is exactly the automatic plan."""
    base = plan_training(gpu, n_items, force_cpu=force_cpu, language=language)
    if preset == FAST:      # about half the passes, half-size adapter
        return replace(base, epochs=compute_epochs(n_items, 200, lo=2, hi=8), lora_r=16, lora_alpha=64)
    if preset == MAXIMUM:   # about twice the passes (lora.md: stay below ~600 examples x epochs), double-size adapter
        return replace(base, epochs=compute_epochs(n_items, 560, lo=4, hi=30), lora_r=64, lora_alpha=256)
    if preset == MANUAL and manual is not None:
        return replace(base, epochs=max(1, manual.epochs), lora_r=max(1, manual.lora_r), lora_alpha=max(1, manual.lora_alpha),
                       lr=manual.lr, grad_accum=max(1, manual.grad_accum))
    return base


def estimate_seconds(plan: TrainPlan, n_items: int, gpu: GpuInfo) -> float:
    """Estimated wall time of the training stage for ``plan`` (see the module docstring)."""
    f = speed_factor(gpu)
    if plan.base_model.endswith("0.6B-Base"):
        f *= 0.55                      # the small model is faster (not measured: rough)
    return LOAD_SEC * (1 if f < CPU_FACTOR else 3) + n_items * plan.epochs * SEC_PER_CLIP_EPOCH_4090 * f


def format_duration(seconds: float, minute: str = "min", second: str = "s", hour: str = "h") -> str:
    """"about 2 min" style text: seconds under a minute, minutes under 2 hours, else hours + minutes."""
    s = max(1, int(round(seconds)))
    if s < 90:
        return f"{max(10, int(round(s / 10.0)) * 10)} {second}"
    m = int(round(s / 60.0))
    if m < 120:
        return f"{m} {minute}"
    return f"{m // 60} {hour} {m % 60:02d} {minute}"


def describe(preset: str, gpu: GpuInfo, n_items: int, manual: Optional[Manual] = None) -> Dict[str, object]:
    """Plan + estimate in one call: ``{"plan", "seconds"}``."""
    plan = build_plan(preset, gpu, n_items, manual=manual)
    return {"plan": plan, "seconds": estimate_seconds(plan, n_items, gpu)}
