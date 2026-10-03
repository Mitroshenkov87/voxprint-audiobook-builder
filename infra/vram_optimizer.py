"""Choose LoRA training parameters for the available VRAM (pure logic + optional monitoring through torch).

The defaults mirror the Alexandria training script (``train_lora.py``): batch size 1 with gradient accumulation,
r=32 / alpha=128, a tiny learning rate, eager attention, gradient checkpointing, bf16 on the GPU.  The plan degrades
gracefully with less VRAM (8-bit Adam, the 0.6B base model) and falls back to the CPU.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional

from core.i18n import tr

log = logging.getLogger("voxprint.vram")

MODEL_1_7B = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_0_6B = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


@dataclass(frozen=True)
class GpuInfo:
    """What we know about the first CUDA device: availability, name and total/free memory in GiB."""
    available: bool
    name: str = ""
    total_gb: float = 0.0
    free_gb: float = 0.0


#: Defaults taken from the Alexandria documentation and training script (train_lora.py / lora.md).
LORA_R = 32
LORA_ALPHA = 128
TARGET_PASSES = 320          # lora.md rule: examples x epochs ~ 250-400 (above ~600 it overfits)
LOSS_WARN_BELOW = 3.5        # lora.md: loss < ~3.5 means garbled speech / missing EOS (confirmed for English only, not for Russian)
MAX_AUDIO_SECONDS = 30.0     # same as --max_audio_seconds in train_lora.py


@dataclass(frozen=True)
class TrainPlan:
    """Complete, immutable set of training hyper-parameters chosen for this machine and dataset size."""
    device: str                    # "cuda:0" | "cpu"
    base_model: str
    batch_size: int                # always 1 (as in train_lora.py)
    grad_accum: int
    gradient_checkpointing: bool
    use_8bit_adam: bool
    dtype: str                     # "bfloat16" | "float32"
    lora_r: int
    lora_alpha: int
    lr: float
    epochs: int
    max_seconds_per_item: float
    language: str = "russian"      # the trainer's --language flag (lower case; Alexandria's default is english)
    attn_implementation: str = "eager"   # as in train_lora.py; flash-attn is not used on Windows
    warnings: List[str] = field(default_factory=list)

    @property
    def effective_batch(self) -> int:
        """Examples per optimizer step (``batch_size * grad_accum``)."""
        return self.batch_size * self.grad_accum


def detect_gpu() -> GpuInfo:
    """Detect the NVIDIA GPU through torch (lazy import).  Without torch/CUDA returns ``available=False``."""
    try:
        import torch

        if not torch.cuda.is_available():
            return GpuInfo(False)
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        return GpuInfo(True, props.name, total / 1024 ** 3, free / 1024 ** 3)
    except Exception as exc:  # noqa: BLE001
        log.warning("GPU detection failed: %s", exc)
        return GpuInfo(False)


def compute_epochs(n_items: int, target_passes: int = TARGET_PASSES, lo: int = 2, hi: int = 15) -> int:
    """Epochs such that (examples x epochs) lands near ``target_passes`` (the 250-400 rule from lora.md), clamped to [lo, hi]."""
    return max(lo, min(hi, round(target_passes / max(1, n_items))))


def compute_lr(n_items: int) -> float:
    """Learning rate: lora.md suggests 1e-6 for ~60 examples and 2e-6 for ~120; for a new language we take the lower bound."""
    return 1e-6 if n_items < 90 else 2e-6


def compute_grad_accum(n_items: int) -> int:
    """Gradient accumulation steps: 4 for small datasets, 8 from 100 examples on."""
    return 4 if n_items < 100 else 8


def plan_training(gpu: GpuInfo, n_items: int, *, force_cpu: bool = False, language: str = "russian") -> TrainPlan:
    """Automatic configuration.

    Batch 1 + accumulation 4-8 (as in Alexandria), r=32/alpha=128, lr 1e-6..2e-6, epochs by the "examples x epochs ~ 320"
    rule, eager attention, checkpointing, bf16 on the GPU.  VRAM tiers: >= 14 GB -> 1.7B model; >= 10 GB -> 1.7B with
    8-bit Adam; >= 6 GB -> 0.6B with 8-bit Adam; less (or no CUDA, or ``force_cpu``) -> CPU with the 0.6B model.
    """
    warnings: List[str] = []
    n = max(1, n_items)
    common = dict(lora_r=LORA_R, lora_alpha=LORA_ALPHA, lr=compute_lr(n), batch_size=1,
                  grad_accum=compute_grad_accum(n), gradient_checkpointing=True, language=language,
                  max_seconds_per_item=MAX_AUDIO_SECONDS)
    if n < 20:
        warnings.append(tr("warn.few_segments"))

    if force_cpu or not gpu.available:
        warnings.append(
            tr("warn.no_gpu"))
        return TrainPlan(device="cpu", base_model=MODEL_0_6B, use_8bit_adam=False, dtype="float32",
                         epochs=compute_epochs(n, target_passes=250), warnings=warnings, **common)

    total = gpu.total_gb
    if total >= 14:            # 16 GB (RTX 4090 Mobile) and above
        model, use8 = MODEL_1_7B, False
    elif total >= 10:
        model, use8 = MODEL_1_7B, True
        warnings.append(tr("warn.low_vram"))
    elif total >= 6:
        model, use8 = MODEL_0_6B, True
        warnings.append(tr("warn.very_low_vram"))
    else:
        return plan_training(GpuInfo(False), n_items, force_cpu=True, language=language)
    return TrainPlan(device="cuda:0", base_model=model, use_8bit_adam=use8, dtype="bfloat16",
                     epochs=compute_epochs(n), warnings=warnings, **common)


def reduce_after_oom(plan: TrainPlan) -> Optional[TrainPlan]:
    """Next, more frugal plan after an out-of-memory error (None when nothing more can be reduced).

    The batch size is already 1, so the steps are: 8-bit optimizer -> the lighter 0.6B model -> (the UI offers the CPU).
    """
    if plan.device == "cpu":
        return None
    if not plan.use_8bit_adam:
        return replace(plan, use_8bit_adam=True,
                       warnings=plan.warnings + [tr("warn.oom_8bit")])
    if plan.base_model == MODEL_1_7B:
        return replace(plan, base_model=MODEL_0_6B,
                       warnings=plan.warnings + [tr("warn.oom_small")])
    return None


class VramMonitor:
    """Background VRAM sampler (every ``interval`` seconds, tracks the peak).  Does nothing without CUDA."""

    def __init__(self, interval: float = 5.0, on_sample: Optional[Callable[[float, float], None]] = None) -> None:
        """``on_sample(used_gb, total_gb)`` is called from the monitor thread for every sample."""
        self.interval = interval
        self.on_sample = on_sample
        self.peak_gb = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def snapshot() -> Optional[tuple]:
        """``(used_gb, total_gb)`` of the first CUDA device, or None."""
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            free, total = torch.cuda.mem_get_info(0)
            return ((total - free) / 1024 ** 3, total / 1024 ** 3)
        except Exception:  # noqa: BLE001
            return None

    def _run(self) -> None:
        """Thread body: sample until stopped, remembering the peak."""
        while not self._stop.wait(self.interval):
            s = self.snapshot()
            if s:
                self.peak_gb = max(self.peak_gb, s[0])
                log.info("VRAM %.1f / %.1f GB (peak %.1f)", s[0], s[1], self.peak_gb)
                if self.on_sample:
                    self.on_sample(*s)

    def start(self) -> "VramMonitor":
        """Start the daemon sampling thread; returns self for chaining."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="vram-monitor")
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop sampling and wait briefly for the thread to finish."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
