"""Подбор параметров обучения LoRA под объём VRAM (чистая логика + необязательный мониторинг через torch)."""
from __future__ import annotations

from core.i18n import tr
import logging
import threading
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional

log = logging.getLogger("voxprint.vram")

MODEL_1_7B = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_0_6B = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


@dataclass(frozen=True)
class GpuInfo:
    available: bool
    name: str = ""
    total_gb: float = 0.0
    free_gb: float = 0.0


#: Значения по умолчанию взяты из документации и скрипта Alexandria (train_lora.py / lora.md).
LORA_R = 32
LORA_ALPHA = 128
TARGET_PASSES = 320          # правило lora.md: примеров x эпох ~ 250-400 (больше ~600 - переобучение)
LOSS_WARN_BELOW = 3.5        # lora.md: loss < ~3.5 - "каша" / нет EOS (для английского; для русского не подтверждено)
MAX_AUDIO_SECONDS = 30.0     # как --max_audio_seconds в train_lora.py


@dataclass(frozen=True)
class TrainPlan:
    device: str                    # "cuda:0" | "cpu"
    base_model: str
    batch_size: int                # всегда 1 (как в train_lora.py)
    grad_accum: int
    gradient_checkpointing: bool
    use_8bit_adam: bool
    dtype: str                     # "bfloat16" | "float32"
    lora_r: int
    lora_alpha: int
    lr: float
    epochs: int
    max_seconds_per_item: float
    language: str = "russian"      # флаг --language обучения (строчными буквами; по умолчанию у Alexandria - english)
    attn_implementation: str = "eager"   # как в train_lora.py; flash-attn на Windows не используется
    warnings: List[str] = field(default_factory=list)

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum


def detect_gpu() -> GpuInfo:
    """Определяет NVIDIA GPU через torch (ленивый импорт). Без torch/CUDA -> available=False."""
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
    """Эпохи так, чтобы (примеров x эпох) оказалось около target_passes (правило 250-400 из lora.md)."""
    return max(lo, min(hi, round(target_passes / max(1, n_items))))


def compute_lr(n_items: int) -> float:
    """lora.md: 1e-6 для ~60 примеров, 2e-6 для ~120. Для нового языка берём нижнюю границу."""
    return 1e-6 if n_items < 90 else 2e-6


def compute_grad_accum(n_items: int) -> int:
    return 4 if n_items < 100 else 8


def plan_training(gpu: GpuInfo, n_items: int, *, force_cpu: bool = False, language: str = "russian") -> TrainPlan:
    """Автоконфиг. batch 1 + накопление 4-8 (как у Alexandria), r=32/alpha=128, lr 1e-6..2e-6,
    эпохи по правилу «примеров x эпох ~ 320», eager-внимание, checkpointing, bf16 на GPU."""
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
    if total >= 14:            # 16 ГБ (RTX 4090 Mobile) и выше
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
    """Следующий, более экономный план после OOM (или None, если дальше снижать нечего).
    Размер пакета уже 1, поэтому: 8-битный оптимизатор -> облегчённая модель 0.6B -> (UI предложит CPU)."""
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
    """Фоновый мониторинг VRAM (раз в interval секунд). Без CUDA ничего не делает."""

    def __init__(self, interval: float = 5.0, on_sample: Optional[Callable[[float, float], None]] = None) -> None:
        self.interval = interval
        self.on_sample = on_sample
        self.peak_gb = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def snapshot() -> Optional[tuple]:
        """(used_gb, total_gb) или None."""
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            free, total = torch.cuda.mem_get_info(0)
            return ((total - free) / 1024 ** 3, total / 1024 ** 3)
        except Exception:  # noqa: BLE001
            return None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            s = self.snapshot()
            if s:
                self.peak_gb = max(self.peak_gb, s[0])
                log.info("VRAM %.1f / %.1f GB (peak %.1f)", s[0], s[1], self.peak_gb)
                if self.on_sample:
                    self.on_sample(*s)

    def start(self) -> "VramMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True, name="vram-monitor")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
