"""LoRA adapter strength ("scale") at inference time (Qt-free).

The voice adapter's contribution is ``scale x (lora_alpha / r) x B @ A``.  Full strength (1.0, what every voice used before)
tends to "over-dry" a fine-tuned Qwen3-TTS voice: strained delivery, noise, a missed end-of-speech token.  Community
measurements on the 1.7B model found the best point at 0.3-0.4 (Instavar's LoRA write-up; QwenLM/Qwen3-TTS issue #343),
and Voxprint's adapters are trained with alpha / r = 4 (theirs: 2), i.e. they are "louder" by default.

* new voices get :data:`DEFAULT_SCALE` unless the quick preview or the automatic checkpoint pick chose another value;
* voices without the field (trained before) keep :data:`LEGACY_SCALE` = 1.0, so they sound exactly as before;
* the value lives in ``voice.json`` (``adapter_scale``) and can be changed in the voice's Properties.

NEEDS VALIDATION ON A GPU: the default and the candidate values are research-based, not measured on our own voices yet.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

#: Range offered in the UI (1.0 = the adapter exactly as trained).
MIN_SCALE, MAX_SCALE = 0.1, 1.0
#: Strength of a freshly trained voice when nothing else chose one (research range 0.3-0.5; see the module docstring).
DEFAULT_SCALE = 0.5
#: Voices without ``adapter_scale`` in voice.json: full strength, as before this setting existed.
LEGACY_SCALE = 1.0
#: Values the quick preview synthesizes (the user listens and picks; 1.0 keeps the old behaviour reachable).
PREVIEW_SCALES = (0.35, 0.5, 1.0)
#: Values the automatic checkpoint pick tries for every candidate epoch.
PICK_SCALES = (0.35, 0.5, 0.7, 1.0)


def clamp(value: Any, default: Optional[float] = None) -> Optional[float]:
    """``value`` as a scale in [MIN_SCALE, MAX_SCALE], rounded to 2 decimals; ``default`` if it is not a number."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:                                        # NaN
        return default
    return round(min(MAX_SCALE, max(MIN_SCALE, v)), 2)


def is_full(scale: Optional[float]) -> bool:
    """True for "no scaling" (``None`` or 1.0)."""
    return scale is None or abs(float(scale) - 1.0) < 1e-6


def apply(peft_model: Any, scale: float, adapter: Optional[str] = None) -> int:
    """Set the strength of the adapter on every LoRA layer of ``peft_model``; returns how many layers were changed.

    Uses peft's ``LoraLayer.set_scale`` (scaling = scale x alpha / r), so it works both for live inference and before
    ``merge_and_unload`` (the merged weights then carry the scaled delta).  ``adapter`` defaults to the active adapter(s).
    """
    from peft.tuners.lora import LoraLayer

    names: Iterable[str]
    if adapter:
        names = [adapter]
    else:
        active = getattr(peft_model, "active_adapters", None) or getattr(peft_model, "active_adapter", "default")
        names = [active] if isinstance(active, str) else list(active)
    n = 0
    for module in peft_model.modules():
        if isinstance(module, LoraLayer):
            for name in names:
                module.set_scale(name, float(scale))
            n += 1
    return n
