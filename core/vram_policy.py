"""How much video memory narration may plan for (Voxprint suite rule, shared with the Movie Dubber).

Rule: plan from the memory that is *actually free right now* and never let the whole card go above a fraction of its total
(default 75 %, configurable 70-80 %).  Whatever another program (the Movie Dubber, a game, the desktop) already holds counts
against that budget, so two Voxprint programs never fill the card together, and a short peak (the codec decoder, a long
chunk) still has room.

    >>> plan_batch(free_gb=14.0, total_gb=16.0, per_item_gb=0.9, max_batch=12)   # 16 GB card, 2 GB in use
    11

Budget in GB = ``min(free - (1 - fraction) * total, free - reserve)``; batch = ``budget // per_item`` clamped to
``1..max_batch``.  At least one item is always allowed (the model is already loaded; a single sequence needs very little).
Stdlib only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: Default share of the card's total memory that may be in use at the peak (models + batch + everything else on the card).
DEFAULT_FRACTION = 0.75
MIN_FRACTION = 0.70
MAX_FRACTION = 0.80


def clamp_fraction(value: object, default: float = DEFAULT_FRACTION) -> float:
    """``value`` as a fraction in ``MIN_FRACTION..MAX_FRACTION``; accepts 0.75 or 75 (percent); anything else -> ``default``."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or f <= 0:
        return default
    if f > 1.0:                       # "75" = 75 %
        f /= 100.0
    return min(MAX_FRACTION, max(MIN_FRACTION, f))


@dataclass(frozen=True)
class VramPlan:
    batch: int
    budget_gb: float
    free_gb: float
    total_gb: float
    fraction: float

    def describe(self) -> str:
        return (f"batch {self.batch} (budget {self.budget_gb:.1f} GB of {self.free_gb:.1f} GB free / {self.total_gb:.1f} GB total, "
                f"cap {self.fraction:.0%})")


def budget_gb(free_gb: float, total_gb: float, fraction: float = DEFAULT_FRACTION, reserve_gb: float = 0.0) -> float:
    """GB that may still be taken: what keeps the card at or below ``fraction`` of ``total`` and ``reserve`` below full."""
    fraction = clamp_fraction(fraction)
    free_gb, total_gb = max(0.0, float(free_gb)), max(0.0, float(total_gb))
    if total_gb <= 0:
        return max(0.0, free_gb - reserve_gb)
    return max(0.0, min(free_gb - (1.0 - fraction) * total_gb, free_gb - reserve_gb))


def plan(free_gb: float, total_gb: float, per_item_gb: float, max_batch: int, fraction: float = DEFAULT_FRACTION,
         reserve_gb: float = 0.0) -> VramPlan:
    """The batch plan for the current free / total memory (see the module docstring)."""
    fraction = clamp_fraction(fraction)
    b = budget_gb(free_gb, total_gb, fraction, reserve_gb)
    n = int(b // per_item_gb) if per_item_gb > 0 else max_batch
    return VramPlan(max(1, min(int(max_batch), n)), b, float(free_gb), float(total_gb), fraction)


def plan_batch(free_gb: float, total_gb: float, per_item_gb: float, max_batch: int, fraction: float = DEFAULT_FRACTION,
               reserve_gb: float = 0.0) -> int:
    """Only the batch size of :func:`plan`."""
    return plan(free_gb, total_gb, per_item_gb, max_batch, fraction, reserve_gb).batch
