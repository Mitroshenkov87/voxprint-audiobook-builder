"""How much video memory narration may plan for (Voxprint suite rule, shared with the Movie Dubber).

The batch comes from the memory that is free *right now*, measured again before every batch group. A reserve is left
untouched so a short peak (the codec decoder, a long chunk) and anything else on the card still have room::

    reserve = max(2.0 GB, 0.08 * total)
    budget  = max(0, free_now - reserve)
    batch   = clamp(budget // per_item, 1, max_batch)

    >>> plan_batch(free_gb=11.0, total_gb=16.0, per_item_gb=0.9, max_batch=12)   # 16 GB card, model loaded
    10

``gpu.vram_fraction`` / ``VOXPRINT_VRAM_FRACTION`` is an optional extra cap (0.70-0.80 of the card's total). It is off
unless the user sets it. At least one item is always allowed. Stdlib only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

#: Never plan into the last 2 GB, and never into the last 8 % of the card (whichever is larger).
RESERVE_FLOOR_GB = 2.0
RESERVE_RATIO = 0.08
#: Kept for the optional user cap: a set fraction is clamped into this range. The cap itself defaults to off.
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


def reserve_gb(total_gb: float) -> float:
    """GB left free on a card of ``total_gb``: ``max(2 GB, 8 % of the total)``."""
    total = max(0.0, float(total_gb))
    return max(RESERVE_FLOOR_GB, RESERVE_RATIO * total)


@dataclass(frozen=True)
class VramPlan:
    """The narration batch that fits in the free video memory, and the numbers behind it."""

    batch: int
    budget_gb: float
    free_gb: float
    total_gb: float
    fraction: Optional[float]
    reserve_gb: float

    def describe(self) -> str:
        """One line naming the batch, the budget, the free memory and the reserve."""
        cap = f", cap {self.fraction:.0%}" if self.fraction else ""
        return (f"batch {self.batch} (budget {self.budget_gb:.1f} GB of {self.free_gb:.1f} GB free / "
                f"{self.total_gb:.1f} GB total, reserve {self.reserve_gb:.1f} GB{cap})")


def budget_gb(free_gb: float, total_gb: float, fraction: Optional[float] = None) -> float:
    """GB that may still be taken: free memory minus the reserve, and below ``fraction`` of the card when that cap is set."""
    free = max(0.0, float(free_gb))
    total = max(0.0, float(total_gb))
    budget = max(0.0, free - reserve_gb(total))
    if fraction is not None:
        room = free - (1.0 - clamp_fraction(fraction)) * total
        budget = max(0.0, min(budget, room))
    return budget


def plan(free_gb: float, total_gb: float, per_item_gb: float, max_batch: int,
         fraction: Optional[float] = None) -> VramPlan:
    """The batch plan for the current free / total memory (see the module docstring)."""
    free = max(0.0, float(free_gb))
    total = max(0.0, float(total_gb))
    reserve = reserve_gb(total)
    b = budget_gb(free, total, fraction)
    # Floor division in decimal GB. ``9.0 // 0.9`` is 9 with binary floats (0.9 is slightly over 9/10),
    # which would plan one item short of the gigabytes that actually fit.
    n = int(math.floor(b / per_item_gb + 1e-9)) if per_item_gb > 0 else int(max_batch)
    frac = clamp_fraction(fraction) if fraction is not None else None
    return VramPlan(max(1, min(int(max_batch), n)), b, free, total, frac, reserve)


def plan_batch(free_gb: float, total_gb: float, per_item_gb: float, max_batch: int,
               fraction: Optional[float] = None) -> int:
    """Only the batch size of :func:`plan`."""
    return plan(free_gb, total_gb, per_item_gb, max_batch, fraction).batch
