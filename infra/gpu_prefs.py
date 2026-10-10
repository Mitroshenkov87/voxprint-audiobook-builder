"""GPU settings of this program (``state/gpu.json``): the optional VRAM cap and the optional fast decode path.

* ``vram_fraction`` - optional extra cap on the card (0.70-0.80 of the total). Off unless set
  (:mod:`core.vram_policy` plans from free memory minus a reserve either way). Environment: ``VOXPRINT_VRAM_FRACTION``
  (``0.75``, ``75``, or ``off``).
* ``fast_decode`` - ``off`` (default: batched generation) or ``graphs`` (one chunk at a time through faster-qwen3-tts with
  CUDA Graphs; :mod:`core.fast_decode`).  Environment: ``VOXPRINT_FAST_DECODE``.  Off until it is benchmarked on a real PC
  (``voxprint bench``).

Which GPU to use is a suite-wide setting (``state/suite.json``, :mod:`infra.suite_settings`), not stored here.
Unknown keys are kept; nothing here raises on a damaged file.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from core import vram_policy
from infra import paths

log = logging.getLogger("voxprint.gpu_prefs")

FAST_DECODE_MODES = ("off", "graphs")
ENV_FRACTION = "VOXPRINT_VRAM_FRACTION"
ENV_FAST_DECODE = "VOXPRINT_FAST_DECODE"
_ALIASES = {"0": "off", "false": "off", "no": "off", "none": "off", "batched": "off",
            "1": "graphs", "on": "graphs", "true": "graphs", "yes": "graphs", "cuda_graphs": "graphs", "cuda-graphs": "graphs"}


def _file():
    return paths.state_dir() / "gpu.json"


def _read() -> Dict[str, Any]:
    try:
        data = json.loads(_file().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(key: str, value: Any) -> None:
    data = _read()
    if value is None:
        data.pop(key, None)
    else:
        data[key] = value
    f = _file()
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, f)


def normalize_mode(value: object) -> str:
    """``off`` / ``graphs`` (aliases such as ``on``, ``cuda_graphs``, ``batched``); ``ValueError`` otherwise."""
    v = str(value).strip().lower()
    v = _ALIASES.get(v, v)
    if v not in FAST_DECODE_MODES:
        raise ValueError(value)
    return v


_OFF = ("off", "none", "false", "no")


def vram_fraction() -> Optional[float]:
    """The optional user cap, or None when it is off (the default, and ``off`` in the environment or the file)."""
    env = os.environ.get(ENV_FRACTION, "").strip()
    if env:
        if env.lower() in _OFF:
            return None
        return vram_policy.clamp_fraction(env)
    raw = _read().get("vram_fraction", None)
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in ("", *_OFF)):
        return None
    return vram_policy.clamp_fraction(raw)


def set_vram_fraction(value: object) -> Optional[float]:
    """Store the cap (clamped to 70-80 %), or clear it with ``off``. ``ValueError`` when ``value`` is not a number or ``off``."""
    if isinstance(value, str) and value.strip().lower() in _OFF:
        _write("vram_fraction", None)
        return None
    try:
        float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(value) from exc
    f = vram_policy.clamp_fraction(value)
    _write("vram_fraction", f)
    return f


def fast_decode() -> str:
    """The fast-decode mode from the environment or the saved setting (``off`` when it is unknown)."""
    for raw in (os.environ.get(ENV_FAST_DECODE, ""), _read().get("fast_decode", "off")):
        if str(raw).strip():
            try:
                return normalize_mode(raw)
            except ValueError:
                log.warning("unknown fast decode mode %r; using off", raw)
                return "off"
    return "off"


def set_fast_decode(value: object) -> str:
    """Save ``value`` as the fast-decode mode and return the stored name."""
    mode = normalize_mode(value)
    _write("fast_decode", mode)
    return mode
