"""Feature flags that can be switched without code changes.

Currently one flag: **AAC / M4B export** (``aac_m4b``).  AAC is patent-encumbered and Voxprint ships no patent licence, so
the option is offered as a convenience only and can be hidden/disabled completely:

* environment variable ``VOXPRINT_ENABLE_AAC=0`` (or ``false``/``no``/``off``) - highest priority, handy for builds and CI;
* ``state/features.json`` containing ``{"aac_m4b": false}`` - per-user switch;
* otherwise :data:`AAC_DEFAULT` (``True``: the option is visible, unchecked, with a disclaimer).

A distributor who does not want the option at all can set :data:`AAC_DEFAULT` to ``False`` (or set the environment
variable in the launcher); the UI then does not show the M4B/AAC entry and the narrator refuses to produce it.
"""
from __future__ import annotations

import json
import os

from infra import paths

#: Default when neither the environment variable nor the state file says anything.
AAC_DEFAULT = True
ENV_AAC = "VOXPRINT_ENABLE_AAC"
_FALSE = {"0", "false", "no", "off", "disable", "disabled"}
_TRUE = {"1", "true", "yes", "on", "enable", "enabled"}


def aac_enabled() -> bool:
    """True if the M4B (AAC) export option may be offered."""
    env = os.environ.get(ENV_AAC, "").strip().lower()
    if env in _FALSE:
        return False
    if env in _TRUE:
        return True
    try:
        data = json.loads((paths.state_dir() / "features.json").read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("aac_m4b"), bool):
            return data["aac_m4b"]
    except (OSError, ValueError):
        pass
    return AAC_DEFAULT
