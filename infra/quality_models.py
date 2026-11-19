"""Small model of the automatic voice check that is not a Hugging Face model folder: DNSMOS P.835 (``sig_bak_ovr.onnx``).

DNSMOS (Microsoft, Reddy et al., ICASSP 2021/2022) predicts the mean opinion score of speech (signal / background / overall)
from the waveform alone; Voxprint uses the overall score (OVRL) as a "does it sound clean and natural" hint.  The file is
1.16 MB and runs on the CPU through onnxruntime (already a dependency).  Licence: the model files of microsoft/DNS-Challenge
are under CC BY 4.0 (attribution; see NOTICE), the reference code under MIT.

The file is pinned by size and SHA-256 in the shared model store (manifest row ``dnsmos``,
``dnsmos/sig_bak_ovr.onnx``) and arrives through VOXPrint MDL (:mod:`infra.mdl_models`).  It is part of the Components
"download all" step (:func:`ui.modules_dialog.default_extras`) and of the first-run model download; the voice check never
downloads by itself (no file = no MOS value).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional, cast

from infra import mdl_models

log = logging.getLogger("voxprint.quality_models")

DNSMOS_FILE = "sig_bak_ovr.onnx"
DNSMOS_META = {"size": 1157965, "sha256": "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"}
DNSMOS_LABEL = "DNSMOS P.835"
MODEL_ID = "dnsmos"


def dnsmos_path(models_dir: Optional[Path] = None) -> Path:
    """Where the DNSMOS file lives: ``<view>/dnsmos/sig_bak_ovr.onnx``."""
    if models_dir is not None:
        return Path(models_dir) / "dnsmos" / DNSMOS_FILE
    return mdl_models.view_dir(MODEL_ID) / "dnsmos" / DNSMOS_FILE


def dnsmos_ready(models_dir: Optional[Path] = None) -> bool:
    """True when the store holds a complete, verified view of the file (a stub counts as missing)."""
    if models_dir is not None:
        p = dnsmos_path(models_dir)
        try:
            return p.is_file() and p.stat().st_size == int(cast(Any, DNSMOS_META["size"]))
        except OSError:
            return False
    return mdl_models.is_ready(MODEL_ID)


def missing_bytes(models_dir: Optional[Path] = None) -> int:
    """Download size still needed (0 when the file is there)."""
    return 0 if dnsmos_ready(models_dir) else int(cast(Any, DNSMOS_META["size"]))


def _silent_progress(f: float, m: str = "") -> None:
    pass


def ensure_dnsmos(progress: Callable[[float, str], None] = _silent_progress, models_dir: Optional[Path] = None,
                  opener=None, timeout: float = 30.0) -> Path:
    """Fetch the DNSMOS file through VOXPrint MDL and return its path.

    ``progress(fraction, message)``; raises :class:`core.errors.ModelDownloadError` when the store cannot serve it."""
    del opener, timeout
    if models_dir is not None:
        return dnsmos_path(models_dir)
    view = mdl_models.ensure(MODEL_ID, progress=progress)
    path = view / "dnsmos" / DNSMOS_FILE
    log.info("DNSMOS ready at %s", path)
    return path
