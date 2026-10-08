"""Small models of the automatic voice check that are not Hugging Face model folders: DNSMOS P.835 (``sig_bak_ovr.onnx``).

DNSMOS (Microsoft, Reddy et al., ICASSP 2021/2022) predicts the mean opinion score of speech (signal / background / overall)
from the waveform alone; Voxprint uses the overall score (OVRL) as a "does it sound clean and natural" hint.  The file is 1.16 MB
and runs on the CPU through onnxruntime (already a dependency).  Licence: the model files of microsoft/DNS-Challenge are under
CC BY 4.0 (attribution; see NOTICE), the reference code under MIT.

The file is pinned by size and SHA-256 and fetched from the upstream repository at a fixed commit, with a Hugging Face mirror
(same bytes, fixed revision) as the fallback.  Resumable, stdlib only, through :mod:`infra.net` (proxy / offline settings).
It is part of the Components "download all" step (:func:`ui.modules_dialog.default_extras`) and of the first-run model
download; the voice check never downloads by itself (no file = no MOS value).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from core.i18n import tr
from infra import external_models, model_release

log = logging.getLogger("voxprint.quality_models")

DNSMOS_FILE = "sig_bak_ovr.onnx"
DNSMOS_META = {"size": 1157965, "sha256": "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"}
#: Upstream at a fixed commit first, then a Hugging Face mirror at a fixed revision (byte-identical, checked 2026-10-08).
DNSMOS_URLS = (
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/DNSMOS/DNSMOS/"
    + DNSMOS_FILE,
    "https://huggingface.co/TigreGotico/dnsmos-onnx/resolve/27691a53aa069b27be6ac957013d43b3c442da9d/" + DNSMOS_FILE,
)
DNSMOS_LABEL = "DNSMOS P.835"


def dnsmos_path(models_dir: Optional[Path] = None) -> Path:
    """Where the DNSMOS file lives: ``<models folder>/dnsmos/sig_bak_ovr.onnx``."""
    if models_dir is not None:
        return Path(models_dir) / "dnsmos" / DNSMOS_FILE
    from infra import paths

    local = paths.models_dir() / "dnsmos" / DNSMOS_FILE
    try:
        if local.is_file() and local.stat().st_size == int(DNSMOS_META["size"]):
            return local
    except OSError:
        pass
    ext = external_models.verified_file("dnsmos/" + DNSMOS_FILE)
    return ext if ext is not None else local


def dnsmos_ready(models_dir: Optional[Path] = None) -> bool:
    """True when the file is present with the pinned size (the hash was checked when it was downloaded)."""
    p = dnsmos_path(models_dir)
    try:
        return p.is_file() and p.stat().st_size == DNSMOS_META["size"]
    except OSError:
        return False


def missing_bytes(models_dir: Optional[Path] = None) -> int:
    """Download size still needed (0 when the file is there)."""
    return 0 if dnsmos_ready(models_dir) else int(DNSMOS_META["size"])


def ensure_dnsmos(progress: Callable[[float, str], None] = lambda f, m="": None, models_dir: Optional[Path] = None,
                  opener=None, timeout: float = 30.0) -> Path:
    """Download the DNSMOS file if it is missing (size + SHA-256 checked, resumable) and return its path.

    ``progress(fraction, message)``; raises :class:`infra.model_release.ReleaseError` when every source failed."""
    target = dnsmos_path(models_dir)
    if dnsmos_ready(models_dir):
        return target
    opener = opener or model_release._open
    size = int(DNSMOS_META["size"])
    errors = []
    for url in DNSMOS_URLS:
        done = [0]

        def on_bytes(n: int) -> None:  # called within this iteration (late binding of ``done`` is fine)
            done[0] += n  # noqa: B023
            progress(min(0.99, done[0] / size), tr("progress.downloading", short=DNSMOS_LABEL,  # noqa: B023
                                                    pct=int(100 * min(1.0, done[0] / size))))  # noqa: B023
        try:
            model_release.fetch_file(url, target, DNSMOS_META, on_bytes, opener, timeout)
            progress(1.0, tr("progress.model_verifying", short=DNSMOS_LABEL))
            log.info("DNSMOS downloaded from %s", url)
            return target
        except model_release.ReleaseError as exc:
            log.warning("DNSMOS download from %s failed: %s", url, exc)
            errors.append(str(exc))
    raise model_release.ReleaseError("; ".join(errors))
