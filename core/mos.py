"""Predicted mean opinion score (MOS) of a sample with DNSMOS P.835 (Qt-free, CPU, onnxruntime).

Follows Microsoft's reference ``DNSMOS/dnsmos_local.py`` (DNS-Challenge, MIT): 16 kHz mono, 9.01 s windows with a 1 s hop
(shorter audio is repeated to fill one window), the network's raw SIG / BAK / OVRL outputs mapped through the published
third-order polynomials, then averaged over the windows.  Scores are on the 1-5 ACR scale; Voxprint uses OVRL.

DNSMOS was trained on noisy / enhanced human speech, not on TTS: treat the value as a coarse "clean and natural?" hint, and
compare voices with each other rather than against an absolute bar.  NEEDS VALIDATION ON A GPU run with real voices (the
warning threshold in :mod:`core.voice_check` is a first guess).  One CPU thread, ~1 MB model: cheap next to the TTS itself.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

log = logging.getLogger("voxprint.mos")

SAMPLE_RATE = 16000
INPUT_SECONDS = 9.01
MAX_WINDOWS = 12                 # a long sample: evenly spaced windows are enough for a hint (keeps the cost bounded)
# Mapping of the raw network outputs to MOS (non-personalised model), from dnsmos_local.py ``get_polyfit_val``
_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])


class DnsMos:
    """``DnsMos(path).score(audio, sr) -> {"ovrl", "sig", "bak"}``; ``session`` is injectable for tests."""

    def __init__(self, model_path: Optional[Path] = None, session=None, threads: int = 1) -> None:
        if session is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, int(threads))   # modest: the check runs next to the GPU work
            opts.inter_op_num_threads = 1
            session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._sess = session
        self._input = session.get_inputs()[0].name

    def score(self, audio: np.ndarray, sr: int) -> Optional[Dict[str, float]]:
        """MOS of ``audio`` (any rate, mono); ``None`` for silence / empty input."""
        from core.audio_utils import resample

        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size == 0 or not np.any(np.abs(x) > 1e-6):
            return None
        if sr != SAMPLE_RATE:
            x = resample(x, sr, SAMPLE_RATE)
        win = int(INPUT_SECONDS * SAMPLE_RATE)
        while len(x) < win:                    # as the reference: repeat a short clip until one window is full
            x = np.concatenate([x, x])
        n = int(np.floor(len(x) / SAMPLE_RATE) - INPUT_SECONDS) + 1
        starts = [i * SAMPLE_RATE for i in range(n)]
        if len(starts) > MAX_WINDOWS:
            starts = [starts[round(i * (len(starts) - 1) / (MAX_WINDOWS - 1))] for i in range(MAX_WINDOWS)]
        sig, bak, ovr = [], [], []
        for s in starts:
            seg = x[s:s + win]
            if len(seg) < win:
                continue
            raw = np.asarray(self._sess.run(None, {self._input: seg[np.newaxis, :].astype(np.float32)})[0]).reshape(-1)
            sig.append(_P_SIG(raw[0]))
            bak.append(_P_BAK(raw[1]))
            ovr.append(_P_OVR(raw[2]))
        if not ovr:
            return None
        return {"ovrl": round(float(np.mean(ovr)), 3), "sig": round(float(np.mean(sig)), 3), "bak": round(float(np.mean(bak)), 3)}


def default_mos() -> Optional[DnsMos]:
    """The scorer if the DNSMOS file is downloaded (never downloads), else ``None``; also ``None`` when switched off with
    ``VOXPRINT_NO_MOS=1`` or when onnxruntime cannot load it."""
    import os

    if os.environ.get("VOXPRINT_NO_MOS", "").strip() in ("1", "true", "yes"):
        return None
    from infra import quality_models

    if not quality_models.dnsmos_ready():
        return None
    try:
        return DnsMos(quality_models.dnsmos_path())
    except Exception as exc:  # noqa: BLE001 - a broken onnxruntime must not break the voice check
        log.warning("DNSMOS unavailable: %s", exc)
        return None
