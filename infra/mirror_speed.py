"""Pick the faster model mirror from a short throughput sample.

voxprint-common 0.1.1 tries the caller's URL list once and does not measure speed.
A latency probe that calls Hugging Face "slow" then stays on a ModelScope mirror
at about 84 KB/s. This module prefers Hugging Face when it is not much slower,
and treats a ModelScope feed under 100 KB/s as the slow one when Hugging Face
already delivered bytes faster. Upstream this race when the library grows one.

    from infra.mirror_speed import choose_primary

    choose_primary(500 * 1024, 84 * 1024)  # "hf"
"""
from __future__ import annotations

import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Optional, Tuple

from infra import modelscope_mirror, net

log = logging.getLogger("voxprint.models")

#: A ModelScope feed slower than this is not worth keeping when Hugging Face is faster.
SLOW_BPS = 100 * 1024
#: How long a transfer may stay under ``SLOW_BPS`` before the caller switches source.
SLOW_AFTER = 4.0
#: Seconds spent reading each host. A timeout counts as zero bytes.
PROBE_SECONDS = 0.75
PROBE_BYTES = 64 * 1024

Sample = Callable[[str], Tuple[float, float]]
_sample: Optional[Sample] = None


def choose_primary(hf_bps: float, ms_bps: float) -> Optional[str]:
    """``"hf"``, ``"ms"``, or None when both samples are empty (the caller keeps its latency probe).

    Hugging Face wins when it is at least half as fast as ModelScope, and also when
    ModelScope is under ``SLOW_BPS`` and Hugging Face is the faster of the two.
    """
    if hf_bps <= 0 and ms_bps <= 0:
        return None
    if hf_bps <= 0:
        return "ms"
    if ms_bps <= 0:
        return "hf"
    if hf_bps >= ms_bps * 0.5:
        return "hf"
    if ms_bps < SLOW_BPS and hf_bps > ms_bps:
        return "hf"
    return "ms"


def order_for(hf_bps: float, ms_bps: float) -> Optional[list]:
    """Source order, or None when there is nothing to measure."""
    chosen = choose_primary(hf_bps, ms_bps)
    if chosen is None:
        return None
    if chosen == "hf":
        return ["hf", "hfm", "ms"]
    return ["ms", "hf", "hfm"]


def too_slow(speed: float, elapsed: float, slow_bps: Optional[float], slow_after: Optional[float], grew: bool) -> bool:
    """True when a source that already sent bytes is still under ``slow_bps`` after ``slow_after`` seconds."""
    if not grew or slow_bps is None or slow_after is None or slow_bps <= 0:
        return False
    return elapsed >= slow_after and 0 < speed < slow_bps


def measure(repo_id: str) -> Tuple[float, float]:
    """``(hugging_face_bytes_per_sec, modelscope_bytes_per_sec)``. A dead host is 0."""
    fn = _sample or _sample_network
    try:
        hf_bps, ms_bps = fn(repo_id)
    except Exception:  # noqa: BLE001 - a probe must not stop the download
        log.info("mirror speed probe failed", exc_info=True)
        return 0.0, 0.0
    return max(0.0, float(hf_bps)), max(0.0, float(ms_bps))


def _sample_network(repo_id: str) -> Tuple[float, float]:
    hf_url = f"{modelscope_mirror.hf_endpoint()}/{repo_id}/resolve/main/config.json"
    ms_url = f"{modelscope_mirror.BASE_URL}/api/v1/models/{repo_id}/repo?" + urllib.parse.urlencode(
        {"Revision": modelscope_mirror.REVISION, "FilePath": "config.json"})
    box: dict = {"hf": 0.0, "ms": 0.0}

    def run(key: str, url: str) -> None:
        box[key] = _bytes_per_sec(url, PROBE_SECONDS)

    threads = [threading.Thread(target=run, args=pair, daemon=True) for pair in (("hf", hf_url), ("ms", ms_url))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(PROBE_SECONDS + 0.35)
    return box["hf"], box["ms"]


def _bytes_per_sec(url: str, timeout: float) -> float:
    req = urllib.request.Request(url, headers={"User-Agent": "Voxprint", "Range": f"bytes=0-{PROBE_BYTES - 1}"})
    started = time.monotonic()
    try:
        with net.urlopen(req, timeout) as response:  # noqa: S310 - huggingface.co or modelscope.cn only
            payload = response.read(PROBE_BYTES)
    except (OSError, urllib.error.URLError, ValueError):
        return 0.0
    elapsed = max(time.monotonic() - started, 0.05)
    return len(payload) / elapsed
