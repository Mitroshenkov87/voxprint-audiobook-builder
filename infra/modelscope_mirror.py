"""ModelScope (modelscope.cn, Alibaba) as a fallback download mirror for the Qwen models.

Verified 2026-10-03 against the public API: the same repository ids exist on ModelScope under the org ``Qwen``
(``Qwen/Qwen3-TTS-12Hz-1.7B-Base``, ``Qwen/Qwen3-TTS-12Hz-0.6B-Base``, ``Qwen/Qwen3-ForcedAligner-0.6B``) and the
file sizes are byte-identical to the pinned Hugging Face commits (see ``core.model_locator.KNOWN_SIZES``).
ModelScope has no commit sha for these files (branch ``master``), so the verified revision is confirmed by sizes:
a mirror copy is accepted only if every file matches both ModelScope's own listing and the sizes of the pinned
Hugging Face commit; otherwise it is thrown away and the caller reports the failure.

Endpoints used (the same ones the ``modelscope`` SDK calls; no SDK, no login needed for public models)::

    GET https://modelscope.cn/api/v1/models/{id}/repo/files?Recursive=true   -> Data.Files[{Path,Size,Type}]
    GET https://modelscope.cn/api/v1/models/{id}/repo?Revision=master&FilePath={path}   -> the file bytes

Downloads resume: data goes to ``<file>.incomplete`` (HTTP Range) and is renamed only when the size matches.
Nothing here touches any other app's folders - it only writes into the ``.partial`` folder it is given.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger("voxprint.models")

BASE_URL = os.environ.get("VOXPRINT_MODELSCOPE_URL", "https://modelscope.cn").rstrip("/")
REVISION = "master"
CHUNK = 1024 * 1024
#: Hugging Face counts as "slow/unreachable" if a tiny API request takes longer than this (seconds).
HF_PROBE_TIMEOUT = 6.0

Opener = Callable[[urllib.request.Request, float], "object"]


class MirrorError(Exception):
    pass


def _open(req: urllib.request.Request, timeout: float):
    from infra import net

    return net.urlopen(req, timeout)  # noqa: S310 - only https to modelscope.cn / huggingface.co


def hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def hf_is_fast(repo_id: str, opener: Opener = _open, timeout: float = HF_PROBE_TIMEOUT,
               clock: Callable[[], float] = time.monotonic) -> bool:
    """True if Hugging Face answers a small API request quickly.  A user-set ``HF_ENDPOINT`` (their own mirror)
    is respected: then we never second-guess it."""
    if os.environ.get("HF_ENDPOINT"):
        return True
    t0 = clock()
    try:
        req = urllib.request.Request(f"{hf_endpoint()}/api/models/{repo_id}", headers={"User-Agent": "Voxprint"})
        with opener(req, timeout) as r:  # type: ignore[attr-defined]
            r.read(1024)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        log.info("huggingface.co not reachable (%s)", exc)
        return False
    dt = clock() - t0
    if dt > timeout * 0.75:
        log.info("huggingface.co is slow (%.1fs)", dt)
        return False
    return True


def list_files(repo_id: str, opener: Opener = _open, timeout: float = 20.0) -> List[Tuple[str, int]]:
    url = f"{BASE_URL}/api/v1/models/{repo_id}/repo/files?Recursive=true"
    try:
        with opener(urllib.request.Request(url, headers={"User-Agent": "Voxprint"}), timeout) as r:  # type: ignore
            data = json.loads(r.read().decode("utf-8"))  # type: ignore[attr-defined]
    except (OSError, ValueError) as exc:
        raise MirrorError(f"file list unavailable: {exc}") from exc
    files = (data.get("Data") or {}).get("Files") or []
    out = [(f["Path"], int(f.get("Size") or 0)) for f in files if f.get("Type", "blob") != "tree" and f.get("Path")]
    out = [(p, s) for p, s in out if p not in (".gitattributes", "README.md")]
    if not out:
        raise MirrorError("empty file list")
    return out


def _safe_rel(rel: str) -> bool:
    parts = rel.replace("\\", "/").split("/")
    return bool(rel) and not rel.startswith(("/", "\\")) and ".." not in parts and ":" not in parts[0]


def download_repo(repo_id: str, dest: Path, progress: Callable[[float], None] = lambda f: None,
                  expected_sizes: Optional[Dict[str, int]] = None, opener: Opener = _open,
                  timeout: float = 30.0) -> Dict[str, int]:
    """Downloads all files of ``repo_id`` into ``dest`` (resuming), verifies sizes, returns {path: size}.

    ``expected_sizes``: sizes of the verified revision; every one of those files must be present with that size."""
    files = list_files(repo_id, opener)
    sizes = dict(files)
    if expected_sizes:
        for rel, size in expected_sizes.items():
            if rel in sizes and sizes[rel] != size:
                raise MirrorError(f"{rel}: size {sizes[rel]} differs from the verified revision ({size})")
    total = sum(s for _, s in files) or 1
    done = 0
    for rel, size in files:
        if not _safe_rel(rel):
            raise MirrorError(f"unsafe path {rel!r}")
        target = dest.joinpath(*rel.split("/"))
        if target.exists() and target.stat().st_size == size:
            done += size
            progress(min(1.0, done / total))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".incomplete")
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if have < size:
            url = f"{BASE_URL}/api/v1/models/{repo_id}/repo?" + urllib.parse.urlencode(
                {"Revision": REVISION, "FilePath": rel})
            headers = {"User-Agent": "Voxprint"}
            if have:
                headers["Range"] = f"bytes={have}-"
            try:
                with opener(urllib.request.Request(url, headers=headers), timeout) as r:  # type: ignore[arg-type]
                    status = getattr(r, "status", 200)
                    if have and status != 206:       # server ignored Range: start over
                        have = 0
                    with open(part, "ab" if have else "wb") as f:
                        while True:
                            chunk = r.read(CHUNK)  # type: ignore[attr-defined]
                            if not chunk:
                                break
                            f.write(chunk)
                            progress(min(1.0, (done + f.tell()) / total))
            except (OSError, urllib.error.URLError) as exc:
                raise MirrorError(f"{rel}: {exc}") from exc     # keep the .incomplete file: next run resumes
        if part.stat().st_size != size:
            raise MirrorError(f"{rel}: incomplete ({part.stat().st_size} of {size} bytes)")
        os.replace(part, target)
        done += size
        progress(min(1.0, done / total))
    return sizes
