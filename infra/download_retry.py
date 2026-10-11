"""Retry a download that voxprint-common 0.1.1 would give up after one pass.

The library resumes with HTTP Range (``<file>.partial``) and tries each source
once. A dropped connection on the last source becomes ``DownloadError``. This
wrapper calls it again, keeping the partial, with a growing pause. It does not
retry a full disk or a digest mismatch. Upstream the loop when the library
grows one.

    from infra.download_retry import download_with_retry

    download_with_retry(dest, [url], sha256=digest, size=nbytes)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from voxprint_common.download import DiskSpaceError, download as library_download

log = logging.getLogger("voxprint.models")

#: Transient failures (reset, timeout) before the error is raised. Partials stay on disk.
TRIES = 5
BACKOFF = 0.5
_sleep = time.sleep

DownloadFn = Callable[..., Path]


def _permanent(exc: BaseException) -> bool:
    """A retry cannot help: the disk is full, the digest is wrong, or the server said the file is gone."""
    if isinstance(exc, DiskSpaceError) or type(exc).__name__ == "HashMismatch":
        return True
    seen: Optional[BaseException] = exc
    for _ in range(5):
        if seen is None:
            break
        code = getattr(seen, "code", None)
        if isinstance(code, int) and 400 <= code < 500 and code not in (408, 429):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def download_with_retry(destination: Path, sources: Sequence[str], *, sha256: str, size: Optional[int] = None,
                        download: Optional[DownloadFn] = None, tries: int = TRIES, **kwargs) -> Path:
    """Download ``destination``, resuming the partial and retrying a transient error ``tries`` times."""
    fetch = download or library_download
    last: Optional[BaseException] = None
    for attempt in range(max(1, tries)):
        try:
            return fetch(destination, sources, sha256=sha256, size=size, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the caller sees the last failure
            last = exc
            if _permanent(exc) or attempt + 1 >= tries:
                raise
            wait = BACKOFF * (2 ** attempt)
            log.warning("download of %s failed (%s) - retry %d of %d in %.1f s",
                        Path(destination).name, exc, attempt + 2, tries, wait)
            _sleep(wait)
    if last is not None:
        raise last
    raise RuntimeError("download was not attempted")
