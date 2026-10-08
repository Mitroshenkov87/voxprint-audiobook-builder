"""Multi-connection HTTP Range downloader for model weight files (every size by default).

``huggingface_hub.snapshot_download`` transfers each file on a single connection
(and with XET off - see :mod:`infra.netroute.disable_xet` - that is the only
reliable path on networks that block the xet host).  Browsers and tools like
aria2 open several ``Range`` requests in parallel against the CDN and finish
multi-GB ``*.safetensors`` much faster on a typical link.

This module is that fast path, standard library only:

* by default every file is split into up to ``CONNECTIONS`` equal Ranges (``PARALLEL_MIN`` defaults to 0);
  set ``VOXPRINT_DL_PARALLEL_MIN`` higher to keep tiny files on one connection; each part is
  written to ``<name>.incomplete.parts/<i>`` and skipped on resume when the
  size already matches; when every part is complete the parts are concatenated
  into ``<name>.incomplete`` and renamed onto the target;
* a server that ignores ``Range`` (HTTP 200 instead of 206) falls back to a
  single-stream download of the whole file;
* ``cancel`` (:class:`threading.Event`) is checked between reads so the stall
  watchdog of :mod:`infra.download_watch` can abandon a source;
* SHA-256 is the caller's job (``model_mirrors.json`` / release manifest).

Environment:

* ``VOXPRINT_DL_CONNECTIONS`` - max parallel Ranges per file (default 8);
* ``VOXPRINT_DL_PARALLEL_MIN`` - smallest file that uses more than one connection
  (default 0 = every file);
* ``VOXPRINT_NO_PARALLEL_DL=1`` - force the single-stream path everywhere.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from infra.download_watch import Stalled

log = logging.getLogger("voxprint.models")

CHUNK = 256 * 1024
DEFAULT_CONNECTIONS = 8
DEFAULT_PARALLEL_MIN = 0
ENV_DISABLE = "VOXPRINT_NO_PARALLEL_DL"
ENV_CONNECTIONS = "VOXPRINT_DL_CONNECTIONS"
ENV_PARALLEL_MIN = "VOXPRINT_DL_PARALLEL_MIN"

Opener = Callable[[urllib.request.Request, float], Any]
ProgressBytes = Callable[[int], None]


class ParallelError(OSError):
    """The multi-connection download failed; the caller may fall back or report."""


def enabled() -> bool:
    """True unless parallel downloads are switched off with ``VOXPRINT_NO_PARALLEL_DL=1``."""
    return os.environ.get(ENV_DISABLE, "").strip().lower() not in ("1", "true", "yes", "on")


def connections() -> int:
    """Max parallel Range requests per file (at least 1)."""
    try:
        n = int(os.environ.get(ENV_CONNECTIONS, str(DEFAULT_CONNECTIONS)) or DEFAULT_CONNECTIONS)
    except ValueError:
        n = DEFAULT_CONNECTIONS
    return max(1, min(32, n))


def parallel_min() -> int:
    """Byte size at/above which a file is split across several connections."""
    try:
        n = int(os.environ.get(ENV_PARALLEL_MIN, str(DEFAULT_PARALLEL_MIN)) or DEFAULT_PARALLEL_MIN)
    except ValueError:
        n = DEFAULT_PARALLEL_MIN
    return max(0, n)


def _open(req: urllib.request.Request, timeout: float):
    from infra import net

    return net.urlopen(req, timeout)  # noqa: S310 - https to HF / ModelScope / mirrors only


def _ua() -> Dict[str, str]:
    return {"User-Agent": "Voxprint"}


def hf_file_url(repo_id: str, filename: str, revision: Optional[str] = None,
                endpoint: Optional[str] = None) -> str:
    """HTTPS URL of one file of a Hugging Face (or ``HF_ENDPOINT``) repository."""
    from infra import modelscope_mirror

    base = (endpoint or modelscope_mirror.hf_endpoint()).rstrip("/")
    rev = revision or "main"
    return f"{base}/{repo_id}/resolve/{rev}/{urllib.parse.quote(filename)}"


def _safe_rel(rel: str) -> bool:
    parts = rel.replace("\\", "/").split("/")
    return bool(rel) and not rel.startswith(("/", "\\")) and ".." not in parts and ":" not in parts[0]


def _parts_dir(target: Path) -> Path:
    return target.with_name(target.name + ".incomplete.parts")


def _incomplete(target: Path) -> Path:
    return target.with_name(target.name + ".incomplete")


def _plan_ranges(size: int, n: int) -> List[Tuple[int, int]]:
    """``(start, end_inclusive)`` byte ranges covering ``0..size-1`` with about equal length."""
    n = max(1, min(n, size))
    out: List[Tuple[int, int]] = []
    for i in range(n):
        start = (size * i) // n
        end = (size * (i + 1)) // n - 1
        if end >= start:
            out.append((start, end))
    return out


def _check_cancel(cancel: Optional[threading.Event]) -> None:
    if cancel is not None and cancel.is_set():
        raise Stalled("download abandoned (no data)")


#: Windows "file in use" (32) / "access denied" (5): another handle (an abandoned download thread, an antivirus scan)
#: still holds the file for a moment.  Retried ``LOCK_RETRIES`` times, ``LOCK_PAUSE`` seconds apart.
LOCK_RETRIES = 5
LOCK_PAUSE = 0.5


def retry_locked(fn: Callable[[], Any], what: str = "") -> Any:
    """Run ``fn`` again a few times while Windows reports the file as locked (WinError 32 / 5); other errors pass."""
    for attempt in range(LOCK_RETRIES + 1):
        try:
            return fn()
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32) or attempt >= LOCK_RETRIES:
                raise
            log.info("%s is locked (%s) - retrying", what or "file", exc)
            time.sleep(LOCK_PAUSE)
    return None  # pragma: no cover - the loop returns or raises


def open_retry(path: Path, mode: str):
    """``open(path, mode)`` with :func:`retry_locked`."""
    return retry_locked(lambda: open(path, mode), Path(path).name)


def replace_retry(src: Path, dst: Path) -> None:
    """``os.replace(src, dst)`` with :func:`retry_locked`."""
    retry_locked(lambda: os.replace(src, dst), Path(src).name)


def _read_into(resp: Any, fh: Any, expected: Optional[int], on_bytes: ProgressBytes,
               cancel: Optional[threading.Event]) -> int:
    """Copy from an HTTP response into an open file; return bytes written."""
    written = 0
    while True:
        _check_cancel(cancel)
        chunk = resp.read(CHUNK)
        if not chunk:
            break
        fh.write(chunk)
        written += len(chunk)
        on_bytes(len(chunk))
        if expected is not None and written > expected:
            raise ParallelError(f"received more than the requested range ({written} > {expected})")
    return written


def _single_stream(url: str, part: Path, size: int, have: int, opener: Opener, timeout: float,
                   on_bytes: ProgressBytes, cancel: Optional[threading.Event]) -> None:
    """One-connection resume download into ``part`` (``.incomplete``); raises ParallelError."""
    headers = _ua()
    if have:
        headers["Range"] = f"bytes={have}-"
        on_bytes(have)
    try:
        with opener(urllib.request.Request(url, headers=headers), timeout) as r:
            status = getattr(r, "status", 200)
            if have and status != 206:
                have = 0
                part.unlink(missing_ok=True)
            mode = "ab" if have else "wb"
            with open_retry(part, mode) as f:
                expect = None if size <= 0 else max(0, size - have)
                _read_into(r, f, expect, on_bytes, cancel)
    except Stalled:
        raise                           # abandoned by the watchdog: no fallback, the caller switches the source
    except (OSError, urllib.error.URLError) as exc:
        raise ParallelError(f"{url}: {exc}") from exc
    got = part.stat().st_size if part.exists() else 0
    if size > 0 and got != size:
        raise ParallelError(f"incomplete ({got} of {size} bytes)")


def _fetch_part(url: str, part_path: Path, start: int, end: int, opener: Opener, timeout: float,
                on_bytes: ProgressBytes, cancel: Optional[threading.Event]) -> None:
    """Download one inclusive byte Range into ``part_path`` (resume if a shorter part exists)."""
    need = end - start + 1
    have = part_path.stat().st_size if part_path.exists() else 0
    if have == need:
        on_bytes(have)
        return
    if have > need:
        part_path.unlink()
        have = 0
    headers = _ua()
    # resume inside the part: ask for the remaining absolute range of the whole file
    abs_start = start + have
    headers["Range"] = f"bytes={abs_start}-{end}"
    if have:
        on_bytes(have)
    try:
        with opener(urllib.request.Request(url, headers=headers), timeout) as r:
            status = getattr(r, "status", 200)
            if status == 200:
                # server ignored Range: cannot do multi-conn for this URL
                raise ParallelError("server ignored Range (HTTP 200)")
            if status != 206:
                raise ParallelError(f"unexpected HTTP status {status} for Range")
            with open_retry(part_path, "ab" if have else "wb") as f:
                _read_into(r, f, need - have, on_bytes, cancel)
    except (ParallelError, Stalled):
        raise
    except (OSError, urllib.error.URLError) as exc:
        raise ParallelError(f"{url} [{start}-{end}]: {exc}") from exc
    got = part_path.stat().st_size if part_path.exists() else 0
    if got != need:
        raise ParallelError(f"part incomplete ({got} of {need} bytes at {start}-{end})")


def _assemble(parts_dir: Path, part_paths: List[Path], incomplete: Path) -> None:
    """Concatenate part files into ``incomplete`` (atomic replace of a temp name)."""
    tmp = incomplete.with_name(incomplete.name + ".assembling")
    try:
        with open_retry(tmp, "wb") as out:
            for p in part_paths:
                with open(p, "rb") as src:
                    shutil.copyfileobj(src, out, length=1 << 20)
        replace_retry(tmp, incomplete)
    finally:
        tmp.unlink(missing_ok=True)


def download_file(url: str, target: Path, size: int, *, opener: Optional[Opener] = None,
                  on_bytes: ProgressBytes = lambda n: None, cancel: Optional[threading.Event] = None,
                  timeout: float = 30.0, conn: Optional[int] = None) -> None:
    """Download ``url`` into ``target`` (resumable).  Multi-connection when ``size >= parallel_min()``.

    ``on_bytes(n)`` is called with newly counted bytes (including already-on-disk bytes once at
    the start of a resumed part).  Existing ``target`` with the right ``size`` is left alone.
    """
    opener = opener or _open
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if size > 0 and target.is_file() and target.stat().st_size == size:
        on_bytes(size)
        return
    if target.exists():
        target.unlink()

    n_conn = conn if conn is not None else connections()
    use_parallel = enabled() and size >= parallel_min() and n_conn > 1 and size > 0
    incomplete = _incomplete(target)
    parts_dir = _parts_dir(target)

    if not use_parallel:
        shutil.rmtree(parts_dir, ignore_errors=True)
        have = incomplete.stat().st_size if incomplete.exists() else 0
        if size > 0 and have > size:
            incomplete.unlink()
            have = 0
        if have < (size or 1) or size <= 0:
            _single_stream(url, incomplete, size, have, opener, timeout, on_bytes, cancel)
        if size > 0 and incomplete.stat().st_size != size:
            raise ParallelError(f"{target.name}: incomplete ({incomplete.stat().st_size} of {size} bytes)")
        replace_retry(incomplete, target)
        return

    ranges = _plan_ranges(size, n_conn)
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_paths = [parts_dir / f"{i:04d}" for i in range(len(ranges))]
    # drop a leftover single-stream incomplete; parts are the source of truth now
    incomplete.unlink(missing_ok=True)

    lock = threading.Lock()

    def counted(n: int) -> None:
        with lock:
            on_bytes(n)

    try:
        with ThreadPoolExecutor(max_workers=len(ranges), thread_name_prefix="vx-dl") as pool:
            futs = [pool.submit(_fetch_part, url, part_paths[i], ranges[i][0], ranges[i][1],
                                opener, timeout, counted, cancel) for i in range(len(ranges))]
            for fut in as_completed(futs):
                fut.result()
    except ParallelError as exc:
        if "ignored Range" in str(exc):
            log.info("CDN ignored Range for %s - falling back to a single connection", target.name)
            shutil.rmtree(parts_dir, ignore_errors=True)
            _single_stream(url, incomplete, size, 0, opener, timeout, on_bytes, cancel)
            if incomplete.stat().st_size != size:
                raise ParallelError(f"{target.name}: incomplete after single-stream fallback") from exc
            replace_retry(incomplete, target)
            return
        raise

    _assemble(parts_dir, part_paths, incomplete)
    if incomplete.stat().st_size != size:
        raise ParallelError(f"{target.name}: assembled size {incomplete.stat().st_size} != {size}")
    replace_retry(incomplete, target)
    shutil.rmtree(parts_dir, ignore_errors=True)


def download_listed(files: Dict[str, int], dest: Path, url_for: Callable[[str], str], *,
                    opener: Optional[Opener] = None, progress: Callable[[float], None] = lambda f: None,
                    cancel: Optional[threading.Event] = None, timeout: float = 30.0,
                    patterns: Optional[Iterable[str]] = None) -> None:
    """Download every ``{relative_path: size}`` into ``dest`` (resuming).  ``url_for(name)`` builds the URL.

    ``progress`` receives an overall 0..1 fraction.  Glob ``patterns`` (like HF ``allow_patterns``)
    select a subset.  Raises :class:`ParallelError` on the first failed file (partials stay for resume).
    """
    import fnmatch

    opener = opener or _open
    selected = dict(files)
    pats = list(patterns or ())
    if pats:
        selected = {n: s for n, s in selected.items() if any(fnmatch.fnmatchcase(n, p) for p in pats)}
        if not selected:
            raise ParallelError(f"no file matches {', '.join(pats)}")
    total = sum(int(s) for s in selected.values()) or 1
    done = 0
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, size in sorted(selected.items()):
        if not _safe_rel(name):
            raise ParallelError(f"unsafe path {name!r}")
        target = dest.joinpath(*name.split("/"))
        size = int(size)
        if target.is_file() and (size <= 0 or target.stat().st_size == size):
            done += size if size > 0 else target.stat().st_size
            progress(min(1.0, done / total))
            continue
        base = done
        lock = threading.Lock()

        def on_bytes(n: int, _b=[0]) -> None:  # noqa: B006 - per-file counter
            with lock:  # noqa: B023 - called within this iteration
                _b[0] += n
                progress(min(1.0, (base + _b[0]) / total))  # noqa: B023

        download_file(url_for(name), target, size, opener=opener, on_bytes=on_bytes,
                      cancel=cancel, timeout=timeout)
        done += size if size > 0 else target.stat().st_size
        progress(min(1.0, done / total))
