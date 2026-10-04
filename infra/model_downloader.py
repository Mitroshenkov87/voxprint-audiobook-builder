"""Automatic download of the models from Hugging Face (with a ModelScope fallback) into the app's model folder.

A download goes into ``<name>.partial``; only after verification is the folder renamed, so a partially downloaded
model is never considered ready, and the next run resumes from the ``.partial`` data.  The commit sha of the
downloaded revision is stored in the ``.revision`` file.  Copies that other programs already downloaded (HF cache,
Pinokio/Alexandria, ModelScope ...) are reused in place, read-only - see :mod:`core.model_locator`.
"""
from __future__ import annotations

from core.i18n import tr
import fnmatch
import gc
import logging
import os
import shutil
import stat
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.errors import ModelDownloadError
from core.events import ProgressCallback, Stage, noop_progress
from infra import download_watch, model_mirrors, modelscope_mirror, netroute, paths

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
netroute.disable_xet()      # plain HTTP: the xet/CAS transfer path stalls on networks that block its host (VOXPRINT_ALLOW_XET=1 keeps it)
log = logging.getLogger("voxprint.models")

ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"
ASR_REPO = "Qwen/Qwen3-ASR-0.6B"   # speech recognition for the no-transcript mode (downloaded on first use)
#: Approximate download size in GB (for the free-disk-space check; an estimate, not an exact value).
APPROX_SIZE_GB = {
    ALIGNER_REPO: 2.0,
    ASR_REPO: 1.9,
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": 4.5,
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": 2.5,
    "ai-forever/sage-fredt5-distilled-95m": 0.5,         # optional text clean-up model (see infra/text_models.py)
    "Helsinki-NLP/opus-mt-ru-en": 0.4,                    # optional translation models (Opus-MT, one per direction)
    "Helsinki-NLP/opus-mt-en-ru": 0.4,
    "Helsinki-NLP/opus-mt-de-en": 0.4,
    "Helsinki-NLP/opus-mt-en-de": 0.4,
}


def hf_url(repo_id: str) -> str:
    """Web page of the repository on Hugging Face (shown to the user in error messages)."""
    return f"https://huggingface.co/{repo_id}"


def local_dir_for(repo_id: str, root: Optional[Path] = None) -> Path:
    """Folder of a model inside the models directory: ``Qwen/X`` -> ``Qwen--X``."""
    return (root or paths.models_dir()) / repo_id.replace("/", "--")


def verify_local_model(path: Path) -> bool:
    """A model counts as ready when it has ``config.json`` and at least one weights file."""
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    return has_weights


def pinned_revision(repo_id: str) -> Optional[str]:
    """Commit pinned for this model in the bundled «verified by Voxprint» manifest (None if not pinned)."""
    try:
        from infra.verified_manifest import load_bundled

        return load_bundled().models.get(repo_id)
    except Exception:  # noqa: BLE001 - the manifest is optional
        return None


#: Set to 1 to disable the ModelScope fallback mirror.
ENV_NO_MIRROR = "VOXPRINT_NO_MIRROR"


def mirror_enabled() -> bool:
    """True unless the ModelScope mirror is disabled with ``VOXPRINT_NO_MIRROR=1``."""
    return os.environ.get(ENV_NO_MIRROR, "").strip().lower() not in ("1", "true", "yes", "on")


def _default_mirror_download(repo_id, dest, progress, expected_sizes):
    """Default mirror downloader (injectable in tests): ModelScope with size verification."""
    return modelscope_mirror.download_repo(repo_id, dest, progress, expected_sizes)


def external_model(repo_id: str, revision: Optional[str] = None):
    """A copy of the model downloaded by another program (read-only), or None.

    See :mod:`core.model_locator`: file completeness and the pinned revision are checked; nothing is copied or modified.
    Voxprint NEVER updates or deletes such a folder.
    """
    from core import model_locator

    try:
        return model_locator.find_model(repo_id, revision if revision else pinned_revision(repo_id))
    except Exception as exc:  # noqa: BLE001 - looking for foreign copies must never break a normal download
        log.warning("external model search failed for %s: %s", repo_id, exc)
        return None


def _import_existing(repo_id: str, revision: Optional[str], progress: ProgressCallback, stage: Stage, short: str) -> Optional[Path]:
    """Import the model from the user's "existing models folder" (:mod:`infra.existing_models`) into the own models folder.

    Link or copy, verified by hash, before anything is downloaded.  Returns the new folder, or None if no such folder is
    configured, it holds no usable copy or the import failed (the normal path - in-place reuse, then download - follows)."""
    from core.errors import CancelledByUser
    from infra import existing_models

    try:
        found = existing_models.find(repo_id, revision if revision else pinned_revision(repo_id))
        if found is None:
            return None
        progress(stage, 0.0, tr("progress.model_importing", short=short, where=str(found.location)))
        path = existing_models.import_model(found, lambda f, m="": progress(stage, f, m))
        progress(stage, 1.0, tr("progress.model_imported", short=short))
        return path
    except CancelledByUser:
        raise
    except Exception as exc:  # noqa: BLE001 - a failed import must never block the download path
        log.warning("importing %s from the existing models folder failed: %s", repo_id, exc)
        return None


#: Model states reported to the UI (see :func:`model_state`).
STATE_MISSING, STATE_PARTIAL, STATE_READY = "missing", "partial", "ready"


def partial_dir_has_data(repo_id: str, root: Optional[Path] = None) -> bool:
    """True if an interrupted download (``.partial`` folder with content) exists and can be resumed."""
    p = local_dir_for(repo_id, root)
    p = p.with_name(p.name + ".partial")
    try:
        return p.is_dir() and any(p.iterdir())
    except OSError:
        return False


def model_state(repo_id: str) -> str:
    """ready: usable now (Voxprint's own folder or another app's copy); partial: an interrupted download that will
    be resumed (ours, or an unfinished one in another app's cache - which we never touch); missing: nothing yet."""
    if verify_local_model(local_dir_for(repo_id)) or external_model(repo_id) is not None:
        return STATE_READY
    try:
        from infra import existing_models

        if existing_models.find(repo_id, pinned_revision(repo_id)) is not None:
            return STATE_READY
    except Exception:  # noqa: BLE001
        pass
    if partial_dir_has_data(repo_id):
        return STATE_PARTIAL
    try:
        from core import model_locator

        if model_locator.has_unfinished_download(repo_id):
            return STATE_PARTIAL
    except Exception:  # noqa: BLE001
        pass
    return STATE_MISSING


def model_states(repos) -> Dict[str, str]:
    """``{repo_id: state}`` for several repositories (see :func:`model_state`)."""
    return {r: model_state(r) for r in repos}


def local_revision(repo_id: str, root: Optional[Path] = None) -> Optional[str]:
    """Commit sha recorded in the model's ``.revision`` file, or None."""
    f = local_dir_for(repo_id, root) / ".revision"
    try:
        return f.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def local_revisions(repos, root: Optional[Path] = None) -> Dict[str, str]:
    """``{repo_id: sha}`` for the repositories that are complete locally and have a recorded revision."""
    out = {}
    for r in repos:
        d = local_dir_for(r, root)
        rev = local_revision(r, root)
        if rev and verify_local_model(d):
            out[r] = rev
    return out


class _ByteProgress:
    """Sums the progress of all per-file byte progress bars of huggingface_hub into one 0..1 value."""

    def __init__(self, cb: Callable[[float], None]) -> None:
        """``cb(fraction)`` is called whenever the overall progress changes."""
        self.cb = cb
        self.lock = threading.Lock()
        self.total = 0
        self.done = 0
        self.cancel: Optional[threading.Event] = None      # set by the stall watchdog: the next update aborts the download

    def make_tqdm_class(self):
        """Build a tqdm subclass that reports byte progress to this tracker (non-byte bars are disabled)."""
        from huggingface_hub.utils import tqdm as hf_tqdm

        tracker = self

        class _Tqdm(hf_tqdm):  # type: ignore[misc, valid-type]
            """tqdm replacement that feeds the shared tracker."""
            def __init__(self, *a: Any, **kw: Any) -> None:
                """Register this bar's total size with the tracker (only for byte bars)."""
                kw["disable"] = True if kw.get("unit") != "B" else kw.get("disable", False)
                super().__init__(*a, **kw)
                self._is_bytes = kw.get("unit") == "B"
                if self._is_bytes and self.total:
                    with tracker.lock:
                        tracker.total += int(self.total)

            def update(self, n: float = 1) -> Optional[bool]:
                """Count the transferred bytes and report the overall fraction."""
                r = super().update(n)
                if tracker.cancel is not None and tracker.cancel.is_set():
                    raise download_watch.Stalled("download abandoned (no data)")
                if getattr(self, "_is_bytes", False):
                    with tracker.lock:
                        tracker.done += int(n)
                        if tracker.total:
                            tracker.cb(min(1.0, tracker.done / tracker.total))
                return r

        return _Tqdm


def _prepare_network() -> None:
    """Pick a network route that reaches Hugging Face (VPN / odd adapters: see :mod:`infra.netroute`); never fails."""
    try:
        from infra import netroute

        netroute.prepare_hf()
    except Exception as exc:  # noqa: BLE001
        log.debug("network route preparation skipped: %s", exc)


# ------------------------------------------------------------------------------------------------ finished downloads
#: Renaming ``<name>.partial`` -> ``<name>`` can fail on Windows while an antivirus scanner or a stale process still holds a
#: handle inside the folder: try again for about half a minute, then copy instead.
FINALIZE_ATTEMPTS = 10
_sleep = time.sleep


def _expected_sizes(repo_id: str, revision: Optional[str], mirror_manifest: Optional[Path],
                    patterns: Optional[List[str]]) -> Optional[Dict[str, int]]:
    """``{file: size}`` of the pinned revision (core.model_locator.KNOWN_SIZES or the backup-mirror manifest), if known."""
    from core import model_locator

    sizes: Optional[Dict[str, int]] = None
    known = model_locator.KNOWN_SIZES.get(repo_id)
    if known and (revision is None or known[0] == revision):
        sizes = dict(known[1])
    else:
        entry = model_mirrors.entry_for(repo_id, mirror_manifest)
        if entry is not None and revision and entry.source_revision == revision:
            try:
                sizes = {n: int(m["size"]) for n, m in entry.downloadable(patterns).items()}
            except Exception:  # noqa: BLE001
                sizes = None
    if sizes and patterns:
        sizes = {n: v for n, v in sizes.items() if any(fnmatch.fnmatchcase(n, p) for p in patterns)}
    return sizes or None


def partial_is_complete(partial: Path, repo_id: str, revision: Optional[str], mirror_manifest: Optional[Path] = None,
                        patterns: Optional[List[str]] = None) -> bool:
    """True if ``partial`` already holds every file of the pinned revision with the right size and no unfinished leftovers
    (nothing is left to download).  Unknown sizes = cannot be proven = False."""
    from core import model_locator

    if not verify_local_model(partial):
        return False
    expected = _expected_sizes(repo_id, revision, mirror_manifest, patterns)
    if not expected:
        return False
    for rel, size in expected.items():
        if rel in model_locator._OPTIONAL_FILES:
            continue
        try:
            if (partial / Path(*rel.split("/"))).stat().st_size != size:
                return False
        except OSError:
            return False
    for root, _dirs, files in os.walk(partial):
        if any(f.endswith(".incomplete") for f in files):
            return False
    return True


def _rmtree(folder: Path) -> None:
    """Remove a folder tree, clearing read-only attributes (Windows) on the way; never raises."""
    if not folder.exists():
        return
    shutil.rmtree(folder, ignore_errors=True)
    if folder.exists():
        _make_writable(folder)
        shutil.rmtree(folder, ignore_errors=True)


def _make_writable(folder: Path) -> None:
    for root, dirs, files in os.walk(folder):
        for n in dirs + files:
            try:
                full = os.path.join(root, n)
                os.chmod(full, os.stat(full).st_mode | stat.S_IWRITE | stat.S_IREAD)
            except OSError:
                pass


def _holders(folder: Path) -> str:
    """Names/PIDs of processes that have a file open inside ``folder`` (needs psutil; best effort, for the log)."""
    try:
        import psutil

        out = []
        prefix = str(folder).lower()
        for p in psutil.process_iter(["pid", "name"]):
            try:
                if any(str(f.path).lower().startswith(prefix) for f in p.open_files()):
                    out.append(f"{p.info['name']}({p.info['pid']})")
            except Exception:  # noqa: BLE001 - access denied, process gone
                continue
        return ", ".join(out) or "none found"
    except Exception:  # noqa: BLE001
        return "unknown"


def finalize_download(partial: Path, target: Path) -> None:
    """Make the finished ``.partial`` folder the model folder.  Closes our own handles first, drops the hub's ``.cache``
    (lock files), then renames with retries and backoff; if Windows keeps refusing, copies the tree instead.  Raises
    ``OSError`` (with the reason of the last failure) only if everything failed."""
    gc.collect()                                              # closes file objects of finished downloads (sessions, locks)
    _rmtree(partial / ".cache")
    _rmtree(target)
    last: Optional[BaseException] = None
    for attempt in range(1, FINALIZE_ATTEMPTS + 1):
        try:
            partial.rename(target)
            if attempt > 1:
                log.info("model folder renamed on attempt %d", attempt)
            return
        except OSError as exc:
            last = exc
            log.warning("renaming %s -> %s failed (attempt %d of %d): %s", partial.name, target.name, attempt,
                        FINALIZE_ATTEMPTS, exc)
            if attempt == 3:
                log.warning("files in %s are open in: %s", partial.name, _holders(partial))
                _make_writable(partial)
            if target.exists() and not partial.exists():
                return                                         # another process finished it
            gc.collect()
            _sleep(min(5.0, 0.5 * 1.6 ** (attempt - 1)))
    log.warning("rename kept failing (%s) - copying the model folder instead", last)
    try:
        _rmtree(target)
        shutil.copytree(partial, target)
    except OSError as exc:
        shutil.rmtree(target, ignore_errors=True)
        raise OSError(f"{last}; copy failed too: {exc}") from exc
    _rmtree(partial)


_NETWORK_WORDS = ("Timeout", "Connection", "SSL", "Proxy", "HTTPError", "URLError", "Stalled", "Offline", "gaierror")


def _is_network_failure(exc: BaseException) -> bool:
    """A network problem (timeout, reset, DNS, proxy, stall) - as opposed to a disk / permission / logic error."""
    return any(any(w in c.__name__ for w in _NETWORK_WORDS) for c in type(exc).__mro__)


#: Longest pre-download check of "is Hugging Face usable" (seconds); a slower answer counts as "slow".
PROBE_CAP = 5.0
_SOURCE_NAMES = {"hf": "Hugging Face", "ms": "ModelScope", "hfm": "Hugging Face mirror"}


def _hf_fast(repo_id: str, hf_probe: Optional[Callable[[str], bool]]) -> bool:
    """Is Hugging Face usable?  The verdict is remembered for the session (later models do not probe again) and the
    probe itself is capped at ``PROBE_CAP`` seconds."""
    if os.environ.get("HF_ENDPOINT"):
        return True
    if hf_probe is None:
        known = modelscope_mirror.hf_verdict()
        if known is not None:
            return known
    fn = hf_probe or modelscope_mirror.hf_is_fast
    box: Dict[str, bool] = {}

    def run() -> None:
        try:
            box["fast"] = bool(fn(repo_id))
        except Exception:  # noqa: BLE001
            box["fast"] = False

    th = threading.Thread(target=run, daemon=True, name="vx-hf-probe")
    th.start()
    th.join(PROBE_CAP)
    fast = box.get("fast", False)
    if th.is_alive():
        log.info("huggingface.co did not answer within %.0f s", PROBE_CAP)
    if hf_probe is None:
        modelscope_mirror.remember_hf(fast)
    return fast


# ------------------------------------------------------------------------------------------------ one download per model
class ModelLock:
    """Cross-process lock for one model folder (an OS file lock: it disappears with a crashed process, so it is never stale).

    The background workers (``auto_quality_worker`` and the main worker's prefetch) may ask for the same model at the same
    moment; a second request waits here and then finds the finished model instead of downloading and renaming it in parallel.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: Any = None

    def try_acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            fh.close()

    def acquire(self, on_wait: Callable[[float], None], poll: float = 0.5) -> None:
        t0 = time.monotonic()
        waited = False
        while not self.try_acquire():
            if not waited:
                log.info("another process is downloading %s - waiting for it", self.path.name)
                waited = True
            on_wait(time.monotonic() - t0)             # may raise (the user cancelled)
            _sleep(poll)
        if waited:
            log.info("the other download of %s ended after %.0f s", self.path.name, time.monotonic() - t0)


def ensure_model(*args: Any, **kwargs: Any) -> Path:
    """Same as :func:`_ensure_model` (see there) but only one process at a time works on a given model."""
    import inspect

    bound = inspect.signature(_ensure_model).bind(*args, **kwargs)
    bound.apply_defaults()
    a = bound.arguments
    repo_id, root, progress, stage = a["repo_id"], a["root"], a["progress"], a["stage"]
    target = local_dir_for(repo_id, root)
    if verify_local_model(target):
        return target
    lock = ModelLock(target.parent / f".{target.name}.lock")
    short = repo_id.split("/")[-1]

    def waiting(seconds: float) -> None:
        progress(stage, 0.0, tr("progress.model_wait", short=short))

    lock.acquire(waiting)
    try:
        return _ensure_model(*args, **kwargs)              # re-checks: the other process may have finished the model
    finally:
        lock.release()



def _ensure_model(
    repo_id: str,
    progress: ProgressCallback = noop_progress,
    root: Optional[Path] = None,
    snapshot_download: Optional[Callable[..., str]] = None,
    get_remote_sha: Optional[Callable[[str], Optional[str]]] = None,
    stage: Stage = Stage.MODEL,
    revision: Optional[str] = None,
    reuse_external: bool = True,
    mirror_download: Optional[Callable[..., Any]] = None,
    hf_probe: Optional[Callable[[str], bool]] = None,
    hf_mirror_fetch: Optional[Callable[..., Any]] = None,
    mirror_manifest: Optional[Path] = None,
    allow_patterns: Optional[List[str]] = None,
    mirror_only: bool = False,
) -> Path:
    """Return the path of the local model, downloading it on first use (automatically).

    Before downloading we look for a complete copy left by another program (HF cache, Pinokio/Alexandria, ModelScope ...);
    it is used in place, read-only (:mod:`core.model_locator`).  This applies only to the main models folder
    (``root=None``): an update through a staging folder always downloads its own copy.

    ``revision`` is the HF commit; by default the verified revision from the "verified by Voxprint" manifest is used.
    If the pinned revision is unavailable the latest one is tried once (and logged).  Sources are tried in order
    Hugging Face -> ModelScope (reversed when Hugging Face is slow/unreachable); if both fail, the project's Hugging
    Face backup mirror (:mod:`infra.model_mirrors`, every file verified by SHA-256) is the last resort.  The result lands in ``<name>.partial``
    first and is renamed only after verification.

    ``allow_patterns`` (glob list) restricts the Hugging Face download to some files (a repository that also holds TF / Rust /
    Flax copies of the weights); ModelScope, which copies whole repositories, is not used then, but the backup mirror is (it
    fetches only the files matching the patterns, each checked by SHA-256).  ``mirror_only`` skips the original sources and
    goes straight to the backup mirror (used when a download from the original failed the caller's hash check).
    """
    target = local_dir_for(repo_id, root)
    if verify_local_model(target):
        return target

    short = repo_id.split("/")[-1]
    if root is None:
        imported = _import_existing(repo_id, revision, progress, stage, short)
        if imported is not None:
            return imported
    if reuse_external and root is None:
        found = external_model(repo_id, revision)
        if found is not None:
            progress(stage, 1.0, tr("progress.model_reused", short=short, where=str(found.location)))
            return found.path
    if partial_dir_has_data(repo_id, root):
        progress(stage, 0.0, tr("progress.model_resume", short=short))
    else:
        progress(stage, 0.0, tr("progress.first_download", short=short))
    need_gb = APPROX_SIZE_GB.get(repo_id, 3.0)
    try:
        free_gb = shutil.disk_usage(target.parent).free / 1024 ** 3
    except OSError:
        free_gb = None
    if free_gb is not None and free_gb < need_gb * 1.2:
        raise ModelDownloadError(
            tr("err.disk_model", short=short, need=f"{need_gb:.0f}", free=f"{free_gb:.1f}"),
            url=hf_url(repo_id))

    if snapshot_download is None and modelscope_mirror.hf_verdict() is not False:   # "slow" is remembered: no more route searching
        _prepare_network()
    if revision is None:
        revision = pinned_revision(repo_id)
    sha: Optional[str] = revision
    if not revision:
        try:
            if get_remote_sha is not None:
                sha = get_remote_sha(repo_id)
            else:
                from huggingface_hub import HfApi

                sha = HfApi().model_info(repo_id).sha
        except Exception as exc:  # noqa: BLE001 - the network may be down; the real error surfaces below
            log.warning("remote sha unavailable: %s", exc)

    partial = target.with_name(target.name + ".partial")   # survives between runs: the download resumes where it stopped
    cur: Dict[str, Any] = {"f": 0.0, "meter": None, "cancel": threading.Event()}

    def report(f: float) -> None:
        """One progress report of the running source: records it, aborts an abandoned download, shows the detail line."""
        if cur["cancel"].is_set():
            raise download_watch.Stalled("download abandoned (no data)")
        cur["f"] = f
        meter = cur["meter"]
        text = meter.text(tr, short, int(f * 100)) if meter is not None else tr("progress.downloading", short=short, pct=int(f * 100))
        progress(stage, f, text)

    tracker = _ByteProgress(report)
    state = {"sha": sha}

    def _download(rev: Optional[str]) -> None:
        """One ``snapshot_download`` call into the ``.partial`` folder (``rev`` None = latest)."""
        sd = snapshot_download
        if sd is None:
            try:
                from huggingface_hub import snapshot_download as sd  # type: ignore[no-redef]
            except ImportError as exc:
                raise ModelDownloadError(tr("err.hub_missing"), url=hf_url(repo_id), details=str(exc)) from exc
        kwargs: Dict[str, Any] = {"repo_id": repo_id, "local_dir": str(partial)}
        if rev:
            kwargs["revision"] = rev
        if allow_patterns:
            kwargs["allow_patterns"] = list(allow_patterns)
        try:
            sd(tqdm_class=tracker.make_tqdm_class(), **kwargs)
        except TypeError:  # some huggingface_hub versions have no tqdm_class parameter
            sd(**kwargs)

    def _from_hf() -> None:
        """Download from Hugging Face; fall back once to the latest revision if the pinned one is unavailable."""
        try:
            _download(revision)
        except download_watch.Stalled:
            raise                       # the watchdog gave up on this source: the caller switches to the next one
        except Exception as exc:  # noqa: BLE001
            if not revision:
                raise
            log.warning("pinned revision %s of %s unavailable (%s) - trying latest", revision[:8], repo_id, exc)
            if not isinstance(exc, OSError):    # network failure: the files are still good for resuming; a revision error: discard them
                shutil.rmtree(partial, ignore_errors=True)
            state["sha"] = None
            _download(None)
            try:
                state["sha"] = (get_remote_sha(repo_id) if get_remote_sha is not None else None)
            except Exception:  # noqa: BLE001
                state["sha"] = None

    def _from_modelscope() -> None:
        """Download from the ModelScope mirror, accepting the pinned revision only if all file sizes matched."""
        from core import model_locator

        known = model_locator.KNOWN_SIZES.get(repo_id)
        expected = known[1] if known and revision and known[0] == revision else None
        progress(stage, 0.0, tr("progress.mirror_modelscope", short=short))
        mirror_download(repo_id, partial, report, expected)
        # ModelScope has no commit sha: the revision is confirmed only when the sizes matched the pinned commit
        state["sha"] = revision if expected else None

    def _from_hf_mirror() -> None:
        """Last resort: the project's own backup mirror on Hugging Face; every file is verified against the SHA-256 manifest."""
        entry = model_mirrors.entry_for(repo_id, mirror_manifest)
        if entry is None:
            raise ModelDownloadError("no backup mirror for this model", url=hf_url(repo_id))
        if revision and revision != entry.source_revision:
            raise ModelDownloadError(f"the backup mirror holds {entry.source_revision[:8]}, not the requested {revision[:8]}",
                                     url=hf_url(repo_id))
        progress(stage, 0.0, tr("progress.mirror_hf", short=short))
        state["sha"] = model_mirrors.download(
            entry, partial, report, hf_mirror_fetch, allow_patterns)

    def complete() -> bool:
        return partial_is_complete(partial, repo_id, revision, mirror_manifest, allow_patterns)

    finished = False
    if partial.is_dir() and complete():
        # an earlier run downloaded everything but could not rename the folder: nothing to download, only finish
        log.info("%s: the .partial folder is already complete - finishing without a download", repo_id)
        finished = True
        state["sha"] = revision
    mirror_ok = mirror_enabled() and root is None and not allow_patterns and not finished   # updates through the staging folder always go straight to Hugging Face
    if mirror_ok:
        mirror_download = mirror_download or _default_mirror_download
        fast = _hf_fast(repo_id, hf_probe)
        order = ["hf", "ms"] if fast else ["ms", "hf"]
        if not fast:
            log.warning("Hugging Face is slow or unreachable - trying ModelScope first for %s", repo_id)
    else:
        order = ["hf"]
    if mirror_only or finished:
        order = []
    if root is None and not finished and model_mirrors.entry_for(repo_id, mirror_manifest) is not None:
        order.append("hfm")                     # the project's Hugging Face backup mirror, always last
    errors: List[str] = []
    ok_source = "partial" if finished else ""
    funcs = {"hf": _from_hf, "hfm": _from_hf_mirror, "ms": _from_modelscope}

    def tick(meter: "download_watch.Meter") -> None:
        progress(stage, cur["f"], meter.text(tr, short, int(cur["f"] * 100)))

    for src in order:
        label = _SOURCE_NAMES[src]
        meter = download_watch.Meter(label, lambda: tracker.total or (
            int(cur["meter"].done / cur["f"]) if cur["meter"] is not None and cur["f"] > 0.01 else 0))
        cur["meter"], cur["f"], cur["cancel"] = meter, 0.0, threading.Event()
        tracker.cancel = cur["cancel"]
        tracker.total = tracker.done = 0
        try:
            if src == "hfm":
                log.warning("download of %s from the original sources failed - trying the Hugging Face backup mirror", repo_id)
            elif src == "ms" and errors:
                log.warning("download of %s from Hugging Face failed - trying ModelScope", repo_id)
            log.info("downloading %s from %s", repo_id, label)
            download_watch.run_watched(funcs[src], partial, meter, cur["cancel"], on_tick=tick, idle_ok=complete)
            ok_source = src
            if src == "hf" and mirror_ok:
                modelscope_mirror.remember_hf(True)
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{src}: {type(exc).__name__}: {exc}")
            log.warning("download of %s from %s failed: %s", repo_id, src, exc)
            if src == "hf" and mirror_ok and _is_network_failure(exc) and not complete():
                modelscope_mirror.remember_hf(False)      # measured: no data / network error; later models go to ModelScope first
    if not ok_source:
        # the .partial folder stays on disk: the next attempt continues where this one stopped
        raise ModelDownloadError(
            tr("err.download_failed", short=short, url=hf_url(repo_id)),
            url=hf_url(repo_id), details=" | ".join(errors))
    sha = state["sha"]
    if not verify_local_model(partial):
        shutil.rmtree(partial, ignore_errors=True)
        raise ModelDownloadError(tr("err.model_partial", short=short),
                                 url=hf_url(repo_id))
    if sha:
        (partial / ".revision").write_text(sha, encoding="utf-8")
    try:
        finalize_download(partial, target)
    except OSError as exc:
        log.error("could not finish %s: %s", repo_id, exc)
        raise ModelDownloadError(tr("err.model_finalize", short=short), url=hf_url(repo_id), details=str(exc)) from exc
    progress(stage, 1.0, tr("progress.model_done", short=short))
    return target


def ensure_aligner_model(progress: ProgressCallback = noop_progress, **kw: Any) -> Path:
    """Ensure the forced-aligner model (Qwen3-ForcedAligner) is available locally."""
    return ensure_model(ALIGNER_REPO, progress, **kw)


def ensure_tts_models(base_repo: str, progress: ProgressCallback = noop_progress, **kw: Any) -> Dict[str, Path]:
    """The TTS base model used for training.

    The audio tokenizer (``speech_tokenizer/``) lives inside the Base model repository (checked against the HF file
    list), so no separate tokenizer repository is needed.
    """
    return {"base": ensure_model(base_repo, progress, stage=Stage.MODEL, **kw)}
