"""Locate models that VOXPrint MDL has already installed.

The weights are downloaded by ``voxprint-mdl`` (segmented Range, mirror switch,
SHA-256). This module only finds a complete copy: the models folder, an imported
folder, or another program's copy (:mod:`core.model_locator`). A missing model
raises :class:`core.errors.ModelDownloadError` and names ``voxprint-mdl``.
Licence: Apache-2.0.

Credit: VOXPrint AI Audiobook Builder https://github.com/Mitroshenkov87/voxprint-audiobook-builder
"""
from __future__ import annotations

from core.i18n import tr
import fnmatch
import gc
import glob
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

from infra import download_watch, model_mirrors, model_release, modelscope_mirror, netroute, paths
from infra.mdl_gate import MDL_HINT

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
netroute.disable_xet()      # plain HTTP: the xet/CAS transfer path stalls on networks that block its host (VOXPRINT_ALLOW_XET=1 keeps it)
log = logging.getLogger("voxprint.models")

ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"
ASR_REPO = "Qwen/Qwen3-ASR-0.6B"   # speech recognition (first-run / --prefetch download-all; also voice check, A/B, spoken consent, no-transcript)
ASR_LARGE_REPO = "Qwen/Qwen3-ASR-1.7B"   # the same on GPUs with ~8 GB of VRAM or more (which one: infra/asr_choice.py)
#: Approximate download size in GB (for the free-disk-space check; an estimate, not an exact value).
APPROX_SIZE_GB = {
    ALIGNER_REPO: 2.0,
    ASR_REPO: 1.9,
    ASR_LARGE_REPO: 4.7,
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": 4.5,
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": 2.5,
    "ai-forever/sage-fredt5-distilled-95m": 0.5,         # Russian text clean-up (first-run download-all; see infra/text_models.py)
    "Helsinki-NLP/opus-mt-ru-en": 0.4,                    # 2020 Opus-MT fallback (ru<->en only)
    "Helsinki-NLP/opus-mt-en-ru": 0.4,
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmw": 0.91,   # en->de, same snapshot as Movie Dubber
    "Helsinki-NLP/opus-mt-tc-bible-big-gmw-deu_eng_fra_por_spa": 0.91,   # de->en
    "Helsinki-NLP/opus-mt-tc-big-en-zle": 0.48,           # Opus-MT tc-big (safetensors only): en->ru, ru->en, ru->de, de->ru
    "Helsinki-NLP/opus-mt-tc-big-zle-en": 0.48,
    "Helsinki-NLP/opus-mt-tc-big-zle-de": 0.48,
    "Helsinki-NLP/opus-mt-tc-big-de-zle": 0.48,
}


def hf_url(repo_id: str) -> str:
    """Web page of the repository on Hugging Face (shown to the user in error messages)."""
    return f"https://huggingface.co/{repo_id}"


def local_dir_for(repo_id: str, root: Optional[Path] = None) -> Path:
    """Folder of a model inside the models directory: ``Qwen/X`` -> ``Qwen--X``.

    With the user's own models folder (installer page "Models folder", :func:`infra.paths.models_dir`) a model that is
    complete only in the DEFAULT folder (``%LOCALAPPDATA%\\Voxprint\\models``, an earlier install) is used from there - it is
    not downloaded again; everything new goes to the chosen folder."""
    name = repo_id.replace("/", "--")
    if root is not None:
        return Path(root) / name
    target = paths.models_dir() / name
    if not verify_local_model(target):
        default = paths.default_models_dir() / name
        if default != target and verify_local_model(default):
            return default
    return target


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


def _default_mirror_download(repo_id, dest, progress, expected_sizes, on_total=None):
    """Default mirror downloader (injectable in tests): ModelScope with size verification."""
    return modelscope_mirror.download_repo(repo_id, dest, progress, expected_sizes, on_total=on_total)


#: Seconds the Hugging Face file-size lookup (for the stable "N MB of TOTAL") may take; it is optional.
SIZES_TIMEOUT = 6.0


def _hf_sizes(repo_id: str, revision: Optional[str], patterns: Optional[List[str]]) -> Optional[Dict[str, int]]:
    """``{file: size}`` from the Hugging Face API (best effort, short timeout), limited by the glob ``patterns``."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True, timeout=SIZES_TIMEOUT)
        sizes = {f.rfilename: int(f.size) for f in (info.siblings or []) if getattr(f, "size", None)}
    except Exception as exc:  # noqa: BLE001 - optional; without it the line shows only what was downloaded
        log.debug("file sizes of %s unavailable: %s", repo_id, exc)
        return None
    if patterns:
        sizes = {n: v for n, v in sizes.items() if any(fnmatch.fnmatchcase(n, p) for p in patterns)}
    return sizes or None


def external_model(repo_id: str, revision: Optional[str] = None):
    """A copy of the model downloaded by another program (read-only), or None.

    See :mod:`core.model_locator`: file completeness and the pinned revision are checked; nothing is copied or modified.
    Voxprint NEVER updates or deletes such a folder.
    """
    from core import model_locator

    rev = revision if revision else pinned_revision(repo_id)
    try:
        found = model_locator.find_model(repo_id, rev)
        if found is None and paths.configured_models_dir() is not None:
            # the folder the user CHOSE as models folder may hold models in another layout (a Hugging Face cache ...):
            # they are picked up in place; an explicit choice is not "guessing" (ignore_disabled).  A Voxprint BACKUP is
            # never used in place: paths.models_dir() refuses it and infra.existing_models restores it into the live folder.
            own = paths.models_dir()
            roots = [("folder", own)] if own.is_dir() else []
            found = model_locator.find_model(repo_id, rev, roots, ignore_disabled=True)
        return found
    except Exception as exc:  # noqa: BLE001 - looking for foreign copies must never break a normal download
        log.warning("external model search failed for %s: %s", repo_id, exc)
        return None


def _import_existing(repo_id: str, revision: Optional[str], progress: ProgressCallback, stage: Stage, short: str) -> Optional[Path]:
    """Import the model from the user's "existing models folder" (:mod:`infra.existing_models`) into the own models folder.

    Link or copy, verified by hash, before anything is downloaded.  Returns the new folder, or None if no such folder is
    configured, it holds no usable copy or the import failed (the normal path - in-place reuse, then download - follows)."""
    from core.errors import CancelledByUser
    from infra import existing_models

    try:   # a Voxprint backup chosen as the source is restored as a whole (once) - models, voices, ffmpeg
        existing_models.adopt_backup_choice()
        if existing_models.restore_pending():
            def _restore_progress(f: float, m: str = "") -> None:
                progress(stage, f, m)

            existing_models.restore_backup(_restore_progress)
            restored = local_dir_for(repo_id)
            if verify_local_model(restored):
                progress(stage, 1.0, tr("progress.model_imported", short=short))
                return restored
    except CancelledByUser:
        raise
    except Exception as exc:  # noqa: BLE001 - a failed restore must never block the per-model import / download
        log.warning("restoring the backup source failed: %s", exc)
    try:
        found = existing_models.find(repo_id, revision if revision else pinned_revision(repo_id))
        if found is None:
            return None
        progress(stage, 0.0, tr("progress.model_importing", short=short, where=str(found.location)))
        def _import_progress(f: float, m: str = "") -> None:
            progress(stage, f, m)

        path = existing_models.import_model(found, _import_progress)
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


def ready_model_path(repo_id: str) -> Optional[Path]:
    """Path of a model that is already complete (Voxprint's folder or another program's copy), or ``None``.

    Never downloads.  Features that expect the first-run / ``--prefetch`` download-all to have fetched the model
    (voice check, A/B, spoken consent) use this instead of :func:`ensure_model`, so a missing model is a clear skip
    / error rather than a surprise background download.
    """
    target = local_dir_for(repo_id)
    if verify_local_model(target):
        return target
    found = external_model(repo_id)
    return found.path if found is not None else None


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

        class _Tqdm(hf_tqdm):
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
#: handle inside the folder: try again for about half a minute, then move the finished files one by one instead.
FINALIZE_ATTEMPTS = 10
#: Seconds a new download attempt waits for an abandoned download thread of this process to let go of the ``.partial``
#: folder before it continues in a fresh ``<name>.partial-N`` folder instead (the hub's lock files would block it).
FRESH_PARTIAL_AFTER = 5.0
_sleep = time.sleep


class HeldByUs(OSError):
    """The model folder could not be finished because a download thread of THIS program still holds files in it."""


def _unfinished(name: str, is_dir: bool) -> bool:
    """Download leftovers that are never part of a model: the hub's ``.cache`` and the unfinished pieces of a transfer."""
    if is_dir:
        return name == ".cache" or name.endswith(".incomplete.parts")
    return name.endswith((".incomplete", ".assembling"))


def _move_finished(src: Path, dst: Path) -> int:
    """Move every finished file of ``src`` into ``dst`` (same tree layout), skipping the hub's ``.cache`` and unfinished
    pieces; a file that is already in ``dst`` stays.  Moving single closed files works on Windows even while other files
    of the folder are held open.  Returns the number of files moved; raises ``OSError`` if one could not be moved."""
    moved = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if not _unfinished(d, True)]
        rel = Path(root).relative_to(src)
        for name in files:
            if _unfinished(name, False):
                continue
            out = Path(dst) / rel / name
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            os.replace(Path(root) / name, out)
            moved += 1
    return moved


def _partials(target: Path) -> List[Path]:
    """``<name>.partial`` and the fresh ``<name>.partial-N`` folders of a model that exist on disk."""
    out = [target.with_name(target.name + ".partial")]
    try:
        out += sorted(target.parent.glob(glob.escape(target.name) + ".partial-*"))
    except OSError:
        pass
    return [p for p in out if p.is_dir()]


def _fresh_partial(target: Path, old: Path) -> Path:
    """A ``<name>.partial-N`` folder no abandoned thread is using, with the finished files of ``old`` moved in."""
    n = 1
    while True:
        cand = target.with_name(f"{target.name}.partial-{n}")
        if cand != old and not download_watch.busy(cand):
            break
        n += 1
    cand.mkdir(parents=True, exist_ok=True)
    try:
        moved = _move_finished(old, cand)
    except OSError as exc:
        moved = -1
        log.warning("could not move all finished files of %s: %s", old.name, exc)
    log.warning("an abandoned download thread still holds files in %s - continuing in %s (%s finished file(s) taken over)",
                old.name, cand.name, moved if moved >= 0 else "some")
    return cand


def _merge_partials(target: Path) -> None:
    """Fold fresh ``<name>.partial-N`` folders of an earlier run into ``<name>.partial`` (so that run's files are resumed)."""
    main = target.with_name(target.name + ".partial")
    for extra in _partials(target):
        if extra == main or download_watch.busy(extra):
            continue
        try:
            main.mkdir(parents=True, exist_ok=True)
            _move_finished(extra, main)
        except OSError as exc:
            log.warning("could not take over %s: %s", extra.name, exc)
            continue
        _rmtree(extra)


def cleanup_partials(target: Path) -> None:
    """After the model folder is complete: delete its leftover ``.partial`` folders that no thread of ours still uses."""
    for p in _partials(target):
        if not download_watch.busy(p):
            _rmtree(p)


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
    for _root, _dirs, files in os.walk(partial):
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
    (lock files), then renames with retries and backoff; if Windows keeps refusing - or a download thread of this program
    that was given up on still holds files in it, which no waiting fixes - the finished files are moved into the model
    folder one by one (the ``.cache`` and unfinished pieces stay behind), and as the last resort copied.  Raises
    ``OSError`` (:class:`HeldByUs` when our own thread is the holder) only if everything failed."""
    gc.collect()                                              # closes file objects of finished downloads (sessions, locks)
    _rmtree(partial / ".cache")
    _rmtree(target)
    ours = download_watch.busy(partial)
    last: Optional[BaseException] = None
    if ours:
        log.info("a download thread of this program still holds files in %s - moving the finished files instead of renaming",
                 partial.name)
    for attempt in range(1, 0 if ours else FINALIZE_ATTEMPTS + 1):
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
            if download_watch.busy(partial):
                ours = True
                break                                          # our own abandoned thread: waiting longer does not help
            gc.collect()
            _sleep(min(5.0, 0.5 * 1.6 ** (attempt - 1)))
    log.warning("finishing %s by moving the finished files (%s)", target.name, last or "held by this program")
    try:
        target.mkdir(parents=True, exist_ok=True)
        _move_finished(partial, target)
        _rmtree(partial)                                       # best effort: a held .cache is cleaned up next time
        return
    except OSError as exc:
        log.warning("moving the finished files failed (%s) - copying them instead", exc)
        last = last or exc
    try:
        shutil.copytree(partial, target, dirs_exist_ok=True,
                        ignore=lambda d, names: [n for n in names if _unfinished(n, os.path.isdir(os.path.join(d, n)))])
    except OSError as exc:
        shutil.rmtree(target, ignore_errors=True)
        cls = HeldByUs if ours or download_watch.busy(partial) else OSError
        raise cls(f"{last}; copy failed too: {exc}") from exc
    _rmtree(partial)


_NETWORK_WORDS = ("Timeout", "Connection", "SSL", "Proxy", "HTTPError", "URLError", "Stalled", "Offline", "gaierror")


def _exact_pattern(name: str) -> str:
    """A glob pattern (``allow_patterns``) that matches exactly ``name``."""
    return "".join(f"[{c}]" if c in "*?[" else c for c in name)


def _locked(exc: BaseException) -> bool:
    """A file was locked / access denied somewhere in the exception chain (a local problem, not the network's)."""
    seen = 0
    e: Optional[BaseException] = exc
    while e is not None and seen < 10:
        if isinstance(e, PermissionError):
            return True
        e, seen = e.__cause__ or e.__context__, seen + 1
    return False


def _is_network_failure(exc: BaseException) -> bool:
    """A network problem (timeout, reset, DNS, proxy, stall) - as opposed to a disk / permission / logic error."""
    if _locked(exc):
        return False
    return any(any(w in c.__name__ for w in _NETWORK_WORDS) for c in type(exc).__mro__)


#: A source that stalled AFTER it had delivered data is resumed this many times (``STALL_PAUSE`` s apart) before the next
#: source is tried (a long file on a flaky route stalls now and then; a source that gives nothing is left at once).
STALL_RETRIES = 3
STALL_PAUSE = 3.0


def _retry_same_source(exc: BaseException, retry: int) -> bool:
    """Resume the same source after a stall that delivered bytes. A source that is merely slow is left for the next one."""
    return (isinstance(exc, download_watch.Stalled) and exc.progressed and not getattr(exc, "slow", False)
            and retry < STALL_RETRIES)


#: Longest pre-download check of "is Hugging Face usable" (seconds); a slower answer counts as "slow".
PROBE_CAP = 5.0
_SOURCE_NAMES = {"gh": "GitHub", "hf": "Hugging Face", "ms": "ModelScope", "hfm": "Hugging Face mirror"}


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

    Background workers (the Components window extras, the first-run prefetch, a task) may ask for the same model at the same
    moment; a second request waits here and then finds the finished model instead of downloading and renaming it in parallel.
    """

    def __init__(self, path: Path) -> None:
        """Remember the lock-file path. The file is created on acquire."""
        self.path = path
        self._fh: Any = None

    def try_acquire(self) -> bool:
        """Take the lock without waiting. False when another process already holds it."""
        for _ in range(5):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fh = open(self.path, "a+b")
            except OSError:
                # Windows cannot open a lock file another thread of this process already holds (the msvcrt lock is
                # process-wide, and a second open raises PermissionError). That means the lock is taken: poll again.
                continue
            try:
                if os.name == "nt":
                    import msvcrt

                    fh.seek(0)
                    getattr(msvcrt, "locking")(fh.fileno(), getattr(msvcrt, "LK_NBLCK"), 1)
                else:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # the holder before us may have deleted the file right after its release (``release(remove=True)``):
                    # a lock on the unlinked file excludes nobody, so try again on the file that is at the path now
                    st = os.fstat(fh.fileno())
                    try:
                        cur = os.stat(self.path)
                        same = (cur.st_ino, cur.st_dev) == (st.st_ino, st.st_dev)
                    except OSError:
                        same = False
                    if not same:
                        fh.close()
                        continue
            except OSError:
                fh.close()
                return False
            self._fh = fh
            return True
        return False

    def release(self, remove: bool = False) -> None:
        """Give the lock up; ``remove`` also deletes the (empty) lock file so a finished model leaves nothing behind.

        On Linux the file is deleted while still locked (a waiter re-checks the path, see :meth:`try_acquire`); on Windows
        it can only go after closing, and the delete simply fails while another process has the file open - it then
        stays until that process (or Check & repair, :func:`sweep_stale_locks`) removes it."""
        fh, self._fh = self._fh, None
        if fh is None:
            return
        if remove and os.name != "nt":
            try:
                self.path.unlink()
            except OSError:
                pass
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                getattr(msvcrt, "locking")(fh.fileno(), getattr(msvcrt, "LK_UNLCK"), 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            fh.close()
        if remove and os.name == "nt":
            try:
                self.path.unlink()
            except OSError:                    # another process opened it meanwhile (it waits for the model): keep it
                pass

    def acquire(self, on_wait: Callable[[float], None], poll: float = 0.5) -> None:
        """Wait until the lock is free. ``on_wait`` is called with the seconds already spent."""
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
        cleanup_partials(target)                       # leftovers of a folder finished while a thread still held them
        return target
    lock = ModelLock(target.parent / f".{target.name}.lock")
    short = repo_id.split("/")[-1]

    def waiting(seconds: float) -> None:
        progress(stage, 0.0, tr("progress.model_wait", short=short))

    lock.acquire(waiting)
    done = False
    try:
        path = _ensure_model(*args, **kwargs)              # re-checks: the other process may have finished the model
        done = True
        return path
    finally:
        lock.release(remove=done)                          # a finished model leaves no empty ``.<name>.lock`` behind


def sweep_stale_locks(root: Optional[Path] = None) -> int:
    """Delete the empty ``.<model>.lock`` files no process holds any more (left by build 667 and older, or a crash).

    A lock that is held right now (a download in progress) is skipped.  Returns how many files were removed."""
    folder = Path(root) if root is not None else paths.models_dir()
    try:
        candidates = [p for p in folder.glob(".*.lock") if p.is_file()]
    except OSError:
        return 0
    removed = 0
    for p in candidates:
        try:
            if p.stat().st_size:
                continue                                   # not one of ours (ours are always empty)
        except OSError:
            continue
        lock = ModelLock(p)
        if not lock.try_acquire():
            continue                                       # a download of that model is running
        lock.release(remove=True)
        if not p.exists():
            removed += 1
    if removed:
        log.info("removed %d stale model lock file(s) in %s", removed, folder)
    return removed



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
    get_remote_sizes: Optional[Callable[[str, Optional[str], Optional[List[str]]], Optional[Dict[str, int]]]] = None,
    release_manifest: Optional[Path] = None,
    release_opener: Optional[Callable[..., Any]] = None,
) -> Path:
    """Return a model that is already on disk.

    VOXPrint MDL downloads the weights (segmented Range, mirror switch, hashes).
    This function reuses a complete local copy, an imported folder or another
    program's copy. A missing model raises :class:`ModelDownloadError` and names
    ``voxprint-mdl``. ``snapshot_download`` is only a test double: production
    never passes it, and it is not a network client of this program.
    """
    del mirror_download, hf_probe, hf_mirror_fetch, mirror_manifest, mirror_only
    del get_remote_sizes, release_manifest, release_opener
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
    if revision is None:
        revision = pinned_revision(repo_id)
    sha: Optional[str] = revision
    if not revision and get_remote_sha is not None:
        try:
            sha = get_remote_sha(repo_id)
        except Exception as exc:  # noqa: BLE001 - the test double reports the real failure below
            log.warning("remote sha unavailable: %s", exc)
            sha = None
    if snapshot_download is None:
        raise ModelDownloadError(
            tr("err.download_failed", short=short, url=MDL_HINT),
            url=hf_url(repo_id), details=MDL_HINT)
    partial = target.with_name(target.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    progress(stage, 0.0, tr("progress.first_download", short=short))
    kwargs: Dict[str, Any] = {"repo_id": repo_id, "local_dir": str(partial)}
    if sha:
        kwargs["revision"] = sha
    if allow_patterns:
        kwargs["allow_patterns"] = list(allow_patterns)
    try:
        try:
            snapshot_download(tqdm_class=None, **kwargs)
        except TypeError:
            snapshot_download(**kwargs)
    except ModelDownloadError:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001 - the test double failed; nothing is left half-installed
        shutil.rmtree(partial, ignore_errors=True)
        raise ModelDownloadError(
            tr("err.download_failed", short=short, url=hf_url(repo_id)),
            url=hf_url(repo_id), details=f"{type(exc).__name__}: {exc}") from exc
    if not verify_local_model(partial):
        raise ModelDownloadError(tr("err.model_partial", short=short), url=hf_url(repo_id))
    if sha:
        (partial / ".revision").write_text(str(sha), encoding="utf-8")
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    partial.rename(target)
    progress(stage, 1.0, tr("progress.first_download", short=short))
    return target


def _caller(depth: int = 3) -> str:
    """``module.function`` of the nearest callers outside the download plumbing (for the log line of a download)."""
    import inspect

    skip = ("infra.model_downloader", "infra.text_models", "infra.auto_steps")
    out = []
    try:
        for fr in inspect.stack(context=0)[1:]:
            mod = fr.frame.f_globals.get("__name__", "?")
            if mod in skip:
                continue
            out.append(f"{mod}.{fr.function}")
            if len(out) >= depth:
                break
    except Exception:  # noqa: BLE001 - diagnostics only
        return "?"
    return " <- ".join(out) or "?"


#: Repository files the hash check after a download ignores: git metadata and pictures / documents.  Hugging Face rewrites
#: ``.gitattributes`` per repository and ModelScope copies may lack such files; they are never loaded by the program.
_UNCHECKED_SUFFIXES = (".md", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".pdf")


def _not_checked(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base.startswith(".") or base.lower().endswith(_UNCHECKED_SUFFIXES)


def manifest_bad_files(folder: Path, repo_id: str, revision: Optional[str], mirror_manifest: Optional[Path] = None,
                       patterns: Optional[List[str]] = None) -> Optional[List[str]]:
    """Files of a downloaded model that are missing or differ from the verified manifest (``infra/model_mirrors.json``:
    size first, then SHA-256).  ``None`` = no verified hashes are known for this repository and revision (nothing to
    compare; the structural check :func:`verify_local_model` is all there is).  The manifest is read even when the backup
    mirror is switched off: the hashes are local data."""
    from core import model_locator

    entry = model_mirrors.load(mirror_manifest).get(repo_id)
    if entry is None or not revision or entry.source_revision != revision:
        return None
    try:
        files = entry.downloadable(patterns)
    except model_mirrors.MirrorError:
        return None
    bad = []
    for name, meta in files.items():
        p = Path(folder) / Path(*name.split("/"))
        if name in model_locator._OPTIONAL_FILES and not p.exists():
            continue
        if _not_checked(name):
            continue
        if not model_release.file_ok(p, meta):
            bad.append(name)
    return bad


def _verify_or_repair(partial: Path, repo_id: str, revision: Optional[str], mirror_manifest: Optional[Path],
                      patterns: Optional[List[str]], short: str, from_mirror: Optional[Callable[[], None]],
                      meter: Any, cur: Dict[str, Any]) -> None:
    """Hash check of a finished download BEFORE it is renamed into place (a model is never marked done unverified).

    Bad files are deleted; the backup mirror (if available) fetches exactly those files, verified; otherwise
    ``ModelDownloadError`` is raised and the ``.partial`` folder stays, so the next attempt downloads only what is missing."""
    bad = manifest_bad_files(partial, repo_id, revision, mirror_manifest, patterns)
    if not bad:
        if bad is None:
            log.info("%s: no verified hashes for revision %s - structural check only", repo_id, (revision or "?")[:8])
        else:
            log.info("%s: every file matches its pinned size + SHA-256", repo_id)
        return
    log.warning("%s: %d file(s) failed the size / SHA-256 check after the download: %s", repo_id, len(bad), ", ".join(bad))
    for name in bad:
        try:
            (partial / Path(*name.split("/"))).unlink()
        except OSError:
            pass
    if from_mirror is not None:
        try:
            cur["cancel"] = threading.Event()
            download_watch.run_watched(from_mirror, partial, meter, cur["cancel"])
            bad = manifest_bad_files(partial, repo_id, revision, mirror_manifest, patterns) or []
        except Exception as exc:  # noqa: BLE001 - reported below as a hash failure
            log.warning("%s: repairing the bad files from the backup mirror failed: %s", repo_id, exc)
    if bad:
        raise ModelDownloadError(tr("err.model_hash", short=short), url=hf_url(repo_id),
                                 details="bad files: " + ", ".join(bad))


def ensure_aligner_model(progress: ProgressCallback = noop_progress, **kw: Any) -> Path:
    """Ensure the forced-aligner model (Qwen3-ForcedAligner) is available locally."""
    return ensure_model(ALIGNER_REPO, progress, **kw)


def ensure_tts_models(base_repo: str, progress: ProgressCallback = noop_progress, **kw: Any) -> Dict[str, Path]:
    """The TTS base model used for training.

    The audio tokenizer (``speech_tokenizer/``) lives inside the Base model repository (checked against the HF file
    list), so no separate tokenizer repository is needed.
    """
    return {"base": ensure_model(base_repo, progress, stage=Stage.MODEL, **kw)}
