"""Automatic download of the models from Hugging Face (with a ModelScope fallback) into the app's model folder.

A download goes into ``<name>.partial``; only after verification is the folder renamed, so a partially downloaded
model is never considered ready, and the next run resumes from the ``.partial`` data.  The commit sha of the
downloaded revision is stored in the ``.revision`` file.  Copies that other programs already downloaded (HF cache,
Pinokio/Alexandria, ModelScope ...) are reused in place, read-only - see :mod:`core.model_locator`.
"""
from __future__ import annotations

from core.i18n import tr
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.errors import ModelDownloadError
from core.events import ProgressCallback, Stage, noop_progress
from infra import modelscope_mirror, paths

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
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
                if getattr(self, "_is_bytes", False):
                    with tracker.lock:
                        tracker.done += int(n)
                        if tracker.total:
                            tracker.cb(min(1.0, tracker.done / tracker.total))
                return r

        return _Tqdm


def ensure_model(
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
) -> Path:
    """Return the path of the local model, downloading it on first use (automatically).

    Before downloading we look for a complete copy left by another program (HF cache, Pinokio/Alexandria, ModelScope ...);
    it is used in place, read-only (:mod:`core.model_locator`).  This applies only to the main models folder
    (``root=None``): an update through a staging folder always downloads its own copy.

    ``revision`` is the HF commit; by default the verified revision from the "verified by Voxprint" manifest is used.
    If the pinned revision is unavailable the latest one is tried once (and logged).  Sources are tried in order
    Hugging Face -> ModelScope (reversed when Hugging Face is slow/unreachable).  The result lands in ``<name>.partial``
    first and is renamed only after verification.
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
    tracker = _ByteProgress(lambda f: progress(stage, f, tr("progress.downloading", short=short, pct=int(f * 100))))
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
        try:
            sd(tqdm_class=tracker.make_tqdm_class(), **kwargs)
        except TypeError:  # some huggingface_hub versions have no tqdm_class parameter
            sd(**kwargs)

    def _from_hf() -> None:
        """Download from Hugging Face; fall back once to the latest revision if the pinned one is unavailable."""
        try:
            _download(revision)
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
        mirror_download(repo_id, partial,
                        lambda f: progress(stage, f, tr("progress.downloading", short=short, pct=int(f * 100))),
                        expected)
        # ModelScope has no commit sha: the revision is confirmed only when the sizes matched the pinned commit
        state["sha"] = revision if expected else None

    mirror_ok = mirror_enabled() and root is None   # updates through the staging folder always go straight to Hugging Face
    if mirror_ok:
        mirror_download = mirror_download or _default_mirror_download
        fast = (hf_probe or modelscope_mirror.hf_is_fast)(repo_id)
        order = ["hf", "ms"] if fast else ["ms", "hf"]
        if not fast:
            log.warning("Hugging Face is slow or unreachable - trying ModelScope first for %s", repo_id)
    else:
        order = ["hf"]
    errors: List[str] = []
    ok_source = ""
    for src in order:
        try:
            if src == "hf":
                _from_hf()
            else:
                if errors:
                    log.warning("download of %s from Hugging Face failed - trying ModelScope", repo_id)
                _from_modelscope()
            ok_source = src
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{src}: {type(exc).__name__}: {exc}")
            log.warning("download of %s from %s failed: %s", repo_id, src, exc)
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
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    partial.rename(target)
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
