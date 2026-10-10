"""Optional soundscape model: ACE-Step 1.5 turbo (MIT code and MIT weights).

Off by default. Nothing is downloaded until the user turns the setting on (:func:`enable`). Narration never
downloads it. The files are hash-pinned in ``infra/model_mirrors.json`` and live in ``models/ace-step-1.5``,
not in the generic ``Owner--Name`` folder, so Check & repair and the full installer do not fetch them.

The weights are about 10.1 GB. The model card says generation fits in under 4 GB of video memory with
offloading; the resident turbo stack is on the order of the weight size when nothing is offloaded. This
module refuses to generate when a user VRAM cap leaves less than 4 GB free.

ACE-Step's Python package is imported only inside :func:`generate`. Machines that never enable the
soundscape do not need it installed.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from core.i18n import tr
from infra import gpu_prefs, model_mirrors, model_release, paths

log = logging.getLogger("voxprint.soundscape")

REPO = "ACE-Step/Ace-Step1.5"
LABEL = "ACE-Step 1.5"
FOLDER_NAME = "ace-step-1.5"
MIN_FREE_BYTES = 4 * 1024 ** 3
#: Kept for callers that cache a clip against the pinned revision.
SOURCE_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"


class SoundscapeModelError(Exception):
    """The soundscape model is missing or refused to run. Narration continues without that cue."""


def _file() -> Path:
    return paths.state_dir() / "soundscape.json"


def enabled() -> bool:
    """True after the user has turned the soundscape on. A missing file means off."""
    try:
        data = json.loads(_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(data.get("enabled")) if isinstance(data, dict) else False


def _save(on: bool) -> None:
    path = _file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"enabled": bool(on)}), encoding="utf-8")


def entry(manifest: Optional[Path] = None) -> model_mirrors.MirrorEntry:
    """The pinned file list. Raises when the manifest has no such model."""
    found = model_mirrors.load(manifest).get(REPO)
    if found is None:
        raise model_release.ReleaseError(f"{REPO} is not in the model manifest")
    return found


def source_revision(manifest: Optional[Path] = None) -> str:
    """Pinned source revision, or the built-in one when the manifest cannot be read."""
    try:
        return entry(manifest).source_revision
    except model_release.ReleaseError:
        return SOURCE_REVISION


def model_dir(models_dir: Optional[Path] = None) -> Path:
    """``<models folder>/ace-step-1.5``."""
    if models_dir is None:
        models_dir = paths.models_dir()
    return Path(models_dir) / FOLDER_NAME


def _files(manifest: Optional[Path] = None):
    """Downloadable files (the model card is left out)."""
    return entry(manifest).downloadable()


def ready(models_dir: Optional[Path] = None, manifest: Optional[Path] = None) -> bool:
    """True when every pinned file is present at the expected size."""
    try:
        files = _files(manifest)
    except model_release.ReleaseError:
        return False
    folder = model_dir(models_dir)
    for name, meta in files.items():
        path = folder.joinpath(*name.split("/"))
        try:
            if not path.is_file() or path.stat().st_size != int(meta["size"]):
                return False
        except OSError:
            return False
    return True


def _silent(frac: float, message: str = "") -> None:
    return None


def ensure(progress: Callable[[float, str], None] = _silent, models_dir: Optional[Path] = None,
           opener=None, timeout: float = 30.0, manifest: Optional[Path] = None) -> Path:
    """Download the pinned files if they are missing. Call this only from :func:`enable`.

    Returns the model folder. Raises :class:`infra.model_release.ReleaseError`.
    """
    e = entry(manifest)
    files = e.downloadable()
    folder = model_dir(models_dir)
    opener = opener or model_release._open
    total = float(sum(int(meta["size"]) for meta in files.values())) or 1.0
    done = 0

    def _report(so_far: int) -> None:
        frac = min(1.0, so_far / total)
        progress(min(0.99, frac), tr("progress.downloading", short=LABEL, pct=int(100 * frac)))

    for name, meta in files.items():
        target = folder.joinpath(*name.split("/"))
        size = int(meta["size"])
        if model_release.file_ok(target, meta):
            done += size
            _report(done)
            continue
        if target.exists():
            target.unlink()
        base = done
        seen = [0]

        def on_bytes(n: int, _seen: list = seen, _base: int = base) -> None:
            _seen[0] += n
            _report(_base + _seen[0])

        url = f"https://huggingface.co/{REPO}/resolve/{e.source_revision}/{name}"
        model_release.fetch_file(url, target, meta, on_bytes, opener, timeout)
        done += size
    progress(1.0, tr("progress.model_verifying", short=LABEL))
    log.info("ACE-Step 1.5 ready in %s", folder)
    return folder


def enable(progress: Callable[[float, str], None] = _silent, models_dir: Optional[Path] = None,
           manifest: Optional[Path] = None) -> Path:
    """Turn the setting on and download the model if it is not already on disk."""
    folder = ensure(progress, models_dir=models_dir, manifest=manifest)
    _save(True)
    return folder


def disable() -> None:
    """Turn the setting off. Downloaded files stay on disk."""
    _save(False)


def generation_allowed() -> bool:
    """False when CUDA is present and the free budget is under 4 GB. Without a GPU, generation is allowed."""
    try:
        import torch  # optional: present in the app, absent in some test environments
    except ImportError:
        return True
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return True
    try:
        free, total = cuda.mem_get_info()
    except Exception:  # noqa: BLE001 - a probe failure must not block a machine we cannot measure
        return True
    frac = gpu_prefs.vram_fraction()
    budget = int(free if frac is None else min(free, frac * total))
    if budget < MIN_FREE_BYTES:
        log.warning("soundscape: %s bytes free is under the 4 GB floor; cues will be skipped", budget)
        return False
    return True


def generate(prompt: str, seconds: float, sample_rate: int, kind: str) -> np.ndarray:
    """Render one clip with ACE-Step. ``sfx`` is rendered by the same call as an accent.

    Raises :class:`SoundscapeModelError` when the model is not installed or there is not enough video memory.
    The package import stays inside this function: it is an optional dependency, not a cycle.
    """
    if not ready():
        raise SoundscapeModelError("ACE-Step 1.5 is not installed")
    if not generation_allowed():
        raise SoundscapeModelError("not enough free video memory for ACE-Step")
    try:
        import acestep  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SoundscapeModelError("the ACE-Step package is not installed") from exc
    pipeline = getattr(acestep, "pipeline", None)
    factory = getattr(pipeline, "ACEStepPipeline", None) if pipeline is not None else None
    if factory is None:
        factory = getattr(acestep, "ACEStepPipeline", None)
    if factory is None:
        raise SoundscapeModelError("the ACE-Step package has no pipeline")
    pipe = factory(checkpoint_dir=str(model_dir()))
    audio = pipe(prompt, audio_duration=float(seconds), infer_step=8)
    samples = np.asarray(getattr(audio, "samples", audio), dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise SoundscapeModelError(f"ACE-Step returned no audio for a {kind} cue")
    return samples
