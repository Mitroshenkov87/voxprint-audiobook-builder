"""Optional soundscape model: ACE-Step 1.5 turbo (MIT code and MIT weights).

Off by default. Nothing is downloaded until the user turns the setting on (:func:`enable`). Narration never
downloads it. The files are the shared-store row ``ace-step`` (``ace-step-1.5/...``,
:mod:`infra.mdl_models`) and are fetched through VOXPrint MDL.

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

from infra import gpu_prefs, mdl_models

log = logging.getLogger("voxprint.soundscape")

REPO = "ACE-Step/Ace-Step1.5"
LABEL = "ACE-Step 1.5"
MODEL_ID = "ace-step"
FOLDER_NAME = "ace-step-1.5"
MIN_FREE_BYTES = 4 * 1024 ** 3
#: Kept for callers that cache a clip against the pinned revision.
SOURCE_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"


def source_revision(manifest: Optional[Path] = None) -> str:
    """Pinned source revision of the ACE-Step repository (the manifest row is keyed by id; the revision pins the cache)."""
    del manifest
    return SOURCE_REVISION


class SoundscapeModelError(Exception):
    """The soundscape model is missing or refused to run. Narration continues without that cue."""


def _state_file() -> Path:
    from infra import paths

    return paths.state_dir() / "soundscape.json"


def enabled() -> bool:
    """True after the user has turned the soundscape on. A missing file means off."""
    try:
        data = json.loads(_state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(data.get("enabled")) if isinstance(data, dict) else False


def _save(on: bool) -> None:
    path = _state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"enabled": bool(on)}), encoding="utf-8")


def model_dir(models_dir: Optional[Path] = None) -> Path:
    """``<view>/ace-step-1.5`` (the folder the pipeline loads from)."""
    if models_dir is None:
        models_dir = mdl_models.view_dir(MODEL_ID)
    return Path(models_dir) / FOLDER_NAME


def ready(models_dir: Optional[Path] = None, manifest: Optional[Path] = None) -> bool:
    """True when the store holds a complete, verified view of the model (a stub counts as missing)."""
    del manifest
    if models_dir is not None:
        folder = model_dir(models_dir)
        try:
            return folder.is_dir() and any(folder.rglob("*.safetensors"))
        except OSError:
            return False
    return mdl_models.is_ready(MODEL_ID)


def _silent(frac: float, message: str = "") -> None:
    return None


def ensure(progress: Callable[[float, str], None] = _silent, models_dir: Optional[Path] = None,
           opener=None, timeout: float = 30.0, manifest: Optional[Path] = None) -> Path:
    """Fetch the model through VOXPrint MDL. Call this only from :func:`enable`.

    Returns the model folder (``<view>/ace-step-1.5``). Raises :class:`core.errors.ModelDownloadError`."""
    del opener, timeout, manifest
    if models_dir is not None:
        return model_dir(models_dir)   # tests pass an explicit root: nothing to fetch
    view = mdl_models.ensure(MODEL_ID, progress=progress)
    folder = view / FOLDER_NAME
    log.info("ACE-Step 1.5 ready in %s", folder)
    return folder


def enable(progress: Callable[[float, str], None] = _silent, models_dir: Optional[Path] = None,
           manifest: Optional[Path] = None) -> Path:
    """Turn the setting on and fetch the model through MDL if it is not already on disk."""
    folder = ensure(progress, models_dir=models_dir, manifest=manifest)
    _save(True)
    return folder


def disable() -> None:
    """Turn the setting off. The store keeps the files (they belong to VOXPrint MDL)."""
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
        import acestep
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
