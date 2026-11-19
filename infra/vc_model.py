"""The optional direct-conversion model: OpenVoice V2 tone-colour converter (MIT).

Not part of the first-run download. The Re-voice window fetches it only when the user presses *Download model*;
the bytes come from the shared model store (manifest row ``openvoice-v2``, files
``myshell-ai--OpenVoiceV2/converter/*``) through VOXPrint MDL (:mod:`infra.mdl_models`).
The checkpoint is about 125 MB and runs on the NVIDIA GPU when CUDA is available, otherwise on the CPU.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from infra import mdl_models

log = logging.getLogger("voxprint.revoice")

REPO = "myshell-ai/OpenVoiceV2"
LABEL = "OpenVoice V2"
MODEL_ID = "openvoice-v2"
#: Only the tone-colour converter. Base-speaker TTS checkpoints are not used and not fetched.
FILES = ("converter/config.json", "converter/checkpoint.pth")


def model_dir(models_dir: Optional[Path] = None) -> Path:
    """``views/audiobook-builder/openvoice-v2-<digest>/myshell-ai--OpenVoiceV2`` (the folder the converter loads from)."""
    if models_dir is not None:
        return Path(models_dir) / "myshell-ai--OpenVoiceV2"
    return mdl_models.view_dir(MODEL_ID) / REPO.replace("/", "--")


def download_bytes() -> int:
    """Bytes of the converter download (the manifest row)."""
    return mdl_models.model_bytes(MODEL_ID)


def download_mb() -> int:
    """The size shown on the button, in megabytes."""
    return max(1, int(round(download_bytes() / 1e6)))


def ready(models_dir: Optional[Path] = None, manifest: Optional[Path] = None) -> bool:
    """True when the store holds a complete, verified view of the converter (a stub counts as missing)."""
    del manifest
    if models_dir is not None:   # tests pass an explicit root with plain files
        folder = model_dir(models_dir)
        try:
            return all((folder / name).is_file() for name in FILES)
        except OSError:
            return False
    return mdl_models.is_ready(MODEL_ID)


def _silent_progress(f: float, m: str = "") -> None:
    pass


def ensure(progress: Callable[[float, str], None] = _silent_progress, models_dir: Optional[Path] = None,
           opener=None, timeout: float = 30.0, manifest: Optional[Path] = None) -> Path:
    """Fetch the converter through VOXPrint MDL, only on the user's explicit request.

    Returns the model folder (``.../myshell-ai--OpenVoiceV2``).  Raises :class:`core.errors.ModelDownloadError`
    when the store cannot serve it."""
    del opener, timeout, manifest
    if models_dir is not None:   # tests pass an explicit root: nothing to fetch
        return model_dir(models_dir)
    view = mdl_models.ensure(MODEL_ID, progress=progress)
    folder = view / REPO.replace("/", "--")
    log.info("OpenVoice V2 converter ready in %s", folder)
    return folder
