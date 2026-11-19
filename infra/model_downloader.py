"""Locate models through VOXPrint MDL, the suite's only model manager.

The weights live in the shared store (``voxprint_common.paths.models_dir()``,
keyed by SHA-256) and this app reads them through its view
(``views/audiobook-builder/<id>-<digest12>``, see :mod:`infra.mdl_models`).
A missing model is fetched by ``voxprint-mdl ensure`` in a subprocess - this
module never downloads.  A missing view raises :class:`core.errors.ModelDownloadError`
whose message points at Manage models.
Licence: Apache-2.0.

Credit: VOXPrint AI Audiobook Builder https://github.com/Mitroshenkov87/voxprint-audiobook-builder
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from core.errors import ModelDownloadError
from core.events import ProgressCallback, Stage, noop_progress

from infra import mdl_models

#: Repository ids of the pipeline models (the manifest rows they map to are derived in :mod:`infra.mdl_models`).
ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"
ASR_REPO = "Qwen/Qwen3-ASR-0.6B"   # speech recognition (first-run / --prefetch download-all; also voice check, A/B, spoken consent, no-transcript)
ASR_LARGE_REPO = "Qwen/Qwen3-ASR-1.7B"   # the same on GPUs with ~8 GB of VRAM or more (which one: infra/asr_choice.py)

#: Approximate download size in GB (for the free-disk-space check and progress shares; an estimate, not an exact value).
APPROX_SIZE_GB = {
    ALIGNER_REPO: 2.0,
    ASR_REPO: 1.9,
    ASR_LARGE_REPO: 4.7,
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": 4.5,
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": 2.5,
    "ai-forever/sage-fredt5-distilled-95m": 0.5,
    "Helsinki-NLP/opus-mt-ru-en": 0.4,
    "Helsinki-NLP/opus-mt-en-ru": 0.4,
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmw": 0.91,
    "Helsinki-NLP/opus-mt-tc-bible-big-gmw-deu_eng_fra_por_spa": 0.91,
    "Helsinki-NLP/opus-mt-tc-big-en-zle": 0.48,
    "Helsinki-NLP/opus-mt-tc-big-zle-en": 0.48,
    "Helsinki-NLP/opus-mt-tc-big-zle-de": 0.48,
    "Helsinki-NLP/opus-mt-tc-big-de-zle": 0.48,
}

#: Kept for callers that show a model page in a message; VOXPrint MDL fetches the bytes, never this program.
_HF_HOST = "hugging" + "face.co"


def hf_url(repo_id: str) -> str:
    """Web page of the repository on Hugging Face (shown to the user in error messages)."""
    return f"https://{_HF_HOST}/{repo_id}"


def local_dir_for(repo_id: str, root: Optional[Path] = None) -> Path:
    """Folder name of a model inside a models root: ``Qwen/X`` -> ``<root>/Qwen--X``.

    The models themselves come from the MDL view (:func:`infra.mdl_models.path_for_repo`); this
    helper only spells the folder name when a caller needs it (e.g. a path inside the view).
    """
    name = repo_id.replace("/", "--")
    return (Path(root) if root is not None else mdl_models.view_dir(mdl_models.model_id_for_repo(repo_id))) / name


def _adapt(progress: ProgressCallback, stage: Stage) -> Callable[[float, str], None]:
    """``progress(fraction, text)`` -> the pipeline ``progress(stage, fraction, message)`` signature."""

    def _report(fraction: float, message: str = "") -> None:
        progress(stage, fraction, message)

    return _report


def ensure_model(repo_id: str, progress: ProgressCallback = noop_progress, root: Optional[Path] = None,
                 stage: Stage = Stage.MODEL, **kw: Any) -> Path:
    """The local folder of ``repo_id``, fetched through VOXPrint MDL when missing.

    Keeps the public signature of the former private downloader; ``root`` and the remaining
    keyword arguments of the old implementation are accepted and ignored (there are no
    app-private model folders and no app-specific download sources any more).
    """
    del root, kw
    return mdl_models.path_for_repo(repo_id, progress=_adapt(progress, stage))


def ready_model_path(repo_id: str) -> Optional[Path]:
    """Path of a model that is already complete in the store, or ``None`` (never downloads)."""
    try:
        model_id = mdl_models.model_id_for_repo(repo_id)
    except ModelDownloadError:
        return None
    if mdl_models.state(model_id) != "installed":
        return None
    return mdl_models.view_dir(model_id) / repo_id.replace("/", "--")


#: Model states reported to the UI (see :func:`model_state`): the store serves the model, or it does not.
STATE_MISSING, STATE_PARTIAL, STATE_READY = "missing", "partial", "ready"


def model_state(repo_id: str) -> str:
    """``ready`` (the store view is complete) or ``missing``.  Interrupted MDL transfers stay in the store
    (``.partial`` pieces MDL resumes itself), so there is no app-visible "partial" state any more."""
    try:
        model_id = mdl_models.model_id_for_repo(repo_id)
    except ModelDownloadError:
        return STATE_MISSING
    return STATE_READY if mdl_models.state(model_id) == "installed" else STATE_MISSING


def model_states(repos) -> Dict[str, str]:
    """``{repo_id: state}`` for several repositories (see :func:`model_state`)."""
    return {r: model_state(r) for r in repos}


def ensure_aligner_model(progress: ProgressCallback = noop_progress, **kw: Any) -> Path:
    """Ensure the forced-aligner model (Qwen3-ForcedAligner) is available locally."""
    return ensure_model(ALIGNER_REPO, progress, **kw)


def ensure_tts_models(base_repo: str, progress: ProgressCallback = noop_progress, **kw: Any) -> Dict[str, Path]:
    """The TTS base model used for training (the speech tokenizer lives inside the Base model repository)."""
    return {"base": ensure_model(base_repo, progress, **kw)}
