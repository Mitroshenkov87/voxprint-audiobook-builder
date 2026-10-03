"""Registry of optional, on-demand downloadable text models (clean-up, stress, translation, roles).

Only models that are small, permissively licensed (MIT / Apache-2.0 / CC-BY) and testable behind a mockable interface are
marked ``integrated``; the rest are listed as ``planned`` so the UI can show them greyed ("coming later") and so the
extension point is visible in one place.  Nothing here is downloaded automatically: the user ticks the option and presses
*Download*; :func:`ensure` then fetches the pinned revision into the models folder (HF -> ModelScope fallback, resumable,
see :mod:`infra.model_downloader`).

Sources of the choices: the research notes on Russian clean-up and translation models (2026-10-03).  Sizes are Hugging Face
file sizes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from core.events import ProgressCallback, noop_progress
from infra import model_downloader as md

KIND_CLEANUP, KIND_PUNCT, KIND_STRESS, KIND_TRANSLATE, KIND_ROLES = "cleanup", "punct", "stress", "translate", "roles"
STATE_READY, STATE_NEEDS_DOWNLOAD, STATE_PLANNED = "ready", "needs_download", "planned"

#: Step keys of the neural stage (UI checkboxes).
STEP_SPELLFIX, STEP_PUNCT, STEP_STRESS, STEP_TRANSLATE, STEP_ROLES = "spellfix", "punct", "stress", "translate", "roles"


@dataclass(frozen=True)
class TextModel:
    """One entry of the registry."""
    key: str
    kind: str
    step: str                       # UI step this model serves
    name: str
    languages: Tuple[str, ...]
    repo: str                       # Hugging Face repository ("" = no model chosen yet)
    revision: str                   # pinned commit ("" = not pinned / not integrated)
    size_mb: int
    license: str
    integrated: bool                # True = code exists (engine + validator + tests); False = placeholder

    @property
    def local_dir(self) -> Path:
        """Folder inside the models directory."""
        return md.local_dir_for(self.repo)


REGISTRY: Tuple[TextModel, ...] = (
    TextModel("sage-ru", KIND_CLEANUP, STEP_SPELLFIX, "SAGE FRED-T5 distilled 95M", ("ru",),
              "ai-forever/sage-fredt5-distilled-95m", "ed51b4a46603931380951a3d8456685c9215864f", 365, "MIT", True),
    TextModel("rupunct-ru", KIND_PUNCT, STEP_PUNCT, "RUPunct big", ("ru",), "RUPunct/RUPunct_big", "", 710, "MIT", False),
    TextModel("spell-en", KIND_CLEANUP, STEP_SPELLFIX, "English spelling correction (BART base)", ("en",),
              "oliverguhr/spelling-correction-english-base", "", 535, "MIT", False),
    TextModel("spell-de", KIND_CLEANUP, STEP_SPELLFIX, "German spelling correction", ("de",),
              "oliverguhr/spelling-correction-german-base", "", 950, "Apache-2.0", False),
    TextModel("stress-ru", KIND_STRESS, STEP_STRESS, "Russian stress and ё", ("ru",), "", "", 0, "", False),
    TextModel("translate-opus", KIND_TRANSLATE, STEP_TRANSLATE, "Opus-MT (ru<->en, de<->en)", ("ru", "en", "de"),
              "Helsinki-NLP/opus-mt-ru-en", "", 300, "CC-BY-4.0", False),
    TextModel("translate-madlad", KIND_TRANSLATE, STEP_TRANSLATE, "MADLAD-400 3B (CTranslate2 int8)", ("ru", "en", "de", "uk", "be"),
              "", "", 2950, "Apache-2.0", False),
    TextModel("roles", KIND_ROLES, STEP_ROLES, "Speaker / role markup", (), "", "", 0, "", False),
)


def get(key: str) -> TextModel:
    """Registry entry by key (``KeyError`` if unknown)."""
    for m in REGISTRY:
        if m.key == key:
            return m
    raise KeyError(key)


def for_step(step: str, language: str = "") -> Optional[TextModel]:
    """Best entry for a UI step and a language code (``""`` = any); integrated models first."""
    cands = [m for m in REGISTRY if m.step == step and (not language or not m.languages or language in m.languages)]
    cands.sort(key=lambda m: not m.integrated)
    return cands[0] if cands else None


def state(model: TextModel) -> str:
    """``ready`` (downloaded), ``needs_download`` or ``planned`` (no code / no model yet)."""
    if not model.integrated or not model.repo:
        return STATE_PLANNED
    return STATE_READY if md.verify_local_model(model.local_dir) else STATE_NEEDS_DOWNLOAD


def ensure(model: TextModel, progress: ProgressCallback = noop_progress, **kw) -> Path:
    """Download the model if needed (pinned revision) and return its folder."""
    if not model.integrated or not model.repo:
        raise ValueError(f"{model.key} is a placeholder")
    return md.ensure_model(model.repo, progress, revision=model.revision or None, reuse_external=False, **kw)


def make_engine(model: TextModel, device: str = ""):
    """Create the engine of an integrated model (no weights are loaded until the first use)."""
    from core.text_cleanup import SageEngine

    if model.key != "sage-ru":
        raise ValueError(f"no engine for {model.key}")
    return SageEngine(model.local_dir, model.revision, device)


def cleanup_engine_for(language: str):
    """Clean-up engine for a language code if its model is downloaded, else ``None`` (the step is skipped)."""
    m = for_step(STEP_SPELLFIX, language)
    if m is None or not m.integrated or state(m) != STATE_READY:
        return None
    return make_engine(m)


def build_plan(rule_steps, neural_steps=frozenset()):
    """:class:`core.book_prep.PrepPlan` from the UI selections (rule step keys, neural step keys)."""
    from core.book_prep import PrepPlan
    from core.text_prep import PrepOptions

    neural = frozenset(neural_steps) & {STEP_SPELLFIX}                 # only integrated neural steps can run
    return PrepPlan(PrepOptions(frozenset(rule_steps)), neural, cleanup_engine_for if neural else None)
