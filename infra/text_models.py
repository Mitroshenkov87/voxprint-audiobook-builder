"""Registry of optional, on-demand downloadable text models (clean-up, stress, translation, roles).

Only models that are small, permissively licensed (MIT / Apache-2.0 / CC-BY) and testable behind a mockable interface are
marked ``integrated``; the rest are listed as ``planned`` so the UI can show them greyed ("coming later") and so the
extension point is visible in one place.  SAGE (Russian clean-up) is part of the first-run / ``--prefetch`` download-all
alongside the main HF models; other integrated models (translation) stay on demand: the user ticks the option and presses
*Download*; :func:`ensure` then fetches the pinned revision into the models folder (HF -> ModelScope fallback, resumable,
see :mod:`infra.model_downloader`).

Sources of the choices: the research notes on Russian clean-up and translation models (2026-10-03).  Sizes are Hugging Face
file sizes.
"""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from core.errors import ModelDownloadError
from core.events import ProgressCallback, noop_progress
from core.i18n import tr
from infra import model_downloader as md
from infra import model_mirrors

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
    #: ``((file, sha256), ...)``: files whose SHA-256 is checked after the download (a mismatch deletes the model).
    sha256: Tuple[Tuple[str, str], ...] = ()
    #: Glob patterns of the files to download ("" = the whole repository).
    files: Tuple[str, ...] = ()
    pair: Tuple[str, str] = ("", "")   # translation models: (source, target) language

    @property
    def local_dir(self) -> Path:
        """Folder inside the models directory."""
        return md.local_dir_for(self.repo)


#: Files of an Opus-MT repository that are needed (the repositories also hold TF / Rust / Flax copies of the weights).
OPUS_FILES: Tuple[str, ...] = ("config.json", "generation_config.json", "tokenizer_config.json", "vocab.json", "source.spm",
                               "target.spm", "pytorch_model.bin")

REGISTRY: Tuple[TextModel, ...] = (
    TextModel("sage-ru", KIND_CLEANUP, STEP_SPELLFIX, "SAGE FRED-T5 distilled 95M", ("ru",),
              "ai-forever/sage-fredt5-distilled-95m", "ed51b4a46603931380951a3d8456685c9215864f", 365, "MIT", True),
    TextModel("rupunct-ru", KIND_PUNCT, STEP_PUNCT, "RUPunct big", ("ru",), "RUPunct/RUPunct_big", "", 710, "MIT", False),
    TextModel("spell-en", KIND_CLEANUP, STEP_SPELLFIX, "English spelling correction (BART base)", ("en",),
              "oliverguhr/spelling-correction-english-base", "", 535, "MIT", False),
    TextModel("spell-de", KIND_CLEANUP, STEP_SPELLFIX, "German spelling correction", ("de",),
              "oliverguhr/spelling-correction-german-base", "", 950, "Apache-2.0", False),
    TextModel("stress-ru", KIND_STRESS, STEP_STRESS, "Russian stress marks and the letter yo", ("ru",), "", "", 0, "", False),
    TextModel("opus-ru-en", KIND_TRANSLATE, STEP_TRANSLATE, "Opus-MT ru -> en", ("ru", "en"), "Helsinki-NLP/opus-mt-ru-en",
              "fbd6dc73284f95536648512cc21d57f19191961a", 300, "CC-BY-4.0", True,
              (("pytorch_model.bin", "535450eb5613f3cc912f9ca3e54cfef6c14d201b319c24a88faf776a65538b5d"),), OPUS_FILES, ("ru", "en")),
    TextModel("opus-en-ru", KIND_TRANSLATE, STEP_TRANSLATE, "Opus-MT en -> ru", ("en", "ru"), "Helsinki-NLP/opus-mt-en-ru",
              "bb09c99d180016eac6819df3dae68edb1690fdee", 300, "Apache-2.0", True,
              (("pytorch_model.bin", "d15fa58c6bc3efd3629c1b6b86d9aa6d15d2751a4620aa4cdd7eed7b5cbe583b"),), OPUS_FILES, ("en", "ru")),
    TextModel("opus-de-en", KIND_TRANSLATE, STEP_TRANSLATE, "Opus-MT de -> en", ("de", "en"), "Helsinki-NLP/opus-mt-de-en",
              "1a922f3b32a8e809e17a47d4b32142d8105924e5", 295, "Apache-2.0", True,
              (("pytorch_model.bin", "e743c3070f61f477cb62fe95ef2c9be2e77f3e488cb6b8030ff8a19e8295c87d"),), OPUS_FILES, ("de", "en")),
    TextModel("opus-en-de", KIND_TRANSLATE, STEP_TRANSLATE, "Opus-MT en -> de", ("en", "de"), "Helsinki-NLP/opus-mt-en-de",
              "6183067f769a302e3861815543b9f312c71b0ca4", 295, "CC-BY-4.0", True,
              (("pytorch_model.bin", "da068344b1b0c20e6d4a8f77f48e06646b90d679029006b18579e68977206bdd"),), OPUS_FILES, ("en", "de")),
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


def _hash_ok(model: TextModel, path: Path) -> bool:
    return all(sha256_of(path / name) == digest for name, digest in model.sha256)


def ensure(model: TextModel, progress: ProgressCallback = noop_progress, **kw) -> Path:
    """Download the model if needed (pinned revision) and return its folder.

    Order: the models folder (already downloaded) -> the original repository (Hugging Face) -> the project's Hugging Face backup mirror
    (:mod:`infra.model_mirrors`, if the model is mirrored).  The SHA-256 of the key files is checked on every path; a folder that fails
    the check (corrupt, tampered, or another revision) is deleted and replaced once by the hash-verified mirror copy, otherwise the
    download fails.
    """
    if not model.integrated or not model.repo:
        raise ValueError(f"{model.key} is a placeholder")
    kw.setdefault("reuse_external", False)
    patterns = list(model.files) or None
    path = md.ensure_model(model.repo, progress, revision=model.revision or None, allow_patterns=patterns, **kw)
    if _hash_ok(model, path):
        return path
    shutil.rmtree(path, ignore_errors=True)                       # a corrupt or tampered download is never kept
    err = ModelDownloadError(tr("err.model_hash", short=model.repo.split("/")[-1]), url=md.hf_url(model.repo))
    if kw.get("root") is not None or not model_mirrors.enabled() or model_mirrors.entry_for(model.repo, kw.get("mirror_manifest")) is None:
        raise err
    try:
        path = md.ensure_model(model.repo, progress, revision=model.revision or None, allow_patterns=patterns, mirror_only=True, **kw)
    except ModelDownloadError:
        raise err from None
    if not _hash_ok(model, path):
        shutil.rmtree(path, ignore_errors=True)
        raise err
    return path


def sha256_of(path: Path) -> str:
    """SHA-256 of a file ("" if it cannot be read)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


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


# --------------------------------------------------------------------------- translation (Opus-MT)
def translate_model(source: str, target: str) -> Optional[TextModel]:
    """The integrated Opus-MT model of one direction, or ``None``."""
    for m in REGISTRY:
        if m.kind == KIND_TRANSLATE and m.integrated and m.pair == (source, target):
            return m
    return None


def translate_models(source: str, target: str) -> List[TextModel]:
    """Models of every hop from ``source`` to ``target`` (``[]`` for the same language; ``ValueError`` if unsupported)."""
    from core import translate as tl

    try:
        hops = tl.route(source, target)
    except tl.TranslateError as exc:
        raise ValueError(str(exc)) from exc
    return [m for m in (translate_model(s, t) for s, t in hops) if m is not None]


def make_translator(source: str, target: str, device: str = ""):
    """Translator of one direction if its model is downloaded, else ``None``."""
    from core.translate import MarianEngine

    m = translate_model(source, target)
    if m is None or state(m) != STATE_READY:
        return None
    return MarianEngine(m.local_dir, m.revision, source, target, device)


def build_translate_plan(target: str, source: str = ""):
    """:class:`core.translate.TranslatePlan` for the UI selection (``None`` if no target language is chosen)."""
    from core.translate import TranslatePlan

    return TranslatePlan(target, source, make_translator) if target else None
