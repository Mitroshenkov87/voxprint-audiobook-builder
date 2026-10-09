"""Registry of optional, on-demand downloadable text models (clean-up, stress, translation, roles).

Only models that are small, permissively licensed (MIT / Apache-2.0 / CC-BY) and testable behind a mockable interface are
marked ``integrated``; the rest are listed as ``planned`` so the UI can show them greyed ("coming later") and so the
extension point is visible in one place.  SAGE (Russian clean-up) is downloaded together with the runtime components in the thin
build's Components window (:data:`COMPONENT_EXTRAS`) and is also part of the first-run / ``--prefetch`` download-all alongside the
main HF models; other integrated models (translation) stay on demand: the user ticks the option and presses *Download*; :func:`ensure` then fetches the pinned revision into the models folder (HF -> ModelScope fallback, resumable,
see :mod:`infra.model_downloader`).

Sources of the choices: the research notes on Russian clean-up and translation models (2026-10-03).  Sizes are Hugging Face
file sizes.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from core.errors import ModelDownloadError
from core.events import ProgressCallback, noop_progress
from core.i18n import tr
from infra import model_downloader as md
from infra import model_mirrors

log = logging.getLogger("voxprint.models")

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
    #: multi-target Opus-MT tc-big models need the target language as a token in front of every sentence (``>>rus<<``)
    target_token: str = ""

    @property
    def local_dir(self) -> Path:
        """Folder inside the models directory."""
        return md.local_dir_for(self.repo)


#: Files of an Opus-MT repository that are needed (the repositories also hold TF / Rust / Flax copies of the weights).
#: Files of an Opus-MT tc-big model (safetensors only: the repositories also carry pytorch_model.bin and tf_model.h5)
TC_BIG_FILES: Tuple[str, ...] = ("config.json", "generation_config.json", "tokenizer_config.json", "vocab.json",
                                 "special_tokens_map.json", "source.spm", "target.spm", "model.safetensors")
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
    # The letter yo is the rule step ``yo`` (:mod:`core.yo`), shipped with the program. This entry stays off:
    # a stress model would emit marks the base Qwen3-TTS model does not read.
    TextModel("stress-ru", KIND_STRESS, STEP_STRESS, "Russian stress marks and the letter yo", ("ru",), "", "", 0, "", False),
    # Opus-MT tc-big (2022-23, same Marian architecture): preferred; listed BEFORE the 2020 models of the same direction,
    # which stay as a fallback for installs that already have them (make_translator takes the first installed one).
    TextModel("opus-big-en-ru", KIND_TRANSLATE, STEP_TRANSLATE, "Opus-MT tc-big en -> ru", ("en", "ru"), "Helsinki-NLP/opus-mt-tc-big-en-zle",
              "708be1d372fe4c358a352f404e6dc9ca0126ba48", 483, "CC-BY-4.0", True,
              (("config.json", "962d235879c9def4cfdf140be81436d4fb9da25271dd69079df2c90b459f1332"),
               ("generation_config.json", "d851d84c804c745d92dae5659bd2d8e8b976b664dd0fbc39b82ee88340147802"),
               ("tokenizer_config.json", "41deeedfc0e3ce366d6bde180dee025a8ca1bcd62b1451889301e8ea4bcbb609"),
               ("vocab.json", "41dbdff4a0b5a6ab125715c3342c5ce6516e93ffd49608813240403f036c5efb"),
               ("special_tokens_map.json", "09059cedc26bc46bc09a52f05b92d4922e11917e87f3b92059bb1a63a59ab2c4"),
               ("model.safetensors", "e68caa9a233c177a3489257b69c18cece6da97767ab2581918ce3fc3c3899416"),
               ("source.spm", "3612abfe04bf08344ba91115f0e15e228a7a15a621ea856bfd548097dbaeb43c"),
               ("target.spm", "22940e744b3a9fd166a04880938fb61f7dfa8ba4b5d2d3f6371a6c4ba8f3b019"),),
              TC_BIG_FILES, ("en", "ru"), ">>rus<<"),
    TextModel("opus-big-ru-en", KIND_TRANSLATE, STEP_TRANSLATE, "Opus-MT tc-big ru -> en", ("ru", "en"), "Helsinki-NLP/opus-mt-tc-big-zle-en",
              "09a40f722d6d8b76aaad6fe51a06c914622a13d1", 482, "CC-BY-4.0", True,
              (("config.json", "c3a0d99762cd5009ab0d9e793fab27b02a8362bb366191a58af677ea712c1263"),
               ("generation_config.json", "aeafe756fb795970bccbb7c16391b79f42a4f9c1d5f32bc6e36b45f17cfa5a35"),
               ("tokenizer_config.json", "1c99d6f6250779f482836b8380d5fdcaa3ab8c7b6f332a6fcec1d8040eeb6c00"),
               ("vocab.json", "92148daf001ba378588442dfcdc7c4529210f2f28635ebc092faecf508190680"),
               ("special_tokens_map.json", "09059cedc26bc46bc09a52f05b92d4922e11917e87f3b92059bb1a63a59ab2c4"),
               ("model.safetensors", "78608083ed48db06dd4b0afdb58b1ac7e2a3648b295fac58449c131c7bf0cc2c"),
               ("source.spm", "a982dbb9362861151e36b0db1595b324cd1ce09acf46ce1f4d6d624e11c5807f"),
               ("target.spm", "e69faed7f1e60eec38c64cceba35cc4fb05ec6478082f4c8a14b141fd6596e9e"),),
              TC_BIG_FILES, ("ru", "en")),
    TextModel("opus-big-ru-de", KIND_TRANSLATE, STEP_TRANSLATE, "Opus-MT tc-big ru -> de", ("ru", "de"), "Helsinki-NLP/opus-mt-tc-big-zle-de",
              "b2e247f0c413ca6aa51a32f2f2be8666cf72405e", 482, "CC-BY-4.0", True,
              (("config.json", "3365e8a0008b014e8360cfb8a48c21bcc250490e623d9fb38ba57372a9d4975a"),
               ("generation_config.json", "3e5f85e6a764107acbf215671ce7b6a18b21cfb52f13892ff01498c71d4baf50"),
               ("tokenizer_config.json", "9946907fad1e49a4ff5e493feaaf3af386211feac30e9d883aaaea61cf822a14"),
               ("vocab.json", "768fd9a234ab432101826b4fd77ebea6aff67b1e461bc242d81877442db9fcfc"),
               ("special_tokens_map.json", "09059cedc26bc46bc09a52f05b92d4922e11917e87f3b92059bb1a63a59ab2c4"),
               ("model.safetensors", "cab19ba28674308c4e51bc05ebf19dd8f2b9cacbc339a350ec389c47cf591a96"),
               ("source.spm", "ce6c0b9887274cbaef5320aec5db60c5afab66f983723ac4051aa68523f9032f"),
               ("target.spm", "7c23b510033f9329f8f2c6200a5c8f2cedc03133690dbde920c0f4f14e6a8bda"),),
              TC_BIG_FILES, ("ru", "de")),
    TextModel("opus-big-de-ru", KIND_TRANSLATE, STEP_TRANSLATE, "Opus-MT tc-big de -> ru", ("de", "ru"), "Helsinki-NLP/opus-mt-tc-big-de-zle",
              "d4db2a2cbaa6c2f1ea57d0ed40924d35767b05f9", 482, "CC-BY-4.0", True,
              (("config.json", "87339bacf38f0f193b9f3b79c1cc0e4de4ed8fcdbd886b87f2be69caa1236a71"),
               ("generation_config.json", "5c3d58c11c2cf673224cd7bb9c7ad3fac618ffb185c4478144a96623940bd3ef"),
               ("tokenizer_config.json", "4cd738433f516c1205d9619098efdb383cde555fafd035f5410601908553f306"),
               ("vocab.json", "1afcad767fa7863a9a935f5b3e0efa0afa6a6918909b1fbffb14c5e2c07f3ff3"),
               ("special_tokens_map.json", "09059cedc26bc46bc09a52f05b92d4922e11917e87f3b92059bb1a63a59ab2c4"),
               ("model.safetensors", "2af78d2ff8d53c73e88dd032506c0dff28d17fb6a828f8b572afe128dbe1647e"),
               ("source.spm", "dde526685600b958138d2b3863bf9b8be10774bf8a99573aede7db619c9b5eb0"),
               ("target.spm", "3341e4c42304e509ef8068c1b0726be116a8ad90b513c62638db210bf47b4fb6"),),
              TC_BIG_FILES, ("de", "ru"), ">>rus<<"),
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


#: Text models that the thin build's Components window downloads in the same pass as the runtime libraries (no separate click
#: in the Narrate window).  The first-run model download (``workers.pipeline_runner.prefetch_models``) lists them too, as a
#: fallback for full builds and for a Components pass whose extra download failed.
COMPONENT_EXTRAS: Tuple[str, ...] = ("sage-ru",
                                     # the translator: every direction between ru / en / de (ru <-> de direct, no pivot)
                                     "opus-big-en-ru", "opus-big-ru-en", "opus-big-ru-de", "opus-big-de-ru",
                                     "opus-de-en", "opus-en-de")


def missing_component_extras() -> List[TextModel]:
    """Integrated :data:`COMPONENT_EXTRAS` that are not on the disk yet."""
    out = []
    for key in COMPONENT_EXTRAS:
        m = get(key)
        if state(m) == STATE_NEEDS_DOWNLOAD:
            out.append(m)
    return out


def ensure_component_extras(progress: ProgressCallback = noop_progress, ensure_fn=None) -> List[str]:
    """Download the missing :data:`COMPONENT_EXTRAS` one after another (pinned revision, hash-checked, existing models folder /
    backup first: see :func:`ensure`).  ``progress(stage, overall_fraction, message)``; returns the keys that were fetched.

    One model that fails does not stop the others: every model is tried, then the first error is raised again (with
    ``fetched`` / ``failed`` key lists attached) so the caller still learns that something is missing.  Cancelling stops at once."""
    from core.errors import CancelledByUser

    ensure_fn = ensure_fn or ensure
    todo = missing_component_extras()
    total = float(sum(max(1, m.size_mb) for m in todo)) or 1.0
    done = 0.0
    fetched: List[str] = []
    failed: List[Tuple[str, BaseException]] = []
    for m in todo:
        def sub(stage, f, msg="", base=done, share=max(1, m.size_mb)):
            progress(stage, min(1.0, (base + share * float(f)) / total), msg)

        try:
            ensure_fn(m, sub)
            fetched.append(m.key)
        except CancelledByUser:
            raise
        except Exception as exc:  # noqa: BLE001 - the next model is tried; the error is raised after the loop
            log.warning("%s could not be downloaded (%s) - continuing with the next model", m.key, exc)
            failed.append((m.key, exc))
        done += max(1, m.size_mb)
    if failed:
        first = failed[0][1]
        try:
            first.fetched = fetched                     # type: ignore[attr-defined]
            first.failed = [k for k, _ in failed]       # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - an exception type without a __dict__
            pass
        raise first
    return fetched


def missing_other_integrated() -> List[TextModel]:
    """Integrated models outside :data:`COMPONENT_EXTRAS` that are not on the disk yet (the 2020 Opus-MT ru <-> en fallbacks):
    part of the complete set of the Full / Quick setup."""
    return [m for m in REGISTRY if m.integrated and m.repo and m.key not in COMPONENT_EXTRAS
            and state(m) == STATE_NEEDS_DOWNLOAD]


def repos() -> List[str]:
    """Repositories of every integrated text model (they are downloaded with their file patterns, never as a whole)."""
    return [m.repo for m in REGISTRY if m.integrated and m.repo]


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
    """The preferred integrated Opus-MT model of one direction (the one to download), or ``None``."""
    cands = translate_candidates(source, target)
    return cands[0] if cands else None


def translate_candidates(source: str, target: str) -> List[TextModel]:
    """Integrated models of one direction, preferred first (tc-big, then the 2020 model kept as a fallback)."""
    return [m for m in REGISTRY if m.kind == KIND_TRANSLATE and m.integrated and m.pair == (source, target)]


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

    m = next((c for c in translate_candidates(source, target) if state(c) == STATE_READY), None)
    if m is None:
        return None
    return MarianEngine(m.local_dir, m.revision, source, target, device, prefix=m.target_token)


def build_translate_plan(target: str, source: str = ""):
    """:class:`core.translate.TranslatePlan` for the UI selection (``None`` if no target language is chosen)."""
    from core.translate import TranslatePlan

    return TranslatePlan(target, source, make_translator) if target else None
