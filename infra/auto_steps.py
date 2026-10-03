"""Automatic quality steps: which of them run by default, the one-click "Maximum quality (auto)" and its model downloads.

Every step below is pre-selected (and marked "recommended") in its window, so a user who touches nothing gets the maximum
quality; the individual check boxes stay where they were.  *Settings > Maximum quality (auto)* re-selects all of them in the
open windows and downloads the models they need (:func:`needed_models`, :func:`download_models`).

Steps that exist (each one has code and tests):

* text preparation, rule based (:mod:`core.text_prep`): layout, noise, quotes, links, headings, numbers, abbreviations,
* neural clean-up of typos (:mod:`core.text_cleanup`, Russian books, model downloaded once),
* quality check of a trained voice (:mod:`core.voice_check`, needs the speech-recognition model),
* A/B comparison of two quick trainings with a recommendation (:mod:`workers.preview_runner`, same model).

Steps that are always on and have no switch: forced alignment of the text onto the audio with its plausibility check
(:mod:`core.aligner`), the audio quality gates of the dataset (:mod:`core.quality`) and the speech-recognition gates of
the audio-only mode (:mod:`core.asr_dataset`).  Not included, because they do not exist yet: punctuation model, stress marks,
translation, speaker roles (:data:`infra.text_models.REGISTRY`, ``integrated=False``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

from core import text_prep
from core.events import Stage
from infra import model_downloader as md
from infra import text_models

log = logging.getLogger("voxprint.auto_steps")

STEP_SPELLFIX = text_models.STEP_SPELLFIX
STEP_CHECK, STEP_COMPARE = "quality_check", "compare"
RULE_STEPS: Tuple[str, ...] = text_prep.STEP_KEYS
#: Every switchable automatic step that is implemented.
STEPS: Tuple[str, ...] = RULE_STEPS + (STEP_SPELLFIX, STEP_CHECK, STEP_COMPARE)
#: Steps without a switch (always on): shown only in the manual.
ALWAYS_ON: Tuple[str, ...] = ("align", "align_check", "audio_gates", "asr_gates")


@dataclass(frozen=True)
class ModelNeed:
    """One heavy model a step needs: ``kind`` is ``"hf"`` (:mod:`infra.model_downloader`) or ``"text"`` (:mod:`infra.text_models`)."""
    key: str
    kind: str
    label: str
    size_gb: float


def needed_models(steps: Optional[Iterable[str]] = None) -> List[ModelNeed]:
    """Models that the enabled ``steps`` (default: all of :data:`STEPS`) need and that are not on the disk yet (a partial
    download counts as missing, it is resumed)."""
    on = set(STEPS if steps is None else steps)
    out: List[ModelNeed] = []
    # alignment is always on: the aligner is part of the first-start download, but it is repaired here if it is missing
    wanted: List[Tuple[str, str]] = [(md.ALIGNER_REPO, "Qwen3-ForcedAligner-0.6B")]
    if STEP_CHECK in on or STEP_COMPARE in on:
        wanted.append((md.ASR_REPO, "Qwen3-ASR-0.6B"))
    for repo, label in wanted:
        if md.model_state(repo) != md.STATE_READY:
            out.append(ModelNeed(repo, "hf", label, md.APPROX_SIZE_GB.get(repo, 2.0)))
    if STEP_SPELLFIX in on:
        m = text_models.get("sage-ru")
        if text_models.state(m) == text_models.STATE_NEEDS_DOWNLOAD:
            out.append(ModelNeed(m.key, "text", m.name, m.size_mb / 1024))
    return out


def download_models(needs: List[ModelNeed], progress: Callable[[float, str], None],
                    ensure_hf: Callable[..., object] = md.ensure_model,
                    ensure_text: Callable[..., object] = text_models.ensure) -> int:
    """Download ``needs`` one after another; ``progress(overall_fraction, message)``.  Returns how many were fetched.

    A failure of one model raises (the caller shows it); models fetched before stay on the disk."""
    total = sum(n.size_gb for n in needs) or 1.0
    done_gb = 0.0
    for n in needs:
        base = done_gb / total
        share = n.size_gb / total

        def sub(_stage: Stage, f: float, msg: str = "", base=base, share=share, n=n) -> None:
            progress(min(1.0, base + share * float(f)), msg or n.label)

        if n.kind == "text":
            ensure_text(text_models.get(n.key), sub)
        else:
            ensure_hf(n.key, sub)
        done_gb += n.size_gb
        progress(min(1.0, done_gb / total), n.label)
    return len(needs)
