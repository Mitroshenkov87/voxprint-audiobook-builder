"""Automatic quality steps: which of them run by default.

Every step below is pre-selected (and marked "recommended") in its window, so a user who touches nothing gets the maximum
quality; the individual check boxes stay where they were.  There is no separate "Maximum quality" button any more: the models
these steps need (speech recognition, SAGE clean-up, the aligner) are part of the first-start download of ALL models
(the Components window of the thin build, then :func:`workers.pipeline_runner.prefetch_models`).

Steps that exist (each one has code and tests):

* text preparation, rule based (:mod:`core.text_prep`): layout, noise, quotes, links, headings, numbers, abbreviations,
  and, for Russian, the letter yo where a dictionary is sure (:mod:`core.yo`),
* neural clean-up of typos (:mod:`core.text_cleanup`, Russian books, model downloaded once),
* quality check of a trained voice (:mod:`core.voice_check`, needs the speech-recognition model),
* A/B comparison of two quick trainings with a recommendation (:mod:`workers.preview_runner`, same model).

Steps that are always on and have no switch: forced alignment of the text onto the audio with its plausibility check
(:mod:`core.aligner`), the audio quality gates of the dataset (:mod:`core.quality`) and the speech-recognition gates of
the audio-only mode (:mod:`core.asr_dataset`).  Not included, because it does not exist yet: a punctuation model.
Russian stress marks are not written: the base speech model does not read them.  Yo restoration is the rule step above.
"""
from __future__ import annotations

import logging
from typing import Tuple

from core import text_prep
from infra import text_models

log = logging.getLogger("voxprint.auto_steps")

STEP_SPELLFIX = text_models.STEP_SPELLFIX
STEP_CHECK, STEP_COMPARE = "quality_check", "compare"
RULE_STEPS: Tuple[str, ...] = text_prep.STEP_KEYS
#: Every switchable automatic step that is implemented.
STEPS: Tuple[str, ...] = RULE_STEPS + (STEP_SPELLFIX, STEP_CHECK, STEP_COMPARE)
#: Steps without a switch (always on): shown only in the manual.
ALWAYS_ON: Tuple[str, ...] = ("align", "align_check", "audio_gates", "asr_gates")
