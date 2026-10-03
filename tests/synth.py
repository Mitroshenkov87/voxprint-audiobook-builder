"""Synthetic "reading": audio with a known ground-truth timeline plus an aligner that mimics the real one.

Used by the dataset-builder, slicer and quality tests, so the whole pipeline can be exercised without any model.
The vocabulary is Russian on purpose (the main target language of the text normalizer); it is test data."""
from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from core import audio_utils as au
from core.aligner import BaseAligner
from core.text_utils import clean_token
from core.types import WordTiming

VOCAB = ("дом лес река небо город человек время слово дорога вечер утро книга окно ветер зима лето "
         "тихий большой новый старый далёкий светлый говорил шёл видел любил помнил ждал").split()


def make_text(n_sentences: int, seed: int = 1) -> str:
    """Deterministic pseudo-random Russian text of ``n_sentences`` sentences with commas and sentence-final punctuation."""
    rnd = random.Random(seed)
    sents = []
    for _ in range(n_sentences):
        n = rnd.randint(5, 14)
        words = [rnd.choice(VOCAB) for _ in range(n)]
        if n > 8:
            words[n // 2] += ","
        sents.append(" ".join(words).capitalize() + rnd.choice([".", ".", "!", "?"]))
    return " ".join(sents)


def gap_after(word: str, char_dur_gap=(0.06, 0.4, 0.7)) -> float:
    """Pause (seconds) after a word: long after ``.!?``, medium after ``,;:``, short otherwise."""
    if word and word[-1] in ".!?…":
        return char_dur_gap[2]
    if word and word[-1] in ",;:":
        return char_dur_gap[1]
    return char_dur_gap[0]


def truth_timeline(text: str, start: float = 0.5, char_dur: float = 0.07) -> List[Tuple[str, float, float]]:
    """Ground truth: ``[(clean_word, start, end)]`` for the text at a constant speaking rate (``char_dur`` s per letter)."""
    out, t = [], start
    for tok in text.split():
        c = clean_token(tok)
        if not c:
            continue
        d = len(c) * char_dur
        out.append((c, t, t + d))
        t += d + gap_after(tok)
    return out


def synth_reading(text: str, sr: int = 16000, seed: int = 0, noise: float = 0.002, tail: float = 0.8):
    """Synthesize audio for the text from :func:`truth_timeline` (harmonic bursts over noise); returns ``(samples, timeline)``."""
    tl = truth_timeline(text)
    total = tl[-1][2] + tail
    rnd = np.random.default_rng(seed)
    x = rnd.normal(0, noise, int(total * sr)).astype(np.float32)
    for _, a, b in tl:
        n0, n1 = int(a * sr), int(b * sr)
        t = np.arange(n1 - n0) / sr
        env = np.minimum(1.0, np.minimum(t, (n1 - n0) / sr - t) / 0.02)
        f = 140 + 60 * rnd.random()
        sig = (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 2 * f * t)) * 0.25 * env
        x[n0:n1] += sig.astype(np.float32)
    return x, tl


class TrueRateAligner(BaseAligner):
    """Behaves like the real aligner: reads the text at the \"true\" speaking rate from the start of speech in the chunk.
    Surplus text collapses into zero-length words at the end of the audio; too little text leaves a tail."""

    def __init__(self):
        """Counts the calls and remembers whether load() was called."""
        self.calls = 0
        self.loaded = False

    def load(self):
        """Mark the aligner as loaded."""
        self.loaded = True

    def align(self, audio, sr, text, language):
        """Time the words as if spoken at the true rate starting at the first voiced sample of ``audio``."""
        self.calls += 1
        vs, _ = au.voiced_bounds(audio, sr)
        end = len(audio) / sr
        out = []
        for w, a, b in truth_timeline(text, start=vs):
            a, b = min(a, end), min(b, end)
            out.append(WordTiming(word=w, start=a, end=b))
        return out
