"""Speech recognition for the "no transcript" mode of the Train window (Qwen3-ASR, offline, no Qt).

The recogniser turns a clip into text; :func:`plausibility` then rates how believable that text is for that clip.  The
ASR model does not expose per-token probabilities through the public API, so the confidence used by the quality gate is
a cheap *plausibility* score computed from the clip and the text (speaking rate, repetition loops, script of the
language, garbage characters).  It reliably catches the typical ASR failures on bad audio (hallucinated loops, empty or
absurdly dense text, wrong script) without a second model; a model-provided confidence, if an engine has one, is
combined by taking the minimum.
"""
from __future__ import annotations

import abc
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger("voxprint.asr")

ASR_REPO = "Qwen/Qwen3-ASR-0.6B"
ASR_SR = 16000
#: A plausible speaking rate in letters per second (normal speech is ~10-18; whispers/fast speech stretch it).
CPS_OK = (4.0, 24.0)


@dataclass
class AsrResult:
    """Recognised ``text`` of one clip; ``confidence`` is the engine's own score in 0..1 (None if it has none)."""
    text: str
    confidence: Optional[float] = None
    language: str = ""


class BaseASR(abc.ABC):
    """Recogniser interface (injectable for tests)."""

    def load(self) -> None:  # noqa: B027 - optional
        """Load the model."""

    def unload(self) -> None:  # noqa: B027 - optional
        """Free the model."""

    @abc.abstractmethod
    def transcribe(self, audio: np.ndarray, sr: int, language: Optional[str] = None) -> AsrResult:
        """Recognise one clip (mono float32); ``language`` is a name such as ``"Russian"`` or None for auto."""


class Qwen3ASR(BaseASR):
    """Qwen3-ASR through the ``qwen_asr`` package (the same package the aligner uses)."""

    def __init__(self, model_path: str = ASR_REPO, device: str = "auto") -> None:
        self.model_path, self.device, self._m = model_path, device, None

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        from core import model_cache

        cuda = torch.cuda.is_available() and self.device in ("auto", "cuda")
        pre = model_cache.take(model_cache.key("asr", self.model_path, cuda))      # "Preload models at startup"
        if pre is not None:
            self._m = model_cache.to_device(pre, "cuda:0") if cuda else pre
            return
        self._m = Qwen3ASRModel.from_pretrained(self.model_path, dtype=torch.bfloat16 if cuda else torch.float32,
                                                device_map="cuda:0" if cuda else "cpu")

    def unload(self) -> None:
        self._m = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def transcribe(self, audio: np.ndarray, sr: int, language: Optional[str] = None) -> AsrResult:
        import soundfile as sf

        from core import audio_utils as au

        if self._m is None:
            self.load()
        x = au.resample(np.asarray(audio, dtype=np.float32), sr, ASR_SR) if sr != ASR_SR else np.asarray(audio, dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:   # the package takes a file path; a 16 kHz wav is the cheapest input
            f = Path(td) / "clip.wav"
            sf.write(str(f), x, ASR_SR)
            r = self._m.transcribe(audio=str(f), language=language)[0]
        return AsrResult(str(getattr(r, "text", "")).strip(), None, str(getattr(r, "language", "") or ""))


class FakeASR(BaseASR):
    """Test double: returns ``texts[i]`` for the i-th call (or a function of the clip)."""

    def __init__(self, texts) -> None:
        self.texts, self.calls = texts, 0

    def transcribe(self, audio: np.ndarray, sr: int, language: Optional[str] = None) -> AsrResult:
        t = self.texts(audio, sr) if callable(self.texts) else self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return t if isinstance(t, AsrResult) else AsrResult(t)


def _letters(text: str) -> str:
    return "".join(ch for ch in text if ch.isalpha())


def plausibility(text: str, seconds: float, language: Optional[str] = None) -> float:
    """How believable ``text`` is as the transcript of a ``seconds``-long clip, 0..1 (see the module docstring)."""
    t = text.strip()
    letters = _letters(t)
    if not letters or seconds <= 0:
        return 0.0
    score = 1.0
    cps = len(letters) / seconds
    lo, hi = CPS_OK
    if cps < lo:
        score *= max(0.0, cps / lo)
    elif cps > hi:
        score *= max(0.0, 1.0 - (cps - hi) / hi)
    words = re.findall(r"\w+", t.lower())
    if len(words) >= 6:
        if len(set(words)) / len(words) < 0.35:   # "и и и и и" / looping hallucination
            score *= 0.3
        run = best = 1
        for a, b in zip(words, words[1:]):
            run = run + 1 if a == b else 1
            best = max(best, run)
        if best >= 4:
            score *= 0.2
    junk = sum(1 for ch in t if not (ch.isalnum() or ch.isspace() or ch in ".,!?;:-\u2014\u2013'\"\u00ab\u00bb()\u2026"))
    score *= max(0.0, 1.0 - 4.0 * junk / max(1, len(t)))
    if language and language.lower() == "russian":
        cyr = sum(1 for ch in letters if "\u0400" <= ch <= "\u04ff")
        score *= min(1.0, (cyr / len(letters)) / 0.8)
    elif language and language.lower() == "english":
        lat = sum(1 for ch in letters if ch.isascii())
        score *= min(1.0, (lat / len(letters)) / 0.8)
    return float(max(0.0, min(1.0, score)))


def make_default_asr(model_path: str, device: str = "auto") -> BaseASR:
    """The recogniser the app uses."""
    return Qwen3ASR(model_path, device)


def split_at_pauses(x: np.ndarray, sr: int, max_s: float = 14.0, min_s: float = 3.0) -> List[tuple]:
    """Cut a long recording into (start, end) sample ranges of at most ``max_s`` seconds at the quietest pauses.

    The last pause inside the allowed window is used (so the pieces are as long as allowed, which gives the trainer few,
    natural clips); with no pause there, the piece is cut hard at ``max_s``.
    """
    from core import audio_utils as au

    n = len(x)
    if n / sr <= max_s:
        return [(0, n)]
    centers = [int(((a + b) / 2) * sr) for a, b in au.find_silences(x, sr, min_len_s=0.2)]
    out, pos = [], 0
    while n - pos > max_s * sr:
        window = [c for c in centers if pos + min_s * sr <= c <= pos + max_s * sr]
        cut = window[-1] if window else int(pos + max_s * sr)
        out.append((pos, cut))
        pos = cut
    if n - pos > 0:
        out.append((pos, n))
    return out
