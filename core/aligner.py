"""Forced alignment of a text onto audio.

Verified against the source of the PyPI package ``qwen-asr`` 0.0.6 and the Hugging Face model card
``Qwen/Qwen3-ForcedAligner-0.6B``::

    from qwen_asr import Qwen3ForcedAligner
    model = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B",
                                               dtype=torch.bfloat16, device_map="cuda:0")
    results = model.align(audio=(np_array, sr), text="...", language="Russian")
    results[0] -> ForcedAlignResult, iterates ForcedAlignItem(text, start_time, end_time)   # seconds

* The unit of alignment is a word (or a character for Chinese); punctuation is dropped.
* The model handles up to ~5 minutes of speech and the package itself cuts at 180 s
  (``MAX_FORCE_ALIGN_INPUT_SECONDS``).  Recordings of 5-15 minutes are therefore cut at pauses into chunks of at most
  150 s and the text is divided by sentences/clauses (see :func:`align_long`).
* Fallback: ``ctc-forced-aligner`` (MahmoudAshraf97, MMS wav2vec2) as :class:`CtcAligner`, used through
  :class:`FallbackAligner` if the primary aligner cannot load or fails.
* The :class:`BaseAligner` interface lets tests plug in :class:`FakeAligner`.
"""
from __future__ import annotations

from core.i18n import tr
import abc
import gc
import logging
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from core import audio_utils as au
from core.errors import AlignmentError, AudioTextMismatchError, OutOfMemoryError_
from core.text_utils import clean_token, count_clean_chars, is_kept_char, split_clauses
from core.types import WordTiming

log = logging.getLogger("voxprint.aligner")

ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
#: Languages supported by Qwen3-ForcedAligner-0.6B (model card).
SUPPORTED_LANGUAGES = (
    "Chinese", "English", "Cantonese", "French", "German", "Italian",
    "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
)
#: Maximum audio length per aligner call (with margin: the model handles up to 300 s, the package itself cuts at 180 s).
MAX_CHUNK_SEC = 150.0
#: ISO-639-3 language codes for ctc-forced-aligner.
ISO3 = {"Russian": "rus", "English": "eng", "Chinese": "cmn", "Cantonese": "yue", "French": "fra", "German": "deu",
        "Italian": "ita", "Japanese": "jpn", "Korean": "kor", "Portuguese": "por", "Spanish": "spa"}


class BaseAligner(abc.ABC):
    """Interface of an aligner: optional ``load``/``unload`` plus ``align``."""

    def load(self) -> None:  # noqa: B027 - optional
        """Load the model (may take long). Default: nothing to do."""

    def unload(self) -> None:  # noqa: B027
        """Free the memory held by the model. Default: nothing to do."""

    @abc.abstractmethod
    def align(self, audio: np.ndarray, sr: int, text: str, language: str) -> List[WordTiming]:
        """Align ``text`` onto ``audio`` (mono float32) and return word timings in seconds from the start of ``audio``."""


class Qwen3Aligner(BaseAligner):
    """Wrapper around ``qwen_asr.Qwen3ForcedAligner`` (transformers backend; bfloat16 on CUDA, float32 on CPU)."""

    def __init__(self, model_path: str = ALIGNER_MODEL_ID, device: str = "auto") -> None:
        """``device``: ``auto`` (CUDA if available, else CPU), ``cuda:0`` or ``cpu``.  The model is loaded lazily by :meth:`load`."""
        self.model_path = model_path
        self.device = device
        self._model = None
        self.resolved_device = "cpu"

    def load(self) -> None:
        """Import qwen-asr/torch lazily and load the model; maps CUDA out-of-memory to ``OutOfMemoryError_``."""
        if self._model is not None:
            return
        try:
            import torch
            from qwen_asr import Qwen3ForcedAligner  # type: ignore
        except ImportError as exc:
            raise AlignmentError(
                tr("err.aligner_missing"),
                details=str(exc),
            ) from exc
        dev = self.device
        if dev == "auto":
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.resolved_device = dev
        dtype = torch.bfloat16 if dev.startswith("cuda") else torch.float32
        try:
            self._model = Qwen3ForcedAligner.from_pretrained(self.model_path, dtype=dtype, device_map=dev)
        except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
            raise OutOfMemoryError_(details=str(exc)) from exc

    def unload(self) -> None:
        """Drop the model and empty the CUDA cache."""
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def align(self, audio: np.ndarray, sr: int, text: str, language: str) -> List[WordTiming]:
        """Align one chunk (the model must receive at most ~3 minutes; see :func:`align_long`). Raises ``AlignmentError`` for unsupported languages."""
        if self._model is None:
            self.load()
        if language not in SUPPORTED_LANGUAGES:
            raise AlignmentError(tr("err.lang_unsupported", language=language))
        try:
            import torch

            oom_type = torch.cuda.OutOfMemoryError  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            oom_type = MemoryError
        try:
            results = self._model.align(audio=(audio.astype(np.float32), int(sr)), text=text, language=language)
        except oom_type as exc:
            raise OutOfMemoryError_(details=str(exc)) from exc
        items = results[0]
        return [WordTiming(word=str(it.text), start=float(it.start_time), end=float(it.end_time)) for it in items]


class CtcAligner(BaseAligner):
    """Backup aligner: ``ctc-forced-aligner`` (``pip install ctc-forced-aligner``; MMS wav2vec2, ~1.2 GB).

    Uses the package's README API: ``load_alignment_model``, ``generate_emissions``, ``preprocess_text``,
    ``get_alignments``, ``get_spans``, ``postprocess_results``; word labels come from the ``text`` field (words are
    separated by spaces).  TODO-needs-GPU-test: not exercised on the development machine (no package/model/network);
    imports are lazy and the result format (dicts with start/end/text) was read from the README and is handled defensively.
    """

    def __init__(self, device: str = "auto", batch_size: int = 8) -> None:
        """``batch_size`` is the number of audio windows the emission model processes at once."""
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """Load the MMS alignment model on the chosen device (fp16 on CUDA)."""
        if self._model is not None:
            return
        try:
            import torch
            from ctc_forced_aligner import load_alignment_model  # type: ignore
        except ImportError as exc:
            raise AlignmentError(tr("err.fallback_missing"), details=str(exc)) from exc
        dev = self.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, self._tokenizer = load_alignment_model(
            dev, dtype=torch.float16 if dev.startswith("cuda") else torch.float32)

    def unload(self) -> None:
        """Drop the model and tokenizer."""
        self._model = self._tokenizer = None
        gc.collect()

    def align(self, audio: np.ndarray, sr: int, text: str, language: str) -> List[WordTiming]:
        """Resample to 16 kHz, run CTC alignment and return the real words (stars/empty tokens are skipped)."""
        if self._model is None:
            self.load()
        import torch
        from ctc_forced_aligner import (generate_emissions, get_alignments, get_spans,  # type: ignore
                                        postprocess_results, preprocess_text)

        wave = torch.from_numpy(np.asarray(au.resample(audio, sr, 16000), dtype=np.float32))
        wave = wave.to(self._model.device).to(self._model.dtype)
        emissions, stride = generate_emissions(self._model, wave.unsqueeze(0) if wave.dim() == 1 else wave,
                                               batch_size=self.batch_size)
        tokens_starred, text_starred = preprocess_text(
            collapse_for_ctc(text), romanize=True, language=ISO3.get(language, "eng"))
        segments, scores, blank = get_alignments(emissions, tokens_starred, self._tokenizer)
        spans = get_spans(tokens_starred, segments, blank)
        res = postprocess_results(text_starred, spans, stride, scores)
        out: List[WordTiming] = []
        for it in res:
            word = str(it["text"])
            if word.strip() in ("", "<star>") or not clean_token(word):
                continue
            out.append(WordTiming(word=clean_token(word), start=float(it["start"]), end=float(it["end"])))
        return out


def collapse_for_ctc(text: str) -> str:
    """Normalize whitespace to single spaces for the CTC pre-processor."""
    return " ".join(text.split())


class FallbackAligner(BaseAligner):
    """Primary aligner plus a backup.

    Switches to the backup if the primary cannot load or raises an alignment error - but *not* on out-of-memory or on
    "audio does not match the text": those are resource/data problems that a different model cannot fix.
    """

    def __init__(self, primary: BaseAligner, fallback: Optional[BaseAligner]) -> None:
        """``fallback`` may be None, in which case a primary failure is simply reported."""
        self.primary, self.fallback = primary, fallback
        self.using_fallback = False

    def load(self) -> None:
        """Load the primary aligner; on failure (other than OOM) try the backup."""
        try:
            self.primary.load()
        except OutOfMemoryError_:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("primary aligner failed to load (%s) - switching to fallback", exc)
            self._switch(exc)

    def _switch(self, exc: BaseException) -> None:
        """Activate the backup; if it is unavailable, re-raise the *original* error (the more useful one)."""
        if self.fallback is None:
            raise exc
        try:
            self.fallback.load()
        except Exception as fb_exc:  # noqa: BLE001 - backup unavailable: report the original cause
            log.warning("fallback aligner unavailable: %s", fb_exc)
            raise exc
        self.using_fallback = True

    def unload(self) -> None:
        """Unload both aligners."""
        self.primary.unload()
        if self.fallback is not None:
            self.fallback.unload()

    def align(self, audio: np.ndarray, sr: int, text: str, language: str) -> List[WordTiming]:
        """Align with the active aligner, switching to the backup on a non-data, non-memory failure."""
        if self.using_fallback:
            return self.fallback.align(audio, sr, text, language)  # type: ignore[union-attr]
        try:
            return self.primary.align(audio, sr, text, language)
        except (OutOfMemoryError_, AudioTextMismatchError):
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("primary aligner failed (%s) - switching to fallback", exc)
            self._switch(exc)
            return self.fallback.align(audio, sr, text, language)  # type: ignore[union-attr]


def make_default_aligner(model_path: str, device: str = "auto") -> "FallbackAligner":
    """Qwen3-ForcedAligner with a ctc-forced-aligner backup (the backup is used only if it is installed)."""
    return FallbackAligner(Qwen3Aligner(model_path, device=device), CtcAligner(device="cpu" if device == "cpu" else "auto"))


class FakeAligner(BaseAligner):
    """A trivial "aligner" without a neural network: words are spread over the voiced part of the audio in
    proportion to their length.  For tests and dry runs (``--fake-aligner``).
    """

    def __init__(self) -> None:
        """``calls`` counts the align() calls (handy in tests)."""
        self.calls = 0

    def align(self, audio: np.ndarray, sr: int, text: str, language: str) -> List[WordTiming]:
        """Distribute the words evenly (by letter count) across the voiced interval of ``audio``."""
        self.calls += 1
        tokens = [clean_token(t) for t in text.split()]
        tokens = [t for t in tokens if t]
        if not tokens:
            return []
        vs, ve = au.voiced_bounds(audio, sr)
        total_chars = sum(len(t) for t in tokens)
        t = vs
        out: List[WordTiming] = []
        for tok in tokens:
            d = (ve - vs) * len(tok) / total_chars
            out.append(WordTiming(word=tok, start=t, end=t + d))
            t += d
        return out


# --------------------------------------------------------------------------- long audio


def _score_alignment(words: Sequence[WordTiming], chunk: np.ndarray, sr: int) -> float:
    """How badly the text fits the audio chunk - lower is better.

    If there is too much text, the trailing words collapse to zero duration at the end of the audio; if there is too
    little, the last words stretch over silence or foreign speech.  The score adds the start/end offsets against the
    voiced interval and penalizes zero-length and very long words.
    """
    if not words:
        return 1e9
    vs, ve = au.voiced_bounds(chunk, sr)
    score = abs(words[0].start - vs) + abs(words[-1].end - ve)
    score += 0.5 * sum(1 for w in words if w.duration < 0.02)
    score += 0.3 * sum(1 for w in words if w.duration > 2.0)
    return float(score)


def _pick_cut(audio: np.ndarray, sr: int, start_s: float, lo_s: float, hi_s: float) -> float:
    """Pick a cut point inside ``[lo_s, hi_s]`` (absolute seconds): the centre of the longest pause (or the quietest frame)."""
    s0, s1 = int(lo_s * sr), int(hi_s * sr)
    seg = audio[s0:s1]
    sil = au.find_silences(seg, sr, min_len_s=0.15)
    if sil:
        a, b = max(sil, key=lambda ab: ab[1] - ab[0])
        return lo_s + (a + b) / 2.0
    fr = au.frame_rms(seg, sr)
    return lo_s + float(np.argmin(fr)) * 0.01 if fr.size else (lo_s + hi_s) / 2.0


def align_long(
    aligner: BaseAligner,
    audio: np.ndarray,
    sr: int,
    text: str,
    language: str,
    max_chunk_sec: float = MAX_CHUNK_SEC,
    on_progress: Optional[Callable[[float, str], None]] = None,
    cancel_check: Optional[Callable[[], None]] = None,
) -> List[WordTiming]:
    """Align a text onto audio of any length; the returned times are from the start of the *whole* audio.

    Algorithm for long recordings (longer than ``max_chunk_sec``), without any ASR:

    1. cut the audio at the longest pause inside the window ``[0.6*max, max]`` from the chunk start;
    2. estimate where that cut falls in the text, proportionally to the *voiced* time (share of voiced frames) and the
       letter counts, then try the neighbouring clause boundaries and keep the one whose alignment looks best
       (:func:`_score_alignment`); the first good candidate (< 0.4) is accepted immediately.

    Short audio goes to the aligner in one call.  TODO-needs-GPU-test: the score thresholds were tuned on synthetic
    data and may need adjustment with the real model.
    """
    cancel_check = cancel_check or (lambda: None)
    on_progress = on_progress or (lambda f, m: None)
    total = len(audio) / float(sr)
    if total <= max_chunk_sec * 1.1:
        cancel_check()
        on_progress(0.0, tr("progress.aligning"))
        words = aligner.align(audio, sr, text, language)
        on_progress(1.0, tr("progress.aligned"))
        return list(words)

    units = split_clauses(text)
    if not units:
        raise AlignmentError(tr("err.no_sentences"))
    unit_chars = [max(1, count_clean_chars(u)) for u in units]
    voiced_total = au.voiced_seconds(audio, sr)
    words_all: List[WordTiming] = []
    t0 = 0.0
    ui = 0  # index of the first clause not yet aligned
    n_units = len(units)

    while True:
        cancel_check()
        remaining = total - t0
        frac_done = min(0.99, t0 / total)
        on_progress(frac_done, tr("progress.aligning_chunk", done=int(t0 // 60), total=int(total // 60) + 1))
        if remaining <= max_chunk_sec * 1.1 or ui >= n_units - 1:
            chunk = audio[int(t0 * sr):]
            chunk_text = " ".join(units[ui:])
            ws = aligner.align(chunk, sr, chunk_text, language)
            for w in ws:
                w.start += t0
                w.end += t0
            words_all.extend(ws)
            break

        cut = _pick_cut(audio, sr, t0, t0 + 0.6 * max_chunk_sec, t0 + max_chunk_sec)
        chunk = audio[int(t0 * sr): int(cut * sr)]
        voiced_chunk = au.voiced_seconds(chunk, sr)
        voiced_rest = max(1e-6, au.voiced_seconds(audio[int(t0 * sr):], sr))
        rest_chars = sum(unit_chars[ui:])
        est_chars = rest_chars * min(1.0, voiced_chunk / voiced_rest)
        # index of the clause where the accumulated letter count is closest to the estimate
        acc, best_k = 0, 1
        best_d = float("inf")
        for k in range(1, n_units - ui):  # always leave at least one clause for the remainder
            acc += unit_chars[ui + k - 1]
            d = abs(acc - est_chars)
            if d < best_d:
                best_d, best_k = d, k
        order = [best_k]
        for delta in (1, 2, 3):
            order += [best_k - delta, best_k + delta]
        order = [k for k in order if 1 <= k <= n_units - ui - 1]

        best: Optional[Tuple[float, int, List[WordTiming]]] = None
        for k in order:
            cancel_check()
            cand_text = " ".join(units[ui:ui + k])
            ws = aligner.align(chunk, sr, cand_text, language)
            sc = _score_alignment(ws, chunk, sr)
            log.debug("chunk t0=%.1f cut=%.1f k=%d score=%.3f", t0, cut, k, sc)
            if best is None or sc < best[0]:
                best = (sc, k, ws)
            if sc < 0.4:
                break
        assert best is not None
        sc, k, ws = best
        for w in ws:
            w.start += t0
            w.end += t0
        words_all.extend(ws)
        ui += k
        t0 = cut
    on_progress(1.0, tr("progress.aligned"))
    return words_all


def check_alignment(words: Sequence[WordTiming], audio: np.ndarray, sr: int) -> List[str]:
    """Sanity-check an alignment result.

    Raises ``AudioTextMismatchError`` if audio and text clearly do not belong together (too many zero-length words,
    an implausible speaking rate, or more than 15 s of unmatched speech at the start/end).  Returns a list of soft warnings.
    """
    warnings: List[str] = []
    if not words:
        raise AudioTextMismatchError(tr("err.no_words_found"))
    vs, ve = au.voiced_bounds(audio, sr)
    voiced = max(0.5, au.voiced_seconds(audio, sr))
    chars = sum(len(clean_token(w.word)) for w in words)
    rate = chars / voiced
    zero = sum(1 for w in words if w.duration < 0.02) / len(words)
    tail_gap = ve - words[-1].end
    head_gap = words[0].start - vs
    if zero > 0.25 or rate < 2.5 or rate > 40:
        raise AudioTextMismatchError(
            tr("err.mismatch_length"),
            details=f"zero_frac={zero:.2f} chars_per_voiced_sec={rate:.1f}",
        )
    if tail_gap > 15 or head_gap > 15:
        raise AudioTextMismatchError(
            tr("err.mismatch_extra"),
            details=f"head_gap={head_gap:.1f} tail_gap={tail_gap:.1f}",
        )
    if zero > 0.05:
        warnings.append(tr("warn.align_inexact"))
    return warnings
