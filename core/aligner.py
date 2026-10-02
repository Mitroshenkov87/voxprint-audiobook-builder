"""Принудительное выравнивание (forced alignment) текста по аудио.

Проверено по исходникам PyPI-пакета qwen-asr 0.0.6 и карточке модели HF
Qwen/Qwen3-ForcedAligner-0.6B (см. README, раздел «Что проверено»):

    from qwen_asr import Qwen3ForcedAligner
    model = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B",
                                               dtype=torch.bfloat16, device_map="cuda:0")
    results = model.align(audio=(np_array, sr), text="...", language="Russian")
    results[0] -> ForcedAlignResult, итерируется по ForcedAlignItem(text, start_time, end_time)  # секунды

* Единица выравнивания - слово (или иероглиф для китайского), пунктуация отбрасывается.
* Модель держит до ~5 минут речи (карточка), пакет сам режет на куски по 180 c (MAX_FORCE_ALIGN_INPUT_SECONDS).
  Поэтому записи 5-15 минут мы режем по паузам на куски <= 150 c (требование: не больше ~4 мин) и
  делим текст по предложениям/клаузам (см. align_long).
* Запасной вариант - ctc-forced-aligner (MahmoudAshraf97, MMS wav2vec2): класс CtcAligner, подключается
  через FallbackAligner, если основной выравниватель не загрузился или упал.
* Интерфейс BaseAligner позволяет подставить FakeAligner в тестах.
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
#: Языки, которые поддерживает Qwen3-ForcedAligner-0.6B (карточка модели).
SUPPORTED_LANGUAGES = (
    "Chinese", "English", "Cantonese", "French", "German", "Italian",
    "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
)
#: Максимальная длина куска аудио для одного вызова (с запасом: модель - до 300 c, пакет режет сам на 180 c).
MAX_CHUNK_SEC = 150.0
#: Код языка ISO-639-3 для ctc-forced-aligner.
ISO3 = {"Russian": "rus", "English": "eng", "Chinese": "cmn", "Cantonese": "yue", "French": "fra", "German": "deu",
        "Italian": "ita", "Japanese": "jpn", "Korean": "kor", "Portuguese": "por", "Spanish": "spa"}


class BaseAligner(abc.ABC):
    """Интерфейс выравнивателя."""

    def load(self) -> None:  # noqa: B027 - необязательно
        """Загрузить модель (может быть долгим)."""

    def unload(self) -> None:  # noqa: B027
        """Освободить память."""

    @abc.abstractmethod
    def align(self, audio: np.ndarray, sr: int, text: str, language: str) -> List[WordTiming]:
        """Выровнять `text` по `audio` (mono float32). Время - секунды от начала `audio`."""


class Qwen3Aligner(BaseAligner):
    """Обёртка над qwen_asr.Qwen3ForcedAligner (transformers-бэкенд)."""

    def __init__(self, model_path: str = ALIGNER_MODEL_ID, device: str = "auto") -> None:
        self.model_path = model_path
        self.device = device
        self._model = None
        self.resolved_device = "cpu"

    def load(self) -> None:
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
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def align(self, audio: np.ndarray, sr: int, text: str, language: str) -> List[WordTiming]:
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
    """Запасной выравниватель: ctc-forced-aligner (pip install ctc-forced-aligner; MMS wav2vec2, ~1.2 ГБ).

    API по README пакета: load_alignment_model, generate_emissions, preprocess_text, get_alignments,
    get_spans, postprocess_results. Метки слов берутся из поля text (слова разделены пробелами).
    TODO-needs-GPU-test: не запускался здесь (нет пакета/модели/сети); импорты ленивые, формат результата
    (dict со start/end/text) прочитан из README и обрабатывается защитно.
    """

    def __init__(self, device: str = "auto", batch_size: int = 8) -> None:
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
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
        self._model = self._tokenizer = None
        gc.collect()

    def align(self, audio: np.ndarray, sr: int, text: str, language: str) -> List[WordTiming]:
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
    return " ".join(text.split())


class FallbackAligner(BaseAligner):
    """Основной выравниватель + запасной. Переключается, если основной не загрузился или бросил ошибку
    выравнивания (но не при нехватке памяти и не при «аудио не совпадает с текстом» - это ошибки данных)."""

    def __init__(self, primary: BaseAligner, fallback: Optional[BaseAligner]) -> None:
        self.primary, self.fallback = primary, fallback
        self.using_fallback = False

    def load(self) -> None:
        try:
            self.primary.load()
        except OutOfMemoryError_:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("primary aligner failed to load (%s) - switching to fallback", exc)
            self._switch(exc)

    def _switch(self, exc: BaseException) -> None:
        if self.fallback is None:
            raise exc
        try:
            self.fallback.load()
        except Exception as fb_exc:  # noqa: BLE001 - запасной недоступен: показываем исходную причину
            log.warning("fallback aligner unavailable: %s", fb_exc)
            raise exc
        self.using_fallback = True

    def unload(self) -> None:
        self.primary.unload()
        if self.fallback is not None:
            self.fallback.unload()

    def align(self, audio: np.ndarray, sr: int, text: str, language: str) -> List[WordTiming]:
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
    """Qwen3-ForcedAligner + запасной ctc-forced-aligner (используется, только если установлен)."""
    return FallbackAligner(Qwen3Aligner(model_path, device=device), CtcAligner(device="cpu" if device == "cpu" else "auto"))


class FakeAligner(BaseAligner):
    """Простейший «выравниватель» без нейросети: слова равномерно по длине букв
    распределяются по участку с речью. Для тестов и сухого прогона (`--fake-aligner`)."""

    def __init__(self) -> None:
        self.calls = 0

    def align(self, audio: np.ndarray, sr: int, text: str, language: str) -> List[WordTiming]:
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


# --------------------------------------------------------------------------- длинное аудио


def _score_alignment(words: Sequence[WordTiming], chunk: np.ndarray, sr: int) -> float:
    """Чем меньше, тем лучше текст «подходит» к куску аудио.

    Если текста слишком много, хвостовые слова схлопываются в нулевую длительность
    у конца аудио; если слишком мало - последние слова растягиваются на тишину/чужую речь.
    """
    if not words:
        return 1e9
    vs, ve = au.voiced_bounds(chunk, sr)
    score = abs(words[0].start - vs) + abs(words[-1].end - ve)
    score += 0.5 * sum(1 for w in words if w.duration < 0.02)
    score += 0.3 * sum(1 for w in words if w.duration > 2.0)
    return float(score)


def _pick_cut(audio: np.ndarray, sr: int, start_s: float, lo_s: float, hi_s: float) -> float:
    """Выбирает точку разреза в [lo_s, hi_s] (абсолютное время) - центр самой длинной паузы."""
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
    """Выравнивает текст по аудио любой длины. Время в результате - от начала всего аудио.

    Алгоритм для длинных записей (аудио > max_chunk_sec), без ASR:
      1. разрез аудио - по самой длинной паузе в окне [0.6*max, max] от начала куска;
      2. позицию разреза в тексте оцениваем пропорционально «речевому времени»
         (доля озвученных кадров) и числу букв, затем перебираем соседние границы
         смысловых единиц (клаузы) и берём ту, где выравнивание выглядит лучше всего
         (_score_alignment). Первый же хороший вариант принимается сразу.
    TODO-needs-GPU-test: пороги score подобраны на синтетике, на реальной модели возможна подстройка.
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
    ui = 0  # индекс первой необработанной клаузы
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
        # индекс клаузы, на которой накопленное число букв ближе всего к оценке
        acc, best_k = 0, 1
        best_d = float("inf")
        for k in range(1, n_units - ui):  # оставляем хотя бы одну клаузу на остаток
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
    """Проверка результата. Бросает AudioTextMismatchError, если аудио и текст явно не совпадают.
    Возвращает список мягких предупреждений."""
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
