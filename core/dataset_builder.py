"""Сборка датасета в формате Alexandria (train_lora.py): аудио + текст -> segment_XXX.wav, ref.wav,
ref_text.txt, metadata.jsonl.

Контракт (прочитан по исходнику Alexandria train_lora.py):
  * metadata.jsonl: строки {"audio","text","ref_audio"}, пути относительно папки датасета;
  * рядом лежат ref.wav и ref_text.txt (точная транскрипция ref.wav);
  * все wav - 24 кГц моно (16 кГц только на входе выравнивателя), у обучающих клипов ~1 с тишины в конце,
    длина 3-12 с, разрезы не посреди слова;
  * `text` - «произносимая» форма (числа и сокращения раскрыты); исходный текст лежит в report.json (text_raw).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core import audio_utils as au
from core import quality
from core.aligner import BaseAligner, MAX_CHUNK_SEC, align_long, check_alignment
from core.errors import AlignmentError, AudioReadError
from core.events import CancelToken, ProgressCallback, Stage, noop_progress
from core.normalizer import NormalizedText, normalize_for_tts
from core.slicer import SliceConfig, cut_segments, slice_words
from core.text_utils import attach_spans, collapse_ws, detect_language, read_text_file, split_sentences
from core.types import Segment

log = logging.getLogger("voxprint.dataset")

ALIGNER_SR = 16000   # только вход выравнивателя
TRAIN_SR = 24000     # Qwen3-TTS-Tokenizer-12Hz, speaker encoder и train_lora.py работают с 24 кГц
TRAIL_SILENCE_SEC = 1.0
METADATA_KEYS = ("audio", "text", "ref_audio")


def training_language(language: str) -> str:
    """Значение флага --language обучения: строчными буквами («russian»), у Alexandria по умолчанию english."""
    return (language or "english").strip().lower()


@dataclass
class BuildConfig:
    sample_rate: int = TRAIN_SR       # частота wav датасета (24 кГц моно)
    language: Optional[str] = None    # None -> определить автоматически
    trail_silence: float = TRAIL_SILENCE_SEC
    normalize: bool = True            # раскрывать числа/сокращения (только русский)
    quality_filter: bool = True       # отбраковка сегментов (клиппинг, тихие, шумные)
    ref_min: float = 5.0
    ref_max: float = 10.0
    max_chunk_sec: float = MAX_CHUNK_SEC
    slice: SliceConfig = field(default_factory=SliceConfig)
    #: переопределение движка нормализации (для тестов): (имя, функция)
    normalizer_engine: Optional[Tuple[str, Callable[[str], str]]] = None


@dataclass
class BuildResult:
    dataset_dir: Path
    n_segments: int
    total_seconds: float
    ref_path: Path
    metadata_path: Path
    language: str
    warnings: List[str] = field(default_factory=list)
    ref_text: str = ""
    n_dropped_quality: int = 0

    @property
    def training_language(self) -> str:
        return training_language(self.language)


# --------------------------------------------------------------------------- jsonl


def write_metadata_jsonl(path, rows: Sequence[Dict[str, Any]]) -> None:
    """Пишет JSONL: UTF-8 без BOM, '\\n' на всех ОС, кириллица без \\u-экранирования."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_metadata_jsonl(path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_rows(segments: Sequence[Segment], ref_name: str = "ref.wav") -> List[Dict[str, Any]]:
    return [{"audio": s.filename, "text": s.text, "ref_audio": ref_name} for s in segments]


# --------------------------------------------------------------------------- ref


@dataclass
class RefChoice:
    start: float
    end: float
    score: float
    samples: np.ndarray
    text: str = ""


def _ref_score(x: np.ndarray, sr: int) -> float:
    """Чистота: SNR (RMS речи / пол шума), штраф за клиппинг и слишком тихую запись."""
    score = au.snr_db(x, sr)
    if au.peak_abs(x) >= 0.99:
        score -= 12.0
    if au.rms(x) < 0.01:
        score -= 6.0
    return score


def select_ref(audio: np.ndarray, sr: int, segments: Sequence[Segment],
               lo: float = 5.0, hi: float = 10.0) -> Optional[RefChoice]:
    """Выбирает самый чистый фрагмент 5-10 с вместе с его ТОЧНЫМ текстом (ref_text.txt).

    Кандидаты: сегменты и склейки двух соседних (подряд идущих по тексту и без разрыва по времени).
    Если ничего не подходит - самый длинный сегмент (с пометкой в отчёте)."""
    cands: List[Tuple[float, float, str]] = []
    for i, s in enumerate(segments):
        if lo <= s.duration <= hi:
            cands.append((s.start, s.end, s.text))
        if i + 1 < len(segments):
            n = segments[i + 1]
            consecutive = n.extra.get("ord", 0) == s.extra.get("ord", -1) + 1
            if consecutive and lo <= n.end - s.start <= hi and n.start - s.end < 0.6:
                cands.append((s.start, n.end, collapse_ws(s.text + " " + n.text)))
    if not cands:
        if not segments:
            return None
        s = max(segments, key=lambda q: q.duration)
        cands = [(s.start, s.end, s.text)]
    best: Optional[RefChoice] = None
    for a, b, t in cands:
        x = audio[int(a * sr): int(b * sr)]
        sc = _ref_score(x, sr)
        if best is None or sc > best.score:
            best = RefChoice(a, b, sc, x, t)
    return best


def with_trailing_silence(x: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    return np.concatenate([x.astype(np.float32), np.zeros(int(round(seconds * sr)), dtype=np.float32)])


# --------------------------------------------------------------------------- builder


class DatasetBuilder:
    def __init__(self, aligner: BaseAligner, config: Optional[BuildConfig] = None,
                 save_stage: Stage = Stage.SAVE) -> None:
        self.aligner = aligner
        self.cfg = config or BuildConfig()
        #: этап, под которым показывается запись файлов (при обучении LoRA этап SAVE - в самом конце)
        self.save_stage = save_stage

    def run(self, audio_path, text_path, out_dir,
            progress: ProgressCallback = noop_progress,
            cancel: Optional[CancelToken] = None) -> BuildResult:
        cancel = cancel or CancelToken()
        cfg = self.cfg
        out = Path(out_dir)
        warnings: List[str] = []

        # --- чтение входов
        txt = read_text_file(text_path)
        warnings.extend(txt.warnings)
        language = cfg.language or detect_language(txt.text)
        norm: NormalizedText
        if cfg.normalize:
            norm = normalize_for_tts(txt.text, language, cfg.normalizer_engine)
        else:
            norm = NormalizedText(raw=txt.text, spoken=txt.text, engine="none")
        text = norm.spoken
        sentences = split_sentences(text)
        rates = tuple(sorted({ALIGNER_SR, cfg.sample_rate}))
        audios = au.load_audio_multi(audio_path, rates)
        a_align = audios[ALIGNER_SR]
        total = au.duration(a_align, ALIGNER_SR)
        if total < 10:
            raise AudioReadError("Запись слишком короткая (меньше 10 секунд). Нужно хотя бы несколько минут.")
        if total < 120:
            warnings.append("Запись короче 2 минут: для хорошего голоса лучше 5-15 минут.")
        log.info("audio %.1fs, text %d chars, %d sentences, language=%s, normalizer=%s (changed=%s)",
                 total, len(text), len(sentences), language, norm.engine, norm.changed)
        cancel.check()

        # --- загрузка модели
        progress(Stage.MODEL, 0.0, "Загружаю модель выравнивания…")
        self.aligner.load()
        progress(Stage.MODEL, 1.0, "Модель готова")
        cancel.check()

        # --- выравнивание (по «произносимому» тексту; 16 кГц - только здесь)
        words = align_long(
            self.aligner, a_align, ALIGNER_SR, text, language, cfg.max_chunk_sec,
            on_progress=lambda f, m: progress(Stage.ALIGN, f, m),
            cancel_check=cancel.check,
        )
        warnings.extend(check_alignment(words, a_align, ALIGNER_SR))
        warnings.extend(attach_spans(words, text))
        cancel.check()

        # --- нарезка
        progress(Stage.SLICE, 0.0, "Нарезаю на фрагменты по паузам…")
        sres = slice_words(words, text, total, cfg.slice)
        warnings.extend(sres.warnings)
        if not sres.segments:
            raise AlignmentError("Не получилось нарезать запись на фрагменты. Проверьте, что запись и текст совпадают.")
        for k, seg in enumerate(sres.segments):
            seg.extra["ord"] = k
            a, b = seg.extra.get("char_span", (None, None))
            if norm.changed and a is not None:
                seg.extra["text_raw"] = norm.raw_for_span(a, b)

        # --- автоматическая отбраковка по качеству (24 кГц)
        main_audio = audios[cfg.sample_rate]
        pieces = cut_segments(main_audio, cfg.sample_rate, sres.segments)
        n_dropped_q = 0
        q_summary: Dict[str, int] = {}
        if cfg.quality_filter:
            qr = quality.assess_segments(pieces, cfg.sample_rate)
            n_dropped_q, q_summary = qr.n_dropped, qr.summary()
            if n_dropped_q:
                log.info("quality filter dropped %d of %d segments: %s", n_dropped_q, len(pieces), q_summary)
                warnings.append(f"Автоматически отброшено фрагментов с плохим звуком: {n_dropped_q}.")
            if qr.snr_disabled:
                warnings.append("Запись ровная по громкости: проверка шума отключена.")
            kept = [(sg, p) for sg, p, ok in zip(sres.segments, pieces, qr.keep) if ok]
            if not kept:
                raise AlignmentError("Все фрагменты записи оказались слишком тихими или с искажениями. "
                                     "Проверьте микрофон и перезапишите.")
            segments = [sg for sg, _ in kept]
            pieces = [p for _, p in kept]
            for k, sg in enumerate(segments, start=1):
                sg.index = k
        else:
            segments = sres.segments
        progress(Stage.SLICE, 1.0, f"Получилось фрагментов: {len(segments)}")

        # --- сохранение
        progress(self.save_stage, 0.0, "Сохраняю датасет…")
        self._clean_old(out)
        out.mkdir(parents=True, exist_ok=True)
        for seg, x in zip(segments, pieces):
            au.write_wav(out / seg.filename, with_trailing_silence(x, cfg.sample_rate, cfg.trail_silence),
                         cfg.sample_rate)
        ref = select_ref(main_audio, cfg.sample_rate, segments, cfg.ref_min, cfg.ref_max)
        assert ref is not None
        if not (cfg.ref_min <= ref.end - ref.start <= cfg.ref_max):
            warnings.append("Не нашлось фрагмента 5-10 с для образца голоса; взят самый длинный.")
        au.write_wav(out / "ref.wav", au.apply_fade(ref.samples, cfg.sample_rate), cfg.sample_rate)
        (out / "ref_text.txt").write_text(ref.text.strip() + "\n", encoding="utf-8", newline="\n")

        meta_path = out / "metadata.jsonl"
        write_metadata_jsonl(meta_path, make_rows(segments))
        total_seg = float(sum(s.duration for s in segments))
        report = {
            "language": language,
            "training_language": training_language(language),
            "sample_rate": cfg.sample_rate,
            "trailing_silence_sec": cfg.trail_silence,
            "audio_seconds": round(total, 2),
            "segments": len(segments),
            "segments_seconds": round(total_seg, 2),
            "sentences_in_text": len(sentences),
            "normalizer": {"engine": norm.engine, "changed": norm.changed},
            "quality_dropped": n_dropped_q,
            "quality_dropped_reasons": q_summary,
            "ref": {"start": round(ref.start, 3), "end": round(ref.end, 3), "score_db": round(ref.score, 1),
                    "text": ref.text},
            "warnings": warnings,
            "segment_times": [{"file": s.filename, "start": round(s.start, 3), "end": round(s.end, 3),
                               "text": s.text, "text_raw": s.extra.get("text_raw", s.text)}
                              for s in segments],
        }
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        progress(self.save_stage, 1.0, "Датасет сохранён")
        return BuildResult(out, len(segments), total_seg, out / "ref.wav", meta_path, language, warnings,
                           ref_text=ref.text, n_dropped_quality=n_dropped_q)

    @staticmethod
    def _clean_old(out: Path) -> None:
        """Удаляет только наши прежние файлы (segment_NNN.wav, ref.wav, ref_text.txt, metadata.jsonl, report.json)."""
        if not out.exists():
            return
        for p in out.iterdir():
            if p.is_file() and (re.fullmatch(r"segment_\d+\.wav", p.name) or p.name in
                                ("ref.wav", "ref_text.txt", "metadata.jsonl", "report.json")):
                p.unlink()
        t = out / "train_24k"       # устаревшая папка прежних версий
        if t.is_dir():
            for p in t.glob("*.wav"):
                p.unlink()
