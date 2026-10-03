"""Dataset builder in the Alexandria format (``train_lora.py``): audio + text -> ``segment_XXX.wav``, ``ref.wav``,
``ref_text.txt``, ``metadata.jsonl`` (plus ``report.json`` with diagnostics).

The contract, read from Alexandria's ``train_lora.py``:

* ``metadata.jsonl`` rows are ``{"audio", "text", "ref_audio"}`` with paths relative to the dataset folder;
* ``ref.wav`` and ``ref_text.txt`` (the exact transcript of ``ref.wav``) sit next to it;
* every wav is 24 kHz mono (16 kHz exists only as the aligner's input); training clips carry ~1 s of trailing silence,
  are 3-12 s long and are never cut in the middle of a word;
* ``text`` is the *spoken* form (numbers and abbreviations expanded); the original text is kept in ``report.json``
  (``text_raw``).

Pipeline inside :meth:`DatasetBuilder.run`: read text -> normalize -> load audio (16 + 24 kHz) -> align -> slice ->
quality filter -> choose the reference clip -> write files.
"""
from __future__ import annotations

from core.i18n import tr
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

ALIGNER_SR = 16000   # only the aligner's input is 16 kHz
TRAIN_SR = 24000     # Qwen3-TTS-Tokenizer-12Hz, the speaker encoder and train_lora.py all work at 24 kHz
TRAIL_SILENCE_SEC = 1.0
METADATA_KEYS = ("audio", "text", "ref_audio")


def training_language(language: str) -> str:
    """Value for the trainer's ``--language`` flag: lower case (``"russian"``); Alexandria defaults to English."""
    return (language or "english").strip().lower()


@dataclass
class BuildConfig:
    """Options of :class:`DatasetBuilder`; the defaults are what the GUI uses."""
    sample_rate: int = TRAIN_SR       # sample rate of the dataset wavs (24 kHz mono)
    language: Optional[str] = None    # None -> detect automatically
    trail_silence: float = TRAIL_SILENCE_SEC
    normalize: bool = True            # expand numbers/abbreviations (Russian only)
    quality_filter: bool = True       # drop bad segments (clipping, too quiet, too noisy)
    ref_min: float = 5.0
    ref_max: float = 10.0
    max_chunk_sec: float = MAX_CHUNK_SEC
    slice: SliceConfig = field(default_factory=SliceConfig)
    #: override of the normalization engine (for tests): (name, function)
    normalizer_engine: Optional[Tuple[str, Callable[[str], str]]] = None


@dataclass
class BuildResult:
    """What :meth:`DatasetBuilder.run` returns: the folder, counts, language, warnings and the reference sample."""
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
        """The detected language in the form the trainer expects (see :func:`training_language`)."""
        return training_language(self.language)


# --------------------------------------------------------------------------- jsonl


def write_metadata_jsonl(path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write JSONL: UTF-8 without BOM, ``\\n`` line ends on every OS, non-ASCII text without ``\\u`` escapes."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_metadata_jsonl(path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts (blank lines are skipped)."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_rows(segments: Sequence[Segment], ref_name: str = "ref.wav") -> List[Dict[str, Any]]:
    """Build the ``metadata.jsonl`` rows (audio file, spoken text, reference file) for the segments."""
    return [{"audio": s.filename, "text": s.text, "ref_audio": ref_name} for s in segments]


# --------------------------------------------------------------------------- reference sample


@dataclass
class RefChoice:
    """The chosen reference clip: time range, quality score, samples and its exact text."""
    start: float
    end: float
    score: float
    samples: np.ndarray
    text: str = ""


def _ref_score(x: np.ndarray, sr: int) -> float:
    """Cleanliness score of a candidate reference: SNR, minus penalties for clipping and a very quiet recording."""
    score = au.snr_db(x, sr)
    if au.peak_abs(x) >= 0.99:
        score -= 12.0
    if au.rms(x) < 0.01:
        score -= 6.0
    return score


#: How much (dB) a reference candidate loses for pauses and for an unusual speaking rate.  Found in the first real-voice
#: test: the plain "cleanest" clip was a counting list ("one, two, three ...") with long pauses (34 % voiced), and the
#: voice cloning prompt built from it gave an off-tone voice.
REF_PAUSE_PENALTY_DB = 20.0
REF_RATE_PENALTY_DB = 20.0


def _ref_naturalness_penalty(x: np.ndarray, sr: int, text: str, median_wps: float) -> float:
    """Penalty (dB, >= 0) for a clip full of pauses or spoken at an unusual words-per-second rate."""
    penalty = 0.0
    try:
        voiced = float(np.mean(au.voiced_mask(x, sr)))
    except Exception:  # noqa: BLE001 - never let a heuristic stop the dataset
        voiced = 1.0
    penalty += REF_PAUSE_PENALTY_DB * max(0.0, 0.8 - voiced) / 0.8
    dur = len(x) / sr
    words = len(re.findall(r"\w+", text))
    if median_wps > 0 and dur > 0 and words:
        penalty += REF_RATE_PENALTY_DB * min(1.0, abs(words / dur - median_wps) / median_wps)
    return penalty


def select_ref(audio: np.ndarray, sr: int, segments: Sequence[Segment],
               lo: float = 5.0, hi: float = 10.0) -> Optional[RefChoice]:
    """Choose a clean, natural 5-10 s clip together with its *exact* text (``ref_text.txt``).

    The score is the SNR minus a penalty for long pauses (lists, counting) and for an unusual speaking rate compared
    with the rest of the recording.

    Candidates are single segments and merges of two consecutive segments (adjacent in the text and with a gap of less
    than 0.6 s).  If nothing fits the length window, the longest segment is used (and a warning is reported).
    """
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
    rates = sorted(len(re.findall(r"\w+", q.text)) / q.duration for q in segments if q.duration > 0)
    median_wps = rates[len(rates) // 2] if rates else 0.0
    best: Optional[RefChoice] = None
    for a, b, t in cands:
        x = audio[int(a * sr): int(b * sr)]
        sc = _ref_score(x, sr) - _ref_naturalness_penalty(x, sr, t, median_wps)
        if best is None or sc > best.score:
            best = RefChoice(a, b, sc, x, t)
    return best


def with_trailing_silence(x: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """Append ``seconds`` of digital silence (the training clips end with ~1 s of it)."""
    return np.concatenate([x.astype(np.float32), np.zeros(int(round(seconds * sr)), dtype=np.float32)])


# --------------------------------------------------------------------------- builder


class DatasetBuilder:
    """Runs the whole audio+text -> dataset pipeline with a given aligner."""
    def __init__(self, aligner: BaseAligner, config: Optional[BuildConfig] = None,
                 save_stage: Stage = Stage.SAVE) -> None:
        """``save_stage`` is the progress stage reported while the dataset files are written (the trainer reports SAVE itself)."""
        self.aligner = aligner
        self.cfg = config or BuildConfig()
        #: the stage under which file writing is shown (when training a LoRA, SAVE is reserved for the very end)
        self.save_stage = save_stage

    def run(self, audio_path, text_path, out_dir,
            progress: ProgressCallback = noop_progress,
            cancel: Optional[CancelToken] = None) -> BuildResult:
        """Build the dataset in ``out_dir`` and return a :class:`BuildResult`.

        Reports progress per stage, honours ``cancel`` at every stage boundary and raises friendly
        ``DatasetMakerError`` subclasses (empty text, too short audio, text/audio mismatch, nothing left after the quality
        filter, ...).  Existing Voxprint files in ``out_dir`` are replaced; unrelated files are left alone.
        """
        cancel = cancel or CancelToken()
        cfg = self.cfg
        out = Path(out_dir)
        warnings: List[str] = []

        # --- read the inputs
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
            raise AudioReadError(tr("err.audio_short"))
        if total < 120:
            warnings.append(tr("warn.audio_short_2min"))
        log.info("audio %.1fs, text %d chars, %d sentences, language=%s, normalizer=%s (changed=%s)",
                 total, len(text), len(sentences), language, norm.engine, norm.changed)
        cancel.check()

        # --- load the model
        progress(Stage.MODEL, 0.0, tr("progress.aligner_loading"))
        self.aligner.load()
        progress(Stage.MODEL, 1.0, tr("progress.model_ready"))
        cancel.check()

        # --- alignment (on the "spoken" text; 16 kHz is used only here)
        words = align_long(
            self.aligner, a_align, ALIGNER_SR, text, language, cfg.max_chunk_sec,
            on_progress=lambda f, m: progress(Stage.ALIGN, f, m),
            cancel_check=cancel.check,
        )
        warnings.extend(check_alignment(words, a_align, ALIGNER_SR))
        warnings.extend(attach_spans(words, text))
        cancel.check()

        # --- slicing
        progress(Stage.SLICE, 0.0, tr("progress.slicing"))
        sres = slice_words(words, text, total, cfg.slice)
        warnings.extend(sres.warnings)
        if not sres.segments:
            raise AlignmentError(tr("err.slice_failed"))
        for k, seg in enumerate(sres.segments):
            seg.extra["ord"] = k
            a, b = seg.extra.get("char_span", (None, None))
            if norm.changed and a is not None:
                seg.extra["text_raw"] = norm.raw_for_span(a, b)

        # --- automatic quality filter (24 kHz)
        main_audio = audios[cfg.sample_rate]
        pieces = cut_segments(main_audio, cfg.sample_rate, sres.segments)
        n_dropped_q = 0
        q_summary: Dict[str, int] = {}
        if cfg.quality_filter:
            qr = quality.assess_segments(pieces, cfg.sample_rate)
            n_dropped_q, q_summary = qr.n_dropped, qr.summary()
            if n_dropped_q:
                log.info("quality filter dropped %d of %d segments: %s", n_dropped_q, len(pieces), q_summary)
                warnings.append(tr("warn.quality_dropped", n=n_dropped_q))
            if qr.snr_disabled:
                warnings.append(tr("warn.quality_snr_off"))
            kept = [(sg, p) for sg, p, ok in zip(sres.segments, pieces, qr.keep) if ok]
            if not kept:
                raise AlignmentError(tr("err.all_dropped"))
            segments = [sg for sg, _ in kept]
            pieces = [p for _, p in kept]
            for k, sg in enumerate(segments, start=1):
                sg.index = k
        else:
            segments = sres.segments
        progress(Stage.SLICE, 1.0, tr("progress.pieces", n=len(segments)))

        # --- saving
        progress(self.save_stage, 0.0, tr("progress.saving_dataset"))
        self._clean_old(out)
        out.mkdir(parents=True, exist_ok=True)
        for seg, x in zip(segments, pieces):
            au.write_wav(out / seg.filename, with_trailing_silence(x, cfg.sample_rate, cfg.trail_silence),
                         cfg.sample_rate)
        ref = select_ref(main_audio, cfg.sample_rate, segments, cfg.ref_min, cfg.ref_max)
        assert ref is not None
        if not (cfg.ref_min <= ref.end - ref.start <= cfg.ref_max):
            warnings.append(tr("warn.no_ref_5_10"))
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
        progress(self.save_stage, 1.0, tr("progress.dataset_saved"))
        return BuildResult(out, len(segments), total_seg, out / "ref.wav", meta_path, language, warnings,
                           ref_text=ref.text, n_dropped_quality=n_dropped_q)

    @staticmethod
    def _clean_old(out: Path) -> None:
        """Delete only our own earlier outputs (``segment_NNN.wav``, ``ref.wav``, ``ref_text.txt``, ``metadata.jsonl``, ``report.json``)."""
        if not out.exists():
            return
        for p in out.iterdir():
            if p.is_file() and (re.fullmatch(r"segment_\d+\.wav", p.name) or p.name in
                                ("ref.wav", "ref_text.txt", "metadata.jsonl", "report.json")):
                p.unlink()
        t = out / "train_24k"       # obsolete folder from earlier versions
        if t.is_dir():
            for p in t.glob("*.wav"):
                p.unlink()
