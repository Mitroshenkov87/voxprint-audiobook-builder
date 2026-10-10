"""Dataset from audio only ("no transcript" mode): many files -> ASR -> quality gates -> one merged dataset.

Each file is handled on its own: a short clip is used as it is, a long recording is cut at pauses into pieces of at most
``MAX_CLIP_SEC``; every piece is recognised separately, so a transcript always belongs to exactly its own audio and no
forced alignment is needed.  The pieces are then filtered (empty / too short / too long / low plausibility / duplicate /
bad signal quality), loudness-normalised and written exactly like :class:`core.dataset_builder.DatasetBuilder` writes its
dataset (``segment_NNN.wav``, ``ref.wav``, ``ref_text.txt``, ``metadata.jsonl``, ``report.json``), so training does not
care where the dataset came from.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from core import audio_utils as au
from core import quality
from core.asr import BaseASR, plausibility, split_at_pauses
from core.dataset_builder import (BuildResult, DatasetBuilder, TRAIL_SILENCE_SEC, TRAIN_SR, make_rows, select_ref, training_language,
                                  with_trailing_silence, write_metadata_jsonl)
from core.errors import AlignmentError, AudioReadError
from core.events import CancelToken, ProgressCallback, Stage, noop_progress
from core.i18n import tr
from core.text_utils import detect_language
from core.types import Segment

log = logging.getLogger("voxprint.asr_dataset")

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".mp4", ".webm", ".mka", ".aiff", ".aif")
MIN_CLIP_SEC = 1.5
MAX_CLIP_SEC = 20.0
SPLIT_SEC = 14.0                #: long recordings are cut into pieces of at most this length
MIN_CONFIDENCE = 0.5
TARGET_RMS_DBFS = -20.0         #: loudness every kept clip is brought to (peak-limited, +-12 dB at most)
MAX_GAIN_DB = 12.0
GAP_SEC = 0.3                   #: pause between clips on the virtual timeline used for choosing ``ref.wav``
MIN_TOTAL_SEC = 30.0


@dataclass
class AsrConfig:
    """Options of the no-transcript build; defaults are what the GUI uses."""
    language: Optional[str] = None          # "Russian", "English" ... or None = recognise automatically
    min_clip: float = MIN_CLIP_SEC
    max_clip: float = MAX_CLIP_SEC
    split_at: float = SPLIT_SEC
    min_confidence: float = MIN_CONFIDENCE
    sample_rate: int = TRAIN_SR
    trail_silence: float = TRAIL_SILENCE_SEC
    quality_filter: bool = True


@dataclass
class AsrReport:
    """Totals of a no-transcript build (also written to ``report.json``)."""
    files: int = 0
    files_failed: int = 0
    clips: int = 0
    kept: int = 0
    seconds_total: float = 0.0
    seconds_kept: float = 0.0
    dropped: Dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        """Count one dropped clip under ``reason``."""
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    @property
    def share_kept(self) -> float:
        """Share of the audio that was kept, or 0 when nothing was timed."""
        return self.seconds_kept / self.seconds_total if self.seconds_total else 0.0

    def as_dict(self) -> dict:
        """A plain dict of the counts, seconds and drop reasons."""
        return {"files": self.files, "files_failed": self.files_failed, "clips": self.clips, "kept": self.kept,
                "seconds_total": round(self.seconds_total, 1), "seconds_kept": round(self.seconds_kept, 1),
                "share_kept": round(self.share_kept, 3), "dropped": dict(self.dropped)}


def expand_inputs(items: Sequence) -> List[Path]:
    """Files and folders (searched recursively) -> sorted unique list of audio files."""
    out: List[Path] = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.is_file() and q.suffix.lower() in AUDIO_EXTENSIONS)
        elif p.is_file():
            out.append(p)
    seen, res = set(), []
    for p in out:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            res.append(p)
    return res


def normalize_level(x: np.ndarray, target_dbfs: float = TARGET_RMS_DBFS, max_gain_db: float = MAX_GAIN_DB) -> np.ndarray:
    """Bring a clip to about ``target_dbfs`` RMS without clipping (gain limited to +-``max_gain_db``)."""
    rms = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
    if rms < 1e-6:
        return x
    gain_db = max(-max_gain_db, min(max_gain_db, target_dbfs - 20.0 * math.log10(rms)))
    gain = 10 ** (gain_db / 20.0)
    peak = float(np.max(np.abs(x)))
    if peak * gain > 0.95:
        gain = 0.95 / peak
    return (x * gain).astype(np.float32)


def _text_key(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower().replace("\u0451", "\u0435")))


def _audio_key(x16: np.ndarray) -> str:
    return hashlib.sha1((np.clip(x16, -1, 1) * 32767).astype(np.int16).tobytes(), usedforsecurity=False).hexdigest()


def build_from_audio(files: Sequence, out_dir, asr: BaseASR, cfg: Optional[AsrConfig] = None,
                     progress: ProgressCallback = noop_progress, cancel: Optional[CancelToken] = None,
                     script_text: Optional[str] = None) -> BuildResult:
    """Recognise ``files`` (audio files or folders), filter and write one dataset into ``out_dir``.

    ``script_text`` (optional): the recording script that was read aloud.  Every recognised piece is then matched to the script
    lines (:mod:`core.script_match`): stumbles and re-read sentences are tolerated - the best reading of each line is kept,
    pieces that match no line (stumbles, the spoken consent statement, chatter) are dropped, and a kept piece gets the clean
    script line as its transcript.
    """
    cfg = cfg or AsrConfig()
    cancel = cancel or CancelToken()
    paths = expand_inputs(files)
    if not paths:
        raise AudioReadError(tr("err.asr_no_files"))
    rep, warnings = AsrReport(), []
    clips: List[dict] = []
    seen_audio: set[str] = set()
    seen_text: Dict[str, float] = {}
    progress(Stage.MODEL, 0.0, tr("progress.asr_loading"))
    asr.load()
    progress(Stage.MODEL, 1.0, tr("progress.model_ready"))
    rep.files = len(paths)
    for fi, path in enumerate(paths):
        cancel.check()
        try:
            audios = au.load_audio_multi(path, (16000, cfg.sample_rate))
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not stop the batch
            log.warning("cannot read %s: %s", path, exc)
            rep.files_failed += 1
            continue
        a16, a24 = audios[16000], audios[cfg.sample_rate]
        pieces = split_at_pauses(a16, 16000, cfg.split_at) if len(a16) / 16000 > cfg.max_clip else [(0, len(a16))]
        for pi, (s16, e16) in enumerate(pieces):
            cancel.check()
            progress(Stage.ALIGN, (fi + pi / len(pieces)) / len(paths), tr("progress.asr_file", i=fi + 1, n=len(paths), name=path.name))
            x16 = a16[s16:e16]
            x24 = a24[int(round(s16 / 16000 * cfg.sample_rate)):int(round(e16 / 16000 * cfg.sample_rate))]
            sec = len(x16) / 16000
            rep.clips += 1
            rep.seconds_total += sec
            if sec < cfg.min_clip:
                rep.drop("too_short")
                continue
            if sec > cfg.max_clip:
                rep.drop("too_long")
                continue
            key = _audio_key(x16)
            if key in seen_audio:
                rep.drop("duplicate")
                continue
            seen_audio.add(key)
            r = asr.transcribe(x16, 16000, cfg.language)
            text = r.text.strip()
            if not _text_key(text):
                rep.drop("empty")
                continue
            conf = plausibility(text, sec, cfg.language)
            if r.confidence is not None:
                conf = min(conf, float(r.confidence))
            if conf < cfg.min_confidence:
                rep.drop("low_confidence")
                continue
            tk = _text_key(text)
            if tk in seen_text and abs(seen_text[tk] - sec) <= 0.05 * sec:   # same words, same length: the same clip twice
                rep.drop("duplicate")
                continue
            seen_text[tk] = sec
            clips.append({"x": x24, "text": text, "conf": conf, "file": path.name, "sec": sec})
    asr.unload()
    if not clips:
        raise AlignmentError(tr("err.asr_nothing_kept"))
    cancel.check()

    progress(Stage.SLICE, 0.0, tr("progress.slicing"))
    if cfg.quality_filter:
        qr = quality.assess_segments([c["x"] for c in clips], cfg.sample_rate)
        keep = list(qr.keep)
        for ok in keep:
            if not ok:
                rep.drop("quality")
        if qr.snr_disabled:
            warnings.append(tr("warn.quality_snr_off"))
        clips = [c for c, ok in zip(clips, keep) if ok]
        if not clips:
            raise AlignmentError(tr("err.all_dropped"))
    if script_text is not None:
        from core.script_match import match_clips, parse_script_lines
        matches = match_clips([c["text"] for c in clips], parse_script_lines(script_text))
        kept_clips = []
        for c, m in zip(clips, matches):
            if m.status == "matched":
                c["text"], c["conf"] = m.text, min(1.0, m.ratio)
                kept_clips.append(c)
            else:
                rep.drop("repeat" if m.status == "repeat" else "not_in_script")
        clips = kept_clips
        if not clips:
            raise AlignmentError(tr("err.asr_nothing_kept"))
    texts = " ".join(c["text"] for c in clips)
    language = detect_language(texts)

    # one virtual timeline (clips separated by a short pause): lets select_ref() choose ref.wav exactly as for a normal recording
    sr = cfg.sample_rate
    timeline, segments, pos = [], [], 0.0
    for k, c in enumerate(clips, start=1):
        x = normalize_level(c["x"])
        c["x"] = x
        seg = Segment(index=k, start=pos, end=pos + len(x) / sr, text=c["text"], n_words=len(c["text"].split()),
                      extra={"source": c["file"], "confidence": round(c["conf"], 2)})
        segments.append(seg)
        timeline += [x, np.zeros(int(GAP_SEC * sr), dtype=np.float32)]
        pos = seg.end + GAP_SEC
    virtual = np.concatenate(timeline)
    rep.kept = len(segments)
    rep.seconds_kept = float(sum(s.duration for s in segments))
    if rep.seconds_kept < MIN_TOTAL_SEC:
        warnings.append(tr("warn.asr_little_audio", sec=int(rep.seconds_kept)))
    warnings.append(tr("warn.asr_unverified", kept=rep.kept, files=rep.files - rep.files_failed))

    out = Path(out_dir)
    progress(Stage.SLICE, 1.0, tr("progress.pieces", n=len(segments)))
    progress(Stage.SLICE, 0.0, tr("progress.saving_dataset"))
    DatasetBuilder._clean_old(out)
    out.mkdir(parents=True, exist_ok=True)
    for seg, c in zip(segments, clips):
        au.write_wav(out / seg.filename, with_trailing_silence(c["x"], sr, cfg.trail_silence), sr)
    ref = select_ref(virtual, sr, segments, 5.0, 10.0)
    assert ref is not None
    if not (5.0 <= ref.end - ref.start <= 10.0):
        warnings.append(tr("warn.no_ref_5_10"))
    au.write_wav(out / "ref.wav", au.apply_fade(ref.samples, sr), sr)
    (out / "ref_text.txt").write_text(ref.text.strip() + "\n", encoding="utf-8", newline="\n")
    meta = out / "metadata.jsonl"
    write_metadata_jsonl(meta, make_rows(segments))
    report = {"mode": "no_transcript", "language": language, "training_language": training_language(language),
              "sample_rate": sr, "asr": rep.as_dict(), "warnings": warnings,
              "segment_times": [{"file": s.filename, "source": s.extra["source"], "confidence": s.extra["confidence"],
                                 "seconds": round(s.duration, 2), "text": s.text} for s in segments]}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(Stage.SLICE, 1.0, tr("progress.dataset_saved"))
    res = BuildResult(out, len(segments), rep.seconds_kept, out / "ref.wav", meta, language, warnings, ref_text=ref.text,
                      n_dropped_quality=rep.dropped.get("quality", 0))
    res.asr_report = rep   # type: ignore[attr-defined]
    return res
