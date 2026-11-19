"""Word-timestamp based slicing of a long recording into training clips.

Words are grouped into *phrases* at pauses longer than ``pause_s``; over-long phrases are split at their biggest
pause.  A small dynamic program then groups consecutive phrases into segments so that the total cost is minimal:
the cost prefers durations near ``target_mid`` (8 s), punishes anything below ``min_dur`` very hard and likes
segments that end at a sentence boundary.  Cuts are placed in the middle of the silence between phrases, so no word
is ever clipped.  Segments outside ``[min_dur, max_dur]`` (3-12 s) are dropped and reported as a warning.

Longer clips (:func:`long_clip_config`, up to 15-20 s): the hard maximum is raised and the cost above ``target_mid`` grows
much more slowly for segments that END a sentence, so a long sentence with comma pauses stays whole instead of being cut
mid-sentence; most segments stay near 8 s, a share becomes 12-20 s.  Voices trained only on short clips can fall apart in
long generations (Qwen3-TTS issue #39), and cutting strictly at pauses teaches the model less about comma pauses.
"""
from __future__ import annotations

from core.i18n import tr
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from core.audio_utils import apply_fade
from core.text_utils import text_for_words
from core.types import Segment, WordTiming


@dataclass
class SliceConfig:
    """Tunable parameters of the slicer (all times in seconds)."""
    pause_s: float = 0.30      # a pause longer than this counts as a phrase boundary
    min_dur: float = 3.0       # hard minimum segment length
    max_dur: float = 12.0      # hard maximum segment length
    target_min: float = 4.0    # preferred minimum
    target_mid: float = 8.0    # preferred "middle" length (the cost is quadratic around it)
    pad: float = 0.10          # silence kept on each side (limited to half of the neighbouring pause)
    #: slope of the quadratic cost above ``target_mid`` for segments that end a sentence (0.08 = same as everywhere else)
    long_slope: float = 0.08


#: Longest training clip offered in the Train window (seconds); 12 is the classic cut, the default allows 15.
CLASSIC_MAX_CLIP_S = 12.0
DEFAULT_MAX_CLIP_S = 15.0
MAX_CLIP_S_LIMIT = 20.0
#: Gentle slope of long mode: a whole 16 s sentence (cost 0.015 * 8^2 = 0.96) beats a mid-sentence cut into 8 + 8 s
#: (+1.5 for the half that does not end a sentence); a 20 s one (2.16) usually still loses to two 10 s halves (1.82).
LONG_SLOPE = 0.015


def long_clip_config(max_clip_s: float) -> "SliceConfig":
    """Slicer settings for a longest clip of ``max_clip_s`` seconds (clamped to 12-20; 12 = the classic cut, unchanged)."""
    try:
        m = float(max_clip_s)
    except (TypeError, ValueError):
        m = DEFAULT_MAX_CLIP_S
    m = max(CLASSIC_MAX_CLIP_S, min(MAX_CLIP_S_LIMIT, m))
    if m <= CLASSIC_MAX_CLIP_S:
        return SliceConfig()
    return SliceConfig(max_dur=m, long_slope=LONG_SLOPE)


@dataclass
class SliceResult:
    """Kept segments, user-facing warnings and the number of dropped segments."""
    segments: List[Segment]
    warnings: List[str] = field(default_factory=list)
    dropped: int = 0


@dataclass
class _Phrase:
    """A run of words without a long pause inside; the unit the dynamic program works with."""
    words: List[WordTiming]

    @property
    def start(self) -> float:
        """Start time of the first word."""
        return self.words[0].start

    @property
    def end(self) -> float:
        """End time of the last word."""
        return self.words[-1].end

    @property
    def sentence_end(self) -> bool:
        """True if the phrase ends a sentence (affects the cost)."""
        return self.words[-1].sentence_end


def _split_phrase(words: List[WordTiming], limit: float) -> List[List[WordTiming]]:
    """Recursively split a phrase that is longer than ``limit`` seconds at its biggest pause (near the middle)."""
    if len(words) < 2 or (words[-1].end - words[0].start) <= limit:
        return [words]
    gaps = [(words[i + 1].start - words[i].end, i) for i in range(len(words) - 1)]
    # prefer a cut near the middle among the noticeable pauses
    mid_t = (words[0].start + words[-1].end) / 2
    best = max(gaps, key=lambda g: (g[0] - 0.15 * abs(words[g[1]].end - mid_t), -g[1]))
    i = best[1]
    return _split_phrase(words[: i + 1], limit) + _split_phrase(words[i + 1:], limit)


def build_phrases(words: Sequence[WordTiming], cfg: SliceConfig) -> List[_Phrase]:
    """Group aligned words into phrases (split at pauses > ``cfg.pause_s``, then at most ``max_dur`` long).

    Words without a ``char_start`` (not matched to the text) are ignored.
    """
    ws = [w for w in words if w.char_start is not None]
    phrases: List[List[WordTiming]] = []
    cur: List[WordTiming] = []
    for w in ws:
        if cur and (w.start - cur[-1].end) > cfg.pause_s:
            phrases.append(cur)
            cur = []
        cur.append(w)
    if cur:
        phrases.append(cur)
    limit = cfg.max_dur - 2 * cfg.pad
    out: List[_Phrase] = []
    for ph in phrases:
        for part in _split_phrase(ph, limit):
            out.append(_Phrase(part))
    return out


def _seg_cost(dur: float, ends_sentence: bool, cfg: SliceConfig) -> float:
    """Cost of a candidate segment: quadratic around the target length, a huge penalty below the minimum,
    a small one below the preferred minimum, and +1.5 when it does not end a sentence.
    """
    slope = cfg.long_slope if (ends_sentence and dur > cfg.target_mid) else 0.08
    cost = slope * (dur - cfg.target_mid) ** 2
    if dur < cfg.min_dur:
        cost += 1000.0 + (cfg.min_dur - dur) * 100.0
    elif dur < cfg.target_min:
        cost += 4.0 * (cfg.target_min - dur)
    if not ends_sentence:
        cost += 1.5
    return cost


def slice_words(words: Sequence[WordTiming], text: str, total_duration: float,
                cfg: SliceConfig | None = None) -> SliceResult:
    """Cut ``words`` into segments of 3-12 s (up to ``cfg.max_dur``, see :func:`long_clip_config`).

    ``words`` must carry ``char_start``/``char_end`` (see ``text_utils.attach_spans``) so each segment can take its text
    from the normalized ``text``.  ``total_duration`` bounds the last cut.  Returns the kept segments (re-indexed from 1)
    plus warnings; segments that still end up outside the allowed length are counted in ``dropped``.
    """
    cfg = cfg or SliceConfig()
    phrases = build_phrases(words, cfg)
    n = len(phrases)
    result = SliceResult(segments=[])
    if n == 0:
        result.warnings.append(tr("warn.no_words"))
        return result

    INF = float("inf")
    best = [INF] * (n + 1)
    prev = [-1] * (n + 1)
    best[0] = 0.0
    for j in range(n):
        for i in range(j, -1, -1):
            dur = phrases[j].end - phrases[i].start + 2 * cfg.pad
            if dur > cfg.max_dur and i != j:
                break
            c = _seg_cost(dur, phrases[j].sentence_end or j == n - 1, cfg)
            if dur > cfg.max_dur:
                c += 5000.0
            if best[i] + c < best[j + 1]:
                best[j + 1] = best[i] + c
                prev[j + 1] = i
    groups: List[tuple] = []
    j = n
    while j > 0:
        i = prev[j]
        groups.append((i, j - 1))
        j = i
    groups.reverse()

    kept: List[Segment] = []
    for i, j in groups:
        ws = [w for k in range(i, j + 1) for w in phrases[k].words]
        prev_end = phrases[i - 1].end if i > 0 else 0.0
        next_start = phrases[j + 1].start if j + 1 < n else total_duration
        start = ws[0].start - min(cfg.pad, max(0.0, (ws[0].start - prev_end) / 2))
        end = ws[-1].end + min(cfg.pad, max(0.0, (next_start - ws[-1].end) / 2))
        start, end = max(0.0, start), min(total_duration, end)
        dur = end - start
        if dur < cfg.min_dur - 1e-6 or dur > cfg.max_dur + 1e-6:
            result.dropped += 1
            continue
        t = text_for_words(text, ws)
        if not t:
            result.dropped += 1
            continue
        kept.append(Segment(index=0, start=start, end=end, text=t, n_words=len(ws),
                            ends_sentence=ws[-1].sentence_end,
                            extra={"char_span": (ws[0].char_start, ws[-1].char_end)}))
    for k, seg in enumerate(kept, start=1):
        seg.index = k
    result.segments = kept
    if result.dropped:
        result.warnings.append(
            tr("warn.slice_dropped", n=result.dropped))
    return result


def cut_segments(audio: np.ndarray, sr: int, segments: Sequence[Segment], fade_ms: float = 8.0) -> List[np.ndarray]:
    """Cut the audio of each segment out of ``audio`` (with short fades to avoid clicks)."""
    out = []
    for s in segments:
        a, b = int(round(s.start * sr)), int(round(s.end * sr))
        out.append(apply_fade(audio[a:b].astype(np.float32), sr, fade_ms))
    return out
