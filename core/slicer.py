"""Нарезка по словам с временными метками: паузы > 300 мс, сегменты 3-10 с (мин. 2 c, макс. 10 c)."""
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
    pause_s: float = 0.30      # пауза, считающаяся границей фразы
    min_dur: float = 3.0       # жёсткий минимум длины сегмента (требование: 3-12 с)
    max_dur: float = 12.0      # жёсткий максимум
    target_min: float = 4.0    # желательный минимум
    target_mid: float = 8.0    # желательная «середина»
    pad: float = 0.10          # запас тишины с каждой стороны (ограничен половиной паузы)


@dataclass
class SliceResult:
    segments: List[Segment]
    warnings: List[str] = field(default_factory=list)
    dropped: int = 0


@dataclass
class _Phrase:
    words: List[WordTiming]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def sentence_end(self) -> bool:
        return self.words[-1].sentence_end


def _split_phrase(words: List[WordTiming], limit: float) -> List[List[WordTiming]]:
    """Рекурсивно делит слишком длинную фразу по самой большой паузе между словами."""
    if len(words) < 2 or (words[-1].end - words[0].start) <= limit:
        return [words]
    gaps = [(words[i + 1].start - words[i].end, i) for i in range(len(words) - 1)]
    # предпочитаем разрез ближе к середине среди заметных пауз
    mid_t = (words[0].start + words[-1].end) / 2
    best = max(gaps, key=lambda g: (g[0] - 0.15 * abs(words[g[1]].end - mid_t), -g[1]))
    i = best[1]
    return _split_phrase(words[: i + 1], limit) + _split_phrase(words[i + 1:], limit)


def build_phrases(words: Sequence[WordTiming], cfg: SliceConfig) -> List[_Phrase]:
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
    cost = 0.08 * (dur - cfg.target_mid) ** 2
    if dur < cfg.min_dur:
        cost += 1000.0 + (cfg.min_dur - dur) * 100.0
    elif dur < cfg.target_min:
        cost += 4.0 * (cfg.target_min - dur)
    if not ends_sentence:
        cost += 1.5
    return cost


def slice_words(words: Sequence[WordTiming], text: str, total_duration: float,
                cfg: SliceConfig | None = None) -> SliceResult:
    """Нарезает слова на сегменты. `words` должны иметь char_start/char_end (attach_spans)."""
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
    for gi, (i, j) in enumerate(groups):
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
    out = []
    for s in segments:
        a, b = int(round(s.start * sr)), int(round(s.end * sr))
        out.append(apply_fade(audio[a:b].astype(np.float32), sr, fade_ms))
    return out
