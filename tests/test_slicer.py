import numpy as np

from core.slicer import SliceConfig, build_phrases, cut_segments, slice_words
from core.text_utils import attach_spans, clean_token
from core.types import WordTiming
from tests.synth import make_text, truth_timeline


def _words(text):
    tl = truth_timeline(text)
    ws = [WordTiming(w, a, b) for w, a, b in tl]
    attach_spans(ws, text)
    return ws, tl[-1][2] + 0.8


def test_segment_constraints_and_text_coverage():
    text = make_text(60, seed=3)
    ws, total = _words(text)
    res = slice_words(ws, text, total)
    assert res.segments
    cfg = SliceConfig()
    for s in res.segments:
        assert cfg.min_dur <= s.duration <= cfg.max_dur, s
        assert s.text and "\n" not in s.text
    # сегменты не пересекаются и идут по порядку
    for a, b in zip(res.segments, res.segments[1:]):
        assert a.end <= b.start + 1e-6
    assert [s.index for s in res.segments] == list(range(1, len(res.segments) + 1))
    # покрытие текста: чистые буквы сегментов - непрерывная подпоследовательность текста
    joined = "".join(clean_token(s.text) for s in res.segments)
    full = clean_token(text)
    assert len(joined) >= 0.9 * len(full)
    assert joined in full or res.dropped > 0


def test_most_segments_end_at_sentence_boundaries_and_cut_on_pauses():
    text = make_text(80, seed=5)
    ws, total = _words(text)
    res = slice_words(ws, text, total)
    ends = sum(1 for s in res.segments if s.text[-1] in ".!?")
    assert ends / len(res.segments) > 0.6
    assert sum(1 for s in res.segments if 3.0 <= s.duration <= 12.0) / len(res.segments) > 0.8


def test_long_phrase_without_pauses_is_split():
    # 40 слов подряд без пауз > 300 мс (по 0.5 с на слово = 20 с)
    toks = [f"слово{chr(1072 + i % 20)}" for i in range(40)]
    text = " ".join(toks) + "."
    ws = [WordTiming(t, i * 0.5, i * 0.5 + 0.4) for i, t in enumerate(toks)]
    attach_spans(ws, text)
    res = slice_words(ws, text, 20.2)
    assert len(res.segments) >= 2
    assert all(s.duration <= 12.0 for s in res.segments)


def test_pause_threshold():
    ws = [WordTiming("а", 0, 0.3), WordTiming("б", 0.5, 0.8), WordTiming("в", 1.2, 1.5)]
    attach_spans(ws, "а б в")
    ph = build_phrases(ws, SliceConfig())
    assert [len(p.words) for p in ph] == [2, 1]  # пауза 0.2 < 0.3, пауза 0.4 > 0.3


def test_cut_segments_shapes():
    text = make_text(20, seed=2)
    ws, total = _words(text)
    res = slice_words(ws, text, total)
    audio = np.zeros(int(total * 16000), dtype=np.float32) + 0.1
    pieces = cut_segments(audio, 16000, res.segments)
    for s, p in zip(res.segments, pieces):
        assert abs(len(p) / 16000 - s.duration) < 0.01
