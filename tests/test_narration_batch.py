"""Batched synthesis in the narrator: grouping by length, OOM halving, fallback to single chunks, background writer."""
from core import narration as nr
from core.book_parsers import Book, Chapter
from tests.test_narration import FakeEngine, run


class BatchEngine(FakeEngine):
    """Fake engine with the optional batch API (what core.tts_engine.Qwen3AdapterEngine provides)."""
    def __init__(self, limit=4, oom_above=None, batch_error=None, **kw):
        super().__init__(**kw)
        self.limit, self.oom_above, self.batch_error, self.batches = limit, oom_above, batch_error, []

    def max_batch(self):
        return self.limit

    def synthesize_batch(self, texts):
        self.batches.append(list(texts))
        if self.batch_error:
            raise self.batch_error
        if self.oom_above is not None and len(texts) > self.oom_above:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return [FakeEngine.synthesize(self, t) for t in texts]


def long_book():
    return Book("B", "A", "en", [Chapter("One", " ".join(f"Sentence number {i} is here." + "x" * (i % 5) for i in range(14)))])


def test_chunks_are_batched_and_every_chunk_is_cached(tmp_path):
    eng = BatchEngine(limit=4)
    res, eng, ff, events = run(tmp_path, engine=eng, book=long_book())
    assert eng.batches and max(len(b) for b in eng.batches) <= 4 and eng.closed
    batched = [t for b in eng.batches for t in b]
    assert len(eng.calls) == len(set(eng.calls))                                # each chunk synthesized exactly once
    assert set(batched) <= set(eng.calls) and len(batched) >= len(eng.calls) - 1    # (a lone last chunk may run as a single call)
    assert len(eng.calls) >= 3 and any(len(b) > 1 for b in eng.batches)
    assert events[-1].done == events[-1].total


def test_group_selection_covers_every_chunk_once():
    texts = {i: "x" * n for i, n in enumerate([50, 5, 60, 6, 70, 7, 80, 8, 90, 9, 100, 10])}
    queue = [_chunk(i) for i in range(12)]
    seen = []
    while queue:
        g = nr._next_group(queue, texts, 4)
        assert 0 < len(g) <= 4
        seen += [c.index for c in g]
    assert sorted(seen) == list(range(12))


def _chunk(i):
    return type("C", (), {"index": i})()


def test_first_chunk_of_the_window_is_in_the_group():
    texts = {i: "x" * (100 - i) for i in range(12)}        # book order = descending length
    queue = [_chunk(i) for i in range(12)]
    g = nr._next_group(queue, texts, 4)
    assert 0 in [c.index for c in g] and len(g) == 4


def test_oom_halves_the_batch_and_still_returns_everything():
    eng = BatchEngine(limit=8, oom_above=2)
    audios, limit = nr._synth_group(eng, ["aaa", "bbbb", "ccccc", "dddddd", "e", "ff", "ggg", "hhhh"], list(range(8)), 8)
    assert len(audios) == 8 and all(a.size for a in audios) and limit <= 4


def test_other_batch_errors_fall_back_to_single_chunks():
    eng = BatchEngine(limit=4, batch_error=ValueError("shape mismatch"))
    audios, limit = nr._synth_group(eng, ["aaa", "bbbb", "ccccc"], [0, 1, 2], 4)
    assert len(audios) == 3 and eng.calls == ["aaa", "bbbb", "ccccc"]


def test_engine_without_batch_api_is_used_serially():
    assert nr._batch_limit(FakeEngine()) == 1
    assert nr._batch_limit(BatchEngine(limit=5)) == 5
