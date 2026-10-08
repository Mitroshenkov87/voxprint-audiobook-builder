"""Benchmark the CPU side of narration with a mock GPU engine (no model, no GPU needed).

    python tools/bench_narration.py --chapters 24 --chars 6000 --rtf 0.02 --formats opus_single,mp3_chapters,m4b

The mock engine *sleeps* like GPU synthesis (``rtf`` x the audio length of the longest chunk in a batch; the sleep releases
the GIL like a CUDA wait) and spends ``--cpu-share`` of that time in a Python loop that holds the GIL, like the Python side
of a real ``generate`` loop.  Its audio is deterministic noise-shaped "speech" (24 kHz, ~14 characters per second), so the
real ffmpeg encoders, the FLAC chunk cache and the chapter assembly do their real amount of work.

Prints the wall time of every phase (prepare / synth / assemble / export) and the total, plus a SHA-256 over the produced
files so two runs (e.g. before / after a change) can be compared for byte-identical output.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import audiobook_export as ex  # noqa: E402
from core import narration as nr  # noqa: E402
from core.book_parsers import Book, Chapter  # noqa: E402

SR = 24000
CHARS_PER_SECOND = 14.0
_WORDS_RU = ("старый смотритель маяка каждое утро спускался к берегу и долго смотрел на море это было единственное что он "
             "по-настоящему любил в деревне знали все корабли проходили мимо огонь горел ночью ветер дул с севера").split()


class MockGpuEngine:
    """Sleeps like a GPU (batched), returns deterministic audio."""
    sample_rate = SR
    tag = "mock-gpu"

    def __init__(self, rtf: float, cpu_share: float, batch: int) -> None:
        self.rtf, self.cpu_share, self.batch = rtf, cpu_share, batch
        self.gpu_seconds = 0.0

    def _audio(self, text: str) -> np.ndarray:
        n = int(SR * max(0.3, len(text) / CHARS_PER_SECOND))
        rng = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
        t = np.arange(n, dtype=np.float32) / SR
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)                        # syllable-rate envelope
        tone = np.sin(2 * np.pi * (140 + 40 * np.sin(2 * np.pi * 0.5 * t)) * t)
        return (0.2 * env * (0.7 * tone + 0.3 * rng.standard_normal(n))).astype(np.float32)

    def _work(self, seconds_of_audio: float) -> None:
        busy = self.rtf * seconds_of_audio
        self.gpu_seconds += busy
        spin_until = time.perf_counter() + busy * self.cpu_share
        x = 0
        while time.perf_counter() < spin_until:                              # Python work that holds the GIL
            x += 1
        time.sleep(busy * (1.0 - self.cpu_share))                             # the GPU part: releases the GIL

    def synthesize(self, text: str) -> np.ndarray:
        self._work(len(text) / CHARS_PER_SECOND)
        return self._audio(text)

    def synthesize_batch(self, texts: Sequence[str]) -> List[np.ndarray]:
        self._work(max(len(t) for t in texts) / CHARS_PER_SECOND)            # a batch runs until its longest item ends
        return [self._audio(t) for t in texts]

    def max_batch(self) -> int:
        return self.batch

    def close(self) -> None:
        pass


def make_book(chapters: int, chars: int, seed: int = 7) -> Book:
    """A Russian book with numbers (so the text normalizer has work to do)."""
    rnd = random.Random(seed)
    out = []
    for ci in range(chapters):
        paras, total = [], 0
        while total < chars:
            sent = []
            for _ in range(rnd.randint(4, 9)):
                words = [rnd.choice(_WORDS_RU) for _ in range(rnd.randint(6, 16))]
                if rnd.random() < 0.3:
                    words.insert(rnd.randint(0, len(words)), str(rnd.randint(2, 1999)))
                sent.append(" ".join(words).capitalize() + rnd.choice((".", ".", "!", "?", "...")))
            p = " ".join(sent)
            paras.append(p)
            total += len(p)
        out.append(Chapter(f"Глава {ci + 1}", "\n\n".join(paras)))
    return Book("Бенчмарк", "Voxprint", "ru", out)


def digest(files: Sequence[Path], ffmpeg: str) -> str:
    """SHA-256 over the outputs.  Ogg files get a random stream serial number on every run (ffmpeg's Ogg muxer), so for
    ``.opus`` the decoded samples are hashed instead of the bytes; every other file is hashed byte for byte."""
    import subprocess

    h = hashlib.sha256()
    for f in sorted(files, key=lambda p: p.name):
        h.update(f.name.encode())
        if f.suffix == ".opus":
            h.update(subprocess.run([ffmpeg, "-v", "error", "-i", str(f), "-f", "s16le", "-"], capture_output=True,
                                    check=True).stdout)
        else:
            h.update(Path(f).read_bytes())
    return h.hexdigest()[:16]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chapters", type=int, default=24)
    ap.add_argument("--chars", type=int, default=6000, help="characters per chapter")
    ap.add_argument("--rtf", type=float, default=0.02, help="mock GPU seconds per second of audio")
    ap.add_argument("--cpu-share", type=float, default=0.1, help="part of the GPU time spent holding the GIL")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--formats", default="opus_single,mp3_chapters,m4b")
    ap.add_argument("--keep", action="store_true", help="keep the output folder")
    a = ap.parse_args(argv)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found on PATH", file=sys.stderr)
        return 2
    book = make_book(a.chapters, a.chars)
    engine = MockGpuEngine(a.rtf, a.cpu_share, a.batch)
    out = Path(tempfile.mkdtemp(prefix="vx-bench-"))
    marks: Dict[str, float] = {}
    t0 = time.perf_counter()

    def progress(p: nr.NarrationProgress) -> None:
        marks.setdefault(p.phase, time.perf_counter() - t0)

    nr._cuda_available = lambda: True      # the mock stands for a GPU: size the CPU pool as on a CUDA machine
    opts = nr.NarrationOptions(formats=set(a.formats.split(",")), allow_aac=True)
    res = nr.narrate_book(book, lambda: engine, engine.tag, out, language="Russian", narrator="Mock", options=opts,
                          progress=progress, ffmpeg=ffmpeg)
    total = time.perf_counter() - t0
    phases = ["prepare", "synth", "assemble", "export", "done"]
    seen = [p for p in phases if p in marks]
    starts = {p: marks[p] for p in seen}
    if "synth" not in starts:
        starts["synth"] = 0.0
    order = sorted(starts, key=starts.get)
    print(f"book: {a.chapters} chapters, {sum(len(c.text) for c in book.chapters)} chars, {res.chunks} chunks, "
          f"{res.seconds / 60:.1f} min of audio; mock GPU busy {engine.gpu_seconds:.1f} s")
    print(f"  {'setup':<9} {starts[order[0]]:7.2f} s   (chunking, text normalisation, model load)")
    for i, p in enumerate(order[:-1]):
        print(f"  {p:<9} {starts[order[i + 1]] - starts[p]:7.2f} s")
    print(f"  total     {total:7.2f} s   (after GPU: {total - engine.gpu_seconds:6.2f} s)")
    print(f"  output sha256 {digest(res.files, ffmpeg)} ({len(res.files)} files)")
    if not a.keep:
        shutil.rmtree(out, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
