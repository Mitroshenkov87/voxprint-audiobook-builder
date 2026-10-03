"""Audiobook narration: book -> chunks -> TTS (chunk by chunk, cached on disk) -> chapters -> exported files.

Flow of :func:`narrate_book`::

    Book --chunk_book--> [Chunk]            sentence-sized pieces with pauses (core/chunker.py)
        --synthesize--> ChunkCache          one FLAC per chunk, named by a hash of (engine, text); written atomically
        --assemble----> chapter WAV files   chunks joined with silence, written incrementally (low memory)
        --export------> m4b / mp3 / opus ...  core/audiobook_export.py

**Resumable:** the cache key depends only on the voice/engine identity and the chunk text, so running the same job again
after a crash, a cancel or a pause that became a quit skips every finished chunk.  The engine is created lazily - a fully
cached job never loads the model.  **Pause/cancel** are cooperative and checked between chunks.

The TTS model sits behind the tiny :class:`TTSEngine` protocol (``synthesize(text) -> samples``); the real implementation
is :mod:`core.tts_engine` and unit tests use a fake one.

**Preparation** (``options.prep``, see :mod:`core.book_prep`): before chunking, the whole book goes through the automatic
rule-based steps and the optional neural clean-up; the prepared text is saved to ``<job>/.debug/``.  Chapter titles in
the exported files keep their original spelling.  Extension points (deliberately not implemented yet): the
``preprocessors`` option (a per-chunk ``text -> text`` hook, e.g. translation) and ``Chunk``-level voice selection for
multi-voice role markup (all chunks currently use the job's voice).
"""
from __future__ import annotations

import hashlib
from concurrent.futures import Future, ThreadPoolExecutor
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Set

import numpy as np
import soundfile as sf

from core import audiobook_export as ex
from core.audio_utils import resample
from core.book_parsers import Book
from core.book_prep import PrepPlan, run_preparation
from core.chunker import DEFAULT_MAX_CHARS, Chunk, chunk_book
from core.errors import CancelledByUser, DatasetMakerError, NarrationError
from core.events import CancelToken
from core.i18n import tr

log = logging.getLogger("voxprint.narration")

CHAPTER_TAIL_MS = 1200          # silence appended to every chapter (also separates chapters in single-file exports)


class TTSEngine(Protocol):
    """What the narrator needs from a speech synthesizer."""

    sample_rate: int
    tag: str                    # identity of voice + model + settings; part of every cache key

    def synthesize(self, text: str) -> np.ndarray:
        """Mono float32 samples in [-1, 1] at ``sample_rate`` for ``text``."""

    def close(self) -> None:
        """Release the model (GPU memory)."""


class PauseToken:
    """Pause/resume switch checked between chunks (thread-safe)."""

    def __init__(self) -> None:
        """Start in the running state."""
        self._run = threading.Event()
        self._run.set()

    def pause(self) -> None:
        """Hold the job before the next chunk."""
        self._run.clear()

    def resume(self) -> None:
        """Continue."""
        self._run.set()

    @property
    def paused(self) -> bool:
        """True while paused."""
        return not self._run.is_set()

    def wait(self, cancel: Optional[CancelToken] = None, poll: float = 0.1) -> None:
        """Block while paused; wake up (and raise) when ``cancel`` fires."""
        while not self._run.wait(poll):
            if cancel is not None:
                cancel.check()


@dataclass
class NarrationOptions:
    """User choices for a narration job."""
    formats: Set[str] = field(default_factory=lambda: set(ex.DEFAULT_FORMATS))
    bitrates: ex.Bitrates = field(default_factory=ex.Bitrates)
    max_chars: int = DEFAULT_MAX_CHARS
    speak_titles: bool = True                 # read each chapter title aloud before the chapter
    keep_cache: bool = False                  # keep the per-chunk audio after a successful export
    allow_aac: bool = True                    # False = the M4B (AAC) export is refused (see infra/features.py)
    #: Extension point (later translation): functions ``text -> text`` applied to every chunk.
    preprocessors: List[Callable[[str], str]] = field(default_factory=list)
    #: Automatic book preparation (rules + optional neural clean-up); ``None`` = the text is used as it is.
    prep: Optional[PrepPlan] = None


@dataclass
class NarrationProgress:
    """A progress report: ``done`` of ``total`` chunks, ``eta`` seconds (``None`` = not known yet)."""
    done: int
    total: int
    eta: Optional[float]
    message: str = ""
    phase: str = "synth"            # prepare | synth | assemble | export | done

    @property
    def fraction(self) -> float:
        """Overall progress 0..1 (synthesis is ~90 % of the work)."""
        base = {"prepare": 0.0, "synth": 0.0, "assemble": 0.9, "export": 0.95, "done": 1.0}[self.phase]
        span = {"prepare": 0.0, "synth": 0.9, "assemble": 0.05, "export": 0.05, "done": 0.0}[self.phase]
        return min(1.0, base + span * (self.done / self.total if self.total else 1.0))


@dataclass
class NarrationResult:
    """What a finished job produced."""
    out_dir: Path
    files: List[Path] = field(default_factory=list)
    chapters: int = 0
    chunks: int = 0
    resumed_chunks: int = 0                   # chunks that were already in the cache
    seconds: float = 0.0                      # length of the whole book


ProgressFn = Callable[[NarrationProgress], None]


class ChunkCache:
    """Finished chunks on disk: ``<dir>/<key>.flac`` (lossless); a partial write is never visible."""

    def __init__(self, folder: Path) -> None:
        """Use ``folder`` (created on demand)."""
        self.dir = Path(folder)

    @staticmethod
    def key(engine_tag: str, text: str) -> str:
        """Stable 24-hex-character key of an engine identity + text."""
        return hashlib.sha256(f"{engine_tag}\x00{text}".encode("utf-8")).hexdigest()[:24]

    def path(self, key: str) -> Path:
        """File of a key."""
        return self.dir / f"{key}.flac"

    def has(self, key: str) -> bool:
        """True if a finished chunk exists (cheap check; a corrupt file is detected on load and re-synthesized)."""
        p = self.path(key)
        return p.is_file() and p.stat().st_size > 0

    def save(self, key: str, audio: np.ndarray, sr: int) -> None:
        """Write atomically (temporary file, then rename)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f"{key}.part.flac"
        sf.write(str(tmp), np.asarray(audio, dtype=np.float32), sr, format="FLAC", subtype="PCM_16")
        os.replace(tmp, self.path(key))

    def load(self, key: str) -> Optional[tuple]:
        """``(samples float32, sample rate)`` or ``None`` if missing/corrupt (the corrupt file is removed)."""
        try:
            data, sr = sf.read(str(self.path(key)), dtype="float32", always_2d=False)
        except (OSError, RuntimeError, sf.LibsndfileError):
            self.path(key).unlink(missing_ok=True)
            return None
        return (data if data.ndim == 1 else data.mean(axis=1)), int(sr)

    def clear(self) -> None:
        """Delete the whole cache folder."""
        shutil.rmtree(self.dir, ignore_errors=True)


def prepare_text(text: str, options: NarrationOptions, normalizer: Optional[Callable[[str], str]] = None) -> str:
    """Text actually sent to the engine: preprocessors (extension hook) then the optional language normalizer."""
    for fn in options.preprocessors:
        text = fn(text)
    return normalizer(text) if normalizer else text


def default_normalizer(language: str) -> Optional[Callable[[str], str]]:
    """Russian numbers/abbreviations are spelled out before synthesis (the same normalizer the dataset builder uses)."""
    if (language or "").strip().lower() in ("russian", "ru"):
        from core.normalizer import normalize_for_tts

        return lambda t: normalize_for_tts(t, "Russian").spoken
    return None


def synthesize_chunks(chunks: Sequence[Chunk], engine_factory: Callable[[], TTSEngine], engine_tag: str,
                      cache: ChunkCache, texts: Dict[int, str], progress: ProgressFn,
                      cancel: CancelToken, pause: PauseToken) -> Dict[str, int]:
    """Synthesize every chunk that is not cached yet.  Returns ``{"cached": n, "made": m}``.

    ``texts`` maps a chunk index to the prepared text; ``engine_tag`` identifies the engine without creating it.
    """
    total = len(chunks)
    keys = {c.index: cache.key(engine_tag, texts[c.index]) for c in chunks}
    pending = [c for c in chunks if not cache.has(keys[c.index])]
    cached = total - len(pending)
    done = cached
    engine: Optional[TTSEngine] = None
    spent = 0.0
    chars_done = 0
    remaining_chars = sum(len(texts[c.index]) for c in pending)
    progress(NarrationProgress(done, total, None, tr("narr.resuming", n=cached) if cached else ""))
    # Pipeline: the GPU thread (this one) only generates; finished chunks are encoded to FLAC and written by a helper thread,
    # so the disk/CPU work of chunk N overlaps with the generation of chunk N+1.  Errors of the writer surface at the next check.
    saver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chunk-writer")
    futures: List[Future] = []
    try:
        queue = list(pending)
        batch_limit = 0                                      # 0 = not known yet (the engine is created lazily)
        while queue:
            pause.wait(cancel)
            cancel.check()
            if engine is None:
                progress(NarrationProgress(done, total, None, tr("narr.loading_model")))
                engine = engine_factory()
                batch_limit = _batch_limit(engine)
            group = _next_group(queue, texts, batch_limit)
            t0 = time.monotonic()
            audios, batch_limit = _synth_group(engine, [texts[c.index] for c in group], [c.index for c in group], batch_limit)
            spent += time.monotonic() - t0
            for c, audio in zip(group, audios):
                futures.append(saver.submit(cache.save, keys[c.index], audio, engine.sample_rate))
                chars_done += max(1, len(texts[c.index]))
                remaining_chars -= len(texts[c.index])
                done += 1
            _raise_finished(futures)
            eta = spent / chars_done * max(0, remaining_chars) if chars_done else None
            progress(NarrationProgress(done, total, eta, tr("narr.chunk_progress", done=done, total=total)))
        for f in futures:
            f.result()
    finally:
        saver.shutdown(wait=True)
        if engine is not None:
            try:
                engine.close()
            except Exception:  # noqa: BLE001
                log.warning("engine close failed", exc_info=True)
    return {"cached": cached, "made": len(pending)}


#: Chunks are sorted by length inside a window of this many batches (a batch runs until its longest item ends, so similar
#: lengths waste the least); a small window keeps the finished chunks close to book order for the live player.
SORT_WINDOW_BATCHES = 3


def _batch_limit(engine: TTSEngine) -> int:
    """How many chunks the engine can take at once (1 = no batching: engine without ``synthesize_batch`` or on CPU)."""
    if not callable(getattr(engine, "synthesize_batch", None)) or not callable(getattr(engine, "max_batch", None)):
        return 1
    try:
        return max(1, int(engine.max_batch()))
    except Exception:  # noqa: BLE001
        return 1


def _next_group(queue: List[Chunk], texts: Dict[int, str], limit: int) -> List[Chunk]:
    """Take the next batch off ``queue`` (in place): the shortest ``limit`` texts of the next window, so lengths are similar."""
    if limit <= 1:
        return [queue.pop(0)]
    window = sorted(queue[:limit * SORT_WINDOW_BATCHES], key=lambda c: len(texts[c.index]))
    group = window[:limit] if len(window) <= limit or SORT_WINDOW_BATCHES == 1 else _centered(window, limit)
    for c in group:
        queue.remove(c)
    return group


def _centered(window: List[Chunk], limit: int) -> List[Chunk]:
    """The batch of ``limit`` neighbours (in length order) that contains the window's first chunk in book order."""
    first = min(range(len(window)), key=lambda i: window[i].index)
    lo = max(0, min(first - limit // 2, len(window) - limit))
    return window[lo:lo + limit]


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


def _synth_group(engine: TTSEngine, group_texts: List[str], indexes: List[int], limit: int):
    """Synthesize one batch; returns ``(audios, new_limit)``.  Out of memory halves the limit and splits the batch; any other
    batch failure falls back to chunk-by-chunk synthesis (with the usual retry), so batching can never make a book fail."""
    if len(group_texts) == 1 or limit <= 1:
        return [_synth_with_retry(engine, t, i) for t, i in zip(group_texts, indexes)], limit
    try:
        audios = [np.asarray(a, dtype=np.float32).reshape(-1) for a in engine.synthesize_batch(group_texts)]  # type: ignore[attr-defined]
        if len(audios) == len(group_texts) and all(a.size for a in audios):
            return audios, limit
        log.warning("batch returned empty audio; retrying chunk by chunk")
    except (CancelledByUser, DatasetMakerError):
        raise
    except Exception as exc:  # noqa: BLE001
        log.warning("batch of %d failed (%s); %s", len(group_texts), exc, "halving the batch size" if _is_oom(exc) else "retrying one by one")
        if _is_oom(exc):
            _free_gpu_cache()
            half = max(1, len(group_texts) // 2)
            first, rest = _synth_group(engine, group_texts[:half], indexes[:half], half)
            second, _ = _synth_group(engine, group_texts[half:], indexes[half:], half)
            return first + second, half
    return [_synth_with_retry(engine, t, i) for t, i in zip(group_texts, indexes)], limit


def _free_gpu_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _raise_finished(futures: List[Future]) -> None:
    """Re-raise the error of an already finished background write (so a full disk stops the job early)."""
    for f in futures:
        if f.done() and f.exception() is not None:
            raise f.exception()  # type: ignore[misc]


def _synth_with_retry(engine: TTSEngine, text: str, index: int, attempts: int = 2) -> np.ndarray:
    """One synthesis call, retried once; empty output or an engine exception becomes a :class:`NarrationError`."""
    last: Optional[BaseException] = None
    for _ in range(attempts):
        try:
            audio = np.asarray(engine.synthesize(text), dtype=np.float32).reshape(-1)
        except (CancelledByUser, DatasetMakerError):
            raise
        except Exception as exc:  # noqa: BLE001 - torch/CUDA/engine errors are not typed
            last = exc
            log.warning("chunk %s failed: %s", index, exc)
            continue
        if audio.size:
            return audio
        last = RuntimeError("empty audio")
    raise NarrationError(tr("err.narration_chunk", n=index + 1), details=repr(last))


def assemble_chapters(book: Book, chunks: Sequence[Chunk], engine_tag: str, cache: ChunkCache,
                      texts: Dict[int, str], work_dir: Path) -> List[ex.ChapterAudio]:
    """Join the cached chunks of every chapter with their pauses into lossless chapter WAV files (streamed to disk)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    by_chapter: Dict[int, List[Chunk]] = {}
    for c in chunks:
        by_chapter.setdefault(c.chapter, []).append(c)
    out: List[ex.ChapterAudio] = []
    for ci in sorted(by_chapter):
        wav = work_dir / f"chapter_{len(out) + 1:04d}.wav"
        frames = 0
        sr_out = 0
        sf_out: Optional[sf.SoundFile] = None
        try:
            for c in by_chapter[ci]:
                loaded = cache.load(cache.key(engine_tag, texts[c.index]))
                if loaded is None:
                    raise NarrationError(tr("err.narration_chunk", n=c.index + 1), details="cache entry missing")
                data, sr = loaded
                if sf_out is None:
                    sr_out = sr
                    sf_out = sf.SoundFile(str(wav), "w", samplerate=sr_out, channels=1, subtype="PCM_16")
                elif sr != sr_out:
                    data = resample(data, sr, sr_out)
                sf_out.write(data)
                frames += len(data)
                is_last = c is by_chapter[ci][-1]
                gap = int(sr_out * (CHAPTER_TAIL_MS if is_last else c.pause_ms) / 1000)
                sf_out.write(np.zeros(gap, dtype=np.float32))
                frames += gap
        finally:
            if sf_out is not None:
                sf_out.close()
        title = book.chapters[ci].title.strip() or tr("book.chapter_n", n=ci + 1)
        out.append(ex.ChapterAudio(len(out), title, wav, frames / sr_out if sr_out else 0.0))
    return out


def narrate_book(book: Book, engine_factory: Callable[[], TTSEngine], engine_tag: str, out_dir: Path,
                 language: str = "", narrator: str = "", options: Optional[NarrationOptions] = None,
                 progress: Optional[ProgressFn] = None, cancel: Optional[CancelToken] = None,
                 pause: Optional[PauseToken] = None, ffmpeg: Optional[str] = None,
                 run: Optional[ex.Run] = None, chapters: Sequence[int] = (),
                 on_plan: Optional[Callable[[List[Path]], None]] = None) -> NarrationResult:
    """Run a complete narration job into ``out_dir / <book name>`` and return its :class:`NarrationResult`.

    ``engine_factory`` is called only if some chunk is missing from the cache.  ``chapters`` restricts the job to the
    given 0-based chapter numbers.  ``on_plan`` receives the ordered chunk files (they appear one by one; used by the live
    player).  ``ffmpeg``/``run`` are injectable for tests.
    """
    options = options or NarrationOptions()
    progress = progress or (lambda p: None)
    cancel = cancel or CancelToken()
    pause = pause or PauseToken()
    formats = [f for f in ex.ALL_FORMATS if f in options.formats] or list(ex.DEFAULT_FORMATS)
    if ex.FORMAT_M4B in formats and not options.allow_aac:
        raise NarrationError(tr("err.narration_aac_disabled"))
    job_dir = Path(out_dir) / ex.safe_filename(book.title, 100, fallback="audiobook")
    job_dir.mkdir(parents=True, exist_ok=True)
    if ex.required_encoders(formats):
        if not ffmpeg:
            raise NarrationError(tr("err.narration_no_ffmpeg"))
        missing = ex.missing_encoders(ffmpeg, formats, run)       # fail now, not after hours of synthesis
        if missing:
            raise NarrationError(tr("err.narration_encoder", encoder=", ".join(missing)))

    source_book = book                                        # original titles go into the exported files
    plan = options.prep
    if plan is not None and plan.enabled:
        progress(NarrationProgress(0, 1, None, tr("narr.preparing"), "prepare"))
        book, _report = run_preparation(
            book, plan, language, debug_dir=job_dir / ".debug", cache_dir=job_dir / ".cache",
            progress=lambda f, m: progress(NarrationProgress(int(f * 100), 100, None, tr("narr.preparing_neural", done=m), "prepare")),
            cancel=cancel)
    chunk_list = chunk_book(book, options.max_chars, chapters, options.speak_titles)
    if not chunk_list:
        raise NarrationError(tr("err.book_empty"))
    normalizer = None if (plan is not None and plan.spells_out_numbers) else default_normalizer(language)
    texts = {c.index: prepare_text(c.text, options, normalizer) for c in chunk_list}
    cache = ChunkCache(job_dir / ".cache")
    if on_plan is not None:
        on_plan([cache.path(cache.key(engine_tag, texts[c.index])) for c in chunk_list])
    counts = synthesize_chunks(chunk_list, engine_factory, engine_tag, cache, texts, progress, cancel, pause)

    total = len(chunk_list)
    cancel.check()
    progress(NarrationProgress(total, total, 0.0, tr("narr.assembling"), "assemble"))
    work = job_dir / ".work"
    shutil.rmtree(work, ignore_errors=True)
    chapter_audio = assemble_chapters(source_book, chunk_list, engine_tag, cache, texts, work)
    cancel.check()

    cover_path: Optional[Path] = None
    if source_book.cover:
        cover_path = work / f"cover.{source_book.cover_ext}"
        cover_path.write_bytes(source_book.cover)
    meta = ex.BookMeta(source_book.title, source_book.author, narrator, source_book.language, cover_path)
    result = ex.export_formats(
        ffmpeg, formats, chapter_audio, meta, job_dir, options.bitrates, run,
        progress=lambda f, m: progress(NarrationProgress(total, total, 0.0, tr("narr.exporting"), "export")),
        work_dir=work)
    shutil.rmtree(work, ignore_errors=True)
    if not options.keep_cache:
        cache.clear()
    progress(NarrationProgress(total, total, 0.0, tr("narr.done"), "done"))
    return NarrationResult(job_dir, result.files, len(chapter_audio), total, counts["cached"],
                           sum(c.duration for c in chapter_audio))
