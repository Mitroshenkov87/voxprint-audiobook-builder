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
the exported files keep their original spelling.  **Translation** (``options.translate``, :mod:`core.translate`) comes first:
the book is translated sentence by sentence (cached; the readable result is ``<job>/translation_<lang>.txt``), the job
folder gets the suffix `` (<lang>)`` and the target language is narrated.  **Speakers** (``options.speakers``,
:mod:`core.speakers`): when a male or female voice is selected besides the narrator, each chunk is synthesized with that
voice (one model load per voice, then the chapters are joined in book order).  The ``preprocessors`` option (a per-chunk
``text -> text`` hook) is still unused.
"""
from __future__ import annotations

import datetime
import hashlib
import itertools
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
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
from core.audio_utils import resample, time_stretch, trim_silence
from core.book_parsers import Book
from core.book_prep import PrepPlan, run_preparation
from core import translate as tl
from core import pauses as pz
from core import pace as pc
from core import ai_disclosure
from core import cpu_budget
from core import gpu_lock
from core import gpu_thermal
from core import speakers as spk
from core import sysmem_spill
from core.chunker import DEFAULT_MAX_CHARS, Chunk, chunk_book
from core import ordinals
from core import yo
from core.errors import CancelledByUser, DatasetMakerError, NarrationError
from core.events import CancelToken
from core.i18n import tr

log = logging.getLogger("voxprint.narration")

CHAPTER_TAIL_MS = 1200          # silence appended to every chapter (also separates chapters in single-file exports)
# Windows monotonic_ns can stay on the same value for two writes in one thread. The counter keeps each part name unique.
_PART_SEQ = itertools.count()


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
    #: Explicit silence between the pieces (comma, sentence, ellipsis, dash, paragraph, chapter ...), independent of the model's
    #: prosody (:mod:`core.pauses`).  ``None`` = the earlier packed chunks with fixed pauses.
    pauses: Optional[pz.PauseProfile] = None   # opt-in: explicit pauses can make the model swallow short words
    #: Structured pauses (:class:`core.pauses.PauseLengths`, Settings / ``--pause-*``): a chunk per sentence and strong
    #: transition, every piece trimmed of its own silence and joined with these lengths.  ``None`` = packed chunks.
    pause_lengths: Optional[pz.PauseLengths] = None
    #: Reading speed (:class:`core.pace.Pace`): a tempo factor per chunk, applied by time-stretching.  ``None`` = as spoken.
    pace: Optional[pc.Pace] = None
    #: Machine translation of the book before narration (:mod:`core.translate`); ``None`` = narrate the book as it is.
    translate: Optional[tl.TranslatePlan] = None
    #: "Prepare text for narration" with the AI text model (:mod:`core.llm_text`, a ``LLMPlan``); ``None`` = off.
    llm_prepare: Optional[object] = None
    #: Per-chunk speech-recognition check with regeneration (:mod:`core.chunk_check`); the runner builds the checker.
    check_chunks: bool = False
    check_max_cer: float = 0.15
    check_retries: int = 2
    #: Spoken AI disclosure as the first chunk (:mod:`core.ai_disclosure`); opt-in, the user decides.
    ai_disclosure: bool = False
    disclosure_date: Optional["datetime.date"] = None        # month/year said in the disclosure (None = today; tests)
    #: Ordinal numbers by context ("глава 2" -> "глава вторая", "3-го" -> "третьего", "21st"), :mod:`core.ordinals`;
    #: Settings -> Narration / ``--no-ordinals``.  On by default.
    ordinals: bool = True
    #: Russian letter yo where the safe dictionary is sure (:mod:`core.yo`).  On by default; ``--no-yo`` turns it off.
    #: Stress marks are not written: the base TTS model does not read them.
    yo: bool = True
    #: Speaker marks and the male / female voice ids (:mod:`core.speakers`).  ``None`` = the narrator voice only.
    speakers: Optional[spk.SpeakerCast] = None


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
    chunk_check: Optional[Dict[str, int]] = None   # counts of the per-chunk check (None = not run)
    speaker_warning: str = ""                 # "mismatch", "unparsed", "no_speakers", joined with ";"


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
        """Write atomically (temporary file, then rename).

        The temporary name is unique per write: two chunks with the same text (a repeated verse or refrain) have the same
        key and may be written by two writer threads at once - with one shared ``<key>.part.flac`` the second writer held
        the file the first one was renaming (Windows ``WinError 32``, build 667).  The rename is retried with backoff
        while Windows reports the file as in use (an antivirus scan, a reader); if the finished file is already there
        (the twin chunk won the race) that copy is kept.  Only when the file stays locked does a clear
        :class:`NarrationError` stop the job (the cache keeps every finished chunk, so a new start resumes)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / (
            f"{key}.{os.getpid()}-{threading.get_ident()}-{time.monotonic_ns()}-{next(_PART_SEQ)}.part.flac")
        dst = self.path(key)
        try:
            sf.write(str(tmp), np.asarray(audio, dtype=np.float32), sr, format="FLAC", subtype="PCM_16")
            replace_with_retry(tmp, dst)
        except PermissionError as exc:
            if self.has(key):                       # the twin chunk (same key = same text and engine) is on disk
                log.info("chunk %s is already cached by a parallel write - keeping that copy", key)
                return
            raise NarrationError(tr("err.narration_file_in_use", name=dst.name), details=repr(exc)) from exc
        finally:
            try:
                tmp.unlink(missing_ok=True)            # gone after a successful rename; a leftover after a failure
            except OSError:
                pass

    def sweep_partial(self, min_age: float = 600.0) -> int:
        """Delete ``*.part.flac`` leftovers of an interrupted write that are older than ``min_age`` seconds (a younger
        one may belong to a job that is still writing).  Returns how many were removed."""
        n = 0
        now = time.time()
        for p in self.dir.glob("*.part.flac") if self.dir.is_dir() else []:
            try:
                if now - p.stat().st_mtime >= min_age:
                    p.unlink()
                    n += 1
            except OSError:
                pass
        return n

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


#: A file that Windows reports as in use (WinError 32 / 5) is retried this many times, the pause doubling from
#: ``LOCKED_FIRST_PAUSE`` seconds (0.2 + 0.4 + 0.8 + 1.6 + 3.2 = about 6 s in total).
LOCKED_RETRIES = 5
LOCKED_FIRST_PAUSE = 0.2


def _is_locked(exc: BaseException) -> bool:
    """Windows "file in use" / "access denied" (another handle holds the file for a moment); POSIX never sets winerror."""
    return isinstance(exc, PermissionError) and getattr(exc, "winerror", None) in (5, 32, None)


def replace_with_retry(src: Path, dst: Path, retries: Optional[int] = None, pause: Optional[float] = None,
                       sleep: Callable[[float], None] = time.sleep) -> None:
    """``os.replace(src, dst)``, retried with exponential backoff while the file is in use; the last error is raised."""
    retries = LOCKED_RETRIES if retries is None else retries
    delay = LOCKED_FIRST_PAUSE if pause is None else pause
    for attempt in range(retries + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            if not _is_locked(exc) or attempt >= retries:
                raise
            log.info("%s is in use (%s) - retrying in %.1f s", Path(dst).name, exc, delay)
            sleep(delay)
            delay *= 2


def prepare_text(text: str, options: NarrationOptions, normalizer: Optional[Callable[[str], str]] = None) -> str:
    """Text actually sent to the engine: preprocessors (extension hook) then the optional language normalizer."""
    for fn in options.preprocessors:
        text = fn(text)
    return normalizer(text) if normalizer else text


def chain_steps(*steps: Optional[Callable[[str], str]]) -> Optional[Callable[[str], str]]:
    """One ``text -> text`` function running ``steps`` in order (``None`` entries skipped); ``None`` if there is none."""
    fns = [f for f in steps if f is not None]
    if not fns:
        return None
    if len(fns) == 1:
        return fns[0]

    def run(text: str) -> str:
        for fn in fns:
            text = fn(text)
        return text
    return run


def text_steps(language: Optional[str], book_language: Optional[str], options: NarrationOptions,
               numbers_spelled: bool = False) -> Optional[Callable[[str], str]]:
    """The normalization run on every chunk before synthesis.

    Ordinals by context come first (they need the digits and the noun next to them).  Russian yo restoration comes
    next, while a dotted abbreviation still has the following word in the case the book used (``мед. училище``).
    The language normalizer runs last: it may capitalise after a full stop, and a second yo pass would then miss
    the abbreviation.  The normalizer keeps a yo the dictionary already wrote.
    """
    spoken = language if (language or "").strip().lower() not in ("", "auto") else book_language
    ordinal = ordinals.ordinal_step(spoken) if options.ordinals else None
    letter = yo.as_step(spoken) if options.yo else None
    base = None if numbers_spelled else default_normalizer(language)
    return chain_steps(ordinal, letter, base)


def default_normalizer(language: str) -> Optional[Callable[[str], str]]:
    """Russian numbers/abbreviations are spelled out before synthesis (the same normalizer the dataset builder uses)."""
    if (language or "").strip().lower() in ("russian", "ru"):
        from core.normalizer import normalize_for_tts

        return lambda t: normalize_for_tts(t, "Russian").spoken
    return None


#: FLAC writers of finished chunks, and how many finished-but-unwritten batches may wait for them (bounded memory).
WRITER_THREADS = 2
PENDING_WRITE_BATCHES = 2


def chunk_voice_tag(chunk: Chunk, engine_tag: str, voices: Optional[Dict[str, tuple]] = None) -> str:
    """Cache tag of one chunk: the extra voice's tag when ``chunk.voice_id`` names one, else ``engine_tag``."""
    vid = chunk.voice_id or ""
    if vid and voices and vid in voices:
        return str(voices[vid][1])
    return engine_tag


def synthesize_chunks(chunks: Sequence[Chunk], engine_factory: Callable[[], TTSEngine], engine_tag: str,
                      cache: ChunkCache, texts: Dict[int, str], progress: ProgressFn,
                      cancel: CancelToken, pause: PauseToken, on_saved: Optional[Callable[[Chunk], None]] = None,
                      engine_future: Optional[Future] = None,
                      background_check: Optional[Callable[[], None]] = None, checker=None,
                      voices: Optional[Dict[str, tuple]] = None,
                      thermal: Optional[gpu_thermal.Monitor] = None) -> Dict[str, int]:
    """Synthesize every chunk that is not cached yet.  Returns ``{"cached": n, "made": m}``.

    ``texts`` maps a chunk index to the prepared text; ``engine_tag`` identifies the engine without creating it.
    ``voices`` maps an extra voice id to ``(factory, tag)``. Chunks are spoken one voice at a time (the narrator
    first) so the model is loaded once per voice, then closed before the next voice. With no ``voice_id`` set this is
    the single-voice path. ``on_saved(chunk)`` runs (on a writer thread) once a chunk is on disk.  ``engine_future``:
    an engine already being loaded in the background (used for the narrator voice).  ``background_check()`` raises the
    error of other background work (chapter assembly, encodes) between batches.  ``checker``
    (:class:`core.chunk_check.ChunkChecker`) re-reads each new chunk before it is written and regenerates it when it
    does not say its text; it is closed at the end.
    """
    total = len(chunks)
    keys = {c.index: cache.key(chunk_voice_tag(c, engine_tag, voices), texts[c.index]) for c in chunks}
    pending = [c for c in chunks if not cache.has(keys[c.index])]
    cached = total - len(pending)
    done = cached
    engine: Optional[TTSEngine] = None
    spent = 0.0
    chars_done = 0
    remaining_chars = sum(len(texts[c.index]) for c in pending)
    progress(NarrationProgress(done, total, None, tr("narr.resuming", n=cached) if cached else ""))
    # Pipeline: the GPU thread (this one) only generates; finished chunks are encoded to FLAC and written by helper threads,
    # so the disk/CPU work of batch N overlaps with the generation of batch N+1.  A semaphore bounds the chunks waiting for
    # a writer (their audio is in memory), so a slow disk holds the GPU back instead of filling the RAM.
    saver = ThreadPoolExecutor(max_workers=WRITER_THREADS, thread_name_prefix="chunk-writer")
    slots: Optional[threading.BoundedSemaphore] = None
    futures: List[Future] = []

    def save(c: Chunk, audio: np.ndarray, sr: int) -> None:
        try:
            cache.save(keys[c.index], audio, sr)
        finally:
            slots.release()  # type: ignore[union-attr]
        if on_saved is not None:
            on_saved(c)

    order: List[str] = []
    buckets: Dict[str, List[Chunk]] = {}
    for c in pending:
        vid = c.voice_id or ""
        if vid not in buckets:
            order.append(vid)
            buckets[vid] = []
        buckets[vid].append(c)
    if "" in order:                                          # narrator first, then each extra voice once
        order.remove("")
        order.insert(0, "")
    used_preload = False

    spill: Optional[sysmem_spill.Monitor] = None
    try:
        batch_ceiling: Optional[int] = None                  # lowered by an out-of-memory or a sysmem spill; None = follow the fresh plan
        eta: Optional[float] = None                          # last estimate, repeated in the "batch starts" message
        groups_done = 0
        for vid in order:
            if engine is not None:                           # one model in memory: the previous voice is closed first
                try:
                    engine.close()
                except Exception:  # noqa: BLE001
                    log.warning("engine close failed", exc_info=True)
                engine = None
                if spill is not None:
                    spill.stop()
                    spill = None
                batch_ceiling = None                         # the next voice loads its own model; plan from that free memory
            queue = list(buckets[vid])
            while queue:
                pause.wait(cancel)
                cancel.check()
                if engine is None:
                    progress(NarrationProgress(done, total, None, tr("narr.loading_model")))
                    if not vid and engine_future is not None:
                        engine = _wait_engine(engine_future, cancel)
                        used_preload = True
                    elif vid and voices and vid in voices:
                        engine = voices[vid][0]()
                    else:
                        engine = engine_factory()
                    spill = sysmem_spill.start_after_load()  # baseline is the first sample after this load
                    if thermal is not None:
                        thermal.retarget(getattr(engine, "device", ""))
                if groups_done and thermal is not None and thermal.should_cool():
                    progress(NarrationProgress(done, total, eta, tr("narr.cooling"), "synth"))
                    thermal.cool(cancel, pause)
                measured = _batch_limit(engine)              # re-reads torch.cuda.mem_get_info on a real engine
                limit = measured if batch_ceiling is None else min(measured, batch_ceiling)
                if spill is not None and spill.tripped():
                    _free_gpu_cache()
                    limit = max(1, limit // 2)
                    batch_ceiling = limit
                    grown = spill.growth()
                    spill.mark()
                    log.warning(
                        "GPU shared memory grew by %.0f MB after the model load; treating it as out of memory "
                        "and halving the batch. %s", grown / (1024 * 1024), sysmem_spill.DIAGNOSTIC_NOTE)
                if slots is None:
                    slots = threading.BoundedSemaphore(max(4, PENDING_WRITE_BATCHES * max(1, limit)))
                group = _next_group(queue, texts, limit)
                # Say what is being generated *before* the (possibly minutes-long) batch call: otherwise the UI and the log
                # stay on "model loaded" until the first batch is finished and the job looks frozen.
                first, last = done + 1, done + len(group)
                log.info("Synthesizing chunks %d-%d of %d (%d chars)", first, last, total,
                         sum(len(texts[c.index]) for c in group))
                progress(NarrationProgress(done, total, eta, tr("narr.synth_batch", first=first, last=last, total=total)))
                t0 = time.monotonic()
                audios, new_limit = _synth_group(engine, [texts[c.index] for c in group], [c.index for c in group], limit)
                if new_limit < limit:
                    batch_ceiling = new_limit
                groups_done += 1
                if checker is not None:        # on this (GPU) thread, before the writers see the chunk: the cache holds the winner
                    progress(NarrationProgress(done, total, eta, tr("narr.checking", first=first, last=last, total=total)))
                    audios = [checker.check(engine, texts[c.index], keys[c.index], c.index, a, engine.sample_rate)
                              for c, a in zip(group, audios)]
                spent += time.monotonic() - t0
                for c, audio in zip(group, audios):
                    while not slots.acquire(timeout=0.2):                # type: ignore[union-attr]
                        cancel.check()
                        _raise_finished(futures)                        # a failed write never frees its wait
                    futures.append(saver.submit(save, c, audio, engine.sample_rate))
                    chars_done += max(1, len(texts[c.index]))
                    remaining_chars -= len(texts[c.index])
                    done += 1
                _raise_finished(futures)
                futures = [f for f in futures if not f.done()]           # finished writes: nothing left to check
                if background_check is not None:
                    background_check()
                eta = spent / chars_done * max(0, remaining_chars) if chars_done else None
                progress(NarrationProgress(done, total, eta, tr("narr.chunk_progress", done=done, total=total)))
        for f in futures:
            f.result()
    finally:
        if spill is not None:
            spill.stop()
        saver.shutdown(wait=True)
        if checker is not None:
            checker.close()                                              # the recogniser leaves (V)RAM with the engine
        if engine_future is not None and not used_preload:
            close_when_loaded(engine_future)                             # loaded for the narrator but that voice was cached
        if engine is not None:
            try:
                engine.close()
            except Exception:  # noqa: BLE001
                log.warning("engine close failed", exc_info=True)
    return {"cached": cached, "made": len(pending)}


def _wait_engine(fut: Future, cancel: CancelToken) -> TTSEngine:
    """The engine of a background load; ``cancel`` is honoured while waiting."""
    while True:
        try:
            return fut.result(timeout=0.2)
        except FutureTimeout:
            cancel.check()


def close_when_loaded(fut: Future) -> None:
    """Close the engine of a background load once it is there (without waiting for it now)."""
    def done(f: Future) -> None:
        if not f.cancelled() and f.exception() is None and f.result() is not None:
            try:
                f.result().close()
            except Exception:  # noqa: BLE001
                log.warning("engine close failed", exc_info=True)
    fut.add_done_callback(done)


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


def assemble_chapter(book: Book, position: int, chapter: int, chunks: Sequence[Chunk], engine_tag: str,
                     cache: ChunkCache, texts: Dict[int, str], work_dir: Path,
                     pauses: Optional[pz.PauseProfile] = None, lengths: Optional[pz.PauseLengths] = None,
                     shape: bool = False, tags: Optional[Dict[int, str]] = None) -> ex.ChapterAudio:
    """Join the cached chunks of one chapter (``chunks``, in order) with their pauses into ``chapter_<position+1>.wav``.

    With ``shape`` every piece first loses its own leading / trailing silence (so the inserted pauses are what is heard)
    and is time-stretched by its ``tempo`` (:mod:`core.pace`)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    wav = work_dir / f"chapter_{position + 1:04d}.wav"
    frames = 0
    sr_out = 0
    sf_out: Optional[sf.SoundFile] = None
    try:
        for c in chunks:
            loaded = cache.load(cache.key((tags or {}).get(c.index, engine_tag), texts[c.index]))
            if loaded is None:
                raise NarrationError(tr("err.narration_chunk", n=c.index + 1), details="cache entry missing")
            data, sr = loaded
            if sf_out is None:
                sr_out = sr
                sf_out = sf.SoundFile(str(wav), "w", samplerate=sr_out, channels=1, subtype="PCM_16")
            elif sr != sr_out:
                data = resample(data, sr, sr_out)
            if shape:
                data = time_stretch(trim_silence(data, sr_out), sr_out, c.tempo)
            sf_out.write(data)
            frames += len(data)
            is_last = c is chunks[-1]
            tail_ms = pauses.ms(pz.CHAPTER) if pauses is not None else (lengths.ms(pz.CHAPTER) if lengths else CHAPTER_TAIL_MS)
            gap = int(sr_out * (tail_ms if is_last else c.pause_ms) / 1000)
            sf_out.write(np.zeros(gap, dtype=np.float32))
            frames += gap
    finally:
        if sf_out is not None:
            sf_out.close()
    title = book.chapters[chapter].title.strip() or tr("book.chapter_n", n=chapter + 1)
    return ex.ChapterAudio(position, title, wav, frames / sr_out if sr_out else 0.0)


def _by_chapter(chunks: Sequence[Chunk]) -> Dict[int, List[Chunk]]:
    """``{chapter: [chunks in order]}`` in chapter order."""
    by: Dict[int, List[Chunk]] = {}
    for c in chunks:
        by.setdefault(c.chapter, []).append(c)
    return {ci: by[ci] for ci in sorted(by)}


def assemble_chapters(book: Book, chunks: Sequence[Chunk], engine_tag: str, cache: ChunkCache,
                      texts: Dict[int, str], work_dir: Path,
                      pauses: Optional[pz.PauseProfile] = None, lengths: Optional[pz.PauseLengths] = None,
                      shape: bool = False, tags: Optional[Dict[int, str]] = None) -> List[ex.ChapterAudio]:
    """Join the cached chunks of every chapter with their pauses into lossless chapter WAV files (streamed to disk)."""
    return [assemble_chapter(book, pos, ci, group, engine_tag, cache, texts, work_dir, pauses, lengths, shape, tags)
            for pos, (ci, group) in enumerate(_by_chapter(chunks).items())]


class _ChapterPipeline:
    """Assembles each chapter as soon as its last chunk is on disk and hands it to the exporter - on the CPU pool, while
    the GPU goes on with the next chapters.  Chapters are numbered by their place in the book, never by completion, so
    the files are the same as after a serial run."""

    def __init__(self, book: Book, chunks: Sequence[Chunk], engine_tag: str, cache: ChunkCache, texts: Dict[int, str],
                 work: Path, pauses: Optional[pz.PauseProfile], exporter: ex.Exporter, pool: ThreadPoolExecutor,
                 cancel: CancelToken, pause: PauseToken, lengths: Optional[pz.PauseLengths] = None,
                 shape: bool = False, tags: Optional[Dict[int, str]] = None) -> None:
        self.book, self.engine_tag, self.cache, self.texts, self.work, self.pauses = book, engine_tag, cache, texts, work, pauses
        self.lengths, self.shape, self.tags = lengths, shape, tags or {}
        self.exporter, self.pool, self.cancel, self.pause = exporter, pool, cancel, pause
        self.groups = _by_chapter(chunks)
        self.position = {ci: pos for pos, ci in enumerate(self.groups)}
        self.left = {ci: sum(1 for c in g if not cache.has(cache.key(self.tags.get(c.index, engine_tag), texts[c.index])))
                     for ci, g in self.groups.items()}
        self.futures: Dict[int, Future] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        """Chapters that are complete already (resumed job) start right away."""
        for ci, n in self.left.items():
            if n == 0:
                self._submit(ci)

    def chunk_saved(self, chunk: Chunk) -> None:
        """Called by a cache writer: the last chunk of a chapter starts its assembly."""
        with self._lock:
            self.left[chunk.chapter] -= 1
            ready = self.left[chunk.chapter] == 0
        if ready:
            self._submit(chunk.chapter)

    def _submit(self, ci: int) -> None:
        with self._lock:
            if ci in self.futures:
                return
            self.futures[ci] = self.pool.submit(self._assemble, ci)

    def _assemble(self, ci: int) -> ex.ChapterAudio:
        self.pause.wait(self.cancel)               # paused: background work holds too (the user wants the machine back)
        self.cancel.check()
        ch = assemble_chapter(self.book, self.position[ci], ci, self.groups[ci], self.engine_tag, self.cache, self.texts,
                              self.work, self.pauses, self.lengths, self.shape, self.tags)
        self.exporter.add_chapter(ch)
        return ch

    def check(self) -> None:
        """Raise the error of a finished background job (assembly or encode) early."""
        for f in list(self.futures.values()) + self.exporter.pending():
            if f.done() and not f.cancelled() and f.exception() is not None:
                raise f.exception()  # type: ignore[misc]

    def finish(self) -> List[ex.ChapterAudio]:
        """All chapters in book order (any chapter not started yet is started now)."""
        for ci in self.groups:
            self._submit(ci)
        return [self.futures[ci].result() for ci in self.groups]


def _effective_translation(book: Book, options: NarrationOptions) -> Optional[tl.TranslatePlan]:
    """The translation plan that really applies (``None`` when off or the book is already in the target language)."""
    tplan = options.translate if (options.translate is not None and options.translate.enabled) else None
    if tplan is not None and (tplan.source or tl.detect_book_language(book)) == tplan.target:
        return None
    return tplan


def job_dir_for(book: Book, out_dir: Path, options: Optional[NarrationOptions] = None) -> Path:
    """The job folder of ``book`` inside the working folder ``out_dir``: ``<title>`` or ``<title> (<lang>)`` when translated
    (next to, not over, the original).  The UI uses it to offer copying the book there before the job starts."""
    tplan = _effective_translation(book, options or NarrationOptions())
    folder = ex.safe_filename(book.title, 100, fallback="audiobook")
    if tplan is not None:
        folder = ex.safe_filename(f"{book.title} ({tplan.target})", 100, fallback="audiobook")
    return Path(out_dir) / folder


def narrate_book(book: Book, engine_factory: Callable[[], TTSEngine], engine_tag: str, out_dir: Path,
                 language: str = "", narrator: str = "", options: Optional[NarrationOptions] = None,
                 progress: Optional[ProgressFn] = None, cancel: Optional[CancelToken] = None,
                 pause: Optional[PauseToken] = None, ffmpeg: Optional[str] = None,
                 run: Optional[ex.Run] = None, chapters: Sequence[int] = (),
                 on_plan: Optional[Callable[[List[Path]], None]] = None, checker=None,
                 extra_engines: Optional[Dict[str, tuple]] = None) -> NarrationResult:
    """Run a complete narration job into ``out_dir / <book name>`` and return its :class:`NarrationResult`.

    ``engine_factory`` is called only if some chunk is missing from the cache.  ``chapters`` restricts the job to the
    given 0-based chapter numbers.  ``on_plan`` receives the ordered chunk files (they appear one by one; used by the live
    player).  ``ffmpeg``/``run`` are injectable for tests.  ``checker``: the per-chunk check (:mod:`core.chunk_check`), or None.

    The GPU lock (:mod:`core.gpu_lock`) is held for every stage of this call and removed when it returns. A temperature
    monitor (:mod:`core.gpu_thermal`) starts with the job.
    """
    progress_cb = progress or (lambda _p: None)
    cancel_token = cancel or CancelToken()
    job_name = str(getattr(book, "title", "") or "").strip() or "audiobook"

    def on_busy(owner: str, other: str) -> None:
        progress_cb(NarrationProgress(0, 1, None, tr("narr.gpu_busy", owner=owner, job=other), "prepare"))

    thermal = gpu_thermal.Monitor(started=time.monotonic())
    with gpu_lock.hold(job_name, on_busy=on_busy, cancel=cancel_token) as held:
        thermal.start()

        def report(item: NarrationProgress) -> None:
            held.note(item.eta)
            progress_cb(item)

        try:
            return _narrate_book_work(
                book, engine_factory, engine_tag, out_dir, language, narrator, options, report, cancel_token,
                pause, ffmpeg, run, chapters, on_plan, checker, extra_engines, thermal)
        finally:
            thermal.stop()


def _narrate_book_work(book: Book, engine_factory: Callable[[], TTSEngine], engine_tag: str, out_dir: Path,
                       language: str = "", narrator: str = "", options: Optional[NarrationOptions] = None,
                       progress: Optional[ProgressFn] = None, cancel: Optional[CancelToken] = None,
                       pause: Optional[PauseToken] = None, ffmpeg: Optional[str] = None,
                       run: Optional[ex.Run] = None, chapters: Sequence[int] = (),
                       on_plan: Optional[Callable[[List[Path]], None]] = None, checker=None,
                       extra_engines: Optional[Dict[str, tuple]] = None,
                       thermal: Optional[gpu_thermal.Monitor] = None) -> NarrationResult:
    """The narration stages. :func:`narrate_book` holds the GPU lock around this."""
    options = options or NarrationOptions()
    t_job = time.monotonic()
    stage_s: Dict[str, float] = {}
    progress = progress or (lambda p: None)
    cancel = cancel or CancelToken()
    pause = pause or PauseToken()
    formats = [f for f in ex.ALL_FORMATS if f in options.formats] or list(ex.DEFAULT_FORMATS)
    if ex.FORMAT_M4B in formats and not options.allow_aac:
        raise NarrationError(tr("err.narration_aac_disabled"))
    tplan = _effective_translation(book, options)
    job_dir = job_dir_for(book, out_dir, options)
    job_dir.mkdir(parents=True, exist_ok=True)
    if ex.required_encoders(formats):
        if not ffmpeg:
            raise NarrationError(tr("err.narration_no_ffmpeg"))
        missing = ex.missing_encoders(ffmpeg, formats, run)       # fail now, not after hours of synthesis
        if missing:
            raise NarrationError(tr("err.narration_encoder", encoder=", ".join(missing)))

    if tplan is not None:
        progress(NarrationProgress(0, 1, None, tr("narr.translating", pct=0), "prepare"))
        book, _src = tl.ensure_translation(
            book, tplan, job_dir, cancel=cancel, only=set(chapters) if chapters else None,
            progress=lambda f, m: progress(NarrationProgress(int(f * 100), 100, None, m, "prepare")))
        language = tl.LANGUAGE_NAMES.get(tplan.target, language)      # narrate with the target language
    if options.llm_prepare is not None:           # the AI model runs (and is closed again) before the voice model loads
        from core import llm_text

        progress(NarrationProgress(0, 1, None, tr("narr.llm_preparing", pct=0), "prepare"))
        book = llm_text.prepare_book(
            book, options.llm_prepare, tplan.target if tplan is not None else (tl.detect_book_language(book) or "en"),
            job_dir, lambda f: progress(NarrationProgress(int(f * 100), 100, None, tr("narr.llm_preparing", pct=int(f * 100)),
                                                          "prepare")), cancel, set(chapters) if chapters else ())
    source_book = book                                        # titles of the exported files (translated ones when translating)
    plan = options.prep
    if plan is not None and plan.enabled:
        progress(NarrationProgress(0, 1, None, tr("narr.preparing"), "prepare"))
        book, _report = run_preparation(
            book, plan, language, debug_dir=job_dir / ".debug", cache_dir=job_dir / ".cache",
            progress=lambda f, m: progress(NarrationProgress(int(f * 100), 100, None, tr("narr.preparing_neural", done=m), "prepare")),
            cancel=cancel)
    speaker_lines: Optional[List[spk.SpeakerLine]] = None
    speaker_note = ""
    tag_warning = ""
    tag_raw: Optional[str] = None
    cast = options.speakers if (options.speakers is not None and extra_engines) else None
    if cast is not None:
        speaker_lines = cast.lines
        if speaker_lines is None and cast.tagger is not None:
            code = tplan.target if tplan is not None else (tl.detect_book_language(book) or "en")
            progress(NarrationProgress(0, 1, None, tr("spk.tagging", pct=0), "prepare"))
            tagged = spk.tag_paragraphs(
                [text for _ci, text in spk.paragraphs(book)], code, cast.tagger,
                lambda f: progress(NarrationProgress(int(f * 100), 100, None, tr("spk.tagging", pct=int(f * 100)), "prepare")))
            speaker_lines = list(tagged)
            tag_warning = tagged.warning
            tag_raw = tagged.raw
    chunk_list = chunk_book(book, options.max_chars, chapters, options.speak_titles, options.pauses, options.pause_lengths,
                            options.pace)
    shape = options.pause_lengths is not None or options.pace is not None
    if not chunk_list:
        raise NarrationError(tr("err.book_empty"))
    if cast is not None and speaker_lines:
        voice_ids = {"male": cast.male_id, "female": cast.female_id}
        chunk_list, speaker_note = spk.assign(chunk_list, book, speaker_lines, voice_ids, cast.narrator_id,
                                              per_line=cast.voices_for(speaker_lines))
        codes = [part for part in (tag_warning, speaker_note) if part]
        speaker_note = ";".join(codes)
        debug = job_dir / ".debug"
        debug.mkdir(parents=True, exist_ok=True)
        note = spk.describe(speaker_lines)
        if "mismatch" in codes:
            note += "\nmismatch\n"
            progress(NarrationProgress(0, 1, None, tr("spk.mismatch"), "prepare"))
        (debug / "speakers.txt").write_text(note, encoding="utf-8")
        if tag_raw is not None:
            (debug / "speakers-raw.txt").write_text(tag_raw, encoding="utf-8")
    if options.ai_disclosure:              # in the narrated language (the translation target when translating)
        chunk_list = ai_disclosure.prepend(chunk_list, ai_disclosure.phrase(
            language or book.language, narrator, options.disclosure_date))
    cache = ChunkCache(job_dir / ".cache")
    cache.sweep_partial()                  # leftovers of an interrupted write would count as "cached" below
    # A fresh job (nothing cached yet) certainly needs the model: start loading it now, on its own thread, so the
    # loading (disk + GPU upload) overlaps with the text normalisation below.  With any cached chunk the load stays
    # lazy - a fully cached job must never load the model.
    loader: Optional[ThreadPoolExecutor] = None
    engine_future: Optional[Future] = None
    if not (cache.dir.is_dir() and any(not p.name.endswith(".part.flac") for p in cache.dir.glob("*.flac"))):
        loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-load")
        engine_future = loader.submit(engine_factory)
    budget = cpu_budget.plan(use_cuda=_cuda_available())
    pool = ThreadPoolExecutor(max_workers=budget.workers, thread_name_prefix="narration-cpu")
    try:
        normalizer = text_steps(language, book.language, options, bool(plan is not None and plan.spells_out_numbers))
        texts = {c.index: prepare_text(c.text, options, normalizer) for c in chunk_list}
        tags = {c.index: chunk_voice_tag(c, engine_tag, extra_engines) for c in chunk_list}
        if on_plan is not None:
            on_plan([cache.path(cache.key(tags[c.index], texts[c.index])) for c in chunk_list])
        work = job_dir / ".work"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        cover_path: Optional[Path] = None
        if source_book.cover:                               # before any chapter encode: MP3 chapters embed it
            cover_path = work / f"cover.{source_book.cover_ext}"
            cover_path.write_bytes(source_book.cover)
        meta = ex.BookMeta(source_book.title, source_book.author, narrator, source_book.language, cover_path)
        n_chapters = len({c.chapter for c in chunk_list})
        exporter = ex.Exporter(ffmpeg, formats, meta, job_dir, n_chapters, options.bitrates, run, work, pool.submit)
        chapters_pipe = _ChapterPipeline(source_book, chunk_list, engine_tag, cache, texts, work, options.pauses, exporter,
                                         pool, cancel, pause, options.pause_lengths, shape, tags)
        chapters_pipe.start()
        handed_over, engine_future = engine_future, None          # from here on synthesize_chunks owns (and closes) it
        counts = synthesize_chunks(chunk_list, engine_factory, engine_tag, cache, texts, progress, cancel, pause,
                                   on_saved=chapters_pipe.chunk_saved, engine_future=handed_over,
                                   background_check=chapters_pipe.check, checker=checker, voices=extra_engines,
                                   thermal=thermal)

        stage_s["synthesis"] = time.monotonic() - t_job
        total = len(chunk_list)
        cancel.check()
        progress(NarrationProgress(total, total, 0.0, tr("narr.assembling"), "assemble"))
        chapter_audio = chapters_pipe.finish()
        stage_s["assembly_wait"] = time.monotonic() - t_job - stage_s["synthesis"]
        cancel.check()
        result = exporter.finish(
            chapter_audio, progress=lambda f, m: progress(NarrationProgress(total, total, 0.0, tr("narr.exporting"), "export")))
        stage_s["export_wait"] = time.monotonic() - t_job - stage_s["synthesis"] - stage_s["assembly_wait"]
    finally:
        # On an error or cancel: queued background jobs are dropped, running ones (a chapter, one ffmpeg) end first
        pool.shutdown(wait=True, cancel_futures=True)
        if engine_future is not None:                   # failed before the synthesis started
            close_when_loaded(engine_future)
        if loader is not None:
            loader.shutdown(wait=False)
    shutil.rmtree(work, ignore_errors=True)
    if not options.keep_cache:
        cache.clear()
    progress(NarrationProgress(total, total, 0.0, tr("narr.done"), "done"))
    log.info("narration done: %d chunks (%d cached), %d chapters, %.0f s of audio, %.0f s total, stages %s, chunk check %s",
             total, counts["cached"], len(chapter_audio), sum(c.duration for c in chapter_audio), time.monotonic() - t_job,
             {k: round(v, 1) for k, v in stage_s.items()}, checker.stats if checker is not None else "off")
    return NarrationResult(job_dir, result.files, len(chapter_audio), total, counts["cached"],
                           sum(c.duration for c in chapter_audio),
                           chunk_check=dict(checker.stats) if checker is not None else None,
                           speaker_warning=speaker_note)


def _cuda_available() -> bool:
    """True if synthesis will most likely run on a CUDA GPU (sizes the CPU pool; see :func:`core.cpu_budget.cuda_likely`)."""
    return cpu_budget.cuda_likely()
