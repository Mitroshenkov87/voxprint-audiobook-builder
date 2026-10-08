"""Qt-free glue for the "Narrate a book" scenario: wires the real TTS engine, ffmpeg and feature flags into ``narrate_book``.

The UI worker calls :func:`run_narration`; tests replace it with a fake runner (or call ``core.narration.narrate_book``
directly with a fake engine), so nothing here needs a GPU in the test-suite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from core import narration as nr
from core import languages, tts_engine
from core.audio_utils import ensure_ffmpeg
from core.book_parsers import Book
from core.events import CancelToken
from core.voice_library import VoiceRecord
from infra import features


@dataclass
class NarrationJob:
    """Everything a narration run needs, as chosen in the UI."""
    book: Book
    voice: VoiceRecord
    out_dir: Path
    options: nr.NarrationOptions = field(default_factory=nr.NarrationOptions)
    chapters: Sequence[int] = ()          # empty = all chapters
    on_plan: Optional[Callable[[list], None]] = None     # receives the ordered chunk files (live player)


def run_narration(job: NarrationJob, progress: Callable[[nr.NarrationProgress], None], cancel: CancelToken,
                  pause: nr.PauseToken) -> nr.NarrationResult:
    """Narrate ``job.book`` with ``job.voice`` (real engine).  Raises :class:`DatasetMakerError` subclasses on failure."""
    job.options.allow_aac = features.aac_enabled()
    tp = job.options.translate
    target = tp.target if tp is not None and tp.enabled else ""
    # The token tells Qwen which language the TEXT is in (not the voice's own language): a Russian voice reading an
    # English book must get "English".  Decided once per book (core.languages.narration_language).
    language = languages.narration_language(job.book, job.voice.language, target)
    factory = tts_engine.make_engine_factory(job.voice, language)
    checker = None
    if job.options.check_chunks:      # per-chunk ASR check + regeneration (core/chunk_check.py); None if no recogniser installed
        from core import chunk_check

        checker = chunk_check.make_default_checker(
            language, chunk_check.ChunkCheckOptions(job.options.check_max_cer, job.options.check_retries))
    # narrate_book uses its ``language`` for text preparation and the AI disclosure; "" there means "the book's own tag"
    return nr.narrate_book(job.book, factory, tts_engine.engine_tag(job.voice, language), job.out_dir,
                           language="" if language == languages.AUTO else language,
                           narrator=job.voice.name, options=job.options, progress=progress, cancel=cancel,
                           pause=pause, ffmpeg=ensure_ffmpeg(), chapters=job.chapters, on_plan=job.on_plan,
                           checker=checker)


def default_output_dir() -> Path:
    """Default parent folder of audiobooks: ``~/Documents/Voxprint/Audiobooks``."""
    from infra import paths

    return paths.default_results_dir() / "Audiobooks"


def format_eta(seconds: Optional[float]) -> str:
    """Human-readable duration for the ETA label: ``"1 h 05 min"``, ``"7 min"``, ``"40 s"`` (empty if unknown)."""
    if seconds is None or seconds < 0:
        return ""
    s = int(round(seconds))
    if s >= 3600:
        return f"{s // 3600} h {(s % 3600) // 60:02d} min"
    if s >= 90:
        return f"{round(s / 60)} min"
    return f"{max(s, 1)} s"
