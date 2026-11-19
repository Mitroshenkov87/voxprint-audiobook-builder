"""Re-voice, text path: turn recorded or imported speech into editable text, then narrate it with one of your voices.

The audio is decoded by ffmpeg (:func:`core.audio_utils.load_audio`: MP3, WAV, M4B, M4A, FLAC, Opus ...), cut at pauses into
pieces the recogniser handles well (:func:`core.asr.split_at_pauses`), recognised by the installed Qwen3-ASR model, and written
as a plain-text book: one chapter per file (``# <file name>`` headings, which :mod:`core.book_parsers` reads as chapters).
The user edits that text before it goes to the normal "Narrate a book" flow. Direct conversion (no text step) is
:mod:`core.voice_convert`.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

from core import asr as asr_mod
from core import audio_utils as au
from core.events import CancelToken

log = logging.getLogger("voxprint.revoice")

#: Pieces of at most this length go to the recogniser (Qwen3-ASR is reliable well past this; pauses keep words whole).
PIECE_S = 25.0
#: Audio files offered by the import dialog.
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4b", ".m4a", ".flac", ".ogg", ".opus", ".aac")


def chapter_title(path: Path) -> str:
    """``01 - The beginning.mp3`` -> ``The beginning`` (leading track numbers dropped, the stem otherwise kept)."""
    t = re.sub(r"^\s*\d+\s*[-._ )]\s*", "", Path(path).stem).replace("_", " ").strip()
    return t or Path(path).stem


def transcribe_file(path: Path, asr: asr_mod.BaseASR, language: Optional[str] = None,
                    progress: Optional[Callable[[float], None]] = None, cancel: Optional[CancelToken] = None) -> str:
    """The recognised text of one audio file (the pieces joined with spaces; narration pauses come from the punctuation)."""
    cancel = cancel or CancelToken()
    x, sr = au.load_audio(path, asr_mod.ASR_SR)
    pieces = asr_mod.split_at_pauses(x, sr, max_s=PIECE_S)
    texts: List[str] = []
    for i, (a, b) in enumerate(pieces):
        cancel.check()
        if au.voiced_seconds(x[a:b], sr) >= 0.3:                # silence only: nothing to recognise
            t = asr.transcribe(x[a:b], sr, language).text.strip()
            if t:
                texts.append(t)
        if progress is not None:
            progress((i + 1) / len(pieces))
    return " ".join(texts)


def transcribe_files(files: Sequence[Path], asr: asr_mod.BaseASR, language: Optional[str] = None,
                     progress: Optional[Callable[[float, str], None]] = None,
                     cancel: Optional[CancelToken] = None) -> List[Tuple[str, str]]:
    """``[(chapter title, text)]``, one entry per file in the given order."""
    out: List[Tuple[str, str]] = []
    n = max(1, len(files))
    for k, f in enumerate(files):
        title = chapter_title(Path(f))
        if progress is None:
            on_piece = None
        else:
            def on_piece(fr: float, done: int = k, name: str = title,
                         cb: Callable[[float, str], None] = progress) -> None:
                cb((done + fr) / n, name)
        text = transcribe_file(Path(f), asr, language, on_piece, cancel)
        log.info("re-voice: %s -> %d characters", Path(f).name, len(text))
        out.append((title, text))
    return out


def to_text(chapters: Sequence[Tuple[str, str]]) -> str:
    """The editable text: ``# title`` then the chapter text, chapters separated by a blank line."""
    return "\n\n".join(f"# {t}\n\n{body.strip()}" for t, body in chapters) + "\n"


def save_text(text: str, folder: Path, name: str) -> Path:
    """Write the (edited) text as ``<folder>/<name>.txt`` for the narrator; returns the path."""
    from core.audiobook_export import safe_filename

    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{safe_filename(name, 80, fallback='re-voice')}.txt"
    p.write_text(text, encoding="utf-8")
    return p


OPUS_KBPS = 48                     # speech, mono: transparent for recognition, ~20x smaller than 16-bit WAV


def to_opus(src: Path, ffmpeg: Optional[str], run: Optional[Callable[..., Any]] = None) -> Path:
    """Store a dictaphone recording as Ogg Opus (``<name>.opus``) and delete the lossless original; the models always get
    PCM decoded in memory.  Returns the file to use - the original when it already is Opus, ffmpeg is missing or fails."""
    import subprocess

    src = Path(src)
    if src.suffix.lower() in (".opus", ".ogg") or not ffmpeg:
        return src
    dst = src.with_suffix(".opus")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-c:a", "libopus", "-b:a", f"{OPUS_KBPS}k",
           "-ac", "1", "-application", "voip", str(dst)]
    try:
        r = (run or subprocess.run)(cmd, capture_output=True, timeout=600,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if getattr(r, "returncode", 1) == 0 and dst.is_file() and dst.stat().st_size > 0:
            src.unlink(missing_ok=True)
            return dst
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("recording kept as %s (Opus conversion failed: %s)", src.suffix, exc)
    dst.unlink(missing_ok=True)
    return src
