"""``Voxprint.exe --selftest-narrate [voice-id]``: narrates two short sentences with the first (or the named) voice of the library and
writes ``<logs>/selftest_narrate.txt``.  Checks the installed program end to end (models, engine, ffmpeg, export) without the GUI.
Exit code 0 = an audio file was produced, 1 = failure, 2 = there is no voice in the library yet."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

SENTENCES = {
    "russian": "Это проверка установленной программы. Голос читает короткий текст.",
    "english": "This is a check of the installed program. The voice reads a short text.",
    "german": "Dies ist eine Prüfung des installierten Programms. Die Stimme liest einen kurzen Text.",
}


def run(voice_id: str = "", out_dir: Optional[Path] = None, library=None, narrate: Optional[Callable] = None) -> int:
    from core import audiobook_export as ex
    from core import narration as nr
    from core.book_parsers import Book, Chapter
    from core.events import CancelToken
    from infra import paths

    log = Path(paths.logs_dir()) / "selftest_narrate.txt"
    out_dir = Path(out_dir or Path(paths.logs_dir()) / "selftest_narrate")

    def write(text: str) -> None:
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(text + "\n", encoding="utf-8")
        except OSError:
            pass
        print(text)

    try:
        if library is None:
            from core.voice_library import VoiceLibrary

            library = VoiceLibrary()
        voices = list(library.list_voices())
        voice = next((v for v in voices if v.id == voice_id), None) if voice_id else (voices[0] if voices else None)
        if voice is None:
            write("NO_VOICE: the library is empty" if not voices else f"NO_VOICE: {voice_id} not found")
            return 2
        if narrate is None:
            from workers.narration_runner import NarrationJob, run_narration

            def narrate(job, progress, cancel, pause):   # noqa: ANN001
                return run_narration(job, progress, cancel, pause)
        else:
            from workers.narration_runner import NarrationJob
        lang = (voice.language or "english").lower()
        book = Book("Selftest", "Voxprint", {"russian": "ru", "german": "de"}.get(lang, "en"),
                    [Chapter("Selftest", SENTENCES.get(lang, SENTENCES["english"]))])
        job = NarrationJob(book, voice, out_dir, nr.NarrationOptions(formats={ex.FORMAT_MP3_CHAPTERS}))
        t0 = time.time()
        res = narrate(job, lambda p: None, CancelToken(), nr.PauseToken())
        files = [str(f) for f in getattr(res, "files", [])]
        ok = bool(files) and all(Path(f).is_file() for f in files)
        write(f"{'OK' if ok else 'FAIL'}: voice={voice.id} audio={getattr(res, 'seconds', 0):.1f}s wall={time.time() - t0:.1f}s files={files}")
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001 - a self test reports, it never crashes the caller
        write(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
