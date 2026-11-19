"""``voxprint --selftest-text``: a tiny pipeline check that needs neither a GPU nor a downloaded model.

Steps (each prints OK / FAIL): text preparation (numbers to words), chunking, translation with a stub engine, a WAV written
and converted to MP3 through ffmpeg.  Used by the Linux CI job and by ``docs/LINUX-TEST-CHECKLIST.md``.
Exit code 0 = all steps passed."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, List, Tuple


class _StubEngine:
    tag = "stub-ru-en@1"

    def translate(self, sentences):
        return [f"[en] {s}" for s in sentences]

    def close(self) -> None:
        pass


def _step_prep() -> str:
    from core import text_prep

    out = text_prep.prepare_text_block("В 2024 году было 25 книг.", "ru")
    if "2024" in out or "25" in out:
        raise AssertionError(f"numbers were not spelled out: {out!r}")
    return out


def _step_chunk() -> str:
    from core import chunker

    parts = chunker.chunk_text("Первое предложение. Второе предложение, подлиннее. " * 20, 120)
    if len(parts) < 2 or any(len(p) > 160 for p, _ in parts):
        raise AssertionError(f"unexpected chunks: {len(parts)}")
    return f"{len(parts)} chunks"


def _step_translate() -> str:
    from core import translate as tl
    from core.book_parsers import Book, Chapter

    with tempfile.TemporaryDirectory() as td:
        book = Book("Маяк", "", "ru", [Chapter("Глава", "Старый маяк стоял на скале. Люди жили рядом много лет.")])
        plan = tl.TranslatePlan("en", "ru", lambda s, d: _StubEngine())
        out = tl.translate_book(book, plan, "ru", tl.TranslationCache(Path(td) / "cache"))
    text = out.chapters[0].text
    if not text.startswith("[en]"):
        raise AssertionError(f"stub translation missing: {text!r}")
    return text[:40]


def _step_ffmpeg() -> str:
    import numpy as np
    import soundfile as sf

    from core import audio_utils

    ff = audio_utils.ensure_ffmpeg()
    if not ff:
        raise AssertionError("ffmpeg not found")
    from infra.updater import run_subprocess

    with tempfile.TemporaryDirectory() as td:
        wav, mp3 = Path(td) / "t.wav", Path(td) / "t.mp3"
        t = np.linspace(0, 0.5, 12000, endpoint=False)
        sf.write(str(wav), 0.2 * np.sin(2 * np.pi * 440 * t), 24000)
        res = run_subprocess([ff, "-y", "-loglevel", "error", "-i", str(wav), str(mp3)])
        rc = res[0] if isinstance(res, tuple) else getattr(res, "returncode", 1)
        if rc != 0 or not mp3.is_file() or mp3.stat().st_size < 500:
            raise AssertionError(f"ffmpeg could not encode MP3 (rc={rc})")
    return ff


STEPS: List[Tuple[str, Callable[[], str]]] = [("text preparation", _step_prep), ("chunking", _step_chunk),
                                              ("translation (stub engine)", _step_translate), ("ffmpeg WAV->MP3", _step_ffmpeg)]


def run() -> int:
    bad = 0
    lines = []
    for name, fn in STEPS:
        try:
            lines.append(f"OK    {name}: {fn()}")
        except Exception as exc:  # noqa: BLE001
            bad += 1
            lines.append(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    lines.append("SELFTEST_TEXT " + ("FAILED" if bad else "OK"))
    text = "\n".join(lines)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"))
    except (OSError, ValueError):
        pass
    return 1 if bad else 0
