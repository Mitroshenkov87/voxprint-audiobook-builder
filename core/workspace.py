"""The book working folder: everything one narration job needs lives in ``<working folder>/<book title>/``.

Layout of a job folder (made by :func:`core.narration.narrate_book`)::

    <book>.fb2 / .txt / .epub   the book file - only if the user agreed to copy/move it here (:func:`place_book`)
    *.m4b, *.mp3, "<book> - MP3/"  the finished audiobook (the result)
    translation_<lang>.txt      the readable (editable) translation          } "prepared text"
    .translation/               translation cache + fingerprint (edits reused) }
    .debug/                     prepared_text.txt, prep_report.json          }
    .cache/ .work/ .export/     temporary parts: synthesized chunks, chapter WAVs, ffmpeg lists

The working folder itself is chosen by the user and remembered in ``state/work_folder.txt`` (Qt-free, no AppData clutter:
nothing of a job is written outside its folder).  After a finished job :func:`scan_job` sorts the files into groups and
:func:`clean_job` removes the groups the user did not keep (the UI asks in a dialog).
"""
from __future__ import annotations

import filecmp
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

log = logging.getLogger("voxprint.workspace")

STATE_FILE = "work_folder.txt"
TEMP_DIRS = (".cache", ".work", ".export")                  # never needed after a successful export
PREPARED_DIRS = (".debug", ".translation")

PLACE_COPY, PLACE_MOVE, PLACE_LEAVE = "copy", "move", "leave"


def _state_file() -> Path:
    from infra import paths

    return paths.state_dir() / STATE_FILE


def load_folder(default: Path) -> Path:
    """The remembered working folder, or ``default`` (never chosen, unreadable, or the drive is gone)."""
    try:
        text = _state_file().read_text(encoding="utf-8").strip()
    except OSError:
        return Path(default)
    p = Path(text) if text else None
    if p is None or not p.is_absolute():
        return Path(default)
    if p.is_dir() or (p.parent.is_dir() and not p.exists()):      # an absent leaf is created on the first job
        return p
    return Path(default)


def save_folder(folder: Path) -> None:
    """Remember the working folder (a failure to write is not an error)."""
    try:
        f = _state_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(str(Path(folder)), encoding="utf-8")
    except OSError as exc:
        log.warning("cannot remember the working folder: %s", exc)


def is_inside(path: Path, folder: Path) -> bool:
    """True if ``path`` lies in ``folder`` (paths resolved; case rules of the OS apply via ``resolve``)."""
    try:
        return Path(path).resolve().is_relative_to(Path(folder).resolve())
    except (OSError, ValueError):
        return False


def place_book(src: Path, job_dir: Path, mode: str) -> Path:
    """Copy or move the book file into ``job_dir`` (``mode`` = copy | move | leave) and return where the book is now.

    An identical file already there is reused; a different one with the same name is never overwritten (`` (2)`` ...)."""
    src = Path(src)
    if mode not in (PLACE_COPY, PLACE_MOVE) or is_inside(src, job_dir):
        return src
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    dst, n = job_dir / src.name, 2
    while dst.exists():
        if dst.is_file() and filecmp.cmp(src, dst, shallow=False):
            if mode == PLACE_MOVE:
                src.unlink(missing_ok=True)
            return dst
        dst, n = job_dir / f"{src.stem} ({n}){src.suffix}", n + 1
    if mode == PLACE_MOVE:
        shutil.move(str(src), str(dst))                  # works across drives too (copy + delete)
    else:
        shutil.copy2(src, dst)
    return dst


@dataclass
class JobFiles:
    """The files of a finished job, by what the user may want to keep."""
    job_dir: Path
    results: List[Path] = field(default_factory=list)      # the audiobook files (and their playlists)
    original: List[Path] = field(default_factory=list)     # the book file copied/moved into the folder
    prepared: List[Path] = field(default_factory=list)     # translation / prepared text (files or folders)
    temp: List[Path] = field(default_factory=list)         # temporary folders (always removable)

    @staticmethod
    def size(items: Iterable[Path]) -> int:
        """Bytes used by files and folders."""
        total = 0
        for p in items:
            try:
                if p.is_dir():
                    total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                elif p.is_file():
                    total += p.stat().st_size
            except OSError:
                pass
        return total


def scan_job(job_dir: Path, result_files: Iterable[Path] = (), book_file: Optional[Path] = None) -> JobFiles:
    """Sort the contents of ``job_dir`` into results / original book / prepared text / temporary parts."""
    job_dir = Path(job_dir)
    out = JobFiles(job_dir, results=[Path(f) for f in result_files if Path(f).exists()])
    if book_file is not None and Path(book_file).is_file() and is_inside(book_file, job_dir):
        out.original.append(Path(book_file))
    for name in PREPARED_DIRS:
        if (job_dir / name).is_dir():
            out.prepared.append(job_dir / name)
    out.prepared += sorted(job_dir.glob("translation_*.txt"))
    out.temp = [job_dir / n for n in TEMP_DIRS if (job_dir / n).is_dir()]
    return out


def _remove(p: Path) -> None:
    try:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)
    except OSError as exc:                     # e.g. a player still holds the file on Windows: keep it, say so in the log
        log.warning("cannot remove %s: %s", p, exc)


def clean_job(files: JobFiles, keep_results: bool = True, keep_original: bool = True, keep_prepared: bool = True) -> int:
    """Remove the temporary parts and every group not kept; empty sub-folders left behind go too.  Returns bytes freed."""
    drop = list(files.temp)
    if not keep_results:
        drop += files.results
    if not keep_original:
        drop += files.original
    if not keep_prepared:
        drop += files.prepared
    freed = JobFiles.size(drop)
    for p in drop:
        _remove(p)
    if not keep_results:                       # chapter folders ("<book> - MP3") become empty
        for d in sorted({p.parent for p in files.results}, key=lambda d: len(d.parts), reverse=True):
            if d != files.job_dir and is_inside(d, files.job_dir):
                try:
                    d.rmdir()
                except OSError:
                    pass
    return freed
