"""Audiobook packaging: chapter files, chapter metadata, playlists and the ffmpeg command lines.

Input is a list of lossless per-chapter WAV files (:class:`ChapterAudio`) written by :mod:`core.narration`; output is any
combination of the formats below.  Everything that is *decided* here (file names, ``ffmetadata`` text, playlist text,
ffmpeg arguments) is a pure function so it can be unit-tested; only :func:`export_formats` runs a process, through an
injectable ``run`` callable.

======================  ===================================================================================
format key              result
======================  ===================================================================================
``opus_single``         one ``<book>.opus`` (Ogg Opus) with ``CHAPTERxxx`` Vorbis-comment chapter markers - open,
                        small, plays in VLC, Audiobookshelf, Voice, Smart AudioBook Player (**default**)
``mp3_chapters``        folder ``<book> - MP3/`` with numbered per-chapter MP3 (ID3v2.3: title, album, artist,
                        track n/N, cover) and an ``.m3u8`` playlist - the most compatible choice, no AAC
``m4b``                 one ``<book>.m4b`` (AAC-LC in an MP4/iPod container) with chapter markers, tags and cover -
                        for Apple Books.  AAC is patent-encumbered, therefore opt-in only
``m4b_opus``            ``<book> (Opus).m4b``: Opus audio in an MP4 container with chapters (few players)
``opus_chapters``       folder with per-chapter ``.opus`` + playlist
``mp3_single``          one ``<book>.mp3`` with ID3 ``CHAP``/``CTOC`` chapter frames
``flac_chapters``       folder with per-chapter FLAC (lossless archive)
``wav_chapters``        folder with per-chapter WAV (lossless archive)
======================  ===================================================================================

Only encoders available in an LGPL ffmpeg build are used: native ``aac``, ``libmp3lame`` (LGPL), ``libopus``,
``flac`` (Opus is the default because it is open and royalty-free).  :func:`required_encoders` / :func:`missing_encoders` let the caller check this before hours of synthesis.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from core.errors import NarrationError
from core.i18n import tr

log = logging.getLogger("voxprint.export")

FORMAT_M4B = "m4b"
FORMAT_M4B_OPUS = "m4b_opus"
FORMAT_MP3_CHAPTERS = "mp3_chapters"
FORMAT_MP3_SINGLE = "mp3_single"
FORMAT_OPUS_SINGLE = "opus_single"
FORMAT_OPUS_CHAPTERS = "opus_chapters"
FORMAT_FLAC_CHAPTERS = "flac_chapters"
FORMAT_WAV_CHAPTERS = "wav_chapters"
ALL_FORMATS: Tuple[str, ...] = (FORMAT_OPUS_SINGLE, FORMAT_MP3_CHAPTERS, FORMAT_M4B, FORMAT_M4B_OPUS, FORMAT_OPUS_CHAPTERS,
                                FORMAT_MP3_SINGLE, FORMAT_FLAC_CHAPTERS, FORMAT_WAV_CHAPTERS)
#: One open, small file with chapter markers.  AAC (M4B) is opt-in: it is patent-encumbered, so it is never the default.
DEFAULT_FORMATS: Tuple[str, ...] = (FORMAT_OPUS_SINGLE,)

#: ffmpeg encoder needed by each format (``None`` = no ffmpeg needed).
ENCODER_FOR_FORMAT: Dict[str, Optional[str]] = {
    FORMAT_M4B: "aac", FORMAT_M4B_OPUS: "libopus", FORMAT_MP3_CHAPTERS: "libmp3lame", FORMAT_MP3_SINGLE: "libmp3lame",
    FORMAT_OPUS_SINGLE: "libopus", FORMAT_OPUS_CHAPTERS: "libopus", FORMAT_FLAC_CHAPTERS: "flac",
    FORMAT_WAV_CHAPTERS: None}

Run = Callable[[List[str]], Tuple[int, str]]
Progress = Callable[[float, str], None]
#: ``submit(fn) -> Future``: runs ``fn`` on a worker pool (``ThreadPoolExecutor.submit``); ``None`` = run inline.
Submit = Callable[[Callable[[], object]], Future]


@dataclass
class Bitrates:
    """Target bitrates in kbit/s; the defaults suit mono speech (clean and small)."""
    aac_kbps: int = 64
    mp3_kbps: int = 96
    opus_kbps: int = 32

    def clamp(self) -> "Bitrates":
        """Return a copy limited to sane ranges (an out-of-range UI value cannot break ffmpeg)."""
        c = lambda v, lo, hi: max(lo, min(hi, int(v)))  # noqa: E731
        return Bitrates(c(self.aac_kbps, 24, 256), c(self.mp3_kbps, 32, 320), c(self.opus_kbps, 12, 128))


#: Quality presets shown in the UI (typical bitrates for mono speech); exact values live under "Advanced".
PRESET_COMPACT, PRESET_STANDARD, PRESET_HIGH = "compact", "standard", "high"
QUALITY_PRESETS = {
    PRESET_COMPACT: Bitrates(aac_kbps=48, mp3_kbps=64, opus_kbps=24),
    PRESET_STANDARD: Bitrates(aac_kbps=64, mp3_kbps=96, opus_kbps=32),
    PRESET_HIGH: Bitrates(aac_kbps=96, mp3_kbps=128, opus_kbps=48),
}
DEFAULT_PRESET = PRESET_STANDARD


def preset_for(bitrates: "Bitrates") -> str:
    """Name of the preset equal to ``bitrates`` or ``""`` (custom values)."""
    for name, b in QUALITY_PRESETS.items():
        if (b.aac_kbps, b.mp3_kbps, b.opus_kbps) == (bitrates.aac_kbps, bitrates.mp3_kbps, bitrates.opus_kbps):
            return name
    return ""


def megabytes_per_hour(kbps: int) -> float:
    """Size of one hour of audio at ``kbps`` (decimal megabytes)."""
    return kbps * 1000 / 8 * 3600 / 1e6


@dataclass
class ChapterAudio:
    """A finished chapter: lossless WAV on disk plus its title and exact duration."""
    index: int                  # 0-based
    title: str
    wav: Path
    duration: float             # seconds


@dataclass
class BookMeta:
    """Tags written into the audio files."""
    title: str
    author: str = ""
    narrator: str = ""          # the voice name
    language: str = ""
    cover: Optional[Path] = None


@dataclass
class ExportResult:
    """Files produced by :func:`export_formats` (``playlists`` are included in ``files`` too)."""
    files: List[Path] = field(default_factory=list)
    folders: List[Path] = field(default_factory=list)


# --------------------------------------------------------------------------- names

_SEPARATORS = re.compile(r'[:/\\|]')
_ILLEGAL = re.compile(r'[<>"?*\x00-\x1f]')


def safe_filename(text: str, limit: int = 80, fallback: str = "untitled") -> str:
    """A file/folder name that is legal on Windows: ``: / \\ |`` become spaces, other illegal characters are dropped, spaces collapsed, no trailing dots."""
    s = _ILLEGAL.sub("", _SEPARATORS.sub(" ", text or ""))
    s = re.sub(r"\s+", " ", s).strip(" .")
    s = s[:limit].strip(" .")
    if re.fullmatch(r"(?i)(con|prn|aux|nul|com\d|lpt\d)(\..*)?", s or ""):
        s = f"_{s}"
    return s or fallback


def chapter_filename(index: int, total: int, title: str, ext: str) -> str:
    """``"03 - Title.mp3"``: 1-based number zero-padded to the width of ``total`` (at least 2 digits)."""
    width = max(2, len(str(total)))
    name = safe_filename(title, 80, fallback=tr("book.chapter_n", n=index + 1))
    return f"{index + 1:0{width}d} - {name}.{ext.lstrip('.')}"


def book_basename(meta: BookMeta) -> str:
    """Base name of single-file outputs: ``"Author - Title"`` or just the title."""
    base = f"{meta.author} - {meta.title}" if meta.author.strip() else meta.title
    return safe_filename(base, 120, fallback="audiobook")


# --------------------------------------------------------------------------- metadata text

def escape_ffmeta(value: str) -> str:
    """Escape a value for an ``ffmetadata`` file (``=``, ``;``, ``#``, ``\\`` and newlines)."""
    out = []
    for ch in value:
        if ch in "=;#\\":
            out.append("\\" + ch)
        elif ch == "\n":
            out.append("\\\n")
        elif ch == "\r":
            continue
        else:
            out.append(ch)
    return "".join(out)


def chapter_timings(chapters: Sequence[ChapterAudio]) -> List[Tuple[int, int, str]]:
    """``[(start_ms, end_ms, title)]`` - contiguous, in order; each chapter starts where the previous one ends."""
    out: List[Tuple[int, int, str]] = []
    t = 0.0
    for c in chapters:
        start = int(round(t * 1000))
        t += max(0.0, c.duration)
        out.append((start, max(start + 1, int(round(t * 1000))), c.title))
    return out


def common_tags(meta: BookMeta) -> Dict[str, str]:
    """Global tags: title/album = book, artist = author (else narrator), composer = narrator, genre = Audiobook."""
    artist = meta.author.strip() or meta.narrator.strip()
    tags = {"title": meta.title, "album": meta.title, "genre": "Audiobook"}
    if artist:
        tags["artist"] = artist
        tags["album_artist"] = artist
    if meta.narrator.strip():
        tags["composer"] = meta.narrator.strip()
        tags["comment"] = tr("export.comment_narrated", voice=meta.narrator.strip())
    if meta.language.strip():
        tags["language"] = meta.language.strip()
    return tags


def build_ffmetadata(meta: BookMeta, chapters: Sequence[ChapterAudio]) -> str:
    """The ``ffmetadata`` document: global tags and one ``[CHAPTER]`` block per chapter (millisecond timebase)."""
    lines = [";FFMETADATA1"]
    lines += [f"{k}={escape_ffmeta(v)}" for k, v in common_tags(meta).items()]
    for start, end, title in chapter_timings(chapters):
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={end}", f"title={escape_ffmeta(title)}"]
    return "\n".join(lines) + "\n"


def build_m3u8(entries: Iterable[Tuple[str, float, str]]) -> str:
    """Extended M3U playlist (UTF-8): ``entries`` are ``(file name, duration seconds, title)``."""
    lines = ["#EXTM3U"]
    for name, dur, title in entries:
        lines += [f"#EXTINF:{int(round(dur))},{title.replace(chr(10), ' ')}", name]
    return "\n".join(lines) + "\n"


def concat_list(wavs: Sequence[Path]) -> str:
    """Text of an ffmpeg concat-demuxer list file."""
    def q(p: Path) -> str:
        return str(p).replace("\\", "/").replace("'", "'\\''")
    return "".join(f"file '{q(w)}'\n" for w in wavs)


# --------------------------------------------------------------------------- ffmpeg command lines

_BASE = ["-y", "-hide_banner", "-loglevel", "error"]


def _tag_args(tags: Dict[str, str]) -> List[str]:
    """``-metadata key=value`` pairs."""
    out: List[str] = []
    for k, v in tags.items():
        if v:
            out += ["-metadata", f"{k}={v}"]
    return out


def chapter_tags(meta: BookMeta, chapter: ChapterAudio, total: int) -> Dict[str, str]:
    """Tags of one per-chapter file: title = chapter, album = book, track = n/N, plus the common tags."""
    tags = dict(common_tags(meta))
    tags.update({"title": chapter.title, "track": f"{chapter.index + 1}/{total}"})
    return tags


def cmd_chapter(ffmpeg: str, fmt: str, chapter: ChapterAudio, total: int, meta: BookMeta, out: Path,
                rates: Bitrates) -> List[str]:
    """ffmpeg arguments for one per-chapter file of ``fmt`` (``mp3_chapters`` / ``opus_chapters`` / ``flac_chapters``)."""
    cmd = [ffmpeg, *_BASE, "-i", str(chapter.wav)]
    cover = meta.cover if (fmt == FORMAT_MP3_CHAPTERS and meta.cover) else None
    if cover:
        cmd += ["-i", str(cover), "-map", "0:a", "-map", "1:v", "-c:v", "copy",
                "-metadata:s:v", "title=Album cover", "-metadata:s:v", "comment=Cover (front)"]
    else:
        cmd += ["-map", "0:a"]
    if fmt == FORMAT_MP3_CHAPTERS:
        cmd += ["-c:a", "libmp3lame", "-b:a", f"{rates.mp3_kbps}k", "-ac", "1", "-id3v2_version", "3"]
    elif fmt == FORMAT_OPUS_CHAPTERS:
        cmd += ["-c:a", "libopus", "-b:a", f"{rates.opus_kbps}k", "-ac", "1", "-application", "voip", "-f", "opus"]
    elif fmt == FORMAT_FLAC_CHAPTERS:
        cmd += ["-c:a", "flac"]
    else:
        raise ValueError(fmt)
    cmd += _tag_args(chapter_tags(meta, chapter, total)) + [str(out)]
    return cmd


def cmd_single(ffmpeg: str, fmt: str, list_file: Path, meta_file: Path, meta: BookMeta, out: Path,
               rates: Bitrates) -> List[str]:
    """ffmpeg arguments for one file with chapter markers (``opus_single`` / ``m4b`` / ``m4b_opus`` / ``mp3_single``).

    Inputs: 0 = concat list of the chapter WAVs, 1 = the ffmetadata file (tags + chapters), 2 = cover (m4b / mp3 only).
    """
    cmd = [ffmpeg, *_BASE, "-f", "concat", "-safe", "0", "-i", str(list_file), "-i", str(meta_file)]
    cover = meta.cover if fmt in (FORMAT_M4B, FORMAT_M4B_OPUS, FORMAT_MP3_SINGLE) else None
    if cover:
        cmd += ["-i", str(cover), "-map", "0:a", "-map", "2:v"]
    else:
        cmd += ["-map", "0:a"]
    cmd += ["-map_metadata", "1", "-map_chapters", "1"]
    if fmt == FORMAT_M4B_OPUS:
        cmd += ["-c:a", "libopus", "-b:a", f"{rates.opus_kbps}k", "-ac", "1", "-application", "voip",
                *(["-c:v", "copy", "-disposition:v:0", "attached_pic"] if cover else []),
                "-movflags", "+faststart", "-f", "mp4"]
    elif fmt == FORMAT_M4B:
        cmd += ["-c:a", "aac", "-b:a", f"{rates.aac_kbps}k", "-ac", "1"]
        if cover:
            cmd += ["-c:v", "copy", "-disposition:v:0", "attached_pic"]
        cmd += ["-movflags", "+faststart", "-f", "ipod"]
    elif fmt == FORMAT_MP3_SINGLE:
        cmd += ["-c:a", "libmp3lame", "-b:a", f"{rates.mp3_kbps}k", "-ac", "1", "-id3v2_version", "3"]
        if cover:
            cmd += ["-c:v", "copy", "-metadata:s:v", "title=Album cover", "-metadata:s:v", "comment=Cover (front)"]
    elif fmt == FORMAT_OPUS_SINGLE:
        cmd += ["-c:a", "libopus", "-b:a", f"{rates.opus_kbps}k", "-ac", "1", "-application", "voip", "-f", "opus"]
    else:
        raise ValueError(fmt)
    cmd.append(str(out))
    return cmd


def required_encoders(formats: Iterable[str]) -> List[str]:
    """ffmpeg encoders needed by ``formats`` (sorted, unique)."""
    return sorted({e for f in formats if (e := ENCODER_FOR_FORMAT.get(f))})


def missing_encoders(ffmpeg: str, formats: Iterable[str], run: Optional[Run] = None) -> List[str]:
    """Encoders required by ``formats`` that this ffmpeg build does not offer (``[]`` = all good)."""
    need = required_encoders(formats)
    if not need:
        return []
    rc, out = (run or default_run)([ffmpeg, "-hide_banner", "-encoders"])
    if rc != 0:
        return need
    return [e for e in need if not re.search(rf"\b{re.escape(e)}\b", out)]


def default_run(cmd: List[str]) -> Tuple[int, str]:
    """Run a command, return ``(exit code, stdout+stderr text)``; no console window on Windows.

    The process runs at below-normal priority: several encoders run in parallel during narration and must not make the
    UI stutter (the priority does not change a single output byte)."""
    flags = 0
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",  # noqa: S603
                         creationflags=flags)
    if sys.platform != "win32":
        try:                                   # renice right after the start (preexec_fn is not safe with threads)
            os.setpriority(os.PRIO_PROCESS, p.pid, 10)
        except (OSError, AttributeError):
            pass
    out, _ = p.communicate()
    return p.returncode, out or ""


# --------------------------------------------------------------------------- the export

#: An output file another program holds open (a player with last run's MP3, an antivirus scan) is waited for this many
#: times, the pause doubling from ``IN_USE_FIRST_PAUSE`` seconds (about 6 s in total), before the job stops with a clear
#: "close the file" message instead of ffmpeg's bare failure.
IN_USE_RETRIES = 5
IN_USE_FIRST_PAUSE = 0.2


def file_in_use(path: Path) -> bool:
    """True if ``path`` exists but cannot be opened for writing (Windows: another process holds it, WinError 32 / 5)."""
    try:
        if not path.is_file():
            return False
        with open(path, "r+b"):
            return False
    except PermissionError:
        return True
    except OSError:
        return False


def _run_checked(run: Run, cmd: List[str], out: Path, sleep: Optional[Callable[[float], None]] = None) -> None:
    """Run ffmpeg and raise :class:`NarrationError` (with the stderr tail as details) if it fails.

    While the output file is held by another program the encode is retried with backoff; if it stays locked the error
    names the file and says to close it (``err.narration_file_in_use``)."""
    import time

    sleep = sleep or time.sleep
    delay = IN_USE_FIRST_PAUSE
    for attempt in range(IN_USE_RETRIES + 1):
        last = attempt >= IN_USE_RETRIES
        if file_in_use(out) and not last:
            log.info("%s is in use by another program - retrying in %.1f s", out.name, delay)
            sleep(delay)
            delay *= 2
            continue
        t0 = time.monotonic()
        rc, text = run(cmd)
        log.info("encode %s: %.1f s (exit %s)", out.name, time.monotonic() - t0, rc)
        if rc == 0 and out.exists():
            return
        if file_in_use(out):
            if last:
                raise NarrationError(tr("err.narration_file_in_use", name=out.name), details=text[-1500:])
            log.info("%s is in use by another program - retrying in %.1f s", out.name, delay)
            sleep(delay)
            delay *= 2
            continue
        raise NarrationError(tr("err.narration_export", name=out.name), details=text[-1500:])


SINGLE_FORMATS = (FORMAT_M4B, FORMAT_M4B_OPUS, FORMAT_MP3_SINGLE, FORMAT_OPUS_SINGLE)
_CHAPTER_EXT = {FORMAT_MP3_CHAPTERS: ("mp3", "MP3"), FORMAT_OPUS_CHAPTERS: ("opus", "Opus"),
                FORMAT_FLAC_CHAPTERS: ("flac", "FLAC"), FORMAT_WAV_CHAPTERS: ("wav", "WAV")}


class Exporter:
    """Writes the requested formats; per-chapter files can be encoded as soon as a chapter exists.

    :meth:`add_chapter` starts the per-chapter encodes of one finished chapter (on ``submit``'s pool, or inline), so the
    narrator can encode chapter 1 while the GPU still speaks chapter 5.  :meth:`finish` waits for them, writes the
    playlists, encodes the single-file formats (in parallel with each other when there is a pool: each is one ffmpeg
    process with its own output) and returns the files in the same fixed order as a serial export.  Every ffmpeg call
    is exactly the one a serial export would make, so the files are byte-identical - only *when* they run changes.
    """

    def __init__(self, ffmpeg: Optional[str], formats: Iterable[str], meta: BookMeta, out_dir: Path, total: int,
                 rates: Optional[Bitrates] = None, run: Optional[Run] = None, work_dir: Optional[Path] = None,
                 submit: Optional[Submit] = None) -> None:
        """``total``: the number of chapters of the finished book (part of every per-chapter tag)."""
        self.ffmpeg, self.meta, self.total = ffmpeg, meta, total
        self.run = run or default_run
        self.rates = (rates or Bitrates()).clamp()
        self.wanted = [f for f in ALL_FORMATS if f in set(formats)]
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.work = Path(work_dir) if work_dir else self.out_dir / ".export"
        if required_encoders(self.wanted) and not ffmpeg:
            raise NarrationError(tr("err.narration_no_ffmpeg"))
        self.base = book_basename(meta)
        self.submit = submit
        self._jobs: Dict[Tuple[str, int], Tuple[Path, Future]] = {}
        self._lock = threading.Lock()

    def _start(self, fn: Callable[[], object]) -> Future:
        if self.submit is not None:
            return self.submit(fn)
        f: Future = Future()
        try:
            f.set_result(fn())
        except BaseException as exc:  # noqa: BLE001 - stored and re-raised by finish(), like a pool would
            f.set_exception(exc)
        return f

    def _folder(self, fmt: str) -> Path:
        return self.out_dir / f"{self.base} - {_CHAPTER_EXT[fmt][1]}"

    def add_chapter(self, ch: ChapterAudio) -> None:
        """Start the per-chapter files of ``ch`` (no-op for formats without per-chapter files, or if already started)."""
        for fmt in self.wanted:
            if fmt in SINGLE_FORMATS:
                continue
            with self._lock:
                if (fmt, ch.index) in self._jobs:
                    continue
                folder = self._folder(fmt)
                folder.mkdir(parents=True, exist_ok=True)
                out = folder / chapter_filename(ch.index, self.total, ch.title, _CHAPTER_EXT[fmt][0])
                if fmt == FORMAT_WAV_CHAPTERS:
                    job = (lambda src=ch.wav, dst=out: shutil.copyfile(src, dst))
                else:
                    cmd = cmd_chapter(self.ffmpeg or "", fmt, ch, self.total, self.meta, out, self.rates)
                    job = (lambda cmd=cmd, out=out: _run_checked(self.run, cmd, out))
                self._jobs[(fmt, ch.index)] = (out, self._start(job))

    def finish(self, chapters: Sequence[ChapterAudio], progress: Optional[Progress] = None) -> ExportResult:
        """Everything that needs the whole book, then the result in the fixed order (formats, then chapters)."""
        for ch in chapters:
            self.add_chapter(ch)
        self.work.mkdir(parents=True, exist_ok=True)
        meta_file, list_file = self.work / "chapters.ffmetadata", self.work / "chapters.txt"
        meta_file.write_text(build_ffmetadata(self.meta, chapters), encoding="utf-8")
        list_file.write_text(concat_list([c.wav for c in chapters]), encoding="utf-8")
        singles: Dict[str, Tuple[Path, Future]] = {}
        for fmt in self.wanted:
            if fmt not in SINGLE_FORMATS:
                continue
            ext = {FORMAT_M4B: "m4b", FORMAT_M4B_OPUS: "m4b", FORMAT_MP3_SINGLE: "mp3", FORMAT_OPUS_SINGLE: "opus"}[fmt]
            out = self.out_dir / (f"{self.base} (Opus).{ext}" if fmt == FORMAT_M4B_OPUS else f"{self.base}.{ext}")
            cmd = cmd_single(self.ffmpeg or "", fmt, list_file, meta_file, self.meta, out, self.rates)

            def _encode(cmd: List[str] = cmd, out: Path = out) -> None:
                _run_checked(self.run, cmd, out)

            singles[fmt] = (out, self._start(_encode))
        result = ExportResult()
        for i, fmt in enumerate(self.wanted):
            if progress:
                progress(i / max(1, len(self.wanted)), fmt)
            if fmt in singles:
                out, fut = singles[fmt]
                fut.result()                         # raises the encoder error of this format
                result.files.append(out)
                continue
            folder = self._folder(fmt)
            folder.mkdir(parents=True, exist_ok=True)
            entries: List[Tuple[str, float, str]] = []
            for ch in chapters:
                out, fut = self._jobs[(fmt, ch.index)]
                fut.result()
                result.files.append(out)
                entries.append((out.name, ch.duration, ch.title))
            playlist = folder / f"{self.base}.m3u8"
            playlist.write_text(build_m3u8(entries), encoding="utf-8")
            result.files.append(playlist)
            result.folders.append(folder)
        if progress:
            progress(1.0, "")
        return result

    def pending(self) -> List[Future]:
        """Futures of the per-chapter jobs started so far (the narrator checks them for early errors)."""
        with self._lock:
            return [f for _, f in self._jobs.values()]


def export_formats(ffmpeg: Optional[str], formats: Iterable[str], chapters: Sequence[ChapterAudio], meta: BookMeta,
                   out_dir: Path, rates: Optional[Bitrates] = None, run: Optional[Run] = None,
                   progress: Optional[Progress] = None, work_dir: Optional[Path] = None,
                   submit: Optional[Submit] = None) -> ExportResult:
    """Write every requested format into ``out_dir`` and return what was produced (see :class:`Exporter`).

    ``ffmpeg`` may be ``None`` only when nothing but ``wav_chapters`` is requested.  Chapter files go to sub-folders
    named after the book and format; single files go directly into ``out_dir``.
    """
    exp = Exporter(ffmpeg, formats, meta, out_dir, len(chapters), rates, run, work_dir, submit)
    return exp.finish(chapters, progress)
