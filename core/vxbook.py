"""Read a Voxprint Book (``.vxbook``, format 1.x).

The file is a ZIP. The first entry is the stored mimetype ``application/vnd.voxprint.book+zip``. ``manifest.json``
names the chapters and the SHA-256 of every other entry. ``book.md`` is the text. ``speakers.json`` and optional
``cast.json`` are the voices. Optional ``sound.json`` is the soundscape; this module only checks that it is a JSON
object and keeps it. :mod:`core.soundscape` decides whether to play it.

A ``.vxbook`` writes the letter yo itself, so :attr:`core.book_parsers.Book.explicit_yo` is set and dictionary
restoration stays off. A stress mark (U+0301) is left in the spoken text for the normalizer.

Example::

    from core.vxbook import load_vxbook
    book = load_vxbook("story.vxbook")
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import struct
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.book_parsers import Book, Chapter
from core.errors import BookParseError
from core.i18n import tr

log = logging.getLogger("voxprint.vxbook")

MIME = b"application/vnd.voxprint.book+zip"
MAX_ENTRIES = 1000
MAX_TOTAL = 512 * 1024 * 1024
MAX_BOOK = 64 * 1024 * 1024
MAX_JSON = 4 * 1024 * 1024
MAX_COVER = 20 * 1024 * 1024
MAX_NOTE = 8 * 1024 * 1024
MAX_RATIO = 100
MAX_PIXELS = 100_000_000
_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_DIRECTIVE = re.compile(r'^<!-- vx:([a-z][a-z0-9_]*)((?: [a-z][a-z0-9_]*="[^"<>]*")*) -->$')
_ATTR_ONE = re.compile(r'([a-z][a-z0-9_]*)="([^"<>]*)"')
_CONF = re.compile(r"^(?:0(?:\.\d{1,2})?|1(?:\.0{1,2})?)$")


def _fail(detail: str) -> None:
    """Raise a book error. ``detail`` is English and is shown inside the translated message."""
    raise BookParseError(tr("err.vxbook", detail=detail), details=detail)


def _version(text: object) -> Tuple[int, int]:
    """``MAJOR.MINOR``. An unknown major is refused; any minor of major 1 is accepted."""
    if not isinstance(text, str) or not re.fullmatch(r"\d+\.\d+", text):
        _fail("format_version must look like 1.0")
    major_s, minor_s = text.split(".")
    major, minor = int(major_s), int(minor_s)
    if major != 1:
        _fail("made by a newer Plotweaver, please update")
    if major == 0:
        _fail("format_version is not supported")
    return major, minor


def _safe_zip_name(name: str) -> bool:
    """Reject absolute paths, drive letters, backslashes and ``..`` (zip-slip)."""
    if not name or name.endswith("/") or "\x00" in name or "\\" in name or name.startswith("/"):
        return False
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False
    head = parts[0]
    if ":" in head:
        return False
    return True


def _limit_for(name: str) -> int:
    """Uncompressed cap for one entry."""
    lower = name.lower()
    if lower.endswith(".json"):
        return MAX_JSON
    if lower.startswith("notes/"):
        return MAX_NOTE
    if lower.startswith("cover."):
        return MAX_COVER
    return MAX_BOOK


def _read_entry(zf: zipfile.ZipFile, info: zipfile.ZipInfo, total: List[int]) -> bytes:
    """Decompress one entry, enforcing the size cap and the compression ratio while reading."""
    if info.flag_bits & 0x1:
        _fail(f"{info.filename} is encrypted")
    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        _fail(f"{info.filename} uses an unsupported compression method")
    cap = min(_limit_for(info.filename), MAX_TOTAL - total[0])
    if cap < 0:
        _fail("the archive is larger than 512 MiB uncompressed")
    if info.file_size > cap:
        _fail(f"{info.filename} is too large")
    out = bytearray()
    compressed = info.compress_size or 0
    try:
        with zf.open(info, "r") as src:
            while True:
                block = src.read(1024 * 1024)
                if not block:
                    break
                out.extend(block)
                if len(out) > cap or total[0] + len(out) > MAX_TOTAL:
                    _fail(f"{info.filename} is too large")
                if compressed and len(out) > max(compressed, 1) * MAX_RATIO and len(out) > 4096:
                    _fail(f"{info.filename} expands too far for its compressed size")
    except BookParseError:
        raise
    except (zipfile.BadZipFile, OSError, RuntimeError):
        _fail(f"{info.filename} could not be read")
    if compressed and len(out) > max(compressed, 1) * MAX_RATIO and len(out) > 4096:
        _fail(f"{info.filename} expands too far for its compressed size")
    total[0] += len(out)
    return bytes(out)


def _json_bytes(data: bytes, name: str) -> Any:
    """Parse one JSON entry. No BOM, no duplicate keys, nesting at most 32."""
    if data.startswith(b"\xef\xbb\xbf"):
        _fail(f"{name} must not start with a BOM")
    try:
        text = data.decode("utf-8")
    except UnicodeError:
        _fail(f"{name} is not UTF-8")

    def pairs(seq: List[Tuple[str, Any]]) -> Dict[str, Any]:
        obj: Dict[str, Any] = {}
        for key, value in seq:
            if key in obj:
                raise ValueError(f"duplicate key {key}")
            obj[key] = value
        return obj

    try:
        value = json.loads(text, object_pairs_hook=pairs)
    except (json.JSONDecodeError, ValueError):
        _fail(f"{name} is not valid JSON")
    if _depth(value) > 32:
        _fail(f"{name} is nested too deeply")
    return value


def _depth(value: Any, level: int = 1) -> int:
    """Nesting depth of a JSON value. A scalar is 1."""
    if isinstance(value, dict):
        return max((level, *(_depth(v, level + 1) for v in value.values())))
    if isinstance(value, list):
        return max((level, *(_depth(v, level + 1) for v in value)))
    return level


def _jpeg_pixels(data: bytes) -> Optional[int]:
    """Width times height from a JPEG SOF marker, or None when the header cannot be read."""
    i = 2
    n = len(data)
    while i + 3 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            return None
        marker = data[i]
        i += 1
        if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if i + 1 >= n:
            return None
        length = struct.unpack(">H", data[i:i + 2])[0]
        if length < 2 or i + length > n:
            return None
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if length < 7:
                return None
            height, width = struct.unpack(">HH", data[i + 3:i + 7])
            return int(width) * int(height)
        i += length
    return None


def _pixels(data: bytes) -> Optional[int]:
    """Pixel count of a PNG or JPEG, or None for a cover this reader does not decode."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24 and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        return int(width) * int(height)
    if data[:3] == b"\xff\xd8\xff":
        return _jpeg_pixels(data)
    return None


def _check_text(text: str) -> None:
    """Encoding rules for ``book.md`` and notes: NFC, LF, no tabs or invisible characters."""
    if text.startswith("\ufeff"):
        _fail("book.md must not start with a BOM")
    if unicodedata.normalize("NFC", text) != text:
        _fail("book.md is not Unicode NFC")
    if not text.endswith("\n"):
        _fail("book.md must end with a newline")
    if "\r" in text or "\t" in text:
        _fail("book.md must use LF line endings and must not contain tabs")
    for ch in text:
        if ch == "\n":
            continue
        cat = unicodedata.category(ch)
        if cat == "Cc" or cat == "Cf":
            _fail("book.md contains a forbidden character")


def _attrs(blob: str) -> Dict[str, str]:
    """Directive attributes. The regex already limited the shape."""
    out: Dict[str, str] = {}
    for match in _ATTR_ONE.finditer(blob):
        out[match.group(1)] = match.group(2)
    return out


def _unescape(text: str) -> str:
    """Resolve one CommonMark backslash escape. An unknown escape keeps both characters."""
    out: List[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            out.append(text[i + 1])
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _drop_emphasis(text: str) -> str:
    """Remove ``*emphasis*`` and ``**strong**`` markers, leaving the words."""
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    return re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)


def _spans(line: str, known: Dict[str, Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    """Strip speaker spans. Returns the inner text and the dominant span speaker, if any.

    Nested spans and an unknown speaker id are refused. A span with no letter is refused.
    """
    out: List[str] = []
    i = 0
    best_id = ""
    best_n = -1
    first_id = ""
    while i < len(line):
        if line.startswith("<span", i):
            end = line.find(">", i)
            if end < 0:
                _fail("a speaker span is not closed")
            open_tag = line[i:end + 1]
            if not re.fullmatch(r'<span(?: data-vx-[a-z]+="[^"]*")*>', open_tag):
                _fail("a speaker span has unexpected markup")
            attrs = {}
            body = open_tag[5:-1]
            if body and not body.startswith(" "):
                _fail("a speaker span has unexpected markup")
            for part in body.strip().split(" "):
                if not part:
                    continue
                key, eq, val = part.partition("=")
                if eq != "=" or len(val) < 2 or not (val.startswith('"') and val.endswith('"')):
                    _fail("a speaker span has unexpected markup")
                if not key.startswith("data-vx-"):
                    _fail("a speaker span has an unknown attribute")
                attrs[key] = val[1:-1]
            sid = attrs.get("data-vx-speaker", "")
            if sid not in known:
                _fail(f"unknown speaker id {sid or '(missing)'}")
            close = line.find("</span>", end + 1)
            if close < 0:
                _fail("a speaker span is not closed")
            inner = line[end + 1:close]
            if "<span" in inner or "</span>" in inner:
                _fail("speaker spans must not be nested")
            if not any(ch.isalpha() for ch in inner):
                _fail("a speaker span must contain a letter")
            letters = sum(1 for ch in inner if ch.isalpha())
            if not first_id:
                first_id = sid
            if letters > best_n:
                best_n = letters
                best_id = sid
            out.append(inner)
            i = close + len("</span>")
            continue
        if line.startswith("</span>", i):
            _fail("a speaker span is not opened")
        if line[i] == "<":
            _fail("book.md contains markup other than a speaker span")
        out.append(line[i])
        i += 1
    dominant = best_id or None
    if dominant and best_n == 0:
        dominant = first_id or None
    return "".join(out), dominant


def _plain(spoken_line: str) -> str:
    """Plain text for ``at_text``: emphasis markers, backslash escapes and U+0301 removed."""
    return _drop_emphasis(_unescape(spoken_line)).replace("\u0301", "")


def _spoken(line: str) -> str:
    """Text to narrate. A blockquote marker is removed. The stress mark stays."""
    text = line[2:] if line.startswith("> ") else line
    return _drop_emphasis(_unescape(text))


def _mark(speaker: Dict[str, Any], warnings: List[str]) -> Tuple[str, str]:
    """One section-7 mark: ``(role, name)``."""
    gender = str(speaker.get("gender") or "")
    name = str(speaker.get("name") or "").strip()
    if speaker.get("role") == "narrator" or speaker.get("id") == "narrator" or gender == "unspecified":
        if gender == "unspecified" and speaker.get("id") != "narrator":
            warnings.append(f"speaker {speaker.get('id')} has no gender and is read by the narrator")
        return "narrator", ""
    if gender == "male":
        return "male", name
    if gender == "female":
        return "female", name
    _fail(f"speaker {speaker.get('id')} has an unknown gender")
    return "narrator", ""


def _parse_markdown(text: str, minor: int, chapters: List[Dict[str, Any]],
                    speakers: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """``book.md`` -> chapter bodies, marks, fingerprints and inline sound cues."""
    _check_text(text)
    if text.endswith("\n\n"):
        _fail("book.md has a trailing blank line")
    for lineno, line in enumerate(text.split("\n"), 1):
        if line.endswith(" "):
            _fail(f"book.md line {lineno} has trailing spaces")
    blocks = text[:-1].split("\n\n") if text.endswith("\n") else text.split("\n\n")
    if any(block == "" for block in blocks):
        _fail("book.md blocks must be separated by exactly one blank line")
    warnings: List[str] = []
    bodies: List[List[str]] = []
    titles: List[str] = []
    ids: List[str] = []
    marks: List[Tuple[str, str]] = []
    unnumbered: List[Tuple[int, int]] = []
    paragraphs: List[Tuple[int, str, int, str, str]] = []
    inline: List[Tuple[int, str]] = []
    counted = 0
    seen_ids = set()
    if not blocks:
        _fail("book.md is empty")
    for block in blocks:
        lines = block.split("\n")
        directives: List[Tuple[str, Dict[str, str]]] = []
        content: List[str] = []
        for line in lines:
            match = _DIRECTIVE.match(line)
            if match and not content:
                directives.append((match.group(1), _attrs(match.group(2))))
            else:
                content.append(line)
        if len(content) != 1:
            _fail("each block must be one line of text, optionally with directive lines before it")
        if any(name == "speaker" for name, _a in directives[:-1]):
            _fail("vx:speaker must be the last directive in a block")
        if len(directives) > 1 and minor < 1:
            _fail("stacked directives need format 1.1")
        names = [name for name, _a in directives]
        if names.count("speaker") > 1 or names.count("chapter") > 1:
            _fail("a block has a repeated directive")
        line = content[0]
        if any(name == "chapter" for name, _a in directives):
            if names != ["chapter"] or not line.startswith("# ") or line.startswith("## "):
                _fail("vx:chapter must sit on the line before a level-1 heading")
            cid = directives[0][1].get("id", "")
            if not _ID.match(cid) or cid in seen_ids:
                _fail(f"bad or repeated chapter id {cid!r}")
            seen_ids.add(cid)
            title = line[2:].strip()
            if not title:
                _fail("a chapter heading is empty")
            ids.append(cid)
            titles.append(title)
            bodies.append([])
            continue
        if not bodies:
            _fail("book.md must start with a chapter")
        if line.startswith("# ") and not line.startswith("## "):
            _fail("a chapter heading needs a vx:chapter directive")
        if line.startswith("## "):
            if directives:
                _fail("a subheading cannot carry a directive")
            bodies[-1].append(line)
            unnumbered.append((len(bodies) - 1, len(bodies[-1]) - 1))
            marks.append(("narrator", ""))
            continue
        if line == "* * *":
            if directives:
                _fail("a scene break cannot carry a directive")
            bodies[-1].append(line)
            unnumbered.append((len(bodies) - 1, len(bodies[-1]) - 1))
            marks.append(("narrator", ""))
            continue
        if line.startswith(">"):
            if not line.startswith("> "):
                _fail("a blockquote must start with '> '")
        elif line[:1] in "-+*" or line.startswith("    "):
            _fail("book.md contains a construct this format does not allow")
        stripped, dominant = _spans(line, speakers)
        speaker_id = ""
        for name, attrs in directives:
            if name == "speaker":
                speaker_id = attrs.get("id", "")
                if speaker_id not in speakers:
                    _fail(f"unknown speaker id {speaker_id}")
                src = attrs.get("src")
                if src is not None and src not in ("auto", "user"):
                    _fail("vx:speaker src must be auto or user")
                conf = attrs.get("conf")
                if conf is not None and not _CONF.match(conf):
                    _fail("vx:speaker conf must be between 0 and 1")
            elif name == "sound":
                cue = attrs.get("cue", "")
                if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", cue):
                    _fail("vx:sound cue id is not valid")
            elif name != "chapter":
                warnings.append(f"ignored unknown directive vx:{name}")
        counted += 1
        if not speaker_id:
            speaker_id = dominant or "narrator"
        marks.append(_mark(speakers[speaker_id], warnings))
        spoken = _spoken(stripped)
        bodies[-1].append(spoken)
        raw_fp = hashlib.sha256(line.encode("utf-8")).hexdigest()[:12]
        paragraphs.append((counted, ids[-1], len(bodies) - 1, raw_fp, _plain(stripped)))
        for name, attrs in directives:
            if name == "sound":
                inline.append((counted, attrs["cue"]))
    if [c.get("id") for c in chapters] != ids:
        _fail("manifest chapters do not match book.md")
    for spec, title, cid in zip(chapters, titles, ids):
        if spec.get("title") != title or spec.get("anchor") != cid:
            _fail(f"chapter {cid} does not match its heading")
        offset = text.encode("utf-8").find(f'<!-- vx:chapter id="{cid}" -->'.encode("utf-8"))
        want = spec.get("offset")
        if isinstance(want, int) and want != offset and offset >= 0:
            warnings.append(f"chapter {cid} offset is {offset}, manifest says {want}")
    return {
        "chapters": [Chapter(title, "\n\n".join(body)) for title, body in zip(titles, bodies)],
        "marks": tuple(marks), "unnumbered": tuple(unnumbered), "paragraphs": tuple(paragraphs),
        "inline": tuple(inline), "warnings": warnings, "ids": tuple(ids),
    }


def _speakers(data: Any) -> Dict[str, Dict[str, Any]]:
    """``speakers.json`` as ``{id: object}``. The narrator id is required."""
    if not isinstance(data, dict) or not isinstance(data.get("speakers"), list) or not data["speakers"]:
        _fail("speakers.json needs a speakers list")
    out: Dict[str, Dict[str, Any]] = {}
    for row in data["speakers"]:
        if not isinstance(row, dict):
            _fail("a speaker entry must be an object")
        sid = row.get("id")
        if not isinstance(sid, str) or not _ID.match(sid) or sid in out:
            _fail(f"bad or repeated speaker id {sid!r}")
        role = row.get("role")
        if role not in ("narrator", "character"):
            _fail(f"speaker {sid} has an unknown role")
        if sid == "narrator" and role != "narrator":
            _fail("the narrator id must have the narrator role")
        out[sid] = row
    if "narrator" not in out:
        _fail("speakers.json has no narrator")
    return out


def _cast(data: Any, speakers: Dict[str, Dict[str, Any]], warnings: List[str]) -> Tuple[Dict[str, str], str, Dict[str, str]]:
    """``(name -> voice id, narrator voice id, alias -> name)``. Unknown libraries are skipped with a warning."""
    if not isinstance(data, dict) or not isinstance(data.get("cast"), dict):
        _fail("cast.json needs a cast object")
    voices: Dict[str, str] = {}
    aliases: Dict[str, str] = {}
    narrator = ""
    for sid, row in data["cast"].items():
        if sid not in speakers:
            warnings.append(f"cast entry {sid} is not a speaker and was ignored")
            continue
        if not isinstance(row, dict):
            continue
        if row.get("library") not in (None, "voxprint"):
            warnings.append(f"cast entry {sid} uses an unknown library and was ignored")
            continue
        vid = str(row.get("voice_id") or "").strip()
        if not vid:
            continue
        if sid == "narrator":
            narrator = vid
            continue
        name = str(speakers[sid].get("name") or "").strip()
        if name:
            voices[name] = vid
        for alias in speakers[sid].get("aliases") or []:
            if isinstance(alias, str) and alias.strip():
                voices[alias.strip()] = vid
                if name:
                    aliases[alias.strip()] = name
    return voices, narrator, aliases


def _optional_object(raw: Optional[bytes], name: str, warnings: List[str]) -> Optional[Dict[str, Any]]:
    """Parse an optional JSON object. A schema problem disables that file and does not refuse the book."""
    if raw is None:
        return None
    try:
        value = _json_bytes(raw, name)
    except BookParseError as exc:
        warnings.append(f"{name} was ignored: {exc.details or exc.user_message}")
        return None
    if not isinstance(value, dict):
        warnings.append(f"{name} was ignored because it is not an object")
        return None
    return value


def load_vxbook(path: Path) -> Book:
    """Open one ``.vxbook``. Raises :class:`core.errors.BookParseError` when the archive is not a book we can read."""
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        _fail("the file is not a zip archive")
    with zf:
        infos = zf.infolist()
        if not infos or len(infos) > MAX_ENTRIES:
            _fail("the archive has no entries or more than 1000")
        names = [i.filename for i in infos]
        folded = [n.casefold() for n in names]
        if len(set(folded)) != len(folded):
            _fail("the archive repeats a file name")
        if any(not _safe_zip_name(n) for n in names):
            _fail("the archive contains an unsafe path")
        first = infos[0]
        if first.filename != "mimetype" or first.compress_type != zipfile.ZIP_STORED or first.extra:
            _fail("mimetype must be the first entry, stored, with no extra field")
        total = [0]
        mime = _read_entry(zf, first, total)
        if mime != MIME:
            _fail("mimetype is not application/vnd.voxprint.book+zip")
        blobs: Dict[str, bytes] = {}
        for info in infos[1:]:
            blobs[info.filename] = _read_entry(zf, info, total)
    if "manifest.json" not in blobs or "book.md" not in blobs or "speakers.json" not in blobs:
        _fail("manifest.json, book.md and speakers.json are required")
    manifest = _json_bytes(blobs["manifest.json"], "manifest.json")
    if not isinstance(manifest, dict):
        _fail("manifest.json must be an object")
    major, minor = _version(manifest.get("format_version"))
    files = manifest.get("files")
    if not isinstance(files, dict):
        _fail("manifest.json needs a files object")
    listed = set(files)
    present = set(blobs) - {"manifest.json"}
    if present != listed:
        extra = ", ".join(sorted(present - listed)) or "none"
        missing = ", ".join(sorted(listed - present)) or "none"
        _fail(f"manifest files do not match the archive (unexpected: {extra}; missing: {missing})")
    for name, meta in files.items():
        if not isinstance(meta, dict):
            _fail(f"{name} has no hash")
        digest = str(meta.get("sha256") or "")
        size = meta.get("size")
        blob = blobs[name]
        if size != len(blob) or len(digest) != 64 or hashlib.sha256(blob).hexdigest() != digest:
            _fail(f"{name} does not match its manifest hash")
    speakers_raw = _json_bytes(blobs["speakers.json"], "speakers.json")
    if not isinstance(speakers_raw, dict) or speakers_raw.get("format_version") != manifest.get("format_version"):
        _fail("speakers.json format_version must match the manifest")
    _version(speakers_raw.get("format_version"))
    speakers = _speakers(speakers_raw)
    chapters = manifest.get("chapters")
    if not isinstance(chapters, list) or not chapters or not all(isinstance(c, dict) for c in chapters):
        _fail("manifest chapters are missing")
    try:
        book_text = blobs["book.md"].decode("utf-8")
    except UnicodeError:
        _fail("book.md is not UTF-8")
    parsed = _parse_markdown(book_text, minor, chapters, speakers)
    warnings = list(parsed["warnings"])
    voices: Dict[str, str] = {}
    narrator = ""
    aliases: Dict[str, str] = {}
    if "cast.json" in blobs:
        cast_raw = _json_bytes(blobs["cast.json"], "cast.json")
        voices, narrator, aliases = _cast(cast_raw, speakers, warnings)
    extensions = manifest.get("extensions") or []
    if extensions is None:
        extensions = []
    if not isinstance(extensions, list) or not all(isinstance(x, str) for x in extensions):
        _fail("manifest extensions must be a list of strings")
    ext_tuple = tuple(extensions)
    sound_doc = None
    sound_cast = None
    if "sound.json" in blobs:
        sound_doc = _optional_object(blobs["sound.json"], "sound.json", warnings)
    if "sound-cast.json" in blobs:
        sound_cast = _optional_object(blobs["sound-cast.json"], "sound-cast.json", warnings)
    if ("sound/1" in ext_tuple) != ("sound.json" in blobs):
        warnings.append("sound/1 is declared exactly when sound.json is present; the soundscape stays off")
        sound_doc = None
    cover = b""
    cover_ext = ""
    cover_names = [n for n in blobs if n.lower().startswith("cover.")]
    if len(cover_names) > 1:
        _fail("a book can have only one cover")
    if cover_names:
        raw_cover = blobs[cover_names[0]]
        pixels = _pixels(raw_cover)
        if pixels is None:
            warnings.append(f"{cover_names[0]} could not be read and was ignored")
        elif pixels > MAX_PIXELS:
            _fail("the cover is larger than 100 megapixels")
        else:
            ext = "png" if raw_cover[:8] == b"\x89PNG\r\n\x1a\n" else "jpg"
            cover, cover_ext = raw_cover, ext
    for warning in warnings:
        log.warning("vxbook: %s", warning)
    language = str(manifest.get("language") or "")
    return Book(
        title=str(manifest.get("title") or Path(path).stem),
        author=str(manifest.get("author") or ""),
        language=language,
        chapters=parsed["chapters"],
        cover=cover or None,
        cover_ext=cover_ext,
        explicit_yo=True,
        speaker_marks=parsed["marks"],
        voice_cast=voices,
        narrator_hint=narrator,
        alias_to_name=aliases,
        vxbook_path=str(path),
        chapter_ids=parsed["ids"],
        unnumbered_blocks=parsed["unnumbered"],
        extensions=ext_tuple,
        sound_document=sound_doc if "sound/1" in ext_tuple else None,
        sound_cast_document=sound_cast if "sound/1" in ext_tuple else None,
        sound_paragraphs=parsed["paragraphs"],
        sound_inline=parsed["inline"],
        vxbook_warnings=tuple(warnings),
    )
