"""``.vxbook`` reader: the sound sample, a small v1.0 book, and archives that must be refused."""
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from core.book_parsers import Book, Chapter, load_book
from core.errors import BookParseError
from core.normalizer import normalize_for_tts
from core import speakers, yo
from core.chunker import chunk_book
from core.narration import NarrationOptions, text_steps

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sound-sample.vxbook"
MIME = b"application/vnd.voxprint.book+zip"

SPEAKERS = {
    "format_version": "1.0",
    "speakers": [
        {"id": "narrator", "role": "narrator", "name": "Narrator", "gender": "unspecified"},
        {"id": "anna", "role": "character", "name": "Anna", "gender": "female"},
    ],
}
BOOK_V1 = """<!-- vx:chapter id="ch01" -->
# Chapter One

Hello there friend.

<!-- vx:speaker id="anna" -->
She said one short line.

* * *

Morning came at last.
"""


def _zip(path: Path, members: dict, *, mime: bytes = MIME, stored_first: bool = True, deflate=()) -> None:
    """Write a ``.vxbook``. ``members`` is name -> bytes, after the mimetype entry."""
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo("mimetype")
        info.compress_type = zipfile.ZIP_STORED if stored_first else zipfile.ZIP_DEFLATED
        zf.writestr(info, mime)
        for name, data in members.items():
            kind = zipfile.ZIP_DEFLATED if name in deflate else zipfile.ZIP_STORED
            zf.writestr(name, data, compress_type=kind)


def _manifest(files: dict, *, version: str = "1.0", extensions=None, chapters=None) -> bytes:
    body = {
        "format_version": version,
        "title": "Sample",
        "author": "Tester",
        "language": "en",
        "extensions": list(extensions or []),
        "book": "book.md",
        "speakers": "speakers.json",
        "chapters": chapters or [{"id": "ch01", "title": "Chapter One", "anchor": "ch01", "offset": 0}],
        "files": files,
    }
    return json.dumps(body).encode("utf-8")


def _files(blobs: dict, *, bad: str = "") -> dict:
    out = {}
    for name, data in blobs.items():
        digest = "0" * 64 if name == bad else hashlib.sha256(data).hexdigest()
        out[name] = {"sha256": digest, "size": len(data)}
    return out


def _pack(path: Path, book: str = BOOK_V1, speakers: dict = None, extra: dict = None,
          version: str = "1.0", extensions=None, bad: str = "", chapters=None, deflate=()) -> None:
    speakers = SPEAKERS if speakers is None else speakers
    blobs = {"book.md": book.encode("utf-8"), "speakers.json": json.dumps(speakers).encode("utf-8")}
    if extra:
        blobs.update(extra)
    manifest = _manifest(_files(blobs, bad=bad), version=version, extensions=extensions, chapters=chapters)
    _zip(path, {"manifest.json": manifest, **blobs}, deflate=deflate)


def test_sound_sample_maps_marks_cast_and_paragraphs():
    book = load_book(FIXTURE)
    assert book.explicit_yo and book.extensions == ("sound/1",)
    assert book.narrator_hint == "levi"
    assert book.voice_cast["Анна"] == "miriam" and book.voice_cast["Петя"] == "natan"
    assert book.alias_to_name["Петя"] == "Пётр"
    fingerprints = [row[3] for row in book.sound_paragraphs]
    assert fingerprints == ["791635f0f627", "82b5671d2862", "bcd4b2c89e66", "746e4aee43ff", "062d3efcc474", "e31b535c10d9"]
    assert book.sound_inline == ((2, "c3"),)
    counted = [(mark, text) for mark, (_ci, text) in zip(book.speaker_marks, speakers.paragraphs(book)) if text != "* * *"]
    assert [mark[0] for mark, _text in counted] == ["narrator", "female", "male", "female", "narrator", "narrator"]
    assert counted[1][0][1] == "Анна" and counted[2][0][1] == "Пётр"
    assert len(book.speaker_marks) == len(speakers.paragraphs(book))
    assert "\u0301" in book.chapters[0].text
    spoken = normalize_for_tts(book.chapters[0].text, "Russian").spoken
    assert "\u0301" in spoken
    chunks = chunk_book(book, 400)
    assert [c.paragraph for c in chunks] == [1, 2, 3, 4, 5, 6]
    assert chunks[3].pause_ms == 1500


def test_chapter_and_speaker_ids_allow_digits_and_hyphens(tmp_path):
    """Ids are ``^[a-z0-9][a-z0-9_-]{0,63}$``. ``1``, ``2``, ``ch01`` and ``part-2`` open; 65 characters do not."""
    digit = """<!-- vx:chapter id="1" -->
# One

Hello there friend.

<!-- vx:chapter id="2" -->
# Two

<!-- vx:speaker id="2" -->
She said one short line.
"""
    speakers = {
        "format_version": "1.0",
        "speakers": [
            {"id": "narrator", "role": "narrator", "name": "Narrator", "gender": "unspecified"},
            {"id": "2", "role": "character", "name": "Two", "gender": "female"},
        ],
    }
    raw = digit.encode("utf-8")
    chapters = [
        {"id": "1", "title": "One", "anchor": "1", "offset": raw.find(b'<!-- vx:chapter id="1" -->')},
        {"id": "2", "title": "Two", "anchor": "2", "offset": raw.find(b'<!-- vx:chapter id="2" -->')},
    ]
    path = tmp_path / "digits.vxbook"
    _pack(path, digit, speakers=speakers, chapters=chapters)
    book = load_book(path)
    assert book.chapter_ids == ("1", "2")
    assert book.speaker_marks[1][0] == "female"

    letter = tmp_path / "letters.vxbook"
    _pack(letter)
    opened = load_book(letter)
    assert opened.chapter_ids == ("ch01",)

    hyphen = """<!-- vx:chapter id="part-2" -->
# Part Two

<!-- vx:speaker id="part-2" -->
She said one short line.
"""
    hyphen_speakers = {
        "format_version": "1.0",
        "speakers": [
            {"id": "narrator", "role": "narrator", "name": "Narrator", "gender": "unspecified"},
            {"id": "part-2", "role": "character", "name": "Part", "gender": "female"},
        ],
    }
    cast = {"cast": {"part-2": {"library": "voxprint", "voice_id": "miriam"}}}
    raw = hyphen.encode("utf-8")
    hyphen_chapters = [
        {"id": "part-2", "title": "Part Two", "anchor": "part-2", "offset": raw.find(b'<!-- vx:chapter id="part-2" -->')},
    ]
    hyphen_path = tmp_path / "hyphen.vxbook"
    _pack(
        hyphen_path, hyphen, speakers=hyphen_speakers, chapters=hyphen_chapters,
        extra={"cast.json": json.dumps(cast).encode("utf-8")},
    )
    hyphen_book = load_book(hyphen_path)
    assert hyphen_book.chapter_ids == ("part-2",)
    assert hyphen_book.speaker_marks[0][0] == "female"
    assert hyphen_book.voice_cast["Part"] == "miriam"

    exact = "b" * 64
    bounded = f"""<!-- vx:chapter id="{exact}" -->
# Bounded

Hello there friend.
"""
    bounded_path = tmp_path / "bounded.vxbook"
    raw = bounded.encode("utf-8")
    _pack(
        bounded_path, bounded,
        chapters=[{"id": exact, "title": "Bounded", "anchor": exact,
                   "offset": raw.find(f'<!-- vx:chapter id="{exact}" -->'.encode("utf-8"))}],
    )
    assert load_book(bounded_path).chapter_ids == (exact,)

    long_id = "a" * 65
    too_long = f"""<!-- vx:chapter id="{long_id}" -->
# Too Long

Hello there friend.
"""
    too_long_path = tmp_path / "too-long.vxbook"
    raw = too_long.encode("utf-8")
    _pack(
        too_long_path, too_long,
        chapters=[{"id": long_id, "title": "Too Long", "anchor": long_id,
                   "offset": raw.find(f'<!-- vx:chapter id="{long_id}" -->'.encode("utf-8"))}],
    )
    with pytest.raises(BookParseError) as exc:
        load_book(too_long_path)
    assert "chapter id" in (exc.value.details or "")


def test_explicit_yo_is_not_restored_from_the_dictionary(tmp_path):
    book_md = BOOK_V1.replace("Hello there friend.", "He said еще once.")
    path = tmp_path / "yo.vxbook"
    _pack(path, book_md)
    book = load_book(path)
    assert book.explicit_yo
    assert yo.restore("еще") != "еще"
    step = text_steps("ru", book.language, NarrationOptions(yo=not book.explicit_yo))
    assert step is None or "еще" in step("еще")
    assert "еще" in book.chapters[0].text


def test_v10_book_without_sound_opens_and_stays_silent(tmp_path):
    path = tmp_path / "plain.vxbook"
    _pack(path)
    book = load_book(path)
    assert book.extensions == () and book.sound_document is None
    from core import soundscape
    assert soundscape.plan_for(book) is None
    assert soundscape.plan_for(Book("T", chapters=[Chapter("C", "Hello there.\n\nSecond paragraph here.")])) is None


def test_bad_hash_unknown_speaker_nested_span_and_zip_bomb(tmp_path):
    bad = tmp_path / "bad.vxbook"
    _pack(bad, bad="book.md")
    with pytest.raises(BookParseError) as exc:
        load_book(bad)
    assert "hash" in exc.value.details

    unknown = tmp_path / "unknown.vxbook"
    text = BOOK_V1.replace('id="anna"', 'id="ghost"')
    _pack(unknown, text)
    with pytest.raises(BookParseError) as exc:
        load_book(unknown)
    assert "ghost" in exc.value.details

    nested = tmp_path / "nested.vxbook"
    line = '<span data-vx-speaker="anna">Hello <span data-vx-speaker="anna">there</span></span> friend.'
    _pack(nested, BOOK_V1.replace("Hello there friend.", line))
    with pytest.raises(BookParseError) as exc:
        load_book(nested)
    assert "nested" in exc.value.details

    bomb = tmp_path / "bomb.vxbook"
    note = b"A" * 200_000
    _pack(bomb, extra={"notes/pad.md": note}, deflate={"notes/pad.md"})
    with pytest.raises(BookParseError) as exc:
        load_book(bomb)
    assert "expands" in exc.value.details


def test_newer_major_is_refused_and_sound_needs_both_halves(tmp_path):
    newer = tmp_path / "new.vxbook"
    speakers = json.loads(json.dumps(SPEAKERS))
    speakers["format_version"] = "2.0"
    _pack(newer, speakers=speakers, version="2.0")
    with pytest.raises(BookParseError) as exc:
        load_book(newer)
    assert "newer Plotweaver" in exc.value.details

    declared = tmp_path / "declared.vxbook"
    speakers_11 = json.loads(json.dumps(SPEAKERS))
    speakers_11["format_version"] = "1.1"
    _pack(declared, speakers=speakers_11, version="1.1", extensions=["sound/1"])
    book = load_book(declared)
    assert book.sound_document is None
    assert any("sound/1" in w for w in book.vxbook_warnings)

    slipped = tmp_path / "slip.vxbook"
    blobs = {"book.md": BOOK_V1.encode(), "speakers.json": json.dumps(SPEAKERS).encode(), "../evil.txt": b"no"}
    manifest = _manifest(_files({k: v for k, v in blobs.items() if k != "../evil.txt"}))
    _zip(slipped, {"manifest.json": manifest, **blobs})
    with pytest.raises(BookParseError) as exc:
        load_book(slipped)
    assert "unsafe" in exc.value.details or "files" in exc.value.details
