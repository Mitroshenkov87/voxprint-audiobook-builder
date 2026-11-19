"""WebP is the default picture format: screenshot tools write .webp (JPEG fallback, never PNG for new
outputs), docs screenshots are tracked as .webp, and book covers accept WebP input but are embedded
into M4B/MP3 as JPEG."""
import base64
import hashlib
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path

import pytest

from core import audiobook_export as ex
from core import vxbook
from core.book_parsers import cover_as_jpeg, load_book, parse_fb2

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _webp_bytes(width: int = 60, height: int = 40, color=(20, 60, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, "WEBP", quality=90)
    return buf.getvalue()


# ----------------------------------------------------------------------------- the picture tools

def test_grab_ui_shell_targets_webp_and_never_png():
    text = (ROOT / "tools" / "grab_ui_shell.py").read_text(encoding="utf-8")
    for name in ("simple-en", "progress-en", "expert-en", "train-en"):
        assert f'out / "{name}.webp"' in text
    assert ".png" not in text
    assert '"WEBP"' in text and '"JPG"' in text and "QImageWriter.supportedImageFormats()" in text


def test_manual_builder_expects_webp_shots():
    pytest.importorskip("markdown")
    pytest.importorskip("weasyprint")
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_manual", ROOT / "docs" / "manual" / "build_manual.py")
    bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm)
    p, fallback = bm.shot_path("train_preview", "en")
    assert p == ROOT / "docs" / "manual" / "img" / "en" / "train_preview.webp" and not fallback
    for name in bm.SHOTS:
        shot = bm.SHOT_DIR / f"{bm.SHOTS[name]}.webp"
        assert shot.is_file(), shot
        assert bm.shot_path(name, "en")[0] == shot


def test_no_tracked_docs_png_remains():
    out = subprocess.run(
        ["git", "ls-files", "docs/**/*.png"], capture_output=True, text=True, cwd=ROOT,
    )
    assert out.returncode == 0
    assert out.stdout.strip() == "", out.stdout


def test_docs_screenshot_references_resolve_to_tracked_webp():
    tracked = set(
        subprocess.run(["git", "ls-files", "docs/"], capture_output=True, text=True, cwd=ROOT).stdout.split(),
    )
    names = [ROOT / "docs" / "FEATURES.md", ROOT / "docs" / "README-full.md", ROOT / "mkdocs.yml"]
    refs = set()
    for name in names:
        if name.is_file():
            refs.update(re.findall(r"screenshots/[^\s)\]\"'>]+\.webp", name.read_text(encoding="utf-8")))
    assert refs, "expected screenshot references in the docs"
    for ref in refs:
        assert ("docs/" + ref).replace(".webp", ".png") not in tracked  # the rename really happened
        assert "docs/" + ref in tracked, f"{ref} is not a tracked docs file"


# ----------------------------------------------------------------------------- book covers

def test_webp_cover_is_accepted_by_fb2_epub_and_vxbook(tmp_path):
    webp = _webp_bytes()
    fb2 = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0" xmlns:l="http://www.w3.org/1999/xlink">
<description><title-info><book-title>WebP Book</book-title><lang>en</lang>
<coverpage><image l:href="#cover.webp"/></coverpage></title-info></description>
<body><section><title><p>One</p></title><p>A long enough paragraph to be a chapter of the book.</p></section></body>
<binary id="cover.webp" content-type="image/webp">%s</binary>
</FictionBook>""" % base64.b64encode(webp).decode()
    book = parse_fb2(fb2.encode("utf-8"))
    assert book.cover == webp and book.cover_ext == "webp"

    epub = tmp_path / "b.epub"
    with zipfile.ZipFile(epub, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
        z.writestr("OEBPS/content.opf",
                   '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>WebP Book</dc:title>'
                   '<dc:creator>Jane Roe</dc:creator><dc:language>en</dc:language></metadata><manifest>'
                   '<item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
                   '<item id="cov" href="cover.webp" media-type="image/webp" properties="cover-image"/>'
                   '</manifest><spine><itemref idref="c1"/></spine></package>')
        z.writestr("OEBPS/ch1.xhtml",
                   '<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>One</h1>'
                   '<p>A long enough paragraph to be a chapter of the book.</p></body></html>')
        z.writestr("OEBPS/cover.webp", webp)
    book = load_book(epub)
    assert book.cover == webp and book.cover_ext == "webp"

    vx = tmp_path / "b.vxbook"
    book_md = b'<!-- vx:chapter id="ch01" -->\n# One\n\nA long enough paragraph to be a chapter of the book.\n'
    speakers = json.dumps({"format_version": "1.0", "speakers": [
        {"id": "narrator", "role": "narrator", "name": "Narrator", "gender": "unspecified"},
    ]}).encode("utf-8")
    manifest = {
        "format_version": "1.0", "title": "WebP Book", "author": "Tester", "language": "en",
        "extensions": [], "book": "book.md", "speakers": "speakers.json",
        "chapters": [{"id": "ch01", "title": "One", "anchor": "ch01", "offset": 0}],
        "files": {
            "book.md": {"sha256": hashlib.sha256(book_md).hexdigest(), "size": len(book_md)},
            "speakers.json": {"sha256": hashlib.sha256(speakers).hexdigest(), "size": len(speakers)},
            "cover.webp": {"sha256": hashlib.sha256(webp).hexdigest(), "size": len(webp)},
        },
    }
    with zipfile.ZipFile(vx, "w") as zf:
        info = zipfile.ZipInfo("mimetype")
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, b"application/vnd.voxprint.book+zip")
        zf.writestr("manifest.json", json.dumps(manifest).encode("utf-8"))
        zf.writestr("book.md", book_md)
        zf.writestr("speakers.json", speakers)
        zf.writestr("cover.webp", webp)
    book = load_book(vx)
    assert book.cover == webp and book.cover_ext == "webp"
    assert vxbook._webp_pixels(webp) == 60 * 40


def test_cover_is_embedded_as_jpeg():
    """What narration.py writes next to the export job: JPEG bytes as cover.jpg (mime image/jpeg)."""
    jpeg = cover_as_jpeg(_webp_bytes())
    assert jpeg is not None and jpeg.startswith(b"\xff\xd8")

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    )
    as_jpeg = cover_as_jpeg(png)
    assert as_jpeg is not None and as_jpeg.startswith(b"\xff\xd8")

    assert cover_as_jpeg(jpeg) == jpeg                     # JPEG passes through untouched
    assert cover_as_jpeg(b"not a picture at all") is None  # undecodable: skipped, never fatal


def test_export_cmd_attaches_the_converted_cover():
    work = Path("/w")
    cover = work / "cover.jpg"                 # narration always writes cover.jpg now
    rates = ex.Bitrates()
    m4b = " ".join(ex.cmd_single("ff", ex.FORMAT_M4B, work / "l.txt", work / "m.ffmetadata",
                                 ex.BookMeta("T", "A", "V", "en", cover=cover), Path("o.m4b"), rates))
    assert f"-i {cover} -map 0:a -map 2:v" in m4b and "-c:v copy" in m4b and "-disposition:v:0" in m4b
    mp3 = " ".join(ex.cmd_chapter("ff", ex.FORMAT_MP3_CHAPTERS,
                                  ex.ChapterAudio(1, "Two", work / "c.wav", 1.0), 3,
                                  ex.BookMeta("T", "A", "V", "en", cover=cover), Path("o.mp3"), rates))
    assert f"-i {cover}" in mp3 and "-c:v copy" in mp3
