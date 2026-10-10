"""Book input for the narrator: plain text, Markdown, FictionBook (FB2, also ``.fb2.zip``) and EPUB -> :class:`Book`.

Only the standard library is used (``zipfile``, ``xml.etree``, ``html.parser``).  XML from files is parsed only after a
guard that rejects documents declaring entities (the "billion laughs" family) - ``xml.etree`` does not resolve external
entities, but it would expand internal ones.  Archive members are read with a size cap.

A :class:`Book` is a title, an author, an optional language/cover and an ordered list of :class:`Chapter` (title + plain
text, paragraphs separated by blank lines).  Chapter detection:

* FB2 - the ``<section>`` tree of the main ``<body>``;
* EPUB - the spine documents, titled from the navigation (``nav``/NCX) or the first heading;
* TXT - heading lines such as "Chapter 3", "Part II", Russian "Глава 5", Markdown ``# Title`` or a bare number/roman numeral.

Extension point: nothing here cleans or translates text (that would be a later preprocessing step on ``Chapter.text``).
"""
from __future__ import annotations

import base64
import posixpath
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

from core.errors import BookParseError
from core.i18n import tr
from core.text_utils import decode_bytes

SUPPORTED_EXTENSIONS = (".txt", ".md", ".fb2", ".epub", ".zip")
MAX_MEMBER_BYTES = 80 * 1024 * 1024
MIN_CHAPTER_CHARS = 40          # EPUB documents with less text (title pages, blank pages) are dropped


@dataclass
class Chapter:
    """One chapter: a title and its plain text (paragraphs separated by a blank line)."""
    title: str
    text: str


@dataclass
class Book:
    """A parsed book."""
    title: str
    author: str = ""
    language: str = ""
    chapters: List[Chapter] = field(default_factory=list)
    cover: Optional[bytes] = None
    cover_ext: str = ""          # "jpg" | "png" when ``cover`` is set

    @property
    def total_chars(self) -> int:
        """Number of characters of text in all chapters."""
        return sum(len(c.text) for c in self.chapters)


# --------------------------------------------------------------------------- helpers

def _safe_xml(data: bytes) -> ET.Element:
    """Parse XML bytes; documents that declare entities (``<!ENTITY``) are rejected."""
    if re.search(rb"<!ENTITY", data[:200_000], re.IGNORECASE):
        raise BookParseError(tr("err.book_unsafe"))
    try:
        return ET.fromstring(data)  # nosec B314 - entity declarations rejected above; expat does not fetch external DTDs
    except ET.ParseError as exc:
        raise BookParseError(tr("err.book_read"), details=str(exc)) from exc


def _local(tag: str) -> str:
    """Tag name without the ``{namespace}`` prefix."""
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _image_ext(data: bytes) -> str:
    """``jpg``/``png`` from the magic bytes, else ``""`` (other formats are not used as a cover)."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return ""


def _squash(text: str) -> str:
    """Collapse runs of whitespace inside a line."""
    return re.sub(r"\s+", " ", text).strip()


def _read_member(zf: zipfile.ZipFile, name: str) -> bytes:
    """Read an archive member with a size cap."""
    info = zf.getinfo(name)
    if info.file_size > MAX_MEMBER_BYTES:
        raise BookParseError(tr("err.book_unsafe"), details=f"{name}: too large")
    return zf.read(name)


def _with_default_title(chapters: List[Chapter]) -> List[Chapter]:
    """Give untitled chapters a numbered title."""
    for i, c in enumerate(chapters, 1):
        if not c.title.strip():
            c.title = tr("book.chapter_n", n=i)
    return chapters


# --------------------------------------------------------------------------- TXT

_HEADING_RES = [
    re.compile(r"^\s*#{1,3}\s+\S.*$"),
    re.compile(r"^\s*(?:глава|часть|книга|пролог|эпилог|предисловие|chapter|part|book|prologue|epilogue|kapitel|teil)"
               r"\b.{0,70}$", re.IGNORECASE),
    re.compile(r"^\s*(?:\d{1,3}|[IVXLC]{1,8})[.)]?\s*$"),
]


def _is_heading(line: str, prev_blank: bool, next_blank: bool) -> bool:
    """A short line that looks like a chapter heading and stands alone between blank lines."""
    if not (prev_blank and next_blank) or not line.strip() or len(line) > 90:
        return False
    words = line.split()
    if len(words) > 8 or (len(words) > 3 and line.rstrip()[-1] in ".!?\u2026,;:"):
        return False                                   # a sentence that merely starts with "Chapter" / "Part" / "Глава"
    return any(r.match(line) for r in _HEADING_RES)


def parse_txt(text: str, title: str = "") -> Book:
    """Split plain text into chapters at heading lines; without headings the whole text is one chapter."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\ufeff")
    lines = text.split("\n")
    chapters: List[Chapter] = []
    cur_title = ""
    cur_lines: List[str] = []

    def flush() -> None:
        body = "\n".join(cur_lines).strip()
        if body or cur_title:
            chapters.append(Chapter(cur_title, _paragraphs_from_lines(cur_lines)))

    for i, line in enumerate(lines):
        prev_blank = i == 0 or not lines[i - 1].strip()
        next_blank = i + 1 >= len(lines) or not lines[i + 1].strip()
        if _is_heading(line, prev_blank, next_blank):
            flush()
            cur_title = line.strip().lstrip("#").strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    flush()
    chapters = [c for c in chapters if c.text.strip()]
    lead = ""
    if len(chapters) > 1 and not chapters[0].title and _title_only(chapters[0].text):
        # a short title line before the first heading ("Бытие" above "Глава 1") is the opening line of chapter 1,
        # not a separate "Chapter 1"
        lead = chapters[0].text.strip()
        chapters = chapters[1:]
        chapters[0].text = lead + "\n\n" + chapters[0].text
    if not chapters:
        raise BookParseError(tr("err.book_empty"))
    if len(chapters) == 1 and not chapters[0].title:
        chapters[0].title = title
    return Book(title=title or lead or chapters[0].title, chapters=_with_default_title(chapters))


def _title_only(text: str) -> bool:
    """One short line without a sentence end: a book title, not prose."""
    t = text.strip()
    return "\n" not in t and len(t) <= 100 and len(t.split()) <= 10 and t[-1:] not in ".!?\u2026,;:"


def _paragraphs_from_lines(lines: List[str]) -> str:
    """Join text lines into paragraphs: blank lines separate paragraphs, single line breaks become spaces unless the
    block looks like verse (many short lines) - those keep their line breaks."""
    blocks, cur = [], []
    for ln in lines + [""]:
        if ln.strip():
            cur.append(ln.strip())
        elif cur:
            blocks.append(cur)
            cur = []
    out = []
    for b in blocks:
        short = len(b) > 2 and sum(len(x) for x in b) / len(b) < 45
        out.append("\n".join(b) if short else " ".join(b))
    return "\n\n".join(out)


# --------------------------------------------------------------------------- FB2

def _fb2_text(el: ET.Element) -> str:
    """Text of a paragraph-like element without footnote markers (``<a type="note">``)."""
    parts: List[str] = [el.text or ""]
    for child in el:
        if not (_local(child.tag) == "a" and child.get("type") == "note"):
            parts.append(_fb2_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _fb2_title(el: Optional[ET.Element]) -> str:
    """Title element -> one line."""
    if el is None:
        return ""
    return _squash(" ".join(_squash(_fb2_text(p)) for p in el if _local(p.tag) == "p") or _fb2_text(el))


def _fb2_section_text(sec: ET.Element) -> str:
    """Paragraphs of a section itself (not of its sub-sections)."""
    out: List[str] = []

    def walk(e: ET.Element) -> None:
        for ch in e:
            name = _local(ch.tag)
            if name in ("p", "subtitle", "text-author"):
                t = _squash(_fb2_text(ch))
                if t:
                    out.append(t)
            elif name == "v":
                t = _squash(_fb2_text(ch))
                if t:
                    out.append(t)
            elif name in ("poem", "stanza", "cite", "epigraph", "annotation"):
                walk(ch)
    walk(sec)
    # verse lines inside <stanza> are separate lines of one paragraph
    return "\n\n".join(out)


def _fb2_chapters(sec: ET.Element, out: List[Chapter], parent_title: str = "") -> None:
    """Flatten the section tree: a section's own text first, then each sub-section as its own chapter."""
    title = _fb2_title(next((c for c in sec if _local(c.tag) == "title"), None)) or parent_title
    own = _fb2_section_text(sec)
    subs = [c for c in sec if _local(c.tag) == "section"]
    if own.strip():
        out.append(Chapter(title, own))
    for s in subs:
        _fb2_chapters(s, out, parent_title="")


def parse_fb2(data: bytes) -> Book:
    """Parse FictionBook 2 XML bytes."""
    root = _safe_xml(data)
    desc = next((c for c in root if _local(c.tag) == "description"), None)
    ti = next((c for c in desc if _local(c.tag) == "title-info"), None) if desc is not None else None
    title = author = lang = ""
    cover_href = ""
    if ti is not None:
        for c in ti:
            n = _local(c.tag)
            if n == "book-title":
                title = _squash(c.text or "")
            elif n == "author" and not author:
                author = _squash(" ".join(_squash(x.text or "") for x in c
                                          if _local(x.tag) in ("first-name", "middle-name", "last-name") and x.text))
            elif n == "lang":
                lang = _squash(c.text or "")
            elif n == "coverpage":
                for img in c:
                    cover_href = img.get("{http://www.w3.org/1999/xlink}href") or img.get("href") or cover_href
    bodies = [c for c in root if _local(c.tag) == "body"]
    main = next((b for b in bodies if not b.get("name")), bodies[0] if bodies else None)
    if main is None:
        raise BookParseError(tr("err.book_empty"))
    chapters: List[Chapter] = []
    body_text = _fb2_section_text(main)
    if body_text.strip():
        chapters.append(Chapter(_fb2_title(next((c for c in main if _local(c.tag) == "title"), None)), body_text))
    for sec in (c for c in main if _local(c.tag) == "section"):
        _fb2_chapters(sec, chapters)
    chapters = [c for c in chapters if c.text.strip()]
    if not chapters:
        raise BookParseError(tr("err.book_empty"))
    book = Book(title=title or chapters[0].title, author=author, language=lang, chapters=_with_default_title(chapters))
    if cover_href.startswith("#"):
        for b in root:
            if _local(b.tag) == "binary" and b.get("id") == cover_href[1:] and b.text:
                try:
                    raw = base64.b64decode(b.text.strip(), validate=False)
                except ValueError:
                    break
                ext = _image_ext(raw)
                if ext:
                    book.cover, book.cover_ext = raw, ext
    return book


# --------------------------------------------------------------------------- EPUB

class _HtmlText(HTMLParser):
    """HTML -> plain text with paragraph breaks; collects the first heading as a title candidate."""
    BLOCK = {"p", "div", "br", "li", "tr", "blockquote", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6",
             "pre", "hr", "dt", "dd"}
    SKIP = {"script", "style", "head", "title", "svg", "nav"}

    def __init__(self) -> None:
        """Start with empty buffers."""
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.heading = ""
        self._skip = 0
        self._in_heading = False
        self._heading_buf: List[str] = []

    def handle_starttag(self, tag: str, _attrs) -> None:  # noqa: D401
        """Track skipped regions, headings and paragraph breaks."""
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n\n" if tag != "br" else "\n")
            if tag in ("h1", "h2", "h3") and not self.heading:
                self._in_heading, self._heading_buf = True, []

    def handle_endtag(self, tag: str) -> None:  # noqa: D401
        """Leave skipped regions / headings."""
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n\n" if tag != "br" else "\n")
            if self._in_heading and tag in ("h1", "h2", "h3"):
                self._in_heading = False
                self.heading = _squash("".join(self._heading_buf))

    def handle_data(self, data: str) -> None:  # noqa: D401
        """Collect text."""
        if self._skip:
            return
        self.parts.append(data)
        if self._in_heading:
            self._heading_buf.append(data)

    def text(self) -> str:
        """The collected text: lines trimmed, paragraphs separated by one blank line."""
        raw = "".join(self.parts)
        paras = [_squash(p) for p in re.split(r"\n\s*\n", raw)]
        return "\n\n".join(p for p in paras if p)


def html_to_text(html: str) -> Tuple[str, str]:
    """``(plain text, first heading)`` of an (X)HTML document."""
    p = _HtmlText()
    p.feed(html)
    p.close()
    return p.text(), p.heading


def _epub_nav_titles(zf: zipfile.ZipFile, base: str, manifest: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """``{document path: title}`` from the EPUB 3 navigation document or the NCX."""
    titles: Dict[str, str] = {}
    for item in manifest.values():
        props, mt, href = item.get("properties", ""), item.get("type", ""), item["href"]
        path = posixpath.normpath(posixpath.join(base, href))
        try:
            if "nav" in props.split():
                nav = _read_member(zf, path).decode("utf-8", "replace")
                for m in re.finditer(r'<a\b[^>]*href="([^"#]+)[^"]*"[^>]*>(.*?)</a>', nav, re.S | re.I):
                    t = _squash(re.sub(r"<[^>]+>", "", m.group(2)))
                    titles.setdefault(posixpath.normpath(posixpath.join(posixpath.dirname(path), m.group(1))), t)
            elif mt == "application/x-dtbncx+xml":
                root = _safe_xml(_read_member(zf, path))
                for np_ in root.iter():
                    if _local(np_.tag) != "navPoint":
                        continue
                    label = next((t.text for t in np_.iter() if _local(t.tag) == "text" and t.text), "")
                    content = next((c for c in np_ if _local(c.tag) == "content"), None)
                    src_attr = content.get("src") if content is not None else None
                    if src_attr:
                        src = src_attr.split("#")[0]
                        titles.setdefault(posixpath.normpath(posixpath.join(posixpath.dirname(path), src)),
                                          _squash(label or ""))
        except (KeyError, BookParseError):
            continue
    return {k: v for k, v in titles.items() if v}


def parse_epub(path: Path) -> Book:
    """Parse an EPUB file (spine order; titles from the navigation, else the first heading)."""
    try:
        zf = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise BookParseError(tr("err.book_read"), details=str(exc)) from exc
    with zf:
        try:
            cont = _safe_xml(_read_member(zf, "META-INF/container.xml"))
            rootfile = next(e.get("full-path") for e in cont.iter() if _local(e.tag) == "rootfile")
            opf_path = cast(str, rootfile)
            opf = _safe_xml(_read_member(zf, opf_path))
        except (KeyError, StopIteration) as exc:
            raise BookParseError(tr("err.book_read"), details=str(exc)) from exc
        base = posixpath.dirname(opf_path)
        title = author = lang = cover_id = ""
        manifest: Dict[str, Dict[str, str]] = {}
        spine: List[str] = []
        for e in opf.iter():
            n = _local(e.tag)
            if n == "title" and not title:
                title = _squash(e.text or "")
            elif n == "creator" and not author:
                author = _squash(e.text or "")
            elif n == "language" and not lang:
                lang = _squash(e.text or "")
            elif n == "meta" and e.get("name") == "cover":
                cover_id = e.get("content") or ""
            elif n == "item":
                item_id, href = e.get("id"), e.get("href")
                if item_id and href:
                    manifest[item_id] = {"href": href, "type": e.get("media-type", ""),
                                         "properties": e.get("properties", "")}
            elif n == "itemref":
                idref = e.get("idref")
                if idref:
                    spine.append(idref)
        nav_titles = _epub_nav_titles(zf, base, manifest)
        chapters: List[Chapter] = []
        for idref in spine:
            item = manifest.get(idref)
            if not item or "html" not in item["type"]:
                continue
            doc = posixpath.normpath(posixpath.join(base, item["href"].split("#")[0]))
            try:
                raw = _read_member(zf, doc)
            except KeyError:
                continue
            html = decode_bytes(raw).text
            text, heading = html_to_text(html)
            nav_title = nav_titles.get(doc, "")
            if len(text) < MIN_CHAPTER_CHARS and not nav_title:
                continue
            if heading and text.startswith(heading):
                text = text[len(heading):].lstrip()           # the heading is spoken as the chapter title
            if text.strip():
                chapters.append(Chapter(nav_title or heading, text))
        if not chapters:
            raise BookParseError(tr("err.book_empty"))
        book = Book(title=title or Path(path).stem, author=author, language=lang, chapters=_with_default_title(chapters))
        cover_item = manifest.get(cover_id) or next(
            (i for i in manifest.values() if "cover-image" in i.get("properties", "").split()), None)
        if cover_item:
            try:
                raw = _read_member(zf, posixpath.normpath(posixpath.join(base, cover_item["href"])))
                ext = _image_ext(raw)
                if ext:
                    book.cover, book.cover_ext = raw, ext
            except KeyError:
                pass
    return book


# --------------------------------------------------------------------------- entry point

def load_book(path) -> Book:
    """Load a TXT / Markdown / FB2 / FB2.ZIP / EPUB file.  Raises :class:`BookParseError` with a user-readable message.

    ``.md`` is read as UTF-8 (or the same encodings as TXT) and split like plain text: ``#`` / ``##`` / ``###``
    headings are chapter titles, and a blank line separates paragraphs.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if not p.is_file():
        raise BookParseError(tr("err.book_read"), details=str(p))
    if ext == ".epub":
        return parse_epub(p)
    if ext == ".fb2":
        book = parse_fb2(p.read_bytes())
    elif ext == ".zip":
        try:
            with zipfile.ZipFile(p) as zf:
                name = next((n for n in zf.namelist() if n.lower().endswith(".fb2")), None)
                if name is None:
                    raise BookParseError(tr("err.book_unsupported"))
                book = parse_fb2(_read_member(zf, name))
        except zipfile.BadZipFile as exc:
            raise BookParseError(tr("err.book_read"), details=str(exc)) from exc
    elif ext in (".txt", ".md"):
        data = p.read_bytes()
        if not data.strip():
            raise BookParseError(tr("err.book_empty"))
        return parse_txt(decode_bytes(data).text, title=p.stem)
    else:
        raise BookParseError(tr("err.book_unsupported"))
    if not book.title:
        book.title = p.stem
    return book
