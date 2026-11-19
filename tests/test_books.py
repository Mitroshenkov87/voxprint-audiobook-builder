"""Book parsers (TXT / FB2 / FB2.ZIP / EPUB) and the chunker."""
import base64
import zipfile

import pytest

from core import chunker
from core.book_parsers import Book, Chapter, html_to_text, load_book, parse_fb2, parse_txt
from core.errors import BookParseError

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 20


# ----------------------------------------------------------------------------- TXT

def test_txt_chapters_from_headings():
    text = ("Prologue text before anything else is long enough.\n\n"
            "Chapter 1\n\nFirst paragraph of one.\nstill the same paragraph.\n\nSecond paragraph.\n\n"
            "Глава 2\n\nТекст второй главы.\n\n"
            "III\n\nThird chapter text.\n")
    book = parse_txt(text, title="My Book")
    titles = [c.title for c in book.chapters]
    assert titles[1:] == ["Chapter 1", "Глава 2", "III"] and len(book.chapters) == 4
    assert book.chapters[1].text == "First paragraph of one. still the same paragraph.\n\nSecond paragraph."
    assert book.title == "My Book" and book.total_chars > 50


def test_txt_without_headings_is_one_chapter_and_empty_is_an_error():
    book = parse_txt("Just a story.\n\nIt has two paragraphs.", title="Story")
    assert [c.title for c in book.chapters] == ["Story"] and book.chapters[0].text.count("\n\n") == 1
    with pytest.raises(BookParseError):
        parse_txt("   \n\n  ")


def test_txt_headings_must_stand_alone_and_verse_keeps_lines():
    book = parse_txt("He said:\nChapter and verse were quoted in the middle of a paragraph.\n\nThe end.", "T")
    assert len(book.chapters) == 1
    verse = parse_txt("Roses\nare red\nviolets\nare blue\nsugar\nis sweet", "V")
    assert verse.chapters[0].text.count("\n") == 5


def test_load_book_txt_detects_encoding(tmp_path):
    f = tmp_path / "kniga.txt"
    f.write_bytes("Глава 1\n\nПривет, мир. Это текст в кодировке windows-1251.".encode("cp1251"))
    book = load_book(f)
    assert book.title == "kniga" and "Привет" in book.chapters[0].text


# ----------------------------------------------------------------------------- FB2

FB2 = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0" xmlns:l="http://www.w3.org/1999/xlink">
<description><title-info><author><first-name>Лев</first-name><last-name>Толстой</last-name></author>
<book-title>Война</book-title><lang>ru</lang><coverpage><image l:href="#cover.png"/></coverpage></title-info></description>
<body>
<section><title><p>Часть первая</p></title>
  <section><title><p>Глава 1</p></title><p>Первый абзац<a l:href="#n1" type="note">[1]</a>.</p><p>Второй <emphasis>абзац</emphasis>.</p></section>
  <section><title><p>Глава 2</p></title><poem><stanza><v>Строка раз</v><v>строка два</v></stanza></poem></section>
</section>
<section><title><p>Эпилог</p></title><p>Конец.</p></section>
</body>
<body name="notes"><section id="n1"><p>Это примечание, его не нужно читать.</p></section></body>
<binary id="cover.png" content-type="image/png">%s</binary>
</FictionBook>""" % base64.b64encode(PNG).decode()


def test_fb2_metadata_chapters_notes_and_cover():
    book = parse_fb2(FB2.encode("utf-8"))
    assert book.title == "Война" and book.author == "Лев Толстой" and book.language == "ru"
    assert [c.title for c in book.chapters] == ["Глава 1", "Глава 2", "Эпилог"]
    assert book.chapters[0].text == "Первый абзац.\n\nВторой абзац."          # footnote marker removed
    assert "Строка раз" in book.chapters[1].text and "примечание" not in " ".join(c.text for c in book.chapters)
    assert book.cover == PNG and book.cover_ext == "png"


def test_fb2_windows_1251_and_zip(tmp_path):
    raw = FB2.replace('encoding="utf-8"', 'encoding="windows-1251"').encode("cp1251", errors="replace")
    f = tmp_path / "w.fb2"
    f.write_bytes(raw)
    assert load_book(f).title == "Война"
    z = tmp_path / "b.fb2.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("book.fb2", FB2.encode("utf-8"))
    assert load_book(z).author == "Лев Толстой"


def test_fb2_entity_bomb_and_garbage_are_rejected():
    bomb = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;&a;">]><FictionBook><body/></FictionBook>'
    with pytest.raises(BookParseError):
        parse_fb2(bomb)
    with pytest.raises(BookParseError):
        parse_fb2(b"<not xml")
    with pytest.raises(BookParseError):
        parse_fb2(b'<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"><body><section/></body></FictionBook>')


# ----------------------------------------------------------------------------- EPUB

def make_epub(path, with_nav=True, ncx=False):
    container = ('<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                 '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
    items = ['<item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>',
             '<item id="c2" href="ch2.xhtml" media-type="application/xhtml+xml"/>',
             '<item id="title" href="title.xhtml" media-type="application/xhtml+xml"/>',
             '<item id="cov" href="cover.png" media-type="image/png" properties="cover-image"/>']
    if with_nav:
        items.append('<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>')
    if ncx:
        items.append('<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>')
    opf = ('<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
           '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Test Book</dc:title><dc:creator>Jane Roe</dc:creator>'
           '<dc:language>en</dc:language></metadata><manifest>' + "".join(items) +
           '</manifest><spine><itemref idref="title"/><itemref idref="c1"/><itemref idref="c2"/></spine></package>')
    html = lambda h, body: f'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>x</title></head><body><h1>{h}</h1>{body}</body></html>'
    nav = ('<html xmlns:epub="http://www.idpf.org/2007/ops"><body><nav epub:type="toc"><ol>'
           '<li><a href="ch1.xhtml">The Beginning</a></li><li><a href="ch2.xhtml#s">The End</a></li></ol></nav></body></html>')
    ncx_doc = ('<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap><navPoint><navLabel><text>NCX One</text></navLabel>'
               '<content src="ch1.xhtml"/></navPoint></navMap></ncx>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml", container)
        z.writestr("OEBPS/content.opf", opf)
        z.writestr("OEBPS/title.xhtml", html("T", "<p>x</p>"))
        z.writestr("OEBPS/ch1.xhtml", html("Heading One", "<p>First   paragraph &amp; more text here, enough to count.</p><p>Second one is here too.</p>"))
        z.writestr("OEBPS/ch2.xhtml", html("Heading Two", "<div>Body of the second chapter<br/>with a line break and enough words.</div><script>bad()</script>"))
        z.writestr("OEBPS/nav.xhtml", nav)
        z.writestr("OEBPS/toc.ncx", ncx_doc)
        z.writestr("OEBPS/cover.png", PNG)
    return path


def test_epub_titles_from_nav_text_and_cover(tmp_path):
    book = load_book(make_epub(tmp_path / "b.epub"))
    assert (book.title, book.author, book.language) == ("Test Book", "Jane Roe", "en")
    assert [c.title for c in book.chapters] == ["The Beginning", "The End"]       # title page dropped (too short)
    assert book.chapters[0].text == "First paragraph & more text here, enough to count.\n\nSecond one is here too."
    assert "bad()" not in book.chapters[1].text and "with a line break" in book.chapters[1].text
    assert book.cover == PNG and book.cover_ext == "png"


def test_epub_ncx_and_heading_fallback(tmp_path):
    book = load_book(make_epub(tmp_path / "n.epub", with_nav=False, ncx=True))
    assert [c.title for c in book.chapters] == ["NCX One", "Heading Two"]


def test_html_to_text_blocks_and_heading():
    text, heading = html_to_text("<h2>Title</h2><p>One<br>two</p><ul><li>a</li><li>b</li></ul><style>x{}</style>")
    assert heading == "Title" and text.split("\n\n") == ["Title", "One two", "a", "b"]


def test_bad_inputs(tmp_path):
    with pytest.raises(BookParseError):
        load_book(tmp_path / "missing.txt")
    (tmp_path / "x.docx").write_bytes(b"x")
    with pytest.raises(BookParseError):
        load_book(tmp_path / "x.docx")
    (tmp_path / "bad.epub").write_bytes(b"not a zip")
    with pytest.raises(BookParseError):
        load_book(tmp_path / "bad.epub")
    (tmp_path / "e.txt").write_bytes(b"  \n ")
    with pytest.raises(BookParseError):
        load_book(tmp_path / "e.txt")
    with zipfile.ZipFile(tmp_path / "nofb2.zip", "w") as z:
        z.writestr("a.txt", "x")
    with pytest.raises(BookParseError):
        load_book(tmp_path / "nofb2.zip")


# ----------------------------------------------------------------------------- chunker

def test_chunks_respect_the_limit_and_keep_all_words():
    sent = "This is a sentence number {i} with some ordinary words in it."
    text = "\n\n".join(" ".join(sent.format(i=i) for i in range(p * 5, p * 5 + 5)) for p in range(3))
    out = chunker.chunk_text(text, max_chars=140)
    assert all(len(t) <= 140 for t, _ in out) and len(out) > 3
    assert " ".join(t for t, _ in out).split() == text.split()


def test_pauses_follow_boundaries():
    out = chunker.chunk_text("Short one. Short two.\n\nSecond paragraph.\n\n* * *\n\nAfter the break.", max_chars=60)
    assert out[0] == ("Short one. Short two.", chunker.PAUSE_PARAGRAPH_MS)
    assert out[1] == ("Second paragraph.", chunker.PAUSE_SCENE_MS)               # the scene break lengthens the pause
    assert out[2] == ("After the break.", chunker.PAUSE_PARAGRAPH_MS)
    mid = chunker.chunk_text("One. Two. Three. Four.", max_chars=10)
    assert [p for _, p in mid[:-1]] == [chunker.PAUSE_SENTENCE_MS] * (len(mid) - 1)
    assert mid[-1][1] == chunker.PAUSE_PARAGRAPH_MS


def test_long_sentence_is_cut_at_clauses_then_words():
    long = "Alpha beta gamma, delta epsilon zeta, eta theta iota, kappa lambda mu, nu xi omicron."
    out = chunker.chunk_text(long, max_chars=30)
    assert len(out) > 1 and all(len(t) <= 30 for t, _ in out) and " ".join(t for t, _ in out) == long
    assert out[0][1] == chunker.PAUSE_CLAUSE_MS and out[-1][1] == chunker.PAUSE_PARAGRAPH_MS
    glued = chunker.chunk_text("x" * 100, max_chars=30)                           # no spaces at all
    assert [len(t) for t, _ in glued] == [30, 30, 30, 10]


def test_chunk_book_indexes_chapters_and_titles():
    book = Book("B", chapters=[Chapter("One", "A. B."), Chapter("", "C."), Chapter("Three", "D.")])
    plain = chunker.chunk_book(book, 100)
    assert [c.index for c in plain] == [0, 1, 2] and [c.chapter for c in plain] == [0, 1, 2]
    titled = chunker.chunk_book(book, 100, speak_titles=True)
    assert [c.text for c in titled] == ["One.", "A. B.", "C.", "Three.", "D."]
    only = chunker.chunk_book(book, 100, chapters=[2])
    assert [c.text for c in only] == ["D."] and only[0].index == 0 and only[0].chapter == 2


def test_russian_text_is_chunked_too():
    text = "Он пришёл домой. Т. е. очень поздно, около 3.5 часов ночи. Никто не спал!"
    out = chunker.chunk_text(text, max_chars=200)
    assert len(out) == 1 and out[0][0] == text
