"""Chapter text -> sentence-sized chunks for the TTS engine, each with the pause that should follow it.

A chunk is what the engine synthesizes in one call.  Short chunks keep the voice stable and make resuming cheap, but
too many tiny chunks sound choppy, so sentences of a paragraph are packed greedily up to ``max_chars``.  A sentence
longer than the limit is cut at clause boundaries (:func:`core.text_utils.split_clauses`, the same splitter the aligner
uses) and, as a last resort, by words.

Pauses (milliseconds) follow the kind of boundary: mid-sentence cut < sentence end < paragraph end < scene break
(a paragraph made only of symbols such as ``* * *``).  The assembler adds them as silence between chunks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from core.book_parsers import Book
from core.text_utils import split_clauses, split_sentences

DEFAULT_MAX_CHARS = 260
PAUSE_CLAUSE_MS = 150
PAUSE_SENTENCE_MS = 300
PAUSE_PARAGRAPH_MS = 750
PAUSE_SCENE_MS = 1500
PAUSE_CHAPTER_MS = 2000       # silence between chapters inside a single-file export

_HAS_WORD = re.compile(r"\w", re.UNICODE)


@dataclass(frozen=True)
class Chunk:
    """One synthesis unit: ``index`` is global (0-based), ``chapter`` the 0-based chapter number."""
    index: int
    chapter: int
    text: str
    pause_ms: int


def _wrap_long(text: str, limit: int) -> List[str]:
    """Cut a piece that has no usable clause boundary into <= ``limit`` characters at spaces."""
    out: List[str] = []
    cur = ""
    for w in text.split():
        while len(w) > limit:                       # a "word" longer than the limit (no spaces at all)
            if cur:
                out.append(cur)
                cur = ""
            out.append(w[:limit])
            w = w[limit:]
        if cur and len(cur) + 1 + len(w) > limit:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def _pieces(paragraph: str, limit: int) -> List[Tuple[str, bool]]:
    """``(text, ends_sentence)`` pieces of one paragraph, each <= ``limit`` characters, sentences packed together."""
    pieces: List[Tuple[str, bool]] = []
    cur, cur_ends = "", False
    for sent in split_sentences(paragraph) or [paragraph]:
        parts: List[Tuple[str, bool]]
        if len(sent) <= limit:
            parts = [(sent, True)]
        else:
            parts = []
            for clause in split_clauses(sent):
                for seg in ([clause] if len(clause) <= limit else _wrap_long(clause, limit)):
                    parts.append((seg, False))
            if parts:
                parts[-1] = (parts[-1][0], True)
        for text, ends in parts:
            if cur and len(cur) + 1 + len(text) <= limit:
                cur, cur_ends = f"{cur} {text}", ends
            else:
                if cur:
                    pieces.append((cur, cur_ends))
                cur, cur_ends = text, ends
    if cur:
        pieces.append((cur, cur_ends))
    return pieces


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> List[Tuple[str, int]]:
    """``[(chunk text, pause after in ms)]`` for one chapter's text."""
    out: List[Tuple[str, int]] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if not _HAS_WORD.search(para):               # "* * *", "---": a scene break lengthens the previous pause
            if out:
                out[-1] = (out[-1][0], PAUSE_SCENE_MS)
            continue
        pieces = _pieces(para, max_chars)
        for i, (t, ends_sentence) in enumerate(pieces):
            last = i == len(pieces) - 1
            pause = PAUSE_PARAGRAPH_MS if last else (PAUSE_SENTENCE_MS if ends_sentence else PAUSE_CLAUSE_MS)
            out.append((t, pause))
    return out


PAUSE_AFTER_TITLE_MS = 900


def chunk_book(book: Book, max_chars: int = DEFAULT_MAX_CHARS, chapters: Sequence[int] = (),
               speak_titles: bool = False) -> List[Chunk]:
    """All chunks of the book in reading order with global indexes (optionally only the given chapter numbers).

    With ``speak_titles`` every chapter starts with a chunk that reads its title aloud.
    """
    chunks: List[Chunk] = []
    for ci, ch in enumerate(book.chapters):
        if chapters and ci not in chapters:
            continue
        if speak_titles and ch.title.strip() and _HAS_WORD.search(ch.title):
            title = ch.title.strip()
            chunks.append(Chunk(len(chunks), ci, title if title[-1] in ".!?…" else title + ".", PAUSE_AFTER_TITLE_MS))
        for text, pause in chunk_text(ch.text, max_chars):
            chunks.append(Chunk(len(chunks), ci, text, pause))
    return chunks
