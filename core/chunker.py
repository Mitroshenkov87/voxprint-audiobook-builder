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

from core import pauses as pz
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
    pause_kind: str = ""          # comma | sentence | ellipsis | dash | paragraph | scene | title (explicit pauses only)


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


# --------------------------------------------------------------------------- explicit pauses

#: Smallest piece (characters) that is cut off at a comma / dash / semicolon: shorter ones ("Well,") stay with the next words,
#: because a tiny separate utterance sounds unnatural and some engines are unstable on it.
MIN_INNER_CHARS = 14
_INNER_RE = re.compile(r"(?<=[,;:])\s+|(?<=\.\.\.)\s+(?=[^\W\d_])|(?<=…)\s+(?=[^\W\d_])|(?<=[—–])\s+|(?<=\s-)\s+")
_CLOSERS = "\"'»”’)]}"


def _end_kind(piece: str) -> str:
    """The kind of pause a piece asks for from the punctuation it ends with."""
    t = piece.rstrip().rstrip(_CLOSERS).rstrip()
    if t.endswith("…") or t.endswith("..."):
        return pz.ELLIPSIS
    if t.endswith(("—", "–")) or t.endswith(" -"):
        return pz.DASH
    if t.endswith((",", ";", ":")):
        return pz.COMMA
    return pz.SENTENCE


def _inner_pieces(sentence: str, limit: int) -> List[Tuple[str, str]]:
    """``(text, kind)`` pieces of one sentence cut at commas, dashes and ellipses; the last piece takes the sentence's kind."""
    raw = [p.strip() for p in _INNER_RE.split(sentence) if p and p.strip()] or [sentence]
    merged: List[str] = []
    carry = ""
    for p in raw:
        cur = f"{carry} {p}".strip() if carry else p
        if len(cur) < MIN_INNER_CHARS and p is not raw[-1]:
            carry = cur
            continue
        merged.append(cur)
        carry = ""
    if carry:
        merged.append(carry)
    if len(merged) > 1 and len(merged[-1]) < MIN_INNER_CHARS:      # a short tail joins the previous piece
        tail = merged.pop()
        merged[-1] = f"{merged[-1]} {tail}"
    out: List[Tuple[str, str]] = []
    for p in merged:
        segs = [p] if len(p) <= limit else _wrap_long(p, limit)
        for i, seg in enumerate(segs):                 # a forced cut inside a long piece is a comma-sized pause
            out.append((seg, _end_kind(seg) if i == len(segs) - 1 else pz.COMMA))
    if out:                                            # the sentence end wins over a trailing comma-like mark
        kind = _end_kind(sentence)
        out[-1] = (out[-1][0], pz.SENTENCE if kind == pz.COMMA else kind)
    return out


def chunk_text_pauses(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> List[Tuple[str, str]]:
    """``[(chunk text, pause kind after it)]``: every sentence (and every comma / dash / ellipsis part) is its own chunk."""
    out: List[Tuple[str, str]] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if not _HAS_WORD.search(para):               # "* * *", "---": a scene break lengthens the previous pause
            if out:
                out[-1] = (out[-1][0], pz.SCENE)
            continue
        start = len(out)
        for sent in split_sentences(para) or [para]:
            out.extend(_inner_pieces(sent, max_chars))
        if len(out) > start:
            out[-1] = (out[-1][0], pz.PARAGRAPH)
    return out


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
               speak_titles: bool = False, pauses: "pz.PauseProfile | None" = None) -> List[Chunk]:
    """All chunks of the book in reading order with global indexes (optionally only the given chapter numbers).

    With ``speak_titles`` every chapter starts with a chunk that reads its title aloud.  With ``pauses`` the text is cut at
    every sentence, comma, dash and ellipsis and the silence between the pieces comes from the profile (explicit pauses);
    without it the earlier packed chunks with fixed pauses are produced.
    """
    chunks: List[Chunk] = []
    for ci, ch in enumerate(book.chapters):
        if chapters and ci not in chapters:
            continue
        if speak_titles and ch.title.strip() and _HAS_WORD.search(ch.title):
            title = ch.title.strip()
            ttext = title if title[-1] in ".!?…" else title + "."
            chunks.append(Chunk(len(chunks), ci, ttext, pauses.ms(pz.TITLE) if pauses else PAUSE_AFTER_TITLE_MS,
                                pz.TITLE if pauses else ""))
        if pauses is not None:
            for text, kind in chunk_text_pauses(ch.text, max_chars):
                chunks.append(Chunk(len(chunks), ci, text, pauses.ms(kind), kind))
            continue
        for text, pause in chunk_text(ch.text, max_chars):
            chunks.append(Chunk(len(chunks), ci, text, pause))
    return chunks
