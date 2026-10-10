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
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from core import pauses as pz
from core.pace import Pace
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
    pause_kind: str = ""          # comma | mid | sentence | ellipsis | dash | paragraph | scene | title (not in packed mode)
    tempo: float = 1.0            # reading-speed factor applied after synthesis (core/pace.py); 1.0 = as synthesized
    voice_id: str = ""            # library id of an extra voice; "" = the job's narrator voice (core/speakers.py)
    paragraph: int = 0            # 1-based counted paragraph (headings and scene breaks are 0); 0 for a title chunk


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
    if t.endswith((";", ":")):
        return pz.MID
    if t.endswith(","):
        return pz.COMMA
    return pz.SENTENCE


#: Conjunctions that open a new clause after a comma ("..., и ...", "..., but ..."): a strong transition (``mid`` pause).
CONJUNCTIONS = frozenset("""и а но да или либо однако зато поэтому потому чтобы хотя если когда пока
    і та але чи бо щоб якщо коли and but or so yet nor because although though while whereas
    und aber oder denn sondern doch weil obwohl während""".split())


def _comma_kind(piece: str, following: str) -> str:
    """``mid`` for a comma before a clause-opening conjunction, else the punctuation's own kind."""
    kind = _end_kind(piece)
    if kind == pz.COMMA:
        m = re.match(r"[\W_]*(\w+)", following)
        if m and m.group(1).lower() in CONJUNCTIONS:
            return pz.MID
    return kind


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
    for n, p in enumerate(merged):
        segs = [p] if len(p) <= limit else _wrap_long(p, limit)
        nxt = merged[n + 1] if n + 1 < len(merged) else ""
        for i, seg in enumerate(segs):                 # a forced cut inside a long piece is a comma-sized pause
            out.append((seg, _comma_kind(seg, nxt) if i == len(segs) - 1 else pz.COMMA))
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


# --------------------------------------------------------------------------- structured pauses (the default of the app)

#: A numbered / verse line ("1 В начале...", "2. ...", "1:3 ..."): its own paragraph-sized pause.
_NUMBERED_RE = re.compile(r"^\s*\d{1,3}(?::\d{1,3})?[.):]?\s+\S")
#: Strong transitions a sentence is cut at in structured mode: after a dash, colon, semicolon, or a comma that is followed by
#: a clause-opening conjunction.  Both sides must keep MIN_STRONG_CHARS so the voice never gets a tiny utterance.
MIN_STRONG_CHARS = 20
_STRONG_RE = re.compile(r"(?<=[;:])\s+|(?<=[—–])\s+|(?<=\s-)\s+|(?<=,)\s+(?=[\"'«„“]?(?:%s)\b)" % "|".join(
    sorted(CONJUNCTIONS, key=len, reverse=True)), re.IGNORECASE)


def blocks(text: str) -> List[str]:
    """Paragraphs of a chapter: blank-line separated, and every numbered / verse line on its own."""
    out: List[str] = []
    for para in re.split(r"\n\s*\n", text):
        lines = [ln for ln in para.split("\n") if ln.strip()]
        if len(lines) > 1 and sum(bool(_NUMBERED_RE.match(ln)) for ln in lines) >= 2:
            cur: List[str] = []
            for ln in lines:
                if _NUMBERED_RE.match(ln) and cur:
                    out.append("\n".join(cur))
                    cur = []
                cur.append(ln)
            out.append("\n".join(cur))
        else:
            out.append(para)
    return [b.strip() for b in out if b.strip()]


def _strong_pieces(sentence: str, limit: int) -> List[Tuple[str, str]]:
    """``(text, kind)`` of one sentence cut only at strong transitions (``mid``); a too-long piece is wrapped (``comma``)."""
    raw = [p for p in _STRONG_RE.split(sentence.strip()) if p and p.strip()] or [sentence.strip()]
    merged: List[str] = []
    for p in raw:
        if merged and (len(merged[-1]) < MIN_STRONG_CHARS or len(p.strip()) < MIN_STRONG_CHARS):
            merged[-1] = f"{merged[-1]} {p.strip()}"
        else:
            merged.append(p.strip())
    if len(merged) > 1 and len(merged[-1]) < MIN_STRONG_CHARS:
        tail = merged.pop()
        merged[-1] = f"{merged[-1]} {tail}"
    out: List[Tuple[str, str]] = []
    for p in merged:
        if len(p) <= limit:
            out.append((p, pz.MID))
        else:
            for seg in split_clauses(p) if len(split_clauses(p)) > 1 else [p]:
                for w in ([seg] if len(seg) <= limit else _wrap_long(seg, limit)):
                    out.append((w, pz.COMMA))
            out[-1] = (out[-1][0], pz.MID)
    kind = _end_kind(sentence)
    out[-1] = (out[-1][0], kind if kind in (pz.ELLIPSIS, pz.SENTENCE) else pz.SENTENCE)
    return out


def chunk_text_structured(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> List[Tuple[str, str]]:
    """``[(chunk text, pause kind after it)]``: one chunk per sentence, cut further only at strong transitions; paragraphs and
    numbered / verse lines end with ``paragraph``, a scene break with ``scene``.  Plain commas stay inside a chunk (the
    model phrases them), so pieces never get tiny."""
    out: List[Tuple[str, str]] = []
    for para in blocks(text):
        if not _HAS_WORD.search(para):
            if out:
                out[-1] = (out[-1][0], pz.SCENE)
            continue
        start = len(out)
        for sent in split_sentences(para.replace("\n", " ")) or [para]:
            out.extend(_strong_pieces(sent, max_chars))
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


def _paragraph_blocks(text: str) -> List[str]:
    """Blank-line blocks of one chapter, in order. This is the writer's paragraph, not a verse split."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _paragraph_id(para: str, ci: int, block: int, counter: List[int], skips: set) -> int:
    """1-based counted paragraph, or 0 for a scene break, a ``##`` subheading or a marked skip.

    ``counter`` is ``[next]`` and advances only for a counted paragraph, including chapters that are not spoken.
    """
    if (ci, block) in skips or not _HAS_WORD.search(para) or para.lstrip().startswith("##"):
        return 0
    n = counter[0]
    counter[0] = n + 1
    return n


def _scene_pause(chunks: List[Chunk], ci: int, pauses: "pz.PauseProfile | None",
                 lengths: "pz.PauseLengths | None") -> None:
    """A scene-break block lengthens the pause of the previous chunk in this chapter."""
    if not chunks or chunks[-1].chapter != ci:
        return
    last = chunks[-1]
    if pauses is not None:
        chunks[-1] = replace(last, pause_ms=pauses.ms(pz.SCENE), pause_kind=pz.SCENE)
    elif lengths is not None:
        chunks[-1] = replace(last, pause_ms=lengths.ms(pz.SCENE), pause_kind=pz.SCENE)
    else:
        chunks[-1] = replace(last, pause_ms=PAUSE_SCENE_MS)


def chunk_book(book: Book, max_chars: int = DEFAULT_MAX_CHARS, chapters: Sequence[int] = (),
               speak_titles: bool = False, pauses: "pz.PauseProfile | None" = None,
               lengths: "pz.PauseLengths | None" = None, pace: Optional[Pace] = None) -> List[Chunk]:
    """All chunks of the book in reading order with global indexes (optionally only the given chapter numbers).

    With ``speak_titles`` every chapter starts with a chunk that reads its title aloud.  With ``pauses`` the text is cut at
    every sentence, comma, dash and ellipsis and the silence between the pieces comes from the profile (explicit pauses);
    With ``lengths`` (the app's default) the text is cut per sentence and at strong transitions (structured pauses,
    :func:`chunk_text_structured`) and the silences come from ``lengths``; without both the earlier packed chunks with fixed
    pauses are produced.  ``pace`` (:class:`core.pace.Pace`) gives every chunk its reading-speed factor.
    """
    from core import pace as pc

    style = pc.book_style(book, pace.style) if pace is not None else ""

    def tempo(text: str, kind: str = "") -> float:
        return pc.segment_tempo(text, style, pace.speed, kind) if pace is not None else 1.0

    chunks: List[Chunk] = []
    counter = [1]
    skips = set(getattr(book, "unnumbered_blocks", ()) or ())
    for ci, ch in enumerate(book.chapters):
        blocks = _paragraph_blocks(ch.text)
        if chapters and ci not in chapters:
            for bi, para in enumerate(blocks):
                _paragraph_id(para, ci, bi, counter, skips)
            continue
        if speak_titles and ch.title.strip() and _HAS_WORD.search(ch.title):
            title = ch.title.strip()
            ttext = title if title[-1] in ".!?…" else title + "."
            ms = pauses.ms(pz.TITLE) if pauses else (lengths.ms(pz.TITLE) if lengths else PAUSE_AFTER_TITLE_MS)
            chunks.append(Chunk(len(chunks), ci, ttext, ms, pz.TITLE if (pauses or lengths) else "", tempo(ttext, pz.TITLE)))
        for bi, para in enumerate(blocks):
            pid = _paragraph_id(para, ci, bi, counter, skips)
            if not _HAS_WORD.search(para):
                _scene_pause(chunks, ci, pauses, lengths)
                continue
            if pauses is not None:
                for text, kind in chunk_text_pauses(para, max_chars):
                    chunks.append(Chunk(len(chunks), ci, text, pauses.ms(kind), kind, tempo(text, kind), paragraph=pid))
                continue
            if lengths is not None:
                for text, kind in chunk_text_structured(para, max_chars):
                    chunks.append(Chunk(len(chunks), ci, text, lengths.ms(kind), kind, tempo(text, kind), paragraph=pid))
                continue
            for text, pause in chunk_text(para, max_chars):
                chunks.append(Chunk(len(chunks), ci, text, pause, "", tempo(text), paragraph=pid))
    return chunks
