"""Text handling: reading, normalization, sentence splitting and mapping aligner words back onto the text.

The token "cleaning" rules mirror ``Qwen3ForceAlignProcessor`` from the ``qwen-asr`` package (checked against
qwen-asr 0.0.6): a token is a piece of text between spaces from which only letters (Unicode category ``L*``),
digits (``N*``) and the apostrophe are kept; for Chinese every character is its own token.  Because of that we can
match the aligner's words to the original text through the stream of "clean" characters, independent of language and
tokenizer.
"""
from __future__ import annotations

from core.i18n import tr
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from core.errors import AlignmentError, TextReadError
from core.types import WordTiming

# --------------------------------------------------------------------------- reading


@dataclass
class TextReadResult:
    """Decoded text plus the detected encoding and any user-facing warnings (e.g. "not UTF-8")."""
    text: str
    encoding: str
    warnings: List[str] = field(default_factory=list)


def decode_bytes(data: bytes) -> TextReadResult:
    """Decode bytes, preferring UTF-8 (BOM/UTF-16 aware); falls back to the common Russian code pages cp1251/cp866."""
    warnings: List[str] = []
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        warnings.append(tr("warn.text_utf16"))
        return TextReadResult(data.decode("utf-16"), "utf-16", warnings)
    if data.startswith(b"\xef\xbb\xbf"):
        return TextReadResult(data[3:].decode("utf-8"), "utf-8-sig", warnings)
    try:
        return TextReadResult(data.decode("utf-8"), "utf-8", warnings)
    except UnicodeDecodeError:
        pass
    for enc in ("cp1251", "cp866"):
        try:
            text = data.decode(enc)
        except UnicodeDecodeError:
            continue
        warnings.append(
            tr("warn.text_not_utf8", enc=enc)
        )
        return TextReadResult(text, enc, warnings)
    warnings.append(tr("warn.text_enc_unknown"))
    return TextReadResult(data.decode("utf-8", errors="replace"), "utf-8-replace", warnings)


def read_text_file(path) -> TextReadResult:
    """Read a text file, decode and normalize it.

    Raises ``TextReadError`` (friendly message) if the file cannot be opened, is empty, or has no letters/digits.
    """
    from pathlib import Path

    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise TextReadError(tr("err.text_open", path=p), details=str(exc)) from exc
    if not data.strip():
        raise TextReadError(tr("err.text_empty"))
    res = decode_bytes(data)
    res.text = normalize_text(res.text)
    if not any(is_kept_char(c) for c in res.text):
        raise TextReadError(tr("err.text_no_letters"))
    return res


_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)


def normalize_text(raw: str) -> str:
    """Remove junk (BOM, zero-width and control characters, odd spaces) but keep line breaks; collapse blank lines."""
    text = raw.translate(_ZERO_WIDTH).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def collapse_ws(text: str) -> str:
    """Replace any whitespace run (including newlines) by a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- tokens


def is_kept_char(ch: str) -> bool:
    """True for the characters that survive token cleaning: letters, digits and the apostrophe."""
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


def clean_token(token: str) -> str:
    """The token with everything except letters, digits and apostrophes removed (what the aligner sees)."""
    return "".join(ch for ch in token if is_kept_char(ch))


def count_clean_chars(text: str) -> int:
    """Number of letters/digits/apostrophes in ``text``."""
    return sum(1 for ch in text if is_kept_char(ch))


def detect_language(text: str) -> str:
    """Rough script-based language guess for the aligner (names as in qwen-asr).

    Counts Cyrillic, Latin, Han, kana and Hangul characters; anything not clearly one of the CJK/Russian cases is "English".
    """
    counts: Dict[str, int] = {"cyr": 0, "lat": 0, "han": 0, "kana": 0, "hangul": 0}
    for ch in text:
        o = ord(ch)
        if 0x0400 <= o <= 0x04FF:
            counts["cyr"] += 1
        elif ch.isalpha() and o < 0x250:
            counts["lat"] += 1
        elif 0x4E00 <= o <= 0x9FFF:
            counts["han"] += 1
        elif 0x3040 <= o <= 0x30FF:
            counts["kana"] += 1
        elif 0xAC00 <= o <= 0xD7AF:
            counts["hangul"] += 1
    if counts["kana"] > 0:
        return "Japanese"
    if counts["hangul"] > 0 and counts["hangul"] >= counts["han"]:
        return "Korean"
    if counts["han"] > 0 and counts["han"] >= max(counts["cyr"], counts["lat"]):
        return "Chinese"
    if counts["cyr"] >= counts["lat"] and counts["cyr"] > 0:
        return "Russian"
    return "English"


# --------------------------------------------------------------------------- sentences

_ABBREVIATIONS = {
    # Russian
    "т.д", "т.п", "т.е", "т.к", "т.н", "т.о", "т.ч", "и.о", "др", "пр", "г", "гг", "ул", "им", "см",
    "рис", "стр", "руб", "коп", "тыс", "млн", "млрд", "проф", "акад", "доц", "ред", "изд", "обл",
    "р", "пос", "с", "д", "кв", "тел", "напр", "англ", "нем", "франц", "лат", "в", "вв",
    # English
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "no", "fig",
    "inc", "ltd", "co", "approx",
}
# abbreviations after which a capital letter usually starts a new sentence
_END_CAPABLE = {"т.д", "т.п", "др", "пр", "etc"}
_TERMINATORS = ".!?…。！？"
_CLOSERS = "\"'»”’)]}›"
_BOUNDARY_RE = re.compile(r"([.!?…。！？]+[\"'»”’)\]}›]*)(\s+|$)")


def _last_word_before(text: str, end: int) -> str:
    """The whitespace-delimited token that ends at index ``end``."""
    m = re.search(r"([^\s]+)$", text[:end])
    return m.group(1) if m else ""


def _is_abbreviation(token: str) -> bool:
    """True if a token ending in "." is a known abbreviation, a chain like "т.д."/"e.g.", or an initial."""
    tok = token.strip(_CLOSERS + "«„“\"(").lower()
    if not tok.endswith("."):
        return False
    base = tok.rstrip(".")
    if not base:
        return False
    if base in _ABBREVIATIONS:
        return True
    # chains of single-letter parts such as "т.д.", "т.е.", "e.g."
    if re.fullmatch(r"(?:[^\W\d_]\.)+[^\W\d_]", base):
        return True
    # initials: a single capital letter
    if re.fullmatch(r"[^\W\d_]", base) and token.strip(_CLOSERS + "«„“\"(")[:1].isupper():
        return True
    return False


def split_sentences(text: str) -> List[str]:
    """Split text into sentences (Russian/English/CJK), taking abbreviations and numbers into account.

    A blank line (paragraph) always ends a sentence.  A single line break only ends one if sentence punctuation precedes
    it; lines without any punctuation (poems) are one sentence each.
    """
    text = normalize_text(text)
    if not text:
        return []
    sentences: List[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        para_flat = para.replace("\n", " ") if re.search(r"[.!?…。！？]", para) else para
        if para_flat is para and "\n" in para:
            # poems / lines without punctuation: every line is a separate sentence
            for line in para.split("\n"):
                if line.strip():
                    sentences.append(collapse_ws(line))
            continue
        start = 0
        for m in _BOUNDARY_RE.finditer(para_flat):
            end = m.end(1)
            tail = para_flat[m.end():]
            token = _last_word_before(para_flat, end)
            if tail:
                nxt = tail.lstrip()[:1]
                # a lower-case letter after "." continues the sentence
                if m.group(1).startswith(".") and nxt.islower():
                    continue
                if _is_abbreviation(token) and not (
                        nxt.isupper() and token.strip(_CLOSERS + "«„“\"(").lower().rstrip(".") in _END_CAPABLE):
                    # abbreviation: do not split (except at the very end of the text)
                    continue
                # decimal numbers / numbering: "3.5", "1. Item"
                if re.fullmatch(r"\d+\.", token) and nxt and not nxt.isupper():
                    continue
            piece = para_flat[start:end].strip()
            if piece:
                sentences.append(collapse_ws(piece))
            start = m.end()
        rest = para_flat[start:].strip()
        if rest:
            sentences.append(collapse_ws(rest))
    return sentences


_CLAUSE_RE = re.compile(r"(?<=[,;:])\s+|\s+(?=[—–]\s)|\n+")


def split_clauses(text: str) -> List[str]:
    """Split text into small units (sentences, then commas/dashes/line breaks, then at most ``MAX_CLAUSE_WORDS`` words).

    Used only for planning long recordings: the more cut points exist, the more precisely pauses in the audio can be
    matched to positions in the text.
    """
    out: List[str] = []
    for sent in split_sentences(text):
        parts = [p.strip() for p in _CLAUSE_RE.split(sent) if p and p.strip()]
        for part in (parts if parts else [sent]):
            out.extend(_wrap_words(part))
    return out


#: A clause longer than this many words is split into equal parts: text without punctuation (transcripts, subtitles)
#: would otherwise stay one huge "clause" and a long recording could not be cut into chunks for the aligner.
MAX_CLAUSE_WORDS = 14


def _wrap_words(part: str, limit: int = MAX_CLAUSE_WORDS) -> List[str]:
    """Split a clause with more than ``limit`` words into equally sized pieces."""
    words = part.split()
    if len(words) <= limit:
        return [part]
    n = -(-len(words) // limit)                      # number of parts (ceiling division)
    size = -(-len(words) // n)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


# --------------------------------------------------------------------------- attaching words to text


def attach_spans(words: Sequence[WordTiming], text: str) -> List[str]:
    """Fill ``char_start``/``char_end``/``sentence_end`` of the aligner's words using the stream of clean characters.

    Returns a list of warnings.  Raises ``AlignmentError`` if the aligner's words do not match the text (wrong text or
    a damaged file) or if too much of the text is left unmatched.  Spans are then widened over adjacent punctuation
    and quotes, and ``sentence_end`` is set when sentence punctuation follows the word.
    """
    kept_idx = [i for i, ch in enumerate(text) if is_kept_char(ch)]
    stream = "".join(text[i] for i in kept_idx)
    pos = 0
    warnings: List[str] = []
    for w in words:
        cw = clean_token(w.word)
        if not cw:
            w.char_start = w.char_end = None
            continue
        piece = stream[pos:pos + len(cw)]
        if piece != cw and piece.casefold() != cw.casefold():
            raise AlignmentError(
                tr("err.words_mismatch"),
                details=f"expected {cw!r} got {piece!r} at clean-pos {pos}",
            )
        w.char_start = kept_idx[pos]
        w.char_end = kept_idx[pos + len(cw) - 1] + 1
        pos += len(cw)
    if pos < len(stream):
        left = len(stream) - pos
        if left > max(5, int(0.02 * len(stream))):
            raise AlignmentError(
                tr("err.align_incomplete"),
                details=f"{left} clean chars left of {len(stream)}",
            )
        warnings.append(tr("warn.chars_unlabeled", left=left))
    # extend the spans over adjacent punctuation and set sentence_end
    n = len(text)
    for w in words:
        if w.char_start is None:
            continue
        e = w.char_end
        while e < n and not text[e].isspace() and not is_kept_char(text[e]):
            e += 1
        trail = text[w.char_end:e]
        w.sentence_end = any(c in _TERMINATORS for c in trail) or (e >= n)
        w.char_end = e
        s = w.char_start
        while s > 0 and (unicodedata.category(text[s - 1]) in ("Ps", "Pi") or text[s - 1] in "\"«„“"):
            s -= 1
        w.char_start = s
    return warnings


def text_for_words(text: str, words: Sequence[WordTiming]) -> str:
    """The original text (with punctuation) covered by a contiguous group of words."""
    spans = [(w.char_start, w.char_end) for w in words if w.char_start is not None]
    if not spans:
        return ""
    return collapse_ws(text[spans[0][0]: spans[-1][1]])
