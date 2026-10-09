"""Speaker marks for multi-voice narration.

One pass of the text model (the same Gemma plan as literary translation) labels each paragraph ``narrator``, ``male``
or ``female``. The user can edit those labels before Start. When a male or a female voice is selected as well as the
narrator, narration synthesizes each paragraph with that voice. With only the narrator voice selected, the marks are
kept for the preview and every paragraph is spoken by the narrator.

The model is asked for tags, not a rewrite, so a bad answer cannot change the book. A paragraph it fails to mark stays
with the narrator.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from core.book_parsers import Book
from core.chunker import Chunk
from core.llm_text import NAMES, LLMPlan, fill, load_prompt
from core.pauses import TITLE

ROLES = ("narrator", "male", "female")
_TAG = re.compile(r"^(NARRATOR|MALE|FEMALE)(?:\s*:\s*(.*?))?\s*$", re.IGNORECASE)
_FENCE = re.compile(r"^```\w*\s*|\s*```$")
MAX_PARAS = 12
MAX_CHARS = 3500


@dataclass(frozen=True)
class SpeakerLine:
    """One paragraph: who speaks it, and the character's name when there is one."""
    role: str = "narrator"
    name: str = ""

    def __post_init__(self) -> None:
        role = self.role if self.role in ROLES else "narrator"
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "name", (self.name or "").strip()[:80])


@dataclass
class SpeakerCast:
    """Edited marks plus the library ids of the male and female voices ("" = the narrator's voice)."""
    lines: Optional[List[SpeakerLine]] = None
    male_id: str = ""
    female_id: str = ""
    tagger: Optional[LLMPlan] = None
    narrator_id: str = ""

    def extra_ids(self, narrator_id: str) -> List[str]:
        """Voice ids besides the narrator, in a stable order."""
        out: List[str] = []
        for vid in (self.male_id, self.female_id):
            if vid and vid != narrator_id and vid not in out:
                out.append(vid)
        return out

    def uses_several(self, narrator_id: str) -> bool:
        """True when at least one selected voice is not the narrator."""
        return bool(self.extra_ids(narrator_id))


def paragraphs(book: Book) -> List[Tuple[int, str]]:
    """``(chapter index, paragraph text)`` in reading order. Blank lines separate paragraphs."""
    out: List[Tuple[int, str]] = []
    for ci, ch in enumerate(book.chapters):
        for raw in re.split(r"\n\s*\n", ch.text.strip()):
            text = raw.strip()
            if text:
                out.append((ci, text))
    return out


def parse_tags(answer: str, count: int) -> Optional[List[SpeakerLine]]:
    """The model's answer as ``count`` lines, or ``None`` when the answer is not usable."""
    text = _FENCE.sub("", (answer or "").strip())
    rows = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(rows) != count:
        return None
    out: List[SpeakerLine] = []
    for ln in rows:
        m = _TAG.match(ln)
        if m is None:
            return None
        role = m.group(1).lower()
        if role != "narrator":
            role = "male" if role == "male" else "female"
        out.append(SpeakerLine(role, m.group(2) or ""))
    return out


def _batches(paras: Sequence[str]) -> List[List[int]]:
    out: List[List[int]] = []
    cur: List[int] = []
    size = 0
    for i, text in enumerate(paras):
        if cur and (len(cur) >= MAX_PARAS or size + len(text) > MAX_CHARS):
            out.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += len(text)
    if cur:
        out.append(cur)
    return out


def tag_paragraphs(paras: Sequence[str], language: str, plan: LLMPlan,
                   progress: Optional[Callable[[float], None]] = None) -> List[SpeakerLine]:
    """Mark ``paras``. A block the model fails is narrator. The model is closed before return."""
    progress = progress or (lambda _f: None)
    lines = [SpeakerLine() for _ in paras]
    if not any(any(c.isalpha() for c in p) for p in paras):
        return lines
    template = load_prompt("speaker_markup")
    model = None
    try:
        model = plan.factory()
        batches = _batches(list(paras))
        for bi, block in enumerate(batches):
            text = "\n\n".join(paras[i] for i in block)
            prompt = fill(template, language=NAMES.get(language, language or "English"), text=text)
            answer = model.complete(prompt, max_tokens=min(2048, 48 * len(block) + 32))
            parsed = parse_tags(answer or "", len(block))
            if parsed is not None:
                for i, line in zip(block, parsed):
                    lines[i] = line
            progress((bi + 1) / max(1, len(batches)))
    except Exception:  # noqa: BLE001 - a failed model leaves every paragraph with the narrator
        progress(1.0)
    finally:
        if model is not None:
            model.close()
    return lines


def _norm(text: str) -> str:
    return " ".join(text.split())


def assign(chunks: Sequence[Chunk], book: Book, lines: Sequence[SpeakerLine], voice_ids: Dict[str, str],
           narrator_id: str = "") -> Tuple[List[Chunk], str]:
    """Chunks with ``voice_id`` set from ``lines``.

    ``voice_ids`` maps ``male`` / ``female`` to a library id. The narrator and any role whose voice is the narrator
    stay on ``voice_id`` ``""`` (the job voice). When the paragraph count does not match ``lines``, the chunks are
    returned unchanged and the note is ``"mismatch"``.
    """
    paras = paragraphs(book)
    if len(paras) != len(lines):
        return list(chunks), "mismatch"
    rows = [(ci, _norm(text), line.role) for (ci, text), line in zip(paras, lines)]
    cursor = 0
    out: List[Chunk] = []
    for chunk in chunks:
        role = "narrator"
        if chunk.pause_kind != TITLE:
            wanted = _norm(chunk.text)
            for j in range(cursor, len(rows)):
                ci, ptext, r = rows[j]
                if ci == chunk.chapter and wanted and wanted in ptext:
                    role = r
                    cursor = j
                    break
        vid = voice_ids.get(role, "") if role in ("male", "female") else ""
        if vid == narrator_id:
            vid = ""
        out.append(chunk if chunk.voice_id == vid else replace(chunk, voice_id=vid))
    return out, ""


def describe(lines: Sequence[SpeakerLine]) -> str:
    """A short text file of the marks (written next to the job for debugging)."""
    return dump_marks(lines)


_MARK_PREFIX = re.compile(r"^\d+\.\s+")


def dump_marks(lines: Sequence[SpeakerLine]) -> str:
    """Editable marks: one numbered line per paragraph (``1. NARRATOR``, ``2. MALE: Ann``).

    The same text is written to ``.debug/speakers.txt`` and is what ``voxprint speakers`` writes.
    """
    rows = []
    for i, line in enumerate(lines, start=1):
        if line.role == "narrator":
            rows.append(f"{i}. NARRATOR")
        else:
            rows.append(f"{i}. {line.role.upper()}" + (f": {line.name}" if line.name else ""))
    return "\n".join(rows) + ("\n" if rows else "")


def load_marks(text: str) -> List[SpeakerLine]:
    """Read :func:`dump_marks` (or the same lines without numbers).

    A trailing ``mismatch`` note, as narration appends when the marks no longer fit the book, is ignored.
    Raises ``ValueError`` when a line is not a speaker mark or the file has none.
    """
    rows: List[SpeakerLine] = []
    for raw in (text or "").splitlines():
        ln = raw.strip()
        if not ln or ln.lower() == "mismatch":
            continue
        ln = _MARK_PREFIX.sub("", ln)
        m = _TAG.match(ln)
        if m is None:
            raise ValueError(f"unreadable speaker mark: {raw.strip()}")
        role = m.group(1).lower()
        if role != "narrator":
            role = "male" if role == "male" else "female"
        rows.append(SpeakerLine(role, m.group(2) or ""))
    if not rows:
        raise ValueError("speaker marks file is empty")
    return rows
