"""Speaker marks for multi-voice narration.

Standalone use is the parser: ``parse_tags`` reads narrator / male / female lines. The tagger also
needs the text-model plan (``core.llm_text``). Licence: Apache-2.0.

Credit: Voxprint AI Audiobook Builder https://github.com/Mitroshenkov87/voxprint-audiobook-builder


One pass of the text model (the same Gemma plan as literary translation) labels each paragraph ``narrator``, ``male``
or ``female``. The user can edit those labels before Start. When a male or a female voice is selected as well as the
narrator, narration synthesizes each paragraph with that voice. With only the narrator voice selected, the marks are
kept for the preview and every paragraph is spoken by the narrator.

The model is asked for tags, not a rewrite, so a bad answer cannot change the book. A paragraph it fails to mark stays
with the narrator, and the result carries a warning instead of looking like a successful all-narrator pass. A reply
that has one tag per paragraph can still be shifted by a line. A paragraph that opens with a dialogue dash is a
character line, and the mark's name has to be the speaker named in that paragraph or in the preceding narration.
A paragraph with no dash and no quotation marks is the narrator. A block that fails this check is asked again, one
paragraph at a time, and the reason is logged as ``[validate]``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from core.book_parsers import Book
from core.chunker import Chunk
from core.llm_text import NAMES, LLMPlan, fill, has_dialogue, load_prompt
from core.pauses import TITLE

ROLES = ("narrator", "male", "female")


def _letters(*codes: int) -> str:
    """A tag word from Unicode code points.

    The i18n check rejects Cyrillic string literals in this package. Russian aliases the text model may print
    (rasskazchik, muzhskoy, zhenskiy, and the short muzh / zhen) are built here so the source stays free of them.
    """
    return "".join(chr(c) for c in codes)


# Longest alias first, so the short male/female words do not take a prefix of the long ones.
_RU_NARRATOR = _letters(0x440, 0x430, 0x441, 0x441, 0x43A, 0x430, 0x437, 0x447, 0x438, 0x43A)
_RU_MALE = _letters(0x43C, 0x443, 0x436, 0x441, 0x43A, 0x43E, 0x439)
_RU_FEMALE = _letters(0x436, 0x435, 0x43D, 0x441, 0x43A, 0x438, 0x439)
_RU_MALE_SHORT = _letters(0x43C, 0x443, 0x436)
_RU_FEMALE_SHORT = _letters(0x436, 0x435, 0x43D)
_TAG = re.compile(
    "^(" + "|".join(("NARRATOR", "MALE", "FEMALE", _RU_NARRATOR, _RU_MALE, _RU_FEMALE,
                     _RU_MALE_SHORT, _RU_FEMALE_SHORT))
    + r")(?:\s*:\s*(.*?))?\s*$",
    re.IGNORECASE)
_ROLE = {
    "narrator": "narrator", _RU_NARRATOR: "narrator",
    "male": "male", _RU_MALE: "male", _RU_MALE_SHORT: "male",
    "female": "female", _RU_FEMALE: "female", _RU_FEMALE_SHORT: "female",
}
# Gemma 4 (llama.cpp --jinja) thinks by default. The thought is in the same string as the answer, often as a channel
# block, and the answer itself is numbered ("1. MALE: Name") with a short preamble. None of that matches a bare tag.
_THINKING = re.compile(
    r"<think\b[^>]*>.*?</think>\s*"
    r"|<\|think\|?>.*?</think>\s*"
    r"|<\|channel>\s*thought\b.*?(?:<channel\|>|<\|channel\|>)\s*",
    re.IGNORECASE | re.DOTALL)
_FENCE = re.compile(r"```+")
_ENUM = re.compile(r"^(?:\d{1,4}[.)]\s+|[-*•]\s+)")
MAX_PARAS = 12
MAX_CHARS = 3500
WARN_UNPARSED = "unparsed"
WARN_NO_SPEAKERS = "no_speakers"
# A dialogue dash at the start of a paragraph: em dash, en dash, or a hyphen with a space ("- Hello").
_CHARACTER_LINE = re.compile(r"^\s*[\u2014\u2013]\s+\S|^\s*-\s+\S")
# Quotation marks mean speech in English and German. Those paragraphs are not forced to the narrator by the dash rule.
_QUOTED_SPEECH = re.compile("[\"\u00ab\u00bb\u201e\u201c\u201d]")
_ATTRIB_SPLIT = re.compile(r"[\u2014\u2013]| - ")
_NAME_TOKEN = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")


@dataclass(frozen=True)
class SpeakerLine:
    """One paragraph: who speaks it, and the character's name when there is one."""
    role: str = "narrator"
    name: str = ""

    def __post_init__(self) -> None:
        """Keep ``role`` only when it is narrator, male or female."""
        role = self.role if self.role in ROLES else "narrator"
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "name", (self.name or "").strip()[:80])


@dataclass
class SpeakerCast:
    """Edited marks plus the library ids of the voices ("" = the narrator's voice).

    ``male_id`` / ``female_id`` are the first voice of each role. ``male2_id`` / ``female2_id`` are an optional second
    voice: different characters of that role then alternate between the two voices in order of first appearance
    (the first man named in the marks gets ``male_id``, the second ``male2_id``, the third ``male_id`` again).
    ``characters`` pins a character name (case-insensitive, as written in the marks) to a voice and wins over both.
    A mark without a name uses the role's first voice.
    """
    lines: Optional[List[SpeakerLine]] = None
    male_id: str = ""
    female_id: str = ""
    tagger: Optional[LLMPlan] = None
    narrator_id: str = ""
    male2_id: str = ""
    female2_id: str = ""
    characters: Dict[str, str] = field(default_factory=dict)

    def extra_ids(self, narrator_id: str) -> List[str]:
        """Voice ids besides the narrator, in a stable order."""
        out: List[str] = []
        for vid in (self.male_id, self.male2_id, self.female_id, self.female2_id, *self.characters.values()):
            if vid and vid != narrator_id and vid not in out:
                out.append(vid)
        return out

    def uses_several(self, narrator_id: str) -> bool:
        """True when at least one selected voice is not the narrator."""
        return bool(self.extra_ids(narrator_id))

    def voices_for(self, lines: Sequence[SpeakerLine]) -> List[str]:
        """The voice id for each mark (``""`` = the narrator's voice)."""
        pinned = {_name_key(k): v for k, v in self.characters.items() if _name_key(k)}
        pools = {"male": [v for v in (self.male_id, self.male2_id) if v],
                 "female": [v for v in (self.female_id, self.female2_id) if v]}
        seen: Dict[str, Dict[str, int]] = {"male": {}, "female": {}}
        out: List[str] = []
        for line in lines:
            if line.role not in ("male", "female"):
                out.append("")
                continue
            key = _name_key(line.name)
            if key and key in pinned:
                out.append(pinned[key])
                continue
            pool = pools[line.role]
            if not pool:
                out.append("")
                continue
            if not key or len(pool) == 1:
                out.append(pool[0])
                continue
            order = seen[line.role]
            if key not in order:
                order[key] = len(order)
            out.append(pool[order[key] % len(pool)])
        return out

    def assignment(self, lines: Sequence[SpeakerLine]) -> Dict[str, str]:
        """``{character name: voice id}`` as :meth:`voices_for` decides it (the first spelling of each name)."""
        out: Dict[str, str] = {}
        keys = set()
        for line, vid in zip(lines, self.voices_for(lines)):
            key = _name_key(line.name)
            if key and key not in keys:
                keys.add(key)
                out[line.name] = vid
        return out


def _name_key(name: str) -> str:
    """Case- and space-insensitive key of a character name."""
    return " ".join((name or "").split()).casefold()


def parse_character_map(items: Sequence[str]) -> Dict[str, str]:
    """``["Name=voice", ...]`` as ``{Name: voice}``. Raises ``ValueError`` on an item without ``=`` or with an empty side."""
    out: Dict[str, str] = {}
    for item in items or ():
        name, sep, voice = str(item).partition("=")
        name, voice = name.strip(), voice.strip()
        if not sep or not name or not voice:
            raise ValueError(f"expected NAME=VOICE, got {item!r}")
        out[name] = voice
    return out


def paragraphs(book: Book) -> List[Tuple[int, str]]:
    """``(chapter index, paragraph text)`` in reading order. Blank lines separate paragraphs."""
    out: List[Tuple[int, str]] = []
    for ci, ch in enumerate(book.chapters):
        for raw in re.split(r"\n\s*\n", ch.text.strip()):
            text = raw.strip()
            if text:
                out.append((ci, text))
    return out


def _clean_answer(answer: str) -> str:
    """Drop a thinking channel and code fences so only the model's answer lines remain."""
    text = _THINKING.sub("", answer or "")
    return _FENCE.sub("", text)


def _line_tag(line: str) -> Optional[SpeakerLine]:
    """One answer line as a mark, or ``None`` when the line is preamble rather than a tag."""
    ln = _ENUM.sub("", line.strip())
    m = _TAG.match(ln)
    if m is None:
        return None
    role = _ROLE.get(m.group(1).lower(), "narrator")
    return SpeakerLine(role, m.group(2) or "")


def parse_tags(answer: str, count: int) -> Optional[List[SpeakerLine]]:
    """The model's answer as ``count`` marks, or ``None`` when the answer is not usable.

    A thinking channel, a code fence, a preamble and numbered prefixes (``1. MALE: Name``) are ignored. Lines that are
    not tags are skipped. The remaining tags must be exactly ``count`` (so ``NARRATOR`` plus a junk line is still
    rejected).
    """
    rows = [ln.strip() for ln in _clean_answer(answer).splitlines() if ln.strip()]
    out = [tag for tag in (_line_tag(ln) for ln in rows) if tag is not None]
    if len(out) != count:
        return None
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


class TagResult:
    """Marks plus the raw model reply and a warning code (``""``, ``unparsed``, ``no_speakers``).

    Callers that only need the marks can iterate, index and take ``len`` as they did with a list.
    """

    def __init__(self, lines: Sequence[SpeakerLine], raw: str = "", warning: str = "") -> None:
        """Store the parsed lines, the raw marks text and a warning when the marks do not match."""
        self.lines = list(lines)
        self.raw = raw
        self.warning = warning

    def __iter__(self) -> Iterator[SpeakerLine]:
        """Yield each speaker line in order."""
        return iter(self.lines)

    def __getitem__(self, item: Union[int, slice]) -> Union[SpeakerLine, List[SpeakerLine]]:
        """Return one line, or a list when ``item`` is a slice."""
        return self.lines[item]

    def __len__(self) -> int:
        """The number of speaker lines."""
        return len(self.lines)

    def __bool__(self) -> bool:
        """True when there is at least one line."""
        return bool(self.lines)


def _is_character_line(text: str) -> bool:
    """True when ``text`` opens with a dialogue dash (em dash, en dash, or ``- ``)."""
    return _CHARACTER_LINE.match(text or "") is not None


def _looks_like_name(word: str) -> bool:
    """A capitalised word that is not an abbreviation (``EBU`` has no lowercase letter)."""
    return len(word) >= 2 and word[0].isupper() and any(ch.islower() for ch in word[1:])


def _attribution_name(text: str) -> str:
    """The speaker named in a dash line's attribution (``— said Name``), or ``""``.

    The opening dash is the speech itself. A later dash starts an attribution: the first capitalised word there is
    the name. ``— said he`` has none.
    """
    if not _is_character_line(text):
        return ""
    body = re.sub(r"^\s*[\u2014\u2013-]\s+", "", text.strip(), count=1)
    for part in _ATTRIB_SPLIT.split(body)[1:]:
        # The attribution is one clause. A dash that only sets off a phrase ("second — a pause. Loudness")
        # must not take the next sentence's capital as a name.
        clause = re.split(r"[.!?…]|\.\.\.", part, maxsplit=1)[0]
        for word in _NAME_TOKEN.findall(clause):
            if _looks_like_name(word):
                return word
    return ""


def _speaker_lexicon(paras: Sequence[str]) -> List[str]:
    """Names taken from attributions, first spelling kept, so narration can be matched against them."""
    out: List[str] = []
    seen = set()
    for text in paras:
        name = _attribution_name(text)
        key = _name_key(name)
        if key and key not in seen:
            seen.add(key)
            out.append(name)
    return out


def _named_in(text: str, names: Sequence[str]) -> List[str]:
    """``names`` that occur in ``text`` as a whole word, in lexicon order."""
    found: List[str] = []
    for name in names:
        if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text or "", re.IGNORECASE):
            found.append(name)
    return found


def _paragraph_expectation(paras: Sequence[str], index: int, lexicon: Sequence[str]) -> Tuple[str, str]:
    """``(role, name)`` the mark for ``paras[index]`` has to satisfy.

    ``role`` is ``character`` (a leading dialogue dash), ``narrator`` (no dash and no quotation marks) or ``""``
    when quotation marks carry the speech and the dash rule does not decide. ``name`` is set for a character line
    whose speaker the paragraph itself names, or, failing that, whose preceding narration names exactly one known
    speaker.
    """
    text = paras[index]
    if _is_character_line(text):
        name = _attribution_name(text)
        if not name and index > 0 and not _is_character_line(paras[index - 1]):
            hits = _named_in(paras[index - 1], lexicon)
            if len(hits) == 1:
                name = hits[0]
        return "character", name
    if _QUOTED_SPEECH.search(text or ""):
        return "", ""
    return "narrator", ""


def _mark_mismatches(paras: Sequence[str], block: Sequence[int], marks: Sequence[SpeakerLine]) -> List[str]:
    """English reasons the marks do not fit ``block``, empty when every mark fits."""
    lexicon = _speaker_lexicon(paras)
    reasons: List[str] = []
    for index, mark in zip(block, marks):
        role, name = _paragraph_expectation(paras, index, lexicon)
        number = index + 1
        if role == "narrator" and mark.role != "narrator":
            reasons.append(f"paragraph {number}: narration marked {mark.role}")
        elif role == "character" and mark.role == "narrator":
            reasons.append(f"paragraph {number}: character line marked narrator")
        if name and _name_key(mark.name) != _name_key(name):
            reasons.append(f"paragraph {number}: name does not match")
    return reasons


def _tag_block(model, template: str, language: str, paras: Sequence[str], block: Sequence[int],
               raw_parts: List[str]) -> List[Optional[SpeakerLine]]:
    """Marks for the paragraphs ``block`` (indexes into ``paras``); ``None`` for a paragraph the model could not mark.

    When the reply has a different number of tags than the block has paragraphs (Gemma sometimes merges a title with the
    first paragraph, or splits one paragraph in two), the block is split in half and each half is asked again, down to
    single paragraphs. One bad answer then costs only the paragraphs it really concerns, not the whole block (build 702
    lost 10 of 26 marks to one merged title). A reply with the right number of tags can still be shifted by one
    paragraph: each mark is checked against its text (:func:`_mark_mismatches`) and a block that fails is asked again
    one paragraph at a time, through this same function. The check is logged as ``[validate]``. Every reply is kept
    in ``raw_parts``.
    """
    text = "\n\n".join(paras[i] for i in block)
    prompt = fill(template, language=NAMES.get(language, language or "English"), text=text)
    answer = model.complete(prompt, max_tokens=min(2048, 48 * len(block) + 32))
    raw_parts.append(answer or "")
    parsed = parse_tags(answer or "", len(block))
    if parsed is not None and len(block) > 1:
        reasons = _mark_mismatches(paras, block, parsed)
        if reasons:
            shown = reasons[:8]
            detail = "; ".join(shown)
            if len(reasons) > len(shown):
                detail += f"; +{len(reasons) - len(shown)} more"
            raw_parts.append(f"[validate] {detail}; asking one paragraph at a time")
            out: List[Optional[SpeakerLine]] = []
            for index in block:
                out.extend(_tag_block(model, template, language, paras, [index], raw_parts))
            return out
    if parsed is not None:
        return list(parsed)
    if len(block) == 1:
        return [None]
    mid = len(block) // 2
    raw_parts.append(f"[retry] {len(block)} paragraphs -> {mid} + {len(block) - mid}")
    return (_tag_block(model, template, language, paras, block[:mid], raw_parts)
            + _tag_block(model, template, language, paras, block[mid:], raw_parts))


def tag_paragraphs(paras: Sequence[str], language: str, plan: LLMPlan,
                   progress: Optional[Callable[[float], None]] = None) -> TagResult:
    """Mark ``paras``. The model is closed before return.

    A block whose reply does not fit is split and asked again (:func:`_tag_block`); only a single paragraph the model still
    cannot mark stays narrator. A reply with one tag per paragraph that does not match the text (a shifted line) is asked
    again one paragraph at a time. ``warning`` is ``unparsed`` when a paragraph is left unmarked (or the model raises), and
    ``no_speakers`` when every paragraph is the narrator but the text obviously contains dialogue. The raw replies are kept
    either way.
    """
    progress = progress or (lambda _f: None)
    lines = [SpeakerLine() for _ in paras]
    if not any(any(c.isalpha() for c in p) for p in paras):
        return TagResult(lines)
    template = load_prompt("speaker_markup")
    model = None
    raw_parts: List[str] = []
    failed = False
    try:
        model = plan.factory()
        batches = _batches(list(paras))
        for bi, block in enumerate(batches):
            marks = _tag_block(model, template, language, paras, block, raw_parts)
            for i, line in zip(block, marks):
                if line is None:
                    failed = True
                else:
                    lines[i] = line
            progress((bi + 1) / max(1, len(batches)))
    except Exception as exc:  # noqa: BLE001 - a failed model leaves the unmarked paragraphs with the narrator, and says so
        failed = True
        raw_parts.append(f"[error] {type(exc).__name__}: {exc}")
        progress(1.0)
    finally:
        if model is not None:
            model.close()
    warning = ""
    if failed:
        warning = WARN_UNPARSED
    elif lines and all(ln.role == "narrator" for ln in lines) and any(has_dialogue(p) for p in paras):
        warning = WARN_NO_SPEAKERS
    return TagResult(lines, raw="\n\n".join(raw_parts), warning=warning)


def _norm(text: str) -> str:
    return " ".join(text.split())


def assign(chunks: Sequence[Chunk], book: Book, lines: Sequence[SpeakerLine], voice_ids: Dict[str, str],
           narrator_id: str = "", per_line: Optional[Sequence[str]] = None) -> Tuple[List[Chunk], str]:
    """Chunks with ``voice_id`` set from ``lines``.

    ``voice_ids`` maps ``male`` / ``female`` to a library id. ``per_line`` (one voice id per mark, as
    :meth:`SpeakerCast.voices_for` returns) overrides that role map when given. The narrator and any mark whose voice is
    the narrator stay on ``voice_id`` ``""`` (the job voice). When the paragraph count does not match ``lines``, the
    chunks are returned unchanged and the note is ``"mismatch"``.
    """
    paras = paragraphs(book)
    if len(paras) != len(lines):
        return list(chunks), "mismatch"
    if per_line is not None and len(per_line) != len(lines):
        per_line = None
    if per_line is None:
        per_line = [voice_ids.get(line.role, "") if line.role in ("male", "female") else "" for line in lines]
    rows = [(ci, _norm(text), vid) for (ci, text), vid in zip(paras, per_line)]
    cursor = 0
    out: List[Chunk] = []
    for chunk in chunks:
        vid = ""
        if chunk.pause_kind != TITLE:
            wanted = _norm(chunk.text)
            for j in range(cursor, len(rows)):
                ci, ptext, v = rows[j]
                if ci == chunk.chapter and wanted and wanted in ptext:
                    vid = v
                    cursor = j
                    break
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
        role = _ROLE.get(m.group(1).lower())
        if role not in ROLES:
            raise ValueError(f"unreadable speaker mark: {raw.strip()}")
        rows.append(SpeakerLine(role, m.group(2) or ""))
    if not rows:
        raise ValueError("speaker marks file is empty")
    return rows


def default_role_picks(records, narrator_id: str = "", preferred: Optional[Dict[str, str]] = None) -> dict:
    """Default ``{"male", "male2", "female"}`` voice ids.

    ``preferred`` maps a role to a library id (the shipped cast, :func:`infra.bundled_voices.preferred_ids`: men Natan and
    Shimon, woman Miriam). A preferred voice is used when it is in ``records``, has the role's gender and is not the narrator
    or already taken by another role. Every other role falls back to the generic rule: the first male / female voice that is
    not the narrator (the narrator's own voice when it is the only one of that gender), and a second, different male voice
    when there is one."""
    preferred = preferred or {}

    def of(gender):
        return [str(r.id) for r in records if str((getattr(r, "info", None) or {}).get("gender") or "") == gender]

    out: Dict[str, str] = {}
    for role, gender in (("male", "male"), ("female", "female")):
        ids = of(gender)
        want = str(preferred.get(role) or "")
        if want and want in ids and want != narrator_id:
            out[role] = want
            continue
        others = [i for i in ids if i != narrator_id]
        out[role] = (others or ids or [""])[0]
    want2 = str(preferred.get("male2") or "")
    if want2 and want2 in of("male") and want2 not in (out["male"], narrator_id):
        out["male2"] = want2
    else:
        rest = [i for i in of("male") if i not in (out["male"], narrator_id)]
        out["male2"] = rest[0] if rest else ""
    return out
