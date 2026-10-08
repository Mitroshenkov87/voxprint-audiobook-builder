"""Optional AI text model passes: **literary translation** and **prepare text for narration** (paragraph level).

The model (Gemma 4 12B in llama.cpp, :mod:`infra.llm_tool`) gets 1-4 paragraphs per request with the previous paragraphs as
context and a names glossary, using our prompt templates (``prompts/*.txt``, editable; see :func:`load_prompt`).  Every
answer passes a simple guard (:func:`check`): same number of paragraphs (and verse lines), a sane length ratio, the right
language, no refusal.  A paragraph that fails - or every paragraph after the model crashed or could not start - is left to
the caller's fallback: Opus-MT for translation, the original text for preparation.  Answers are cached per block (resume).
The model is closed (process ended, VRAM freed) when the pass ends, before the voice model is loaded.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence

from core.events import CancelToken

log = logging.getLogger("voxprint.llm")

MAX_BLOCK_CHARS = 1500
MAX_BLOCK_PARAS = 4
CONTEXT_PARAS = 2
MAX_GLOSSARY = 40
#: Language names as written into the prompts.
NAMES = {"en": "English", "ru": "Russian", "de": "German"}
_REFUSAL = re.compile(r"\b(I can(?:no|')t|I'm sorry|I am sorry|As an AI|I'm unable|Извините|К сожалению, я не|"
                      r"Я не могу|Es tut mir leid|Ich kann (?:diesen|das|nicht))", re.I)
#: Dialogue the pause chunker can hear: quotes, guillemets, German low-high quotes, or a line that opens with a dash.
_DIALOGUE = re.compile(r"[\"«»„“”]|^\s*[—–]\s+\S|^\s*-\s+\S", re.MULTILINE)
_WORD = re.compile(r"[^\W\d_]", re.UNICODE)


class ChatModel(Protocol):
    """A started model: one prompt in, the answer text out."""

    def complete(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.2) -> str: ...

    def close(self) -> None: ...


@dataclass
class LLMPlan:
    """How to start the model (``factory()`` returns a started :class:`ChatModel`) and its tag (cache key)."""
    factory: Callable[[], ChatModel]
    tag: str


def load_prompt(name: str, user_dir: Optional[Path] = None, bundled_dir: Optional[Path] = None) -> str:
    """``<data folder>/prompts/<name>.txt`` if the user made one (Narrate window: "Edit prompts"), else ours."""
    from infra import paths

    for d in (user_dir or paths.app_home() / "prompts", bundled_dir or paths.resource_dir() / "prompts"):
        p = Path(d) / f"{name}.txt"
        if p.is_file():
            return p.read_text(encoding="utf-8")
    raise FileNotFoundError(f"prompt {name}.txt not found")


def fill(template: str, **values: str) -> str:
    """``{name}`` placeholders replaced literally (no str.format: the text may contain braces)."""
    for k, v in values.items():
        template = template.replace("{" + k + "}", v)
    return template


def find_names(paras: Sequence[str], limit: int = MAX_GLOSSARY) -> List[str]:
    """Probable names: capitalised words seen at least 3 times, at least once inside a sentence, never in lower case."""
    from core.text_utils import split_sentences

    caps: Counter = Counter()
    mid: Counter = Counter()
    lower = set()
    for p in paras:
        for sent in split_sentences(" ".join(p.split())):
            words = re.findall(r"[^\W\d_][\w'\u2019-]*", sent)
            caps.update(w for w in words if len(w) > 1 and w[0].isupper() and not w.isupper())
            mid.update(w for w in words[1:] if w[0].isupper())       # capitalised inside a sentence: not just a sentence start
            lower.update(w for w in words if w.islower())
    return [w for w, n in caps.most_common() if n >= 3 and mid[w] and w.lower() not in lower][:limit]


def _split_answer(text: str) -> List[str]:
    text = re.sub(r"^```\w*\n|\n```$", "", text.strip())
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def check(src: str, out: str, language: str, lo: float, hi: float) -> bool:
    """The guard for one paragraph (see the module doc)."""
    from core.translate import detect_language

    if not out or (_REFUSAL.search(out) and not _REFUSAL.search(src)):
        return False
    if src.count("\n") != out.count("\n"):                  # a verse keeps its lines
        return False
    if len(src) >= 40 and not lo <= len(out) / len(src) <= hi:
        return False
    found = detect_language(out)
    return found in ("", language) or (language == "ru" and found == "uk")


def keeps_spoken_shape(src: str, out: str) -> bool:
    """True when a narration rewrite still has the pauses and the dialogue the chunker reads.

    A scene-break paragraph (no words) must stay a scene break. A paragraph that shows speech with quotes or with a
    leading dash must still show speech. Wording may change; the shape may not.
    """
    if bool(_WORD.search(src)) != bool(_WORD.search(out)):
        return False
    if _DIALOGUE.search(src) and not _DIALOGUE.search(out):
        return False
    return True


def _blocks(idx: Sequence[int], paras: Sequence[str]) -> List[List[int]]:
    out: List[List[int]] = []
    cur: List[int] = []
    size = 0
    for i in idx:
        if cur and (len(cur) >= MAX_BLOCK_PARAS or size + len(paras[i]) > MAX_BLOCK_CHARS):
            out.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += len(paras[i])
    if cur:
        out.append(cur)
    return out


class _Session:
    """Starts the model only when a block is not cached; after a crash it stays broken (the rest falls back)."""

    def __init__(self, plan: LLMPlan) -> None:
        self.plan, self.model, self.broken = plan, None, False

    def ask(self, prompt: str, max_tokens: int) -> Optional[str]:
        if self.broken:
            return None
        try:
            if self.model is None:
                self.model = self.plan.factory()
            return self.model.complete(prompt, max_tokens=max_tokens)
        except Exception:  # noqa: BLE001 - any model / process / HTTP failure: fall back for the rest
            log.warning("AI text model failed - the rest uses the fallback", exc_info=True)
            self.broken = True
            return None

    def close(self) -> None:
        if self.model is not None:
            try:
                self.model.close()
            finally:
                self.model = None


def run_paragraphs(paras: Sequence[str], template: str, values: Dict[str, str], language: str, plan: LLMPlan, cache,
                   ratio: tuple, progress: Optional[Callable[[float], None]] = None,
                   cancel: Optional[CancelToken] = None, session: Optional[_Session] = None,
                   shape: bool = False) -> Dict[int, str]:
    """``{paragraph index: accepted output}`` for the paragraphs with letters; the others / rejected ones are missing.
    ``cache`` is a :class:`core.translate.TranslationCache`; its key includes the model tag and the prompt."""
    cancel = cancel or CancelToken()
    progress = progress or (lambda f: None)
    tag = f"{plan.tag}|{hashlib.sha1(fill(template, **values).encode('utf-8'), usedforsecurity=False).hexdigest()[:10]}"
    idx = [i for i, p in enumerate(paras) if any(c.isalpha() for c in p)]
    out: Dict[int, str] = {i: cache.get(tag, paras[i]) for i in idx if cache.get(tag, paras[i]) is not None}
    blocks = _blocks([i for i in idx if i not in out], paras)      # only what is not cached yet (resume)
    own = session is None
    session = session or _Session(plan)
    rejected = 0
    try:
        for bi, block in enumerate(blocks):
            cancel.check()
            text = "\n\n".join(paras[i] for i in block)
            ctx = "\n\n".join(out[j] for j in range(block[0] - CONTEXT_PARAS, block[0]) if j in out)
            answer = session.ask(fill(template, **values, context=ctx or "-", text=text),
                                 max_tokens=min(4096, int(len(text) * 1.5) + 256))
            parts = _split_answer(answer or "")
            if len(parts) == len(block):              # each paragraph is judged on its own; a failed one falls back
                for i, o in zip(block, parts):
                    if check(paras[i], o, language, *ratio) and (not shape or keeps_spoken_shape(paras[i], o)):
                        out[i] = o
                        cache.put(tag, paras[i], o)
            rejected += sum(1 for i in block if i not in out) if answer is not None else 0
            progress((bi + 1) / max(1, len(blocks)))
            if bi % 8 == 7:
                cache.flush()
    finally:
        cache.flush()
        if own:
            session.close()
    log.info("AI text pass: %d of %d paragraphs accepted, %d rejected by the guard%s", len(out), len(idx), rejected,
             ", model failed (rest = fallback)" if session.broken else "")
    return out


def glossary(names: Sequence[str], source: str, target: str, plan: LLMPlan, session: _Session, file: Optional[Path]) -> str:
    """``Name = translation`` lines (one request), kept in ``file``; an existing file (maybe edited by the user) is used as is."""
    if file is not None and file.is_file():
        return file.read_text(encoding="utf-8").strip()
    if not names:
        return ""
    if source == target:
        text = "\n".join(names)
    else:
        answer = session.ask(fill(load_prompt("names_glossary"), source_language=NAMES.get(source, source),
                                  target_language=NAMES.get(target, target), names="\n".join(names)), max_tokens=1024) or ""
        wanted = set(names)
        lines = [f"{a.strip()} = {b.strip()}" for a, _, b in (ln.partition("=") for ln in answer.splitlines())
                 if a.strip() in wanted and b.strip()]
        text = "\n".join(lines)
    if file is not None and text:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text + "\n", encoding="utf-8")
    return text


def translate_paragraphs(paras: Sequence[str], source: str, target: str, plan: LLMPlan, cache, glossary_file: Optional[Path],
                         progress=None, cancel=None) -> Dict[int, str]:
    """Literary translation of ``paras``; missing indexes are for the Opus-MT fallback."""
    session = _Session(plan)
    try:
        gl = glossary(find_names(paras), source, target, plan, session, glossary_file)
        values = {"source_language": NAMES.get(source, source), "target_language": NAMES.get(target, target),
                  "glossary": gl or "-"}
        return run_paragraphs(paras, load_prompt("literary_translate"), values, target, plan, cache, (0.5, 2.2),
                              progress, cancel, session)
    finally:
        session.close()


def prepare_paragraphs(paras: Sequence[str], language: str, plan: LLMPlan, cache, glossary_file: Optional[Path],
                       progress=None, cancel=None) -> Dict[int, str]:
    """Narration-ready rewrite of ``paras`` in the same language; missing indexes keep the original text."""
    session = _Session(plan)
    try:
        gl = glossary(find_names(paras), language, language, plan, session, glossary_file)
        values = {"language": NAMES.get(language, language), "glossary": gl or "-"}
        return run_paragraphs(paras, load_prompt("prepare_narration"), values, language, plan, cache, (0.75, 1.9),
                              progress, cancel, session, shape=True)
    finally:
        session.close()


def prepare_book(book, plan: LLMPlan, language: str, job_dir: Path, progress=None, cancel=None, chapters=()):
    """The book with its chapter texts prepared for narration (titles unchanged); the result is also written to
    ``<job>/.debug/llm_prepared.txt`` for comparison."""
    from core.book_parsers import Book, Chapter
    from core.translate import TranslationCache, split_paragraphs

    structure = [split_paragraphs(ch.text) if (not chapters or i in chapters) else None for i, ch in enumerate(book.chapters)]
    flat = [p for paras in structure if paras for p in paras]
    texts = [p.joiner.join(p.units) for p in flat]
    cache = TranslationCache(Path(job_dir) / ".cache" / "llm_prepare.json")
    done = prepare_paragraphs(texts, language, plan, cache, Path(job_dir) / ".translation" / f"names_{language}.txt",
                              progress, cancel)
    k = 0
    out = []
    for ch, paras in zip(book.chapters, structure):
        if paras is None:
            out.append(Chapter(ch.title, ch.text))
            continue
        parts = []
        for _p in paras:
            parts.append(done.get(k, texts[k]))
            k += 1
        out.append(Chapter(ch.title, "\n\n".join(parts)))
    debug = Path(job_dir) / ".debug"
    debug.mkdir(parents=True, exist_ok=True)
    (debug / "llm_prepared.txt").write_text("\n".join(f"=== [{i + 1}] {c.title}\n\n{c.text}\n" for i, c in enumerate(out)),
                                            encoding="utf-8")
    return Book(book.title, book.author, book.language, out, book.cover, book.cover_ext)
