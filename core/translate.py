"""Automatic translation of a book before narration (optional step of the "Narrate a book" flow).

The book is translated **sentence by sentence** by a local, offline machine-translation model (Opus-MT / Marian, see
:mod:`infra.text_models`), then narrated with a voice of the target language.  The structure is preserved: chapters,
chapter titles, paragraphs (blank lines), poem lines and scene breaks stay where they are, so the pauses of
:mod:`core.pauses` work exactly as for an original text.

* **Pairs.**  Opus-MT has one model per direction.  Available here: ru<->en and de<->en; ru<->de goes through English
  (two hops, noticeably lower quality).  The source language is detected from the text (:func:`detect_language`).
* **Resumable.**  Every translated sentence is cached on disk (``<job>/.translation/translation_cache.json``, keyed by model
  tag + sentence), so a resumed job never translates a sentence twice.  The finished translation is also written next to
  the audiobook as ``translation_<lang>.txt``: the user can read and **edit** it; on the next run of the same job the
  edited file is used instead of translating again (delete it to translate anew).
* **Safe.**  Nothing is downloaded or run unless the user ticks the option; the engine is an interface
  (:class:`Translator`), tests use a fake one.  Machine translation can be wrong; see the note in the UI and the manual.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

from core.book_parsers import Book, Chapter
from core.errors import DatasetMakerError
from core.events import CancelToken
from core.i18n import tr
from core.text_utils import split_sentences

log = logging.getLogger("voxprint.translate")

#: Languages the UI offers as a target (and the detector can tell apart).
LANGUAGES: Tuple[str, ...] = ("en", "ru", "de")
#: Directions for which an Opus-MT model exists (see ``infra.text_models``).
DIRECT_PAIRS: frozenset = frozenset({("ru", "en"), ("en", "ru"), ("de", "en"), ("en", "de")})
#: Language names as the TTS engine and the narration code write them.
LANGUAGE_NAMES: Dict[str, str] = {"en": "English", "ru": "Russian", "de": "German"}
MAX_SENTENCE_CHARS = 450        # longer sentences are cut at clause boundaries (Marian is trained on single sentences)
BATCH_SENTENCES = 16
BATCH_CHARS = 3200
CACHE_FLUSH_SECONDS = 15.0
FILE_MARK = re.compile(r"^=== \[(\d+)\][ \t]*(.*)$")


class TranslateError(DatasetMakerError):
    """The translation cannot be done (unsupported pair, missing model); ``user_message`` is shown to the user."""

    kind = "translate"

    def __init__(self, user_message: str, details: str = "") -> None:
        """Keep the message for the UI and the technical details for the log."""
        super().__init__(user_message, details=details)


class Translator(Protocol):
    """A machine-translation model for one direction.  ``tag`` identifies model + revision (cache key)."""
    tag: str

    def translate(self, sentences: Sequence[str]) -> List[str]:
        """Translate every sentence (same order, same length of the result)."""

    def close(self) -> None:
        """Free memory (called when the hop is finished)."""


#: ``factory(source, target)`` -> engine of that direction, or ``None`` if its model is not downloaded.
EngineFactory = Callable[[str, str], Optional[Translator]]
ProgressFn = Callable[[float, str], None]


@dataclass
class TranslatePlan:
    """What to translate: the target language (``en`` / ``ru`` / ``de``), the source (``""`` = detect) and the engines."""
    target: str
    source: str = ""
    engine_factory: Optional[EngineFactory] = None

    @property
    def enabled(self) -> bool:
        """True if a target language is chosen."""
        return bool(self.target)


# --------------------------------------------------------------------------- language detection and routing
_STOP = {
    "en": "the and of to in that is was he she it for with as his her you i on at by this had not are but from they we"
          " have be or an which one all were there their what would when said".split(),
    "de": "der die das und ist nicht ein eine ich zu mit den von sie es auf dem sich er auch für im war als aber wie"
          " noch nach bei aus wir hat dass oder wenn nur über einen".split(),
}
_UKR = set("іїєґІЇЄҐ")


def detect_language(text: str) -> str:
    """``ru`` / ``en`` / ``de`` for a text, ``uk`` for Ukrainian, ``""`` if it is not one of these (or too short)."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 20:
        return ""
    cyr = sum(1 for c in letters if "\u0400" <= c <= "\u04ff")
    if cyr > len(letters) * 0.5:
        return "uk" if any(c in _UKR for c in text) else "ru"
    if sum(1 for c in letters if c.isascii() or c in "äöüßÄÖÜéèàç") < len(letters) * 0.8:
        return ""
    words = re.findall(r"[^\W\d_]+", text.lower())
    if not words:
        return ""
    score = {lang: sum(1 for w in words if w in set(st)) / len(words) for lang, st in _STOP.items()}
    best = max(score, key=score.get)
    other = "de" if best == "en" else "en"
    if score[best] < 0.06 or score[best] < score[other] * 1.15:
        return ""                                                # French, Spanish ... or too close to call
    return best


def detect_book_language(book: Book) -> str:
    """Language of a book, from a sample of its text (the file's own language tag is only a fallback)."""
    sample = " ".join(c.text[:3000] for c in book.chapters[:6])[:12000]
    code = detect_language(sample)
    if code:
        return code
    tag = (book.language or "").split("-")[0].lower()
    return tag if tag in LANGUAGES else ""


def route(source: str, target: str) -> List[Tuple[str, str]]:
    """The model hops from ``source`` to ``target``: ``[]`` (same language), one direct hop or two through English."""
    if source == target:
        return []
    if (source, target) in DIRECT_PAIRS:
        return [(source, target)]
    if (source, "en") in DIRECT_PAIRS and ("en", target) in DIRECT_PAIRS:
        return [(source, "en"), ("en", target)]
    raise TranslateError(tr("err.translate_pair", source=source or "?", target=target))


# --------------------------------------------------------------------------- structure
@dataclass
class _Para:
    """A paragraph: its units (sentences, or lines of a poem) and the string that joins them again."""
    units: List[str]
    joiner: str


def _cut_long(sentence: str, limit: int = MAX_SENTENCE_CHARS) -> List[str]:
    """Pieces of at most ``limit`` characters, cut after a clause mark (or at a space) near the limit."""
    out: List[str] = []
    s = sentence
    while len(s) > limit:
        window = s[:limit]
        cut = max(window.rfind("; "), window.rfind(", "), window.rfind(": "), window.rfind(" \u2014 "))
        cut = cut + 1 if cut > limit // 3 else window.rfind(" ")
        if cut <= 0:
            cut = limit
        out.append(s[:cut].strip())
        s = s[cut:].strip()
    if s:
        out.append(s)
    return out


def split_paragraphs(text: str) -> List[_Para]:
    """Paragraphs of a chapter text (blank-line separated) with their sentences; a verse (lines without sentence
    punctuation) is kept line by line."""
    paras: List[_Para] = []
    for raw in re.split(r"\n\s*\n", text.strip()):
        raw = raw.strip()
        if not raw:
            continue
        if "\n" in raw and not re.search(r"[.!?\u2026]", raw):
            paras.append(_Para([ln.strip() for ln in raw.split("\n") if ln.strip()], "\n"))
            continue
        units: List[str] = []
        for s in split_sentences(" ".join(raw.split())):
            units.extend(_cut_long(s))
        paras.append(_Para(units or [raw], " "))
    return paras


def _translatable(unit: str) -> bool:
    """True if the unit has letters (scene breaks like ``* * *`` and bare numbers are passed through)."""
    return any(c.isalpha() for c in unit)


# --------------------------------------------------------------------------- cache
def _key(tag: str, text: str) -> str:
    """Cache key of one sentence for one model."""
    return hashlib.sha1((tag + "\0" + text).encode("utf-8"), usedforsecurity=False).hexdigest()[:24]


class TranslationCache:
    """Sentence-level translation cache in one JSON file (written atomically; a damaged file is ignored)."""

    def __init__(self, path: Optional[Path]) -> None:
        """Load ``path`` if it exists (``None`` = memory only)."""
        self.path = path
        self.data: Dict[str, str] = {}
        self._dirty = 0
        if path is not None and path.is_file():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(d, dict):
                    self.data = {str(k): str(v) for k, v in d.items()}
            except (OSError, ValueError):
                log.warning("translation cache %s is unreadable - starting empty", path)

    def get(self, tag: str, text: str) -> Optional[str]:
        """Cached translation or ``None``."""
        return self.data.get(_key(tag, text))

    def put(self, tag: str, text: str, translated: str) -> None:
        """Remember one translation."""
        self.data[_key(tag, text)] = translated
        self._dirty += 1

    def flush(self) -> None:
        """Write the file if something changed."""
        if self.path is None or not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)
        self._dirty = 0


# --------------------------------------------------------------------------- the translation itself
def _batches(texts: Sequence[str]) -> Iterable[List[str]]:
    """Batches of similar-length sentences (sorted by length: less padding, faster)."""
    cur: List[str] = []
    chars = 0
    for t in sorted(texts, key=len):
        if cur and (len(cur) >= BATCH_SENTENCES or chars + len(t) > BATCH_CHARS):
            yield cur
            cur, chars = [], 0
        cur.append(t)
        chars += len(t)
    if cur:
        yield cur


def translate_texts(texts: Sequence[str], hops: Sequence[Tuple[str, str]], factory: EngineFactory, cache: TranslationCache,
                    progress: Optional[ProgressFn] = None, cancel: Optional[CancelToken] = None) -> Dict[str, str]:
    """``{source text: translation}`` for the unique ``texts`` through all ``hops`` (one engine at a time)."""
    cancel = cancel or CancelToken()
    progress = progress or (lambda f, m: None)
    current: Dict[str, str] = {t: t for t in dict.fromkeys(texts) if _translatable(t)}
    n_hops = max(1, len(hops))
    for hi, (src, dst) in enumerate(hops):
        engine = factory(src, dst)
        if engine is None:
            raise TranslateError(tr("err.translate_model", source=src, target=dst))
        try:
            todo = list(dict.fromkeys(v for v in current.values() if cache.get(engine.tag, v) is None))
            total, done, last = len(todo), 0, time.monotonic()
            progress(hi / n_hops, tr("narr.translating", pct=int(100 * hi / n_hops)))
            for batch in _batches(todo):
                cancel.check()
                out = engine.translate(batch)
                if len(out) != len(batch):
                    raise TranslateError(tr("err.translate_failed"), details="the engine returned a wrong number of sentences")
                for src_text, res in zip(batch, out):
                    cache.put(engine.tag, src_text, res.strip() or src_text)
                done += len(batch)
                if time.monotonic() - last > CACHE_FLUSH_SECONDS:
                    cache.flush()
                    last = time.monotonic()
                frac = (hi + done / max(1, total)) / n_hops
                progress(frac, tr("narr.translating", pct=int(100 * frac)))
        finally:
            cache.flush()
            engine.close()
        current = {k: cache.get(engine.tag, v) or v for k, v in current.items()}
    return current


def translate_book(book: Book, plan: TranslatePlan, source: str, cache: TranslationCache,
                   progress: Optional[ProgressFn] = None, cancel: Optional[CancelToken] = None,
                   only: Optional[Set[int]] = None) -> Book:
    """The translated copy of ``book`` (title, chapter titles and texts; cover and author are kept).

    ``only`` limits the work to the given 0-based chapters; the others stay in their original language."""
    hops = route(source, plan.target)
    if plan.engine_factory is None:
        raise TranslateError(tr("err.translate_model", source=source, target=plan.target))
    structure: List[Optional[List[_Para]]] = [
        split_paragraphs(ch.text) if (only is None or i in only) else None for i, ch in enumerate(book.chapters)]
    titles = [book.title] + [ch.title for i, ch in enumerate(book.chapters) if only is None or i in only]
    units = [u for t in titles for u in [t]] + [u for paras in structure if paras for p in paras for u in p.units]
    mapping = translate_texts(units, hops, plan.engine_factory, cache, progress, cancel)

    def tx(s: str) -> str:
        return mapping.get(s, s)

    chapters: List[Chapter] = []
    for ch, paras in zip(book.chapters, structure):
        if paras is None:
            chapters.append(Chapter(ch.title, ch.text))
            continue
        text = "\n\n".join(p.joiner.join(tx(u) for u in p.units) for p in paras)
        chapters.append(Chapter(tx(ch.title) if ch.title else ch.title, text))
    return Book(tx(book.title) if book.title else book.title, book.author, plan.target, chapters, book.cover, book.cover_ext)


# --------------------------------------------------------------------------- the editable file next to the audiobook
def source_fingerprint(book: Book, target: str, only: Optional[Set[int]] = None) -> str:
    """Hash of everything the translation depends on (the text, the target and the chapter selection)."""
    h = hashlib.sha256()
    for part in (target, book.title, ",".join(map(str, sorted(only))) if only is not None else "*",
                 *[c.title + "\0" + c.text for c in book.chapters]):
        h.update(part.encode("utf-8"))
        h.update(b"\1")
    return h.hexdigest()


def file_name(target: str) -> str:
    """Name of the readable / editable translation next to the audiobook."""
    return f"translation_{target}.txt"


def write_translation_file(path: Path, book: Book, source: str, target: str, model_note: str) -> None:
    """Write the translated book as plain text: a header, then ``=== [n] Chapter title`` and the chapter text."""
    head = (f"# Voxprint machine translation {source} -> {target} ({model_note}).\n"
            "# Machine translation can contain errors. You may edit this file: if you start the same narration again,\n"
            "# the edited text is narrated (delete the file to translate again). Keep the '=== [n]' lines.\n"
            f"TITLE: {book.title}\n\n")
    parts = [f"=== [{i + 1}] {ch.title}\n\n{ch.text.strip()}\n" for i, ch in enumerate(book.chapters)]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(head + "\n".join(parts), encoding="utf-8")
    tmp.replace(path)


def read_translation_file(path: Path, like: Book, target: str) -> Optional[Book]:
    """Read a translation file back; ``None`` if it does not have exactly the chapters of ``like``."""
    try:
        lines = path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
    except OSError:
        return None
    title = like.title
    chapters: List[Tuple[str, List[str]]] = []
    for ln in lines:
        m = FILE_MARK.match(ln)
        if m:
            chapters.append((m.group(2).strip(), []))
        elif chapters:
            chapters[-1][1].append(ln)
        elif ln.startswith("TITLE:"):
            title = ln[6:].strip() or like.title
    if len(chapters) != len(like.chapters):
        return None
    out = [Chapter(t, "\n".join(body).strip()) for t, body in chapters]
    return Book(title, like.author, target, out, like.cover, like.cover_ext)


def ensure_translation(book: Book, plan: TranslatePlan, job_dir: Path, progress: Optional[ProgressFn] = None,
                       cancel: Optional[CancelToken] = None, only: Optional[Set[int]] = None,
                       model_note: str = "Opus-MT") -> Tuple[Book, str]:
    """Translate ``book`` for a job, or reuse the translation of an earlier run.  Returns ``(book, source_language)``.

    Order: (1) ``translation_<lang>.txt`` of the same source text - user edits included; (2) the sentence cache plus the
    model for what is still missing.  The result is always written back to ``translation_<lang>.txt``."""
    source = plan.source or detect_book_language(book)
    if not source:
        raise TranslateError(tr("err.translate_source"))
    if source == plan.target:
        return book, source
    route(source, plan.target)                                   # fail early for an unsupported pair
    out_file = job_dir / file_name(plan.target)
    meta_file = job_dir / ".translation" / f"translation_{plan.target}.json"
    fp = source_fingerprint(book, plan.target, only)
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    if meta.get("source") == fp and out_file.is_file():
        reused = read_translation_file(out_file, book, plan.target)
        if reused is not None:
            log.info("translation reused from %s", out_file)
            return reused, source
    if out_file.is_file():                                       # a different book / settings or a broken file: keep the old one
        out_file.replace(out_file.with_suffix(".previous.txt"))
    job_dir.mkdir(parents=True, exist_ok=True)
    cache = TranslationCache(job_dir / ".translation" / "translation_cache.json")
    translated = translate_book(book, plan, source, cache, progress, cancel, only)
    write_translation_file(out_file, translated, source, plan.target, model_note)
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(json.dumps({"source": fp, "from": source, "to": plan.target, "chapters": len(book.chapters)}),
                         encoding="utf-8")
    return translated, source


def tidy_quotes(text: str) -> str:
    """Marian puts spaces inside straight quotes (``" Hello "``): remove them (only when the quotes are balanced)."""
    if text.count('"') < 2 or text.count('"') % 2:
        return text
    parts = text.split('"')
    for i in range(1, len(parts)):
        if i % 2:                                   # after an opening quote
            parts[i] = parts[i].lstrip()
        else:                                       # after a closing quote: the text before it loses its trailing space
            parts[i - 1] = parts[i - 1].rstrip()
    return '"'.join(parts)


# --------------------------------------------------------------------------- the real engine (Opus-MT / Marian)
class MarianEngine:
    """Opus-MT (Marian) model of one direction with ``transformers``.  GPU (fp16) if CUDA is available, else CPU.

    The weights are loaded on first use.  The folder must hold ``config.json``, ``pytorch_model.bin`` (or safetensors),
    ``source.spm``, ``target.spm`` and ``vocab.json`` (see ``infra.text_models``)."""

    def __init__(self, model_dir: Path, revision: str, source: str, target: str, device: str = "") -> None:
        """Remember where the model is; nothing is loaded yet."""
        self.dir, self.device_name = Path(model_dir), device
        self.tag = f"opus-mt-{source}-{target}@{(revision or 'local')[:12]}"
        self._model = None
        self._tok = None
        self._torch = None

    @property
    def device(self) -> str:
        """``cuda`` or ``cpu`` (loads torch)."""
        import torch

        return self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self) -> None:
        """Load tokenizer and model."""
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        self._torch = torch
        dev = self.device
        self._tok = MarianTokenizer.from_pretrained(str(self.dir))
        model = MarianMTModel.from_pretrained(str(self.dir)).eval()
        if dev == "cuda":
            model = model.half()
        self._model = model.to(dev)
        log.info("translation model %s loaded on %s", self.tag, dev)

    def translate(self, sentences: Sequence[str]) -> List[str]:
        """Translate a batch (beam search of 4 on GPU, 3 on CPU)."""
        if self._model is None:
            self._load()
        torch, dev = self._torch, self.device
        enc = self._tok(list(sentences), return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
        longest = int(enc["input_ids"].shape[1])
        with torch.inference_mode():
            gen = self._model.generate(**enc, num_beams=4 if dev == "cuda" else 3, max_new_tokens=min(512, int(longest * 1.6) + 16))
        return [tidy_quotes(t) for t in self._tok.batch_decode(gen, skip_special_tokens=True)]

    def close(self) -> None:
        """Free the weights (and GPU memory)."""
        self._model, self._tok = None, None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
