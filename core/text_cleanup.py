"""Neural text clean-up with a safety net: the model *proposes*, a validator *disposes*.

Stage 2 of the preparation pipeline (after :mod:`core.text_prep`, before synthesis).  A small seq2seq model
(``ai-forever/sage-fredt5-distilled-95m`` for Russian) rewrites a sentence with typos and missing commas corrected.  A
language model must never be allowed to rewrite an author, so its output is only a *proposal*: it is compared with the
source word by word and every difference is judged on its own.  Accepted:

* a word replaced by a close spelling (Damerau-Levenshtein distance <= 2, <= 3 for long words) when the source word is
  *rare in this very book* (a typo occurs once or twice; a name or dialect word occurs often) and is not a mid-sentence
  capitalized word (names), digits untouched, same letter case;
* ``е`` -> ``ё`` (helps the engine to stress the word) - not for the ambiguous "все", never in a word that already has a
  ``ё`` (a Russian word has at most one) and never as a side effect of a spelling fix;
* a comma / semicolon / colon / dash *inserted* between two words (never removed or replaced), at most one per four words.

Everything else (deleted words, changed punctuation, case changes, rewording) is dropped, and a paragraph whose accepted
typo corrections exceed 5 % of its words is left untouched.  Results are cached per sentence block in a JSON file next to
the narration cache, so a resumed job gets exactly the same text.

The engine is an interface (:class:`CleanupEngine`); tests use a fake one.  :class:`SageEngine` loads the real model with
``transformers`` - **unverified on real hardware**.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from core.book_parsers import Book, Chapter
from core.events import CancelToken
from core.text_utils import split_sentences

log = logging.getLogger("voxprint.cleanup")

MAX_BLOCK_CHARS = 500           # the model was trained on short inputs (the paper uses blocks of <= 1000 characters)
BATCH = 8
MAX_TYPO_SHARE = 0.05
MAX_COMMA_PER_WORDS = 4
INSERTABLE = {",", ";", ":", "\u2014"}
_TOKEN = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*|[^\w\s]", re.UNICODE)


class CleanupEngine(Protocol):
    """A text-correction model.  ``tag`` identifies model + revision (cache key)."""
    tag: str

    def correct(self, texts: Sequence[str]) -> List[str]:
        """Return a corrected version of every input text (same order)."""

    def close(self) -> None:
        """Free memory (called once when the clean-up pass is finished)."""


@dataclass
class CleanupStats:
    """What the validator did over a whole book."""
    blocks: int = 0
    cached: int = 0
    typos: int = 0
    yo: int = 0
    commas: int = 0
    rejected: int = 0
    untouched_blocks: int = 0

    def as_dict(self) -> Dict[str, int]:
        """Plain dict for the debug report."""
        return dict(self.__dict__)


def damerau(a: str, b: str) -> int:
    """Optimal-string-alignment (Damerau-Levenshtein) distance."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def _is_word(tok: str) -> bool:
    """True for a word token (not punctuation)."""
    return tok[:1].isalnum()


def word_frequencies(book: Book) -> Counter:
    """Lower-case word frequencies over the whole book (the validator's dictionary)."""
    freq: Counter = Counter()
    for ch in book.chapters:
        freq.update(t.lower() for t in _TOKEN.findall(ch.text) if _is_word(t))
    return freq


def _typo_ok(src: str, new: str, freq: Counter, mid_sentence: bool) -> str:
    """Judge one word replacement: ``"typo"``, ``"yo"`` or ``""`` (reject)."""
    if not (src.isalpha() and new.isalpha()) or src == new:
        return ""
    if src.lower() == new.lower():
        return ""                                                   # case-only change
    s_yo, n_yo = src.lower().count("ё"), new.lower().count("ё")
    if n_yo > 1 or (s_yo and n_yo != s_yo):
        return ""                                                   # one yo per word; never move or add a second one
    if src.replace("ё", "е").replace("Ё", "Е") == new.replace("ё", "е").replace("Ё", "Е") and ("ё" in new.lower()):
        return "yo" if src.lower() != "все" and not s_yo else ""
    if n_yo > s_yo:
        return ""                                                   # a "typo" fix that also adds a yo: leave it
    if src[:1].isupper() != new[:1].isupper() or (src.isupper() != new.isupper() and len(src) > 1):
        return ""
    if src[:1].isupper() and mid_sentence:
        return ""                                                   # a name
    limit = 3 if len(src) > 8 else 2
    if len(src) < 3 or damerau(src.lower(), new.lower()) > limit:
        return ""
    if freq.get(src.lower(), 0) > 2:
        return ""                                                   # a "typo" that occurs often is a name or dialect
    return "typo"


def validate(source: str, proposed: str, freq: Counter, stats: Optional[CleanupStats] = None) -> str:
    """Apply only the acceptable differences between ``source`` and the model's ``proposed`` text."""
    stats = stats if stats is not None else CleanupStats()
    if not proposed or proposed.strip() == source.strip():
        return source
    if len(proposed) > 1.5 * len(source) + 20 or len(proposed) < 0.5 * len(source):
        stats.rejected += 1
        return source
    ms = list(_TOKEN.finditer(source))
    ts = [m.group(0) for m in ms]
    tp = _TOKEN.findall(proposed)
    sm = difflib.SequenceMatcher(None, ts, tp, autojunk=False)
    edits: List[Tuple[int, int, str, str]] = []          # (start, end, text, kind)
    words = max(1, sum(1 for t in ts if _is_word(t)))
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        a, b = ts[i1:i2], tp[j1:j2]
        if op == "insert":
            if (all(t in INSERTABLE for t in b) and 0 < i1 < len(ts) and _is_word(ts[i1 - 1]) and _is_word(ts[i1])
                    and source[ms[i1 - 1].end():ms[i1].start()].strip() == ""):
                at = ms[i1 - 1].end()
                edits.append((at, at, "".join((" " + t) if t == "\u2014" else t for t in b), "comma"))
            else:
                stats.rejected += 1
            continue
        if op == "replace" and a and all(_is_word(t) for t in a):
            head = [t for t in b if _is_word(t)]
            trailing_punct = b[len(head):] if b[:len(head)] == head else None
            if len(head) == len(a) and trailing_punct is not None and all(t in INSERTABLE for t in trailing_punct):
                ok_all = True
                kinds = []
                for k, (s_word, n_word) in enumerate(zip(a, head)):
                    mid = i1 + k > 0 and ts[i1 + k - 1] not in (".", "!", "?", "…", "\u2014", '"', "«")
                    kind = _typo_ok(s_word, n_word, freq, mid)
                    if not kind and s_word != n_word:
                        ok_all = False
                        break
                    kinds.append(kind)
                if ok_all:
                    for k, kind in enumerate(kinds):
                        if kind:
                            m = ms[i1 + k]
                            edits.append((m.start(), m.end(), head[k], kind))
                    if trailing_punct and i2 < len(ts) and _is_word(ts[i2 - 1]) and _is_word(ts[i2]):
                        at = ms[i2 - 1].end()
                        edits.append((at, at, "".join(trailing_punct), "comma"))
                    continue
        stats.rejected += 1
    typos = [e for e in edits if e[3] in ("typo", "yo")]
    commas = [e for e in edits if e[3] == "comma"]
    if sum(1 for e in typos if e[3] == "typo") > MAX_TYPO_SHARE * words and sum(1 for e in typos if e[3] == "typo") > 1:
        stats.untouched_blocks += 1
        stats.rejected += len(typos)
        typos = []
    if len(commas) > max(1, words // MAX_COMMA_PER_WORDS):
        stats.rejected += len(commas)
        commas = []
    final = sorted(typos + commas, key=lambda e: e[0], reverse=True)
    out = source
    for start, end, text, kind in final:
        out = out[:start] + text + out[end:]
        if kind == "typo":
            stats.typos += 1
        elif kind == "yo":
            stats.yo += 1
        else:
            stats.commas += 1
    return out


# --------------------------------------------------------------------------- cache and driver

class BlockCache:
    """JSON cache ``{sha256(tag + source): corrected}`` kept next to the narration cache."""

    def __init__(self, path: Optional[Path]) -> None:
        """Load ``path`` if it exists (a damaged file is ignored)."""
        self.path = Path(path) if path else None
        self.data: Dict[str, str] = {}
        self.dirty = 0
        if self.path and self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data = {str(k): str(v) for k, v in loaded.items()}
            except (OSError, ValueError):
                self.data = {}

    @staticmethod
    def key(tag: str, text: str) -> str:
        """Cache key of one block."""
        return hashlib.sha256((tag + "\n" + text).encode("utf-8")).hexdigest()

    def save(self) -> None:
        """Write the cache atomically."""
        if not self.path or not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
        self.dirty = 0


def split_blocks(paragraph: str, limit: int = MAX_BLOCK_CHARS) -> List[str]:
    """Cut a paragraph into blocks of whole sentences, each <= ``limit`` characters (longer sentences stay whole)."""
    blocks: List[str] = []
    cur = ""
    for s in split_sentences(paragraph) or [paragraph]:
        if cur and len(cur) + 1 + len(s) > limit:
            blocks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        blocks.append(cur)
    return blocks


def cleanup_book(book: Book, engine: CleanupEngine, cache_path: Optional[Path] = None,
                 progress: Optional[Callable[[float, str], None]] = None,
                 cancel: Optional[CancelToken] = None) -> Tuple[Book, CleanupStats]:
    """Run the engine over every paragraph of ``book`` and return the book with the validated corrections."""
    stats = CleanupStats()
    cache = BlockCache(cache_path)
    cancel = cancel or CancelToken()
    freq = word_frequencies(book)
    # collect blocks
    layout: List[List[List[str]]] = []                 # chapter -> paragraph -> blocks
    todo: List[str] = []
    for ch in book.chapters:
        paras = []
        for p in ch.text.split("\n\n"):
            blocks = split_blocks(p) if "\n" not in p.strip() else [p.strip()]
            paras.append(blocks)
            for b in blocks:
                stats.blocks += 1
                if cache.key(engine.tag, b) not in cache.data and b not in todo:
                    todo.append(b)
        layout.append(paras)
    stats.cached = stats.blocks - len(todo)
    done = 0
    for i in range(0, len(todo), BATCH):
        cancel.check()
        batch = todo[i:i + BATCH]
        outputs = engine.correct(batch)
        if len(outputs) != len(batch):
            raise ValueError("engine returned a wrong number of texts")
        for src, out in zip(batch, outputs):
            cache.data[cache.key(engine.tag, src)] = out
            cache.dirty += 1
        done += len(batch)
        cache.save()
        if progress:
            progress(done / max(1, len(todo)), f"{done}/{len(todo)}")
    chapters: List[Chapter] = []
    for ch, paras in zip(book.chapters, layout):
        out_paras = []
        for blocks in paras:
            fixed = [validate(b, cache.data.get(cache.key(engine.tag, b), b), freq, stats) for b in blocks]
            out_paras.append(" ".join(fixed) if len(blocks) > 1 or "\n" not in (blocks[0] if blocks else "") else blocks[0])
        chapters.append(Chapter(ch.title, "\n\n".join(out_paras)))
    cache.save()
    return book.carry(chapters=chapters), stats


# --------------------------------------------------------------------------- the real model

class SageEngine:
    """``ai-forever/sage-fredt5-distilled-95m`` (seq2seq, MIT).  Loaded lazily; **unverified on real hardware**."""

    def __init__(self, model_dir: Path, revision: str = "", device: str = "") -> None:
        """``model_dir`` is the local snapshot; ``device`` "" = CUDA if available, else CPU."""
        self.model_dir = Path(model_dir)
        self.revision = revision
        self.device = device
        self.tag = f"sage-fredt5-distilled-95m@{revision or 'local'}"
        self._tok: Any = None
        self._model: Any = None

    def _load(self) -> None:
        """Import transformers and load the tokenizer and model once."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        from core import model_cache

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        pre = model_cache.take(model_cache.key("sage", self.model_dir, dev.startswith("cuda")))   # "Preload models at startup"
        if pre is not None:
            self._tok, model = pre
        else:
            self._tok = AutoTokenizer.from_pretrained(str(self.model_dir))  # nosec B615 - local, hash-verified folder
            model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir))  # nosec B615
        self._model = model.to(dev).eval()

    def correct(self, texts: Sequence[str]) -> List[str]:
        """Greedy decoding, as the model card shows (no sampling)."""
        import torch

        self._load()
        out: List[str] = []
        for text in texts:
            enc = self._tok(text, max_length=None, padding="longest", truncation=False, return_tensors="pt")
            enc = {k: v.to(self._model.device) for k, v in enc.items()}
            with torch.no_grad():
                ids = self._model.generate(**enc, max_length=int(enc["input_ids"].size(1) * 1.5) + 8)
            out.append(self._tok.batch_decode(ids, skip_special_tokens=True)[0])
        return out

    def close(self) -> None:
        """Release the model (the TTS needs the GPU next)."""
        self._model = None
        self._tok = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - torch may be absent in tests
            pass
