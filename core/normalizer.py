"""Russian text normalization before alignment: numbers and abbreviations -> "as pronounced".

Why: the forced aligner (Qwen3-ForcedAligner) splits text on spaces and drops punctuation; it "reads" digits and
abbreviations differently from the speaker ("5 км" -> "пять километров") and the alignment would drift.  So the
*spoken* form is used for alignment and for the ``text`` field of the dataset; the original text is kept only in
``report.json`` (``text_raw``).

Engines, best first, picked automatically (all optional):

1. ``ru-normalizr`` (``pip install ru-normalizr``) - numbers with cases, dates, Roman numerals, abbreviations;
2. ``rutextnorm`` (``pip install rutextnorm``) - a single file, regexps only;
3. the built-in fallback below: abbreviations such as "т.д./т.е./т.к.", integers in the nominative case.

Every engine is applied *per sentence* (otherwise sentence boundaries get lost: engines remove the period after an
abbreviation), and afterwards the built-in pass reads out any digits that are left.

Token map: for every word of the spoken text we store the range of the original word(s) (``difflib`` over tokens),
so the original fragment can be recovered for any segment.  Other languages pass through unchanged.
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from core.text_utils import collapse_ws, split_sentences

log = logging.getLogger("voxprint.normalizer")

# --------------------------------------------------------------------------- built-in fallback (Russian word tables are data)

_ABBR: List[Tuple[str, str]] = [
    (r"\bт\.\s?е\.", "то есть"), (r"\bт\.\s?д\.", "так далее"), (r"\bт\.\s?п\.", "тому подобное"),
    (r"\bт\.\s?к\.", "так как"), (r"\bт\.\s?н\.", "так называемый"), (r"\bт\.\s?ч\.", "том числе"),
    (r"\bи\s+др\.", "и другие"), (r"\bи\s+пр\.", "и прочее"), (r"\bим\.", "имени"),
    (r"\bг\.(?=\s*[А-ЯЁ])", "город"), (r"\bул\.(?=\s*[А-ЯЁ])", "улица"), (r"\bпр\.", "прочее"),
    (r"\bсм\.(?=\s)", "смотри"), (r"\bср\.(?=\s)", "сравни"), (r"\bн\.\s?э\.", "нашей эры"),
    (r"\bдо\s+н\.\s?э\.", "до нашей эры"),

]
_SYMS: List[Tuple[str, str]] = [(r"(?<=\d)\s?%", " процентов"), (r"№\s?", "номер "), (r"&", " и ")]
_ABBR_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in _ABBR]
_SYMS_RE = [(re.compile(p), r) for p, r in _SYMS]

_UNITS_M = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_UNITS_F = ["ноль", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
          "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят",
         "девяносто"]
_HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот",
             "девятьсот"]
_SCALES = [("тысяча", "тысячи", "тысяч", True), ("миллион", "миллиона", "миллионов", False),
           ("миллиард", "миллиарда", "миллиардов", False), ("триллион", "триллиона", "триллионов", False)]


def _triplet(n: int, feminine: bool) -> List[str]:
    """Words for a number 0..999 (``feminine`` selects одна/две, needed for "тысяча")."""
    words: List[str] = []
    h, rest = divmod(n, 100)
    if h:
        words.append(_HUNDREDS[h])
    if rest >= 20:
        t, u = divmod(rest, 10)
        words.append(_TENS[t])
        if u:
            words.append((_UNITS_F if feminine else _UNITS_M)[u])
    elif rest >= 10:
        words.append(_TEENS[rest - 10])
    elif rest:
        words.append((_UNITS_F if feminine else _UNITS_M)[rest])
    return words


def _plural(n: int, forms: Tuple[str, str, str]) -> str:
    """Pick the Russian plural form (1 / 2-4 / 5+ rule, with the 11-14 exception)."""
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return forms[0]
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return forms[1]
    return forms[2]


def number_to_words(n: int) -> str:
    """Integer -> Russian words in the nominative case. Numbers from 10**15 up are read digit by digit."""
    if n < 0:
        return "минус " + number_to_words(-n)
    if n == 0:
        return "ноль"
    if n >= 10 ** 15:
        return " ".join(_UNITS_M[int(c)] for c in str(n))
    parts = [(n // 1000 ** i) % 1000 for i in range(5)]
    words: List[str] = []
    for i in range(4, -1, -1):
        v = parts[i]
        if not v:
            continue
        if i == 0:
            words.extend(_triplet(v, False))
        else:
            one, few, many, fem = _SCALES[i - 1]
            words.extend(_triplet(v, fem))
            words.append(_plural(v, (one, few, many)))
    return " ".join(words)


_NUM_RE = re.compile(r"\d+(?:[ \u00a0]\d{3})*")


def _spell_numbers(text: str) -> str:
    """Replace every integer in the text (spaces allowed as thousands separators) by its spoken form."""
    def repl(m: re.Match) -> str:
        """Regex callback for :func:`_spell_numbers`."""
        return number_to_words(int(re.sub(r"\D", "", m.group(0))))
    return _NUM_RE.sub(repl, text)


def expand_abbreviations(text: str) -> str:
    """Expand common Russian abbreviations ("т.д.", "т.е.", "им.", ...) using ``_ABBR``."""
    for rx, rep in _ABBR_RE:
        text = rx.sub(rep, text)
    return text


def builtin_normalize(text: str) -> str:
    """The complete built-in pass: abbreviations, symbols (%, №, &) and numbers."""
    text = expand_abbreviations(text)
    for rx, rep in _SYMS_RE:
        text = rx.sub(rep, text)
    return _spell_numbers(text)


# --------------------------------------------------------------------------- engines


def _engine_ru_normalizr() -> Optional[Callable[[str], str]]:
    """Return the ``ru_normalizr.normalize`` function if the package works, else ``None``."""
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import ru_normalizr

            ru_normalizr.normalize("тест 1")  # warm-up + check that the dependencies are present
        return ru_normalizr.normalize
    except Exception as exc:  # noqa: BLE001
        log.info("ru-normalizr unavailable: %s", exc)
        return None


def _engine_rutextnorm() -> Optional[Callable[[str], str]]:
    """Return ``rutextnorm.normalize_russian`` if the package works, else ``None``."""
    try:
        import rutextnorm

        rutextnorm.normalize_russian("тест 1")
        return rutextnorm.normalize_russian
    except Exception as exc:  # noqa: BLE001
        log.info("rutextnorm unavailable: %s", exc)
        return None


def pick_engine() -> Tuple[str, Callable[[str], str]]:
    """Choose the best available engine as ``(name, function)``; falls back to ``("builtin", identity)``."""
    for name, factory in (("ru-normalizr", _engine_ru_normalizr), ("rutextnorm", _engine_rutextnorm)):
        fn = factory()
        if fn is not None:
            return name, fn
    return "builtin", lambda s: s


# --------------------------------------------------------------------------- result and token map

_TERMINATORS = ".!?…"
_TOKEN_RE = re.compile(r"\S+")


def _key(tok: str) -> str:
    """Comparison key of a token: lower-case, ``ё`` -> ``е``, only letters and digits."""
    return "".join(ch for ch in tok.casefold().replace("ё", "е") if ch.isalnum())


@dataclass
class TokenSpan:
    """Maps one spoken-text word (``ns``..``ne``) to the original word or group of words (``rs``..``re_``)."""
    ns: int   # range of the word in the "spoken" text
    ne: int
    rs: int   # range of the original word (or group of words) in `raw`
    re_: int


@dataclass
class NormalizedText:
    """Result of normalization: the original text, the spoken form, the engine used and the token map."""
    raw: str                      # original text (for Russian: the sentences joined by a space)
    spoken: str                   # what is pronounced (goes to the aligner and to metadata.jsonl)
    engine: str = "none"
    tokens: List[TokenSpan] = field(default_factory=list)
    changed: bool = False

    def raw_for_span(self, ns: int, ne: int) -> str:
        """The original fragment that corresponds to the range ``[ns, ne)`` of the spoken text."""
        if not self.changed:
            return collapse_ws(self.spoken[ns:ne])
        hit = [t for t in self.tokens if t.ne > ns and t.ns < ne]
        if not hit:
            return ""
        return collapse_ws(self.raw[min(t.rs for t in hit): max(t.re_ for t in hit)])


def _map_sentence(raw: str, spoken: str, raw_off: int, spoken_off: int) -> List[TokenSpan]:
    """Align spoken tokens with original tokens (``difflib``) and return their :class:`TokenSpan` list."""
    rt = [(m.start(), m.end(), _key(m.group())) for m in _TOKEN_RE.finditer(raw)]
    st = [(m.start(), m.end(), _key(m.group())) for m in _TOKEN_RE.finditer(spoken)]
    out: List[TokenSpan] = []
    if not rt or not st:
        return out
    sm = difflib.SequenceMatcher(None, [t[2] for t in rt], [t[2] for t in st], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "delete":
            continue
        for j in range(j1, j2):
            if op == "equal":
                r = rt[i1 + (j - j1)]
                rs, re_ = r[0], r[1]
            elif op == "replace":
                rs, re_ = rt[i1][0], rt[i2 - 1][1]
            else:  # insert: attach to the nearest original word
                k = min(max(i1 - 1, 0), len(rt) - 1)
                rs, re_ = rt[k][0], rt[k][1]
            out.append(TokenSpan(spoken_off + st[j][0], spoken_off + st[j][1], raw_off + rs, raw_off + re_))
    return out


def _skeleton(token: str) -> str:
    """Comparison form of a token: case-folded, yo folded to ye, stress marks removed."""
    return "".join(ch.casefold().replace("ё", "е") for ch in token if ch != "\u0301")


def _author_marks(token: str):
    """``(yo flags, stress flags)`` for each character of ``token`` except U+0301."""
    yos: list[bool] = []
    stresses: list[bool] = []
    for ch in token:
        if ch == "\u0301":
            if stresses:
                stresses[-1] = True
            continue
        yos.append(ch in ("ё", "Ё"))
        stresses.append(False)
    return yos, stresses


def _paint_author_marks(word: str, yos, stresses) -> str:
    """Copy author yo and U+0301 onto ``word`` when it has the same letters aside from those marks."""
    out = []
    i = 0
    for ch in word:
        if ch == "\u0301":
            continue
        if i >= len(yos):
            return word
        if yos[i] and ch in ("е", "Е"):
            ch = "ё" if ch == "е" else "Ё"
        out.append(ch)
        if stresses[i]:
            out.append("\u0301")
        i += 1
    if i != len(yos):
        return word
    return "".join(out)


def _later_has_skeleton(matches, start: int, skeleton: str, limit: int = 8) -> bool:
    """True when a spoken token within ``limit`` of ``start`` has ``skeleton``."""
    for match in matches[start:start + limit]:
        if _skeleton(match.group()) == skeleton:
            return True
    return False


def _keep_author_yo_and_stress(raw: str, norm: str) -> str:
    """Put author yo and U+0301 back onto the words the engine kept.

    The Russian engines may fold yo to ye and drop combining stress. A word the author already
    wrote with yo, and a stress mark the author wrote, are copied onto the spoken token with the
    same letters. Nothing is inserted when the author did not write it. Tokens the engine expanded
    (a number read as words) are left as the engine wrote them.
    """
    if "ё" not in raw and "Ё" not in raw and "\u0301" not in raw:
        return norm
    raw_toks = [match.group() for match in _TOKEN_RE.finditer(raw)]
    matches = list(_TOKEN_RE.finditer(norm))
    pieces = []
    cursor = 0
    i = 0
    j = 0
    while i < len(matches):
        match = matches[i]
        word = match.group()
        skeleton = _skeleton(word)
        if j < len(raw_toks) and _skeleton(raw_toks[j]) == skeleton:
            yos, stresses = _author_marks(raw_toks[j])
            painted = _paint_author_marks(word, yos, stresses) if any(yos) or any(stresses) else word
            pieces.append(norm[cursor:match.start()])
            pieces.append(painted)
            cursor = match.end()
            i += 1
            j += 1
        elif j < len(raw_toks) and _later_has_skeleton(matches, i + 1, _skeleton(raw_toks[j])):
            pieces.append(norm[cursor:match.start()])
            pieces.append(word)
            cursor = match.end()
            i += 1
        elif j < len(raw_toks):
            j += 1
        else:
            pieces.append(norm[cursor:match.start()])
            pieces.append(word)
            cursor = match.end()
            i += 1
    pieces.append(norm[cursor:])
    return "".join(pieces)


def _fix_sentence(raw_sent: str, norm: str) -> str:
    """Tidy an engine's output and restore the sentence terminator the engine removed.

    Author yo and U+0301 are copied back onto words the engine folded. This function does not
    invent either mark.
    """
    norm = _keep_author_yo_and_stress(raw_sent, collapse_ws(norm))
    # the engines drop the period after an abbreviation - restore the sentence terminator
    tail = raw_sent.rstrip("\"'»”’)]} ")[-1:]
    if tail and tail in _TERMINATORS and (not norm or norm[-1] not in _TERMINATORS + "\"'»”’)"):
        norm += tail
    return norm


def normalize_for_tts(text: str, language: str = "Russian",
                      engine: Optional[Tuple[str, Callable[[str], str]]] = None) -> NormalizedText:
    """Return the spoken form plus a token map for Russian text; other languages are returned unchanged.

    ``engine`` may be passed as ``(name, function)`` (used by tests). Each sentence is normalized separately; if nothing
    changed, the text is returned untouched (keeping its original line breaks).
    """
    if language.lower() not in ("russian", "ru", "русский"):
        return NormalizedText(raw=text, spoken=text, engine="none", changed=False)
    name, fn = engine or pick_engine()
    sentences = split_sentences(text)
    if not sentences:
        return NormalizedText(raw=text, spoken=text, engine=name, changed=False)
    spoken_parts: List[str] = []
    tokens: List[TokenSpan] = []
    raw_off = spoken_off = 0
    changed = False
    for sent in sentences:
        pre = expand_abbreviations(sent)   # "т.д.", "т.ч." - before the engine (some engines do not know them)
        try:
            norm = fn(pre) if name != "builtin" else pre
        except Exception as exc:  # noqa: BLE001 - bad input must not break dataset building
            log.warning("normalizer engine %s failed on %r: %s", name, sent[:40], exc)
            norm = pre
        norm = builtin_normalize(norm)  # read out what the engine left alone (digits, %, №)
        norm = _fix_sentence(sent, norm) or sent
        if norm != sent:
            changed = True
        tokens.extend(_map_sentence(sent, norm, raw_off, spoken_off))
        spoken_parts.append(norm)
        raw_off += len(sent) + 1
        spoken_off += len(norm) + 1
    raw = " ".join(sentences)
    spoken = " ".join(spoken_parts)
    if not changed:
        # nothing changed: keep the text as it is (with the original line breaks)
        return NormalizedText(raw=text, spoken=text, engine=name, changed=False)
    return NormalizedText(raw=raw, spoken=spoken, engine=name, tokens=tokens, changed=True)
