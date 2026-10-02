"""Нормализация русского текста перед выравниванием: числа, сокращения -> «как произносится».

Зачем: выравниватель (Qwen3-ForcedAligner) делит текст по пробелам и выбрасывает пунктуацию; цифры и
сокращения он «читает» иначе, чем диктор («5 км» -> «пять километров»), и разметка сдвигается.
Поэтому для выравнивания и для поля `text` датасета используется «произносимая» форма; исходный текст
сохраняется только в report.json (`text_raw`).

Движок (по убыванию качества, выбирается автоматически; всё необязательно):
  1. ru-normalizr  (pip install ru-normalizr)  - числа с падежами, даты, римские цифры, сокращения;
  2. rutextnorm    (pip install rutextnorm)    - один файл, только regexp;
  3. встроенный запасной вариант (ниже): сокращения «т.д./т.е./т.к./…», целые числа в именительном падеже.
Любой движок обрабатывается ПО ПРЕДЛОЖЕНИЯМ (иначе потеряются границы предложений: движки убирают точки
после сокращений), а после него встроенный проход дочитывает оставшиеся цифры.

Карта токенов: для каждого слова «произносимого» текста хранится диапазон исходного слова/слов
(difflib по токенам), по ней можно восстановить исходный фрагмент для любого сегмента.
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from core.text_utils import collapse_ws, split_sentences

log = logging.getLogger("voxprint.normalizer")

# --------------------------------------------------------------------------- встроенный запасной вариант

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
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return forms[0]
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return forms[1]
    return forms[2]


def number_to_words(n: int) -> str:
    """Целое число -> слова (именительный падеж). Числа от 10**15 читаются по цифрам."""
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
    def repl(m: re.Match) -> str:
        return number_to_words(int(re.sub(r"\D", "", m.group(0))))
    return _NUM_RE.sub(repl, text)


def expand_abbreviations(text: str) -> str:
    for rx, rep in _ABBR_RE:
        text = rx.sub(rep, text)
    return text


def builtin_normalize(text: str) -> str:
    """Полный встроенный проход: сокращения, символы (%, №), числа."""
    text = expand_abbreviations(text)
    for rx, rep in _SYMS_RE:
        text = rx.sub(rep, text)
    return _spell_numbers(text)


# --------------------------------------------------------------------------- движки


def _engine_ru_normalizr() -> Optional[Callable[[str], str]]:
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import ru_normalizr  # type: ignore

            ru_normalizr.normalize("тест 1")  # прогрев + проверка, что зависимости на месте
        return ru_normalizr.normalize
    except Exception as exc:  # noqa: BLE001
        log.info("ru-normalizr unavailable: %s", exc)
        return None


def _engine_rutextnorm() -> Optional[Callable[[str], str]]:
    try:
        import rutextnorm  # type: ignore

        rutextnorm.normalize_russian("тест 1")
        return rutextnorm.normalize_russian
    except Exception as exc:  # noqa: BLE001
        log.info("rutextnorm unavailable: %s", exc)
        return None


def pick_engine() -> Tuple[str, Callable[[str], str]]:
    for name, factory in (("ru-normalizr", _engine_ru_normalizr), ("rutextnorm", _engine_rutextnorm)):
        fn = factory()
        if fn is not None:
            return name, fn
    return "builtin", lambda s: s


# --------------------------------------------------------------------------- результат и карта токенов

_TERMINATORS = ".!?…"
_TOKEN_RE = re.compile(r"\S+")


def _key(tok: str) -> str:
    return "".join(ch for ch in tok.casefold().replace("ё", "е") if ch.isalnum())


@dataclass
class TokenSpan:
    ns: int   # диапазон слова в «произносимом» тексте
    ne: int
    rs: int   # диапазон исходного слова (или группы слов) в `raw`
    re_: int


@dataclass
class NormalizedText:
    raw: str                      # исходный текст (для русского - предложения, склеенные пробелом)
    spoken: str                   # то, что произносится (идёт в выравниватель и в metadata.jsonl)
    engine: str = "none"
    tokens: List[TokenSpan] = field(default_factory=list)
    changed: bool = False

    def raw_for_span(self, ns: int, ne: int) -> str:
        """Исходный фрагмент для диапазона [ns, ne) «произносимого» текста."""
        if not self.changed:
            return collapse_ws(self.spoken[ns:ne])
        hit = [t for t in self.tokens if t.ne > ns and t.ns < ne]
        if not hit:
            return ""
        return collapse_ws(self.raw[min(t.rs for t in hit): max(t.re_ for t in hit)])


def _map_sentence(raw: str, spoken: str, raw_off: int, spoken_off: int) -> List[TokenSpan]:
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
            else:  # insert: привязываем к ближайшему исходному слову
                k = min(max(i1 - 1, 0), len(rt) - 1)
                rs, re_ = rt[k][0], rt[k][1]
            out.append(TokenSpan(spoken_off + st[j][0], spoken_off + st[j][1], raw_off + rs, raw_off + re_))
    return out


def _fix_sentence(raw_sent: str, norm: str) -> str:
    norm = collapse_ws(norm.replace("\u0301", ""))
    # движки убирают точку после сокращения - возвращаем знак конца предложения
    tail = raw_sent.rstrip("\"'»”’)]} ")[-1:]
    if tail and tail in _TERMINATORS and (not norm or norm[-1] not in _TERMINATORS + "\"'»”’)"):
        norm += tail
    return norm


def normalize_for_tts(text: str, language: str = "Russian",
                      engine: Optional[Tuple[str, Callable[[str], str]]] = None) -> NormalizedText:
    """Для русского текста - «произносимая» форма + карта токенов; для остальных языков - без изменений."""
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
        pre = expand_abbreviations(sent)   # «т.д.», «т.ч.» - до движка (часть движков их не знает)
        try:
            norm = fn(pre) if name != "builtin" else pre
        except Exception as exc:  # noqa: BLE001 - плохой ввод не должен ронять сборку датасета
            log.warning("normalizer engine %s failed on %r: %s", name, sent[:40], exc)
            norm = pre
        norm = builtin_normalize(norm)  # дочитываем то, что движок не тронул (цифры, %, №)
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
        # ничего не поменялось: оставляем текст как есть (с исходными переносами строк)
        return NormalizedText(raw=text, spoken=text, engine=name, changed=False)
    return NormalizedText(raw=raw, spoken=spoken, engine=name, tokens=tokens, changed=True)
