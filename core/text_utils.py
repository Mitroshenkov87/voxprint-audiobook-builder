"""Работа с текстом: чтение, нормализация, разбиение на предложения, привязка слов к тексту.

Правила «очистки» токенов повторяют Qwen3ForceAlignProcessor из пакета qwen-asr
(проверено по исходникам qwen-asr 0.0.6): токен = кусок текста между пробелами,
из которого оставлены только буквы (категория Unicode L*), цифры (N*) и апостроф.
Для китайского каждый иероглиф - отдельный токен. Благодаря этому мы можем
сопоставить слова выравнивателя с исходным текстом по потоку «чистых» символов,
независимо от языка и токенизатора.
"""
from __future__ import annotations

from core.i18n import tr
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from core.errors import AlignmentError, TextReadError
from core.types import WordTiming

# --------------------------------------------------------------------------- чтение


@dataclass
class TextReadResult:
    text: str
    encoding: str
    warnings: List[str] = field(default_factory=list)


def decode_bytes(data: bytes) -> TextReadResult:
    """Декодирует байты; предпочитает UTF-8, при неудаче пробует типичные русские кодировки."""
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
    """Убирает мусор (BOM, нулевые пробелы, управляющие символы), сохраняя переводы строк."""
    text = raw.translate(_ZERO_WIDTH).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- токены


def is_kept_char(ch: str) -> bool:
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


def clean_token(token: str) -> str:
    return "".join(ch for ch in token if is_kept_char(ch))


def count_clean_chars(text: str) -> int:
    return sum(1 for ch in text if is_kept_char(ch))


def detect_language(text: str) -> str:
    """Грубое определение языка для выравнивателя (названия как в qwen-asr)."""
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


# --------------------------------------------------------------------------- предложения

_ABBREVIATIONS = {
    # русские
    "т.д", "т.п", "т.е", "т.к", "т.н", "т.о", "т.ч", "и.о", "др", "пр", "г", "гг", "ул", "им", "см",
    "рис", "стр", "руб", "коп", "тыс", "млн", "млрд", "проф", "акад", "доц", "ред", "изд", "обл",
    "р", "пос", "с", "д", "кв", "тел", "напр", "англ", "нем", "франц", "лат", "в", "вв",
    # английские
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "no", "fig",
    "inc", "ltd", "co", "approx",
}
# сокращения, после которых заглавная буква обычно начинает новое предложение
_END_CAPABLE = {"т.д", "т.п", "др", "пр", "etc"}
_TERMINATORS = ".!?…。！？"
_CLOSERS = "\"'»”’)]}›"
_BOUNDARY_RE = re.compile(r"([.!?…。！？]+[\"'»”’)\]}›]*)(\s+|$)")


def _last_word_before(text: str, end: int) -> str:
    m = re.search(r"([^\s]+)$", text[:end])
    return m.group(1) if m else ""


def _is_abbreviation(token: str) -> bool:
    tok = token.strip(_CLOSERS + "«„“\"(").lower()
    if not tok.endswith("."):
        return False
    base = tok.rstrip(".")
    if not base:
        return False
    if base in _ABBREVIATIONS:
        return True
    # «т.д.», «т.е.»: цепочка однобуквенных частей
    if re.fullmatch(r"(?:[^\W\d_]\.)+[^\W\d_]", base):
        return True
    # инициалы: одна заглавная буква
    if re.fullmatch(r"[^\W\d_]", base) and token.strip(_CLOSERS + "«„“\"(")[:1].isupper():
        return True
    return False


def split_sentences(text: str) -> List[str]:
    """Разбивает текст на предложения (русский/английский/CJK), учитывая сокращения.

    Пустая строка (абзац) всегда завершает предложение. Одиночный перевод строки -
    только если перед ним уже стоит знак конца предложения.
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
            # стихи/строки без пунктуации: каждая строка - отдельное предложение
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
                # после «.» строчная буква - продолжение предложения
                if m.group(1).startswith(".") and nxt.islower():
                    continue
                if _is_abbreviation(token) and not (
                        nxt.isupper() and token.strip(_CLOSERS + "«„“\"(").lower().rstrip(".") in _END_CAPABLE):
                    # сокращение: не режем, кроме явного конца текста
                    continue
                # десятичные числа/нумерация: «3.5», «1. Пункт»
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
    """Дробит текст на мелкие смысловые единицы (по предложениям, запятым, тире, строкам).

    Используется только для планирования длинного аудио: чем больше точек разреза,
    тем точнее можно сопоставить пауз в аудио и позицию в тексте.
    """
    out: List[str] = []
    for sent in split_sentences(text):
        parts = [p.strip() for p in _CLAUSE_RE.split(sent) if p and p.strip()]
        out.extend(parts if parts else [sent])
    return out


# --------------------------------------------------------------------------- привязка слов


def attach_spans(words: Sequence[WordTiming], text: str) -> List[str]:
    """Заполняет char_start/char_end/sentence_end у слов по потоку «чистых» символов.

    Возвращает список предупреждений. Бросает AlignmentError, если слова
    выравнивателя не совпадают с текстом (значит, что-то пошло не так).
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
    # расширяем диапазоны на примыкающую пунктуацию и ставим sentence_end
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
    """Исходный текст (с пунктуацией) для непрерывной группы слов."""
    spans = [(w.char_start, w.char_end) for w in words if w.char_start is not None]
    if not spans:
        return ""
    return collapse_ws(text[spans[0][0]: spans[-1][1]])
