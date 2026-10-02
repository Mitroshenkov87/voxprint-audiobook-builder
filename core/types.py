"""Общие простые структуры данных."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class WordTiming:
    """Слово с временными метками (секунды от начала всего аудио).

    `word` - токен в том виде, как его вернул выравниватель (без пунктуации).
    `char_start/char_end` - диапазон в исходном (нормализованном) тексте,
    заполняется функцией core.text_utils.attach_spans.
    """

    word: str
    start: float
    end: float
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    sentence_end: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return {"word": self.word, "start": round(self.start, 3), "end": round(self.end, 3)}


@dataclass
class Segment:
    """Сегмент датасета: интервал аудио + чистый текст."""

    index: int  # 1-based
    start: float
    end: float
    text: str
    n_words: int = 0
    ends_sentence: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def filename(self) -> str:
        return f"segment_{self.index:03d}.wav"
