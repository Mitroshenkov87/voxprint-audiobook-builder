"""Исключения приложения. Каждое несёт понятное пользователю сообщение (на языке интерфейса, см. core/i18n)."""
from __future__ import annotations

from core.i18n import tr


class DatasetMakerError(Exception):
    """Базовая ошибка. `user_message` показывается пользователю как есть."""

    kind = "error"

    def __init__(self, user_message: str, *, details: str = "") -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.details = details


class CancelledByUser(DatasetMakerError):
    kind = "cancelled"

    def __init__(self, user_message: str = "") -> None:
        super().__init__(user_message or tr("err.cancelled"))


class AudioReadError(DatasetMakerError):
    kind = "audio_read"


class TextReadError(DatasetMakerError):
    kind = "text_read"


class AlignmentError(DatasetMakerError):
    kind = "alignment"


class AudioTextMismatchError(AlignmentError):
    """Аудио и текст явно не соответствуют друг другу."""

    kind = "mismatch"


class ModelDownloadError(DatasetMakerError):
    """Не удалось скачать модель. `url` - ссылка на страницу модели на Hugging Face."""

    kind = "download"

    def __init__(self, user_message: str, *, url: str = "", details: str = "") -> None:
        super().__init__(user_message, details=details)
        self.url = url


class OutOfMemoryError_(DatasetMakerError):
    """Не хватило памяти GPU. UI предлагает повторить на CPU."""

    kind = "oom"

    def __init__(self, user_message: str = "", *, details: str = "") -> None:
        super().__init__(user_message or tr("err.oom"), details=details)


class TrainingError(DatasetMakerError):
    kind = "training"


class UpdateError(DatasetMakerError):
    kind = "update"


class ExportError(DatasetMakerError):
    """Не удалось собрать универсальную модель (нет места на диске, нет адаптера и т.п.)."""

    kind = "export"
