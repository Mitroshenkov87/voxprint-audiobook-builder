"""Application exceptions.

Every exception carries ``user_message`` - a friendly text in the UI language (see ``core.i18n``) that the GUI and the
CLI show as is.  Technical details go to ``details`` and into the log, never into the message.  ``kind`` is a short
stable string the UI uses to pick a dialog title and, for ``oom``, to offer a retry on the CPU.
"""
from __future__ import annotations

from core.i18n import tr


class DatasetMakerError(Exception):
    """Base class of all expected errors. ``user_message`` is shown to the user unchanged."""

    kind = "error"

    def __init__(self, user_message: str, *, details: str = "") -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.details = details


class CancelledByUser(DatasetMakerError):
    """Raised by ``CancelToken.check()`` when the user pressed Cancel."""
    kind = "cancelled"

    def __init__(self, user_message: str = "") -> None:
        super().__init__(user_message or tr("err.cancelled"))


class AudioReadError(DatasetMakerError):
    """The audio file is missing, unreadable or in an unsupported format."""
    kind = "audio_read"


class TextReadError(DatasetMakerError):
    """The text file is missing, empty or could not be decoded."""
    kind = "text_read"


class AlignmentError(DatasetMakerError):
    """The forced aligner failed or produced an unusable result."""
    kind = "alignment"


class AudioTextMismatchError(AlignmentError):
    """The audio and the text clearly do not belong together (wrong files or a partial reading)."""

    kind = "mismatch"


class ModelDownloadError(DatasetMakerError):
    """A model could not be downloaded. ``url`` points to the model page for a manual download."""

    kind = "download"

    def __init__(self, user_message: str, *, url: str = "", details: str = "") -> None:
        super().__init__(user_message, details=details)
        self.url = url


class OutOfMemoryError_(DatasetMakerError):
    """GPU memory ran out. The UI offers to continue on the CPU.

    The trailing underscore avoids shadowing the built-in ``MemoryError`` naming style of torch's own OOM error.
    """

    kind = "oom"

    def __init__(self, user_message: str = "", *, details: str = "") -> None:
        super().__init__(user_message or tr("err.oom"), details=details)


class TrainingError(DatasetMakerError):
    """LoRA training stopped because of an error."""
    kind = "training"


class UpdateError(DatasetMakerError):
    """A component or model update failed."""
    kind = "update"


class ExportError(DatasetMakerError):
    """The universal (merged) model could not be built: no disk space, no adapter, etc."""

    kind = "export"


class VoiceLibraryError(DatasetMakerError):
    """A voice could not be imported, found or changed in the voice library."""

    kind = "voice"


class VoiceRepositoryError(DatasetMakerError):
    """The online voice repository could not be read, or a downloaded voice failed its integrity check."""

    kind = "voice_repo"


class BookParseError(DatasetMakerError):
    """A book file (TXT / FB2 / EPUB) is unreadable, unsafe or has no text."""

    kind = "book"


class NarrationError(DatasetMakerError):
    """Synthesis or export of an audiobook failed (engine problem, ffmpeg problem, no disk space ...)."""

    kind = "narration"
