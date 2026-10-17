"""Stdout and stderr that cannot raise.

A packaged ``Voxprint.exe`` is a windowed program. PowerShell ``& Voxprint.exe status --json > file`` can leave
stdout as a real object whose write fails with ``OSError`` errno 22 (invalid handle), not ``None``. That error must
not escape: the windowed bootloader shows a traceback window and the process hangs.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional, TextIO


class GuardedStream:
    """Delegates to a stream and swallows a dead handle, including errno 22, on ``write`` and ``flush``."""

    def __init__(self, inner: Optional[TextIO]) -> None:
        """Wrap ``inner``. A missing stream is already treated as dead."""
        self._inner = inner
        self._dead = inner is None

    def write(self, text: str) -> int:
        """Write ``text``, or return 0 when the handle is dead."""
        if self._dead or self._inner is None:
            return 0
        try:
            return self._inner.write(text)
        except Exception:  # noqa: BLE001 - errno 22 and a missing console must not escape
            self._dead = True
            return 0

    def flush(self) -> None:
        """Flush the inner stream, and mark it dead if that fails."""
        if self._dead or self._inner is None:
            return
        try:
            self._inner.flush()
        except Exception:  # noqa: BLE001 - interpreter shutdown flushes stdout
            self._dead = True

    def __getattr__(self, name: str):
        """Forward any other attribute to the inner stream."""
        if self._inner is None:
            raise AttributeError(name)
        return getattr(self._inner, name)


def write_utf8(stream, text: str) -> None:
    """Write ``text`` as UTF-8, even when the console code page cannot be changed.

    A Windows redirect (cp1251, cp1252) replaces Cyrillic with ``?`` once ``reconfigure`` fails.
    The bytes then go to the underlying buffer. A stream that is already UTF-8 is written as text.
    """
    line = text if str(text).endswith("\n") else str(text) + "\n"
    encoding = (getattr(stream, "encoding", None) or "").lower().replace("_", "").replace("-", "")
    if encoding not in ("utf8", "utf8sig"):
        buf = getattr(stream, "buffer", None)
        if buf is not None and hasattr(buf, "write"):
            try:
                buf.write(line.encode("utf-8", errors="replace"))
                flush = getattr(buf, "flush", None)
                if flush is not None:
                    flush()
                return
            except Exception:  # noqa: BLE001 - fall through to the text write
                pass
    stream.write(line)
    flush = getattr(stream, "flush", None)
    if flush is not None:
        flush()


def guard_stdio() -> None:
    """Wrap ``sys.stdout`` and ``sys.stderr`` once. A later flush of a dead handle does not raise."""
    if not isinstance(sys.stdout, GuardedStream):
        sys.stdout = GuardedStream(sys.stdout)
    if not isinstance(sys.stderr, GuardedStream):
        sys.stderr = GuardedStream(sys.stderr)


def install_cli_excepthook() -> None:
    """Log an uncaught CLI exception and do not call the default hook.

    The default hook writes a traceback to stderr. On a windowed exe that write raises again, and the bootloader
    shows the traceback in a window.
    """

    def hook(etype, value, tb):
        if etype is not None and not issubclass(etype, KeyboardInterrupt):
            logging.getLogger("voxprint.crash").error("uncaught exception", exc_info=(etype, value, tb))

    if getattr(sys.excepthook, "_voxprint_cli", False):
        return
    hook._voxprint_cli = True  # type: ignore[attr-defined]
    sys.excepthook = hook
