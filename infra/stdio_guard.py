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
        self._inner = inner
        self._dead = inner is None

    def write(self, text: str) -> int:
        if self._dead or self._inner is None:
            return 0
        try:
            return self._inner.write(text)
        except Exception:  # noqa: BLE001 - errno 22 and a missing console must not escape
            self._dead = True
            return 0

    def flush(self) -> None:
        if self._dead or self._inner is None:
            return
        try:
            self._inner.flush()
        except Exception:  # noqa: BLE001 - interpreter shutdown flushes stdout
            self._dead = True

    def __getattr__(self, name: str):
        if self._inner is None:
            raise AttributeError(name)
        return getattr(self._inner, name)


def guard_stdio() -> None:
    """Wrap ``sys.stdout`` and ``sys.stderr`` once. A later flush of a dead handle does not raise."""
    if not isinstance(sys.stdout, GuardedStream):
        sys.stdout = GuardedStream(sys.stdout)  # type: ignore[assignment]
    if not isinstance(sys.stderr, GuardedStream):
        sys.stderr = GuardedStream(sys.stderr)  # type: ignore[assignment]


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
