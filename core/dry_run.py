"""Validate CLI inputs without loading a model, using the GPU, or downloading.

The process exit codes are the same numbers documented in ``docs/CLI.md``.
A dry-run never returns the missing-model code or the GPU codes: those belong to a real run.

Example::

    from core.dry_run import load_book_input, check_output
    book, fmt = load_book_input(Path("story.vxbook"))
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from core.book_parsers import Book, load_book
from core.errors import BookParseError

EXIT_INTERNAL = 1
EXIT_INPUT = 3

RESULT_KEYS = (
    "type", "ok", "command", "exit_code", "outputs", "warnings",
    "duration_s", "error", "hint", "dry_run",
)


class DryRunFailure(Exception):
    """A dry-run check failed with a documented exit code."""

    def __init__(self, code: int, message: str, *, hint: str = "") -> None:
        """Store the exit code, the message and the suggested fix."""
        super().__init__(message)
        self.code = int(code)
        self.message = message
        self.hint = hint


def book_format(path: Path) -> str:
    """Short format name for a recognised book path (``vxbook``, ``fb2.zip``, ``txt``, ...)."""
    name = path.name.lower()
    if name.endswith(".fb2.zip"):
        return "fb2.zip"
    return path.suffix.lower().lstrip(".")


def load_book_input(path: Path) -> tuple[Book, str]:
    """Open ``path`` and recognise its format, including a ``.vxbook``.

    Raises :class:`DryRunFailure` with exit code 3 when the file is missing or not a book
    this program can read. Parsing a ``.vxbook`` checks the archive; it does not load a model.
    """
    path = Path(path)
    if not path.is_file():
        raise DryRunFailure(
            EXIT_INPUT, f"book not found: {path}",
            hint="Pass a TXT, Markdown, FB2, FB2.ZIP, EPUB or .vxbook file that exists.",
        )
    try:
        book = load_book(path)
    except BookParseError as exc:
        raise DryRunFailure(
            EXIT_INPUT, exc.user_message or str(exc),
            hint="Check the path and the file type. A .vxbook must be a valid Voxprint book archive.",
        ) from exc
    return book, book_format(path)


def check_output(path: Path, *, kind: str) -> str:
    """Accept an output path without creating it.

    ``kind="dir"`` is a directory that already exists, or a path whose parent directory exists.
    ``kind="file"`` is a file path (not an existing directory) whose parent directory exists.
    """
    path = Path(path).expanduser()
    if kind == "dir":
        if path.exists() and not path.is_dir():
            raise DryRunFailure(EXIT_INPUT, f"output path is not a directory: {path}",
                                hint="Pass a folder for --out.")
        parent = path if path.is_dir() else path.parent
        if not parent.is_dir():
            raise DryRunFailure(EXIT_INPUT, f"output directory does not exist: {parent}",
                                hint="Create the folder, or pass a path inside a folder that exists.")
        return str(path)
    if path.exists() and path.is_dir():
        raise DryRunFailure(EXIT_INPUT, f"output path is a directory: {path}",
                            hint="Pass a file path for --out.")
    parent = path.parent
    if parent == Path(""):
        parent = Path(".")
    if not parent.is_dir():
        raise DryRunFailure(EXIT_INPUT, f"output directory does not exist: {parent}",
                            hint="Create the folder that should hold the output file.")
    return str(path)


def existing_input(path: Path, label: str) -> str:
    """Require a file or directory that is already on disk. Does not open it as a model."""
    path = Path(path)
    if not path.exists():
        raise DryRunFailure(EXIT_INPUT, f"{label} not found: {path}", hint="Check the path and run the command again.")
    return str(path)


def schema_error(payload: dict, extra: Sequence[str] = ()) -> str:
    """Empty when ``payload`` is a dry-run result object. Otherwise one sentence naming the gap."""
    if not isinstance(payload, dict):
        return "result is not an object"
    missing = [key for key in RESULT_KEYS if key not in payload]
    if missing:
        return "result is missing " + ", ".join(missing)
    if payload.get("type") != "result":
        return "result type is not result"
    if payload.get("dry_run") is not True:
        return "result dry_run is not true"
    if not isinstance(payload.get("outputs"), list) or not isinstance(payload.get("warnings"), list):
        return "outputs and warnings must be arrays"
    if not isinstance(payload.get("exit_code"), int):
        return "exit_code must be an integer"
    if payload.get("ok") is not (payload["exit_code"] == 0):
        return "ok does not match exit_code"
    if not isinstance(payload.get("duration_s"), (int, float)):
        return "duration_s must be a number"
    if payload["ok"]:
        absent = [key for key in extra if key not in payload]
        if absent:
            return "result is missing " + ", ".join(absent)
    elif payload.get("error") in (None, ""):
        return "a failed result needs an error string"
    return ""
