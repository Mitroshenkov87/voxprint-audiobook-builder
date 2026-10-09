"""Build number baked into a frozen Voxprint.

``BUILD`` is 0 in git. ``tools/embed_build_stamp.py`` writes the CI number into this file before PyInstaller and
appends the same number to ``Voxprint.exe``. The executable trailer wins over this module, which wins over
``credits.json``, so an exe-only patch reports its own build.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

BUILD = 0
CODENAME = ""

MARKER = b"\nVOXPRINT_BUILD_STAMP\x00"
_TAIL = 512


def read_trailer(path: Union[str, Path]) -> Optional[Tuple[int, str]]:
    """``(build, codename)`` from the last bytes of ``path``, or ``None`` when there is no stamp."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - _TAIL))
            tail = fh.read()
    except OSError:
        return None
    idx = tail.rfind(MARKER)
    if idx < 0:
        return None
    parts = tail[idx + len(MARKER):].split(b"\x00", 2)
    if len(parts) < 2:
        return None
    try:
        build = int(parts[0].decode("ascii"))
    except (UnicodeError, ValueError):
        return None
    if build <= 0:
        return None
    try:
        name = parts[1].decode("ascii")
    except UnicodeError:
        name = ""
    return build, name


def write_trailer(path: Union[str, Path], build: int, codename: str) -> None:
    """Append a build stamp, replacing one that is already in the tail. ``build`` <= 0 does nothing.

    The marker is not a PyInstaller onefile cookie. An onedir bootloader ignores bytes after the PE image.
    """
    build = int(build)
    if build <= 0:
        return
    name = "".join(ch for ch in str(codename) if ch.isascii() and (ch.isalnum() or ch in "-_"))
    blob = MARKER + str(build).encode("ascii") + b"\x00" + name.encode("ascii") + b"\x00"
    dest = Path(path)
    data = dest.read_bytes()
    idx = data.rfind(MARKER, max(0, len(data) - _TAIL))
    if idx >= 0:
        data = data[:idx]
    dest.write_bytes(data + blob)
