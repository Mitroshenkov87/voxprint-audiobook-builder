"""Write the CI build number into ``core/build_stamp.py`` and onto ``Voxprint.exe``.

Called from ``build_thin.bat``. The number comes from ``VOXPRINT_BUILD`` / ``VOXPRINT_CODENAME``, or from a
``credits.json`` that ``tools/build_number.py --stamp`` already filled. A local build with no number is left alone.
``BUILD.json`` is not read and not written.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# build_thin.bat runs this file (`python tools\embed_build_stamp.py`), so sys.path[0] is tools/, not the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.build_stamp import write_trailer  # noqa: E402
MODULE = ROOT / "core" / "build_stamp.py"
_BUILD_RE = re.compile(r"^BUILD = \d+[ \t]*$", re.MULTILINE)
_NAME_RE = re.compile(r'^CODENAME = "[^"]*"[ \t]*$', re.MULTILINE)


def resolve_stamp(credits: Path) -> tuple:
    """``(build, codename)`` from the environment, then from ``credits``."""
    env_build = os.environ.get("VOXPRINT_BUILD", "").strip()
    env_name = os.environ.get("VOXPRINT_CODENAME", "").strip()
    build = int(env_build) if env_build.isdigit() else 0
    codename = env_name
    if build <= 0 or not codename:
        try:
            app = json.loads(credits.read_text(encoding="utf-8")).get("app") or {}
        except (OSError, ValueError, AttributeError):
            app = {}
        if build <= 0:
            raw = str(app.get("build", "0") or "0")
            build = int(raw) if raw.isdigit() else 0
        if not codename:
            codename = str(app.get("codename") or "")
    return build, codename


def write_module(path: Path, build: int, codename: str) -> None:
    """Replace the ``BUILD`` and ``CODENAME`` assignments. The rest of the file stays as it is in git."""
    text = path.read_text(encoding="utf-8")
    name = "".join(ch for ch in str(codename) if ch.isascii() and (ch.isalnum() or ch in "-_"))
    text2, n_build = _BUILD_RE.subn(f"BUILD = {int(build)}", text, count=1)
    text3, n_name = _NAME_RE.subn(f'CODENAME = "{name}"', text2, count=1)
    if n_build != 1 or n_name != 1:
        raise RuntimeError(f"could not stamp {path}")
    path.write_text(text3, encoding="utf-8")


def restore_module(path: Path) -> None:
    """Put the git values back so a local thin build does not leave the source tree dirty."""
    write_module(path, 0, "")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Embed the Voxprint build number into the frozen program")
    parser.add_argument("--write-module", nargs="?", const=str(MODULE), default=None,
                        help="Stamp core/build_stamp.py (default path) before PyInstaller")
    parser.add_argument("--restore-module", nargs="?", const=str(MODULE), default=None,
                        help="Set BUILD = 0 again after PyInstaller")
    parser.add_argument("--exe", default=None, help="Append the stamp to this Voxprint.exe")
    parser.add_argument("--credits", default=str(ROOT / "credits.json"))
    args = parser.parse_args(argv)
    if args.restore_module:
        restore_module(Path(args.restore_module))
    build, codename = resolve_stamp(Path(args.credits))
    if args.write_module:
        if build > 0:
            write_module(Path(args.write_module), build, codename)
    if args.exe and build > 0:
        write_trailer(args.exe, build, codename)
    return 0


if __name__ == "__main__":
    sys.exit(main())
