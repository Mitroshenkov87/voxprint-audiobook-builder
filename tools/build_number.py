"""Build number and codename of a CI installer build.  The number is ``GITHUB_RUN_NUMBER`` of the build workflow + ``offset``
in ``BUILD.json`` (the first numbered build, of v0.1.1-beta, is 665), so every later CI build gets the next number
automatically; ``VOXPRINT_BUILD`` overrides it; a local build without either is build 0.  ``codename`` in ``BUILD.json`` is
one Biblical Hebrew word (Latin transliteration, ASCII) describing the build's changes - edit it per release.

    python tools/build_number.py            # prints the number
    python tools/build_number.py --stamp    # also writes number, codename and commit into credits.json (app.build/codename/commit)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_number(env=None) -> int:
    env = os.environ if env is None else env
    if str(env.get("VOXPRINT_BUILD", "")).strip().isdigit():
        return int(env["VOXPRINT_BUILD"])
    run = str(env.get("GITHUB_RUN_NUMBER", "")).strip()
    if not run.isdigit():
        return 0
    return int(run) + int(info()["offset"])


def info() -> dict:
    """``BUILD.json``: ``offset`` and ``codename``."""
    return json.loads((ROOT / "BUILD.json").read_text(encoding="utf-8"))


def stamp(credits: Path, build: int, commit: str = "", codename: str = "") -> None:
    data = json.loads(credits.read_text(encoding="utf-8"))
    data.setdefault("app", {})["build"] = build
    if codename:
        data["app"]["codename"] = codename
    if commit:
        data["app"]["commit"] = commit[:8]
    credits.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    n = build_number()
    if "--stamp" in argv:
        stamp(ROOT / "credits.json", n, os.environ.get("GITHUB_SHA", ""), str(info().get("codename", "")))
    print(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
