"""Build number and codename of an installer build.

The number is the explicit ``build`` field in ``BUILD.json`` (1002 for the next RC), unless the release tag
ends in ``+<number>`` or ``-build<number>`` (``v1.0.0-rc.4+1005``, ``v1.0.0-rc.4-build1002``). A plain version
tag such as ``v1.0.0-rc.3`` does not carry a number. ``VOXPRINT_BUILD`` overrides both. ``GITHUB_RUN_NUMBER``
is ignored, so a failed run does not consume a number. ``offset`` stays in ``BUILD.json`` as history only.

``codename`` in ``BUILD.json`` is one Biblical Hebrew word (Latin transliteration, ASCII) describing the
build's changes. Edit it per release; this file does not choose a new one.

    python tools/build_number.py            # prints the number
    python tools/build_number.py --stamp    # also writes number, codename and commit into credits.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_TAG_BUILD = re.compile(r"(?:\+|-build)(\d+)$", re.IGNORECASE)


def info() -> dict:
    """``BUILD.json``: ``build``, ``codename``, and historical ``offset``."""
    return json.loads((ROOT / "BUILD.json").read_text(encoding="utf-8"))


def release_tag(env) -> str:
    """The tag this run is publishing, or ``""`` when the run is not a release."""
    explicit = str(env.get("VOXPRINT_RELEASE_TAG") or env.get("REL_TAG") or "").strip()
    if explicit:
        return explicit
    ref = str(env.get("GITHUB_REF") or "")
    name = str(env.get("GITHUB_REF_NAME") or "").strip()
    if ref.startswith("refs/tags/") and name:
        return name
    return ""


def build_from_tag(tag: str) -> int:
    """A number carried by the tag (``+1005`` or ``-build1002``). A version-only tag is 0."""
    match = _TAG_BUILD.search(str(tag or "").strip())
    if not match:
        return 0
    return int(match.group(1))


def build_number(env=None) -> int:
    """``VOXPRINT_BUILD``, else the release tag, else ``BUILD.json`` ``build``, else 0.

    ``GITHUB_RUN_NUMBER`` and ``offset`` are not used.
    """
    env = os.environ if env is None else env
    override = str(env.get("VOXPRINT_BUILD", "")).strip()
    if override.isdigit():
        return int(override)
    tagged = build_from_tag(release_tag(env))
    if tagged > 0:
        return tagged
    raw = info().get("build", 0)
    try:
        number = int(raw)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


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
