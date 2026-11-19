"""Check that ``BUILD.json``'s codename does not repeat an earlier release in this repository.

History is read from ``CHANGELOG.md`` headings and from ``docs/RELEASE-NOTES-*.md``.
The file ``BUILD.json`` is only read. The suite codename registry is private and is not stored here.

    python tools/codenames.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_HEADING = re.compile(
    r'^## \[[^\]]+\] - \d{4}-\d{2}-\d{2} - build (\d+) "([A-Za-z][A-Za-z0-9-]*)"',
    re.MULTILINE,
)
_NOTE_FILE = re.compile(r"^RELEASE-NOTES-(\d+)-([A-Za-z0-9-]+)\.md$")
_NOTE_NAME = re.compile(r"codename is ([A-Za-z][A-Za-z0-9-]*)", re.IGNORECASE)


class CodenameError(Exception):
    """``BUILD.json``'s codename collides with release history, or that history disagrees with itself."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("\n".join(problems))


def _root(root: Path | None) -> Path:
    return ROOT if root is None else Path(root)


def history(root: Path | None = None) -> list[tuple[int, str, str]]:
    """``(build number, codename, source)`` from the changelog and the release notes.

    A release note contributes one row when its file name and its "codename is" sentence
    name the same word. A mismatch contributes both spellings so the check can report it.
    """
    base = _root(root)
    rows: list[tuple[int, str, str]] = []
    changelog = base / "CHANGELOG.md"
    if changelog.is_file():
        text = changelog.read_text(encoding="utf-8")
        for build_s, name in _HEADING.findall(text):
            rows.append((int(build_s), name, "CHANGELOG.md"))
    notes = base / "docs"
    if notes.is_dir():
        for path in sorted(notes.glob("RELEASE-NOTES-*.md")):
            match = _NOTE_FILE.match(path.name)
            if match is None:
                continue
            build = int(match.group(1))
            from_file = match.group(2)
            prose = _NOTE_NAME.search(path.read_text(encoding="utf-8"))
            from_prose = prose.group(1) if prose else ""
            source = f"docs/{path.name}"
            if from_prose and from_prose.casefold() == from_file.casefold():
                rows.append((build, from_prose, source))
            else:
                rows.append((build, from_file, source))
                if from_prose:
                    rows.append((build, from_prose, source))
    return rows


def problems(root: Path | None = None) -> list[str]:
    """Human-readable collisions. Empty when the current codename is allowed.

    The codename in ``BUILD.json`` may be new, or it may be the codename of the newest
    build in this history. It may not be the codename of an earlier build. Two builds
    must not share one codename, and one build must not be given two spellings.
    """
    base = _root(root)
    build_path = base / "BUILD.json"
    try:
        current = str(json.loads(build_path.read_text(encoding="utf-8")).get("codename") or "").strip()
    except (OSError, ValueError) as exc:
        return [f"BUILD.json could not be read: {exc}"]
    if not current:
        return ["BUILD.json has no codename"]
    by_build: dict[int, set[str]] = defaultdict(set)
    sources: dict[tuple[int, str], list[str]] = defaultdict(list)
    for build, name, source in history(base):
        folded = name.casefold()
        by_build[build].add(folded)
        sources[(build, folded)].append(source)
    found: list[str] = []
    for build, names in sorted(by_build.items()):
        if len(names) > 1:
            found.append(f"build {build} is named {', '.join(sorted(names))}")
    by_name: dict[str, set[int]] = defaultdict(set)
    for build, names in by_build.items():
        if len(names) == 1:
            by_name[next(iter(names))].add(build)
    for name, builds in sorted(by_name.items()):
        if len(builds) > 1:
            found.append(f"{name} is used by builds {', '.join(str(n) for n in sorted(builds))}")
    if found:
        return found
    latest = max(by_build) if by_build else None
    latest_names = by_build.get(latest, set()) if latest is not None else set()
    folded = current.casefold()
    if folded in by_name and folded not in latest_names:
        earlier = sorted(by_name[folded])
        refs = sources.get((earlier[0], folded), [])
        found.append(
            f"BUILD.json codename {current} is already used by build {', '.join(str(n) for n in earlier)} "
            f"({'; '.join(refs)})"
        )
    return found


def assert_unique(root: Path | None = None) -> None:
    """Raise :class:`CodenameError` when :func:`problems` is not empty."""
    found = problems(root)
    if found:
        raise CodenameError(found)


def main() -> int:
    """Print any collision and return 1. Return 0 when the current codename is allowed."""
    try:
        assert_unique()
    except CodenameError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("codename is unique against this repository's release history")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
