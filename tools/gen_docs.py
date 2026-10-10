"""Write the generated documentation pages.

The command reference comes from ``cli.build_parser()``, so it stays aligned with
the flags the program actually accepts. The public roadmap is copied from the
repository root. API pages are one MkDocs stub per public module.

Example::

    python tools/gen_docs.py

Dependencies: the application imports (for the command reference), the standard
library. Licence: Apache-2.0.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("COLUMNS", "88")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VOXPRINT_ALLOW_NO_GPU", "1")
os.environ.setdefault("VOXPRINT_NO_ENV_PROBE", "1")

import cli as user_cli  # noqa: E402  the path insert above has to come first

DOCS = ROOT / "docs"
API = DOCS / "api"
GENERATED = DOCS / "generated"
MKDOCS = ROOT / "mkdocs.yml"
NAV_START = "# BEGIN GENERATED API NAV"
NAV_END = "# END GENERATED API NAV"

#: The windows the docstring gate and the API pages treat as the UI entry points.
UI_ENTRY = (
    "ui/__init__.py",
    "ui/studio.py",
    "ui/main_window.py",
    "ui/narrate_window.py",
    "ui/voices_window.py",
    "ui/settings_dialog.py",
    "ui/revoice_window.py",
)

DOC_GATE_PATHS = ("core", "infra", "cli.py", "main.py", *UI_ENTRY)


def module_files() -> List[Path]:
    """Python modules that get an API page, in a stable order."""
    files: List[Path] = []
    for folder in ("core", "infra"):
        files.extend(sorted(p for p in (ROOT / folder).rglob("*.py") if p.is_file()))
    files.append(ROOT / "cli.py")
    files.append(ROOT / "main.py")
    files.extend(ROOT / rel for rel in UI_ENTRY)
    return files


def module_name(path: Path) -> str:
    """Import name of ``path`` (``core`` for ``core/__init__.py``, ``cli`` for ``cli.py``)."""
    rel = path.relative_to(ROOT)
    if rel.name == "__init__.py":
        return ".".join(rel.parts[:-1])
    return ".".join(rel.with_suffix("").parts)


def api_page(name: str) -> str:
    """Markdown body for one module. MkDocs fills it from the docstrings at build time."""
    return f"# `{name}`\n\n::: {name}\n"


def api_index(names: Sequence[str]) -> str:
    """Index of the API pages, grouped by top-level package."""
    groups: Dict[str, List[str]] = {}
    for name in names:
        groups.setdefault(name.split(".", 1)[0], []).append(name)
    lines = [
        "# API reference",
        "",
        "Generated from the docstrings in `core/`, `infra/`, `cli.py`, `main.py`",
        "and the main windows. Do not edit the module pages by hand.",
        "",
    ]
    for group in ("core", "infra", "cli", "main", "ui"):
        members = groups.get(group, [])
        if not members:
            continue
        lines.append(f"## {group}")
        lines.append("")
        for name in members:
            lines.append(f"- [`{name}`]({name}.md)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def api_nav(names: Sequence[str]) -> str:
    """The API section of ``mkdocs.yml``, including the marker comments."""
    groups: Dict[str, List[str]] = {}
    for name in names:
        groups.setdefault(name.split(".", 1)[0], []).append(name)
    lines = [f"  {NAV_START}", "  - API reference:", "      - Overview: api/index.md"]
    for group in ("core", "infra", "cli", "main", "ui"):
        members = groups.get(group, [])
        if not members:
            continue
        if group in ("cli", "main"):
            lines.append(f"      - {group}: api/{group}.md")
            continue
        lines.append(f"      - {group}:")
        for name in members:
            title = name.split(".", 1)[1] if "." in name else name
            lines.append(f"          - {title}: api/{name}.md")
    lines.append(f"  {NAV_END}")
    return "\n".join(lines)


def _subparsers(parser: argparse.ArgumentParser) -> List[Tuple[List[str], argparse.ArgumentParser]]:
    """``(names, parser)`` for each direct subcommand. The first name is the canonical one."""
    found: List[Tuple[List[str], argparse.ArgumentParser]] = []
    actions = getattr(parser, "_actions", [])
    for action in actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        seen: Dict[int, List[str]] = {}
        order: List[int] = []
        choices = action.choices or {}
        for name, sub in choices.items():
            if not isinstance(sub, argparse.ArgumentParser):
                continue
            key = id(sub)
            if key not in seen:
                seen[key] = []
                order.append(key)
            seen[key].append(str(name))
        for key in order:
            names = seen[key]
            sub = choices[names[0]]
            if isinstance(sub, argparse.ArgumentParser):
                found.append((names, sub))
    return found


def _stabilize_help(text: str) -> str:
    """Join a usage ellipsis onto the choice list.

    Python 3.12 wraps the ``...`` of a subparser onto its own line. Python 3.14
    keeps ``{choices} ...`` together when the line is wide enough. The page is
    committed, so both interpreters have to emit the same text.
    """
    return re.sub(r"\}\n[ ]+\.\.\.(?=\n)", "} ...", text)


def _help_block(title: str, parser: argparse.ArgumentParser, aliases: Sequence[str]) -> str:
    text = _stabilize_help(parser.format_help()).replace("```", "'''").rstrip()
    alias = ""
    if aliases:
        joined = ", ".join(f"`{name}`" for name in aliases)
        alias = f"\n\nAlias: {joined}.\n"
    return f"## `{title}`\n{alias}\n```text\n{text}\n```\n"


def _walk(prefix: Sequence[str], parser: argparse.ArgumentParser) -> List[str]:
    blocks: List[str] = []
    for names, sub in _subparsers(parser):
        title = " ".join((*prefix, names[0]))
        blocks.append(_help_block(title, sub, names[1:]))
        blocks.extend(_walk([*prefix, names[0]], sub))
    return blocks


def render_cli() -> str:
    """The command reference, rendered from the live parser at a fixed width."""
    os.environ["COLUMNS"] = "88"
    parser = user_cli.build_parser()
    parts = [
        "# Command reference",
        "",
        "Generated from `cli.build_parser()`. Do not edit this file.",
        "Regenerate it with `python tools/gen_docs.py`.",
        "The longer guide, with JSON fields and exit codes, is [CLI.md](../CLI.md).",
        "",
        _help_block("voxprint", parser, ()),
        *_walk(["voxprint"], parser),
    ]
    return "\n".join(parts).rstrip() + "\n"


def render_roadmap() -> str:
    """The public roadmap, with its two doc links rewritten for ``docs/generated/``.

    The root file links to ``docs/USER-GUIDE.md`` and ``docs/CLI.md``, which is
    right when the file is read from the repository root. This copy lives in
    ``docs/generated/``, so those same pages are one directory up.
    """
    text = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    text = text.replace("(docs/USER-GUIDE.md)", "(../USER-GUIDE.md)")
    text = text.replace("(docs/CLI.md)", "(../CLI.md)")
    return text


def write_all() -> None:
    """Write the command reference, the roadmap copy, the API pages and the nav block."""
    GENERATED.mkdir(parents=True, exist_ok=True)
    API.mkdir(parents=True, exist_ok=True)
    (GENERATED / "cli.md").write_text(render_cli(), encoding="utf-8")
    (GENERATED / "roadmap.md").write_text(render_roadmap(), encoding="utf-8")
    names = [module_name(path) for path in module_files()]
    keep = {"index.md"}
    for name in names:
        (API / f"{name}.md").write_text(api_page(name), encoding="utf-8")
        keep.add(f"{name}.md")
    (API / "index.md").write_text(api_index(names), encoding="utf-8")
    for stale in API.glob("*.md"):
        if stale.name not in keep:
            stale.unlink()
    _replace_nav(api_nav(names))


def _replace_nav(block: str) -> None:
    text = MKDOCS.read_text(encoding="utf-8")
    start = text.find(NAV_START)
    end = text.find(NAV_END)
    if start < 0 or end < 0:
        raise SystemExit(f"{MKDOCS} is missing the API nav markers")
    # Keep the two-space indent of the marker lines; ``block`` already includes the markers.
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)
    else:
        line_end += 1
    MKDOCS.write_text(text[:line_start] + block + "\n" + text[line_end:], encoding="utf-8")


def main() -> None:
    """Regenerate every generated documentation page."""
    write_all()


if __name__ == "__main__":
    main()
