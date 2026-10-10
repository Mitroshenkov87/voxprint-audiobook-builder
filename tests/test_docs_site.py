"""Generated docs stay aligned with the code, and public docstrings stay complete.

``--fail-under 100`` is the floor for the public modules this file lists. Do not lower it.
"""
from __future__ import annotations

import subprocess
import sys

from tools.gen_docs import (
    API,
    DOC_GATE_PATHS,
    GENERATED,
    ROOT,
    api_index,
    api_nav,
    api_page,
    module_files,
    module_name,
    render_cli,
    render_roadmap,
)


def test_cli_reference_matches_the_parser() -> None:
    assert (GENERATED / "cli.md").read_text(encoding="utf-8") == render_cli()


def test_roadmap_page_matches_the_root_plan() -> None:
    page = (GENERATED / "roadmap.md").read_text(encoding="utf-8")
    assert page == render_roadmap()
    root = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    assert "(docs/USER-GUIDE.md)" in root and "(../USER-GUIDE.md)" in page
    assert "(docs/CLI.md)" in root and "(../CLI.md)" in page


def test_api_pages_match_the_modules() -> None:
    names = [module_name(path) for path in module_files()]
    assert (API / "index.md").read_text(encoding="utf-8") == api_index(names)
    for name in names:
        assert (API / f"{name}.md").read_text(encoding="utf-8") == api_page(name)
    assert api_nav(names) in (ROOT / "mkdocs.yml").read_text(encoding="utf-8")


def test_public_docstrings_stay_at_full_coverage() -> None:
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--config", str(ROOT / "tools" / "ruff-docs.toml"), *DOC_GATE_PATHS],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert ruff.returncode == 0, ruff.stdout + ruff.stderr
    # 100 is the floor. A new public module, class or function needs a Google-style docstring.
    interrogate = subprocess.run(
        [
            sys.executable, "-m", "interrogate", "--style", "google",
            "-s", "-n", "-C", "-O", "--fail-under", "100", "--quiet",
            *DOC_GATE_PATHS,
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert interrogate.returncode == 0, interrogate.stdout + interrogate.stderr


def test_public_roadmap_keeps_private_notes_out() -> None:
    text = (ROOT / "ROADMAP.md").read_text(encoding="utf-8").lower()
    for banned in ("torah", "tanakh", "aleksandr", "zahav", "tikkun", "menuchah", "signpath"):
        assert banned not in text
