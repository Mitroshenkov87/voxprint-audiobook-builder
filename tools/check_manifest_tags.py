"""Fail when an installer manifest names a GitHub release tag that does not exist.

Local test servers (``127.0.0.1``, ``localhost``) are ignored. A URL such as
``https://github.com/org/repo/releases/download/v0.0.0-dev/manifest.json`` fails
when that tag is missing.

    python tools/check_manifest_tags.py build/online-release
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

TAG_URL = re.compile(
    r"https://github\.com/(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/releases/download/(?P<tag>[^/\"'\s]+)/"
)


def github_urls(text: str) -> list[tuple[str, str]]:
    """Return ``(repo, tag)`` pairs from GitHub release-download URLs in ``text``."""
    found: list[tuple[str, str]] = []
    for match in TAG_URL.finditer(text):
        pair = (match.group("repo"), match.group("tag"))
        if pair not in found:
            found.append(pair)
    return found


def tag_exists(repo: str, tag: str, *, opener: Callable[..., object] | None = None) -> bool:
    """True when ``refs/tags/<tag>`` exists on ``repo``. ``opener`` replaces ``urlopen`` in tests."""
    url = f"https://api.github.com/repos/{repo}/git/ref/tags/{tag}"
    request = urllib.request.Request(url, headers={"User-Agent": "voxprint-manifest-check", "Accept": "application/vnd.github+json"})
    open_url = opener or urllib.request.urlopen
    try:
        response = open_url(request, timeout=20)  # nosec B310 - GitHub API only
        try:
            status = getattr(response, "status", 200)
            return int(status) < 400
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def iter_manifests(paths: Iterable[Path]) -> list[Path]:
    """JSON files under the given files and directories."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            found.append(path)
    return found


def check(paths: Iterable[Path], *, exists: Callable[[str, str], bool] = tag_exists) -> list[str]:
    """Return one error line per release URL whose tag does not exist."""
    errors: list[str] = []
    for path in iter_manifests(paths):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        # A manifest is JSON. Scanning the text catches URLs nested in any string field.
        try:
            json.loads(text)
        except ValueError:
            errors.append(f"{path}: not JSON")
            continue
        for repo, tag in github_urls(text):
            try:
                ok = exists(repo, tag)
            except Exception as exc:  # noqa: BLE001 - the message names the URL that could not be checked
                errors.append(f"{path}: could not check {repo} tag {tag}: {exc}")
                continue
            if not ok:
                errors.append(f"{path}: GitHub tag {tag} does not exist ({repo})")
    return errors


def main(argv: list[str] | None = None) -> int:
    """Check the given manifest files. Exit 1 when a tag is missing."""
    parser = argparse.ArgumentParser(description="Fail if a manifest URL names a missing GitHub release tag.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    errors = check(args.paths)
    for line in errors:
        print(line, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
