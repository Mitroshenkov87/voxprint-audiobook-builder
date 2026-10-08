"""Check that every file of ``infra/runtime_lock.json`` still exists at its upstream address with the pinned size (HEAD requests, no download).

Run by the CI of the thin installer before it publishes a manifest, and by hand after ``make_runtime_lock.py``.
Exit code 0 = all fine, 1 = problems (printed).
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import urllib.request
from pathlib import Path

LOCK = Path(__file__).resolve().parent.parent / "infra" / "runtime_lock.json"


def head(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "voxprint-check/1"})
    with urllib.request.urlopen(req, timeout=60) as r:  # nosec B310 - dev tool, https URLs from our own lists
        return int(r.headers.get("Content-Length", "-1"))


def check(w: dict) -> str:
    try:
        n = head(w["url"])
    except Exception as exc:  # noqa: BLE001
        return f"{w['file']}: {type(exc).__name__}: {exc}"
    return "" if n == w["size"] else f"{w['file']}: size {n} at the upstream site, {w['size']} in the lock"


def main() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    with cf.ThreadPoolExecutor(8) as ex:
        bad = [m for m in ex.map(check, lock["wheels"]) if m]
    for m in bad:
        print("PROBLEM:", m)
    print(f"{len(lock['wheels']) - len(bad)} of {len(lock['wheels'])} files are where the lock says")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
