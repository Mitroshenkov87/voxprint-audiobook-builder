"""Split a PyInstaller ``--onedir`` build into the release assets of the ONLINE installer.

    python tools/make_online_payload.py --dist dist/Voxprint --out build/online --tag v0.1.0-beta \
        --repo Mitroshenkov87/voxprint [--channel beta|stable] [--base-url URL] [--limit-mib 1800]

Writes ``Voxprint-payload-01.zip`` ... (every zip stays under ``--limit-mib`` MiB: a GitHub release asset must be below
2 GiB) and ``manifest-<channel>.json`` (schema 1: id, file, url, size, sha256, unpacked_bytes, markers per component).  The
download address of each part is ``<base-url>/<file>``; by default the assets of the release ``--tag`` of ``--repo``.
The channel is ``beta`` for a tag with a pre-release suffix (``v0.1.0-beta``), ``stable`` otherwise, unless given.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import List, Optional

SCHEMA = 1
DEFAULT_LIMIT_MIB = 1800
PREFIX = "Voxprint-payload"


def channel_for(tag: str) -> str:
    """``beta`` when the tag carries a pre-release suffix (``-beta``, ``-rc1`` ...), else ``stable``."""
    return "beta" if "-" in tag.lstrip("vV") else "stable"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def list_files(dist: Path) -> List[Path]:
    """All files, the program itself first (so that part 01 alone is a recognisable core), then the rest sorted."""
    files = sorted(p for p in dist.rglob("*") if p.is_file())
    files.sort(key=lambda p: (0 if p.parent == dist and p.name.lower() == "voxprint.exe" else 1 if p.parent == dist
                              else 2 if "torch" not in p.parts else 3, str(p).lower()))
    return files


def build(dist: Path, out: Path, tag: str, repo: str, channel: Optional[str] = None, base_url: str = "",
          limit_mib: int = DEFAULT_LIMIT_MIB, app_version: str = "") -> Path:
    """Write the zips and the manifest into ``out``; returns the manifest path."""
    dist, out = Path(dist), Path(out)
    if not dist.is_dir():
        raise SystemExit(f"{dist} is not a folder")
    out.mkdir(parents=True, exist_ok=True)
    channel = channel or channel_for(tag)
    base = (base_url or f"https://github.com/{repo}/releases/download/{tag}").rstrip("/")
    limit = limit_mib * 1024 * 1024
    parts: List[dict] = []
    cur: Optional[zipfile.ZipFile] = None
    cur_path: Optional[Path] = None
    cur_raw, cur_markers = 0, []

    def close() -> None:
        nonlocal cur, cur_path, cur_raw, cur_markers
        if cur is None:
            return
        cur.close()
        assert cur_path is not None
        parts.append({"path": cur_path, "raw": cur_raw, "markers": cur_markers})
        cur, cur_path, cur_raw, cur_markers = None, None, 0, []

    for f in list_files(dist):
        if cur is None:
            cur_path = out / f"{PREFIX}-{len(parts) + 1:02d}.zip"
            cur = zipfile.ZipFile(cur_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True)
        rel = f.relative_to(dist).as_posix()
        cur.write(f, rel)
        cur_raw += f.stat().st_size
        if len(cur_markers) < 3 and f.stat().st_size > 0:
            cur_markers.append(rel)
        if cur_path is not None and cur_path.stat().st_size >= limit:       # rotate after the file that crossed the limit
            close()
    close()
    comps = []
    for i, p in enumerate(parts, 1):
        path = p["path"]
        if path.stat().st_size >= 2 * 1024 ** 3:
            raise SystemExit(f"{path.name} is {path.stat().st_size} bytes: over the 2 GiB asset limit, lower --limit-mib")
        comps.append({"id": f"payload-{i:02d}", "file": path.name, "url": f"{base}/{path.name}", "size": path.stat().st_size,
                      "sha256": sha256_of(path), "unpacked_bytes": p["raw"], "markers": p["markers"]})
    manifest = {"schema": SCHEMA, "channel": channel, "app_version": app_version or tag.lstrip("vV"), "tag": tag, "repo": repo,
                "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "components": comps}
    mp = out / f"manifest-{channel}.json"
    mp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return mp


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repo", default="Mitroshenkov87/voxprint")
    ap.add_argument("--channel", choices=("beta", "stable"))
    ap.add_argument("--base-url", default="")
    ap.add_argument("--limit-mib", type=int, default=DEFAULT_LIMIT_MIB)
    a = ap.parse_args(argv)
    mp = build(Path(a.dist), Path(a.out), a.tag, a.repo, a.channel, a.base_url, a.limit_mib)
    m = json.loads(mp.read_text(encoding="utf-8"))
    for c in m["components"]:
        print(f"{c['file']}  {c['size']:>12}  {c['sha256']}")
    print(f"manifest: {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
