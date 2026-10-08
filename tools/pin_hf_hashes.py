"""Pin the size + SHA-256 of every file of a Hugging Face model revision in ``infra/model_mirrors.json`` (a hashes-only
entry: no backup mirror, it verifies downloads from the original repository and lets Check & repair test the files).

The weights are NOT downloaded: the Hugging Face file listing (``/api/models/<repo>/tree/<revision>``) gives the size and,
for LFS files, the SHA-256.  Small non-LFS files (configs, vocabularies; capped at ``SMALL_MAX``) are fetched and hashed,
and their git blob id is checked against the listing.

    python tools/pin_hf_hashes.py --text-models            # every integrated text model that has no entry yet (dry run)
    python tools/pin_hf_hashes.py --text-models --write    # ... and write them into infra/model_mirrors.json
    python tools/pin_hf_hashes.py REPO REVISION --files config.json model.safetensors --license MIT --write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "infra" / "model_mirrors.json"
SMALL_MAX = 8 * 1024 * 1024
API = "https://huggingface.co"

Fetch = Callable[[str], bytes]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Voxprint-pin"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - https to huggingface.co only
        return r.read()


def listing(repo: str, revision: str, fetch: Fetch = _get) -> List[dict]:
    """Every file of the revision (recursive) as the Hugging Face API lists it."""
    url = f"{API}/api/models/{repo}/tree/{revision}?recursive=true"
    return [e for e in json.loads(fetch(url)) if e.get("type") == "file"]


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 - git's object id, not a security hash


def pin(repo: str, revision: str, files: Optional[Iterable[str]] = None, fetch: Fetch = _get) -> Dict[str, Dict[str, object]]:
    """``{file: {"size", "sha256"}}`` of ``files`` (default: every file) of the revision."""
    entries = {e["path"]: e for e in listing(repo, revision, fetch)}
    wanted = list(files) if files else sorted(entries)
    out: Dict[str, Dict[str, object]] = {}
    for name in wanted:
        e = entries.get(name)
        if e is None:
            raise SystemExit(f"{repo}@{revision[:8]}: no file {name}")
        lfs = e.get("lfs") or {}
        if lfs.get("oid"):
            out[name] = {"sha256": str(lfs["oid"]), "size": int(lfs.get("size", e["size"]))}
            continue
        if int(e["size"]) > SMALL_MAX:
            raise SystemExit(f"{name}: {e['size']} bytes without an LFS hash - not downloading it")
        data = fetch(f"{API}/{repo}/resolve/{revision}/{urllib.parse.quote(name)}")
        if len(data) != int(e["size"]) or git_blob_id(data) != e.get("oid"):
            raise SystemExit(f"{name}: the downloaded file does not match the listing")
        out[name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    return dict(sorted(out.items()))


def add_entry(manifest: dict, repo: str, revision: str, license_id: str, files: Dict[str, Dict[str, object]]) -> None:
    manifest["models"][repo] = {"files": files, "license": license_id, "source_repo": repo, "source_revision": revision}
    manifest["models"] = dict(sorted(manifest["models"].items()))


def text_model_jobs(manifest: dict) -> List[tuple]:
    """(repo, revision, files, licence, expected sha256 by file) for integrated text models without an entry."""
    sys.path.insert(0, str(ROOT))
    from infra import text_models

    jobs = []
    for m in text_models.REGISTRY:
        if m.integrated and m.repo and m.revision and m.files and m.repo not in manifest["models"]:
            jobs.append((m.repo, m.revision, list(m.files), m.license, dict(m.sha256)))
    return jobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo", nargs="?")
    ap.add_argument("revision", nargs="?")
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--license", default="")
    ap.add_argument("--text-models", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    jobs = text_model_jobs(manifest) if a.text_models else [(a.repo, a.revision, a.files, a.license, {})]
    for repo, rev, files, lic, expected in jobs:
        if not repo or not rev:
            ap.error("REPO and REVISION (or --text-models) are required")
        pinned = pin(repo, rev, files)
        for name, digest in expected.items():
            if name in pinned and pinned[name]["sha256"] != digest:
                raise SystemExit(f"{repo} {name}: Hugging Face says {pinned[name]['sha256']}, the registry {digest}")
        total = sum(int(str(m["size"])) for m in pinned.values())
        print(f"{repo}@{rev[:8]}: {len(pinned)} files, {total / 1e6:.1f} MB")
        add_entry(manifest, repo, rev, lic, pinned)
    if a.write:
        MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"written: {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
