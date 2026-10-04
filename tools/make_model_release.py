"""Create / update the GitHub pre-release that hosts the SMALL models (``models-v1``) and write ``infra/model_release.json``.

Small = the whole model is at most ``SMALL_MODEL_MAX`` bytes (Opus-MT translation models, SAGE clean-up).  Large models
(Qwen3 ...) stay on Hugging Face / ModelScope.  The files come from the original Hugging Face repositories at the pinned
commits listed in ``infra/model_mirrors.json`` (that file also holds the expected SHA-256 of every file); each file is
downloaded into ``--work`` (resumable), VERIFIED against that hash, and only then uploaded as one release asset
``<model>--<file>`` (``/`` in a path becomes ``--``).  Besides, ``SHA256SUMS-models.txt`` and ``models-manifest.json`` (a
copy of ``infra/model_release.json``) are attached.  Nothing heavy goes into the git tree.

    python tools/make_model_release.py                  # plan: sizes, disk space (nothing is downloaded)
    python tools/make_model_release.py --fetch          # download + verify into --work, write infra/model_release.json
    python tools/make_model_release.py --fetch --upload # ... and create the release / upload the assets with ``gh``

Needs: Python 3.9+, ``gh`` (logged in) only for ``--upload``.  Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infra import model_mirrors, model_release  # noqa: E402

DEFAULT_REPO = "Mitroshenkov87/voxprint-audiobook-builder"
DEFAULT_TAG = "models-v1"
GITHUB_ASSET_MAX = 2 * 1024 ** 3 - 1


def asset_name(model: str, rel: str) -> str:
    """``Helsinki-NLP/opus-mt-ru-en`` + ``pytorch_model.bin`` -> ``opus-mt-ru-en--pytorch_model.bin``."""
    return model.split("/")[-1] + "--" + rel.replace("/", "--")


def plan(entries: dict, small_max: int = model_release.SMALL_MODEL_MAX) -> dict:
    """``{source_repo: entry}`` of the small models (large ones are left out)."""
    out = {}
    for repo, e in sorted(entries.items()):
        files = e.downloadable()
        total = sum(int(m["size"]) for m in files.values())
        if total <= small_max and all(int(m["size"]) < GITHUB_ASSET_MAX for m in files.values()):
            out[repo] = e
    return out


def manifest_for(chosen: dict, repo: str, tag: str) -> dict:
    models = {}
    for src, e in chosen.items():
        models[src] = {"source_revision": e.source_revision, "license": e.license,
                       "files": {n: {"asset": asset_name(src, n), "size": int(m["size"]), "sha256": m["sha256"]}
                                 for n, m in sorted(e.downloadable().items())}}
    return {"schema": model_release.SCHEMA, "repo": repo, "tag": tag, "models": models}


def fetch_file(src: str, rev: str, rel: str, meta: dict, dest: Path) -> None:
    """Download one file from the original Hugging Face repository (resuming) and verify its SHA-256."""
    if dest.is_file() and dest.stat().st_size == int(meta["size"]) and model_release.sha256_file(dest) == meta["sha256"]:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".incomplete")
    url = f"https://huggingface.co/{src}/resolve/{rev}/{urllib.parse.quote(rel)}"
    have = part.stat().st_size if part.exists() else 0
    if have > int(meta["size"]):
        part.unlink()
        have = 0
    if have < int(meta["size"]):
        req = urllib.request.Request(url, headers={"User-Agent": "voxprint-release-tool", **({"Range": f"bytes={have}-"} if have else {})})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have and r.status == 206 else "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    if model_release.sha256_file(part) != meta["sha256"]:
        part.unlink()
        raise SystemExit(f"SHA-256 mismatch for {src}/{rel} (the file was deleted)")
    os.replace(part, dest)


def gh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], check=check, text=True, capture_output=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", default=str(ROOT / "build" / "model-release"), help="download folder (outside the git tree)")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--fetch", action="store_true", help="download and verify the files")
    ap.add_argument("--upload", action="store_true", help="create the pre-release and upload the assets with gh (implies --fetch)")
    ap.add_argument("--small-max-mb", type=int, default=model_release.SMALL_MODEL_MAX // 10 ** 6)
    a = ap.parse_args(argv)
    work = Path(a.work)
    entries = model_mirrors.load()
    chosen = plan(entries, a.small_max_mb * 10 ** 6)
    total = 0
    for repo, e in entries.items():
        size = sum(int(m["size"]) for m in e.downloadable().values())
        mark = "RELEASE" if repo in chosen else "stays on Hugging Face / ModelScope"
        print(f"{repo:45s} {size / 1e6:8.1f} MB  {mark}")
        total += size if repo in chosen else 0
    free = shutil.disk_usage(work if work.exists() else work.parent if work.parent.exists() else Path("/")).free
    print(f"small models: {len(chosen)}, {total / 1e6:.0f} MB; free disk: {free / 1e9:.1f} GB")
    if not (a.fetch or a.upload):
        return 0
    if free < total * 1.3:
        raise SystemExit("not enough free disk space")
    for src, e in chosen.items():
        for rel, meta in sorted(e.downloadable().items()):
            print(f"  fetch {src}/{rel} ({int(meta['size']) / 1e6:.1f} MB)", flush=True)
            fetch_file(src, e.source_revision, rel, meta, work / asset_name(src, rel))
    manifest = manifest_for(chosen, a.repo, a.tag)
    text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    (ROOT / "infra" / "model_release.json").write_text(text, encoding="utf-8")
    (work / "models-manifest.json").write_text(text, encoding="utf-8")
    sums = "".join(f"{m['sha256']}  {m['asset']}\n" for e in manifest["models"].values() for m in e["files"].values())
    (work / "SHA256SUMS-models.txt").write_text(sums, encoding="utf-8")
    print("verified; infra/model_release.json written")
    if not a.upload:
        return 0
    if gh("release", "view", a.tag, "--repo", a.repo, check=False).returncode != 0:
        gh("release", "create", a.tag, "--repo", a.repo, "--prerelease", "--title", f"Models ({a.tag})",
           "--notes", "Small models (Opus-MT translation, SAGE clean-up) mirrored byte-for-byte from Hugging Face at pinned commits; "
                      "SHA-256 in SHA256SUMS-models.txt and models-manifest.json. Used by Voxprint as the first download source.")
    assets = sorted(p for p in work.iterdir() if p.is_file() and not p.name.endswith(".incomplete"))
    for i in range(0, len(assets), 10):
        gh("release", "upload", a.tag, "--repo", a.repo, "--clobber", *[str(p) for p in assets[i:i + 10]])
        print(f"  uploaded {min(i + 10, len(assets))}/{len(assets)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
