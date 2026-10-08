"""Turn ``infra/runtime_lock.json`` into manifest components of the thin installer.

Every pinned wheel becomes a component whose ``url`` is the ORIGINAL file (PyPI / download.pytorch.org) and whose ``urls`` may hold our
own mirror as a fallback.  Nothing third-party is built, repacked or uploaded: the release holds the small shell and the manifest only.
The exception are the few sdist-only pure-Python packages (``sdist`` in the lock): a wheel is built from the pinned, hash-checked
sdist (seconds) and shipped as a small release asset.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

#: Order of the modules in the Components window / the install order: small ones first, PyTorch last.
ORDER = ("libs", "text", "audio", "torch")


def cid(w: dict) -> str:
    """Component id: ``whl-<dist>-<version>[-<flavor>]`` (the manifest allows ``[A-Za-z0-9._-]{1,64}``)."""
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", f"whl-{w['dist']}-{w['version']}")
    return base[:64]


def marker(w_file: str) -> str:
    """Folder of the wheel's metadata after installation: ``<name>-<version>.dist-info/METADATA``."""
    parts = w_file[:-4].split("-")
    return f"{parts[0]}-{parts[1]}.dist-info/METADATA"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _fetch(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "voxprint-build/1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:  # nosec B310 - dev tool, https URLs from our own lists
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)


def build_sdist_wheel(w: dict, work: Path, out: Path, fetch: Callable[[str, Path], None] = _fetch,
                      run: Callable[..., "subprocess.CompletedProcess"] = subprocess.run) -> Path:
    """Download the pinned sdist, check its SHA-256, build a wheel from it; it must be pure Python (``-none-any``)."""
    work.mkdir(parents=True, exist_ok=True)
    src = work / w["file"]
    fetch(w["url"], src)
    if sha256_of(src) != w["sha256"]:
        raise SystemExit(f"{w['file']}: the SHA-256 of the sdist does not match the lock")
    r = run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-cache-dir", "--wheel-dir", str(out), str(src)],
            capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"building a wheel from {w['file']} failed:\n{(r.stdout or '')[-800:]}{(r.stderr or '')[-800:]}")
    built = sorted(out.glob(re.sub(r"[-_.]+", "_", w["dist"]).lower() + "-*.whl")) or sorted(out.glob("*.whl"))
    cand = [p for p in built if p.name.lower().startswith(re.sub(r"[-_.]+", "_", w["dist"]).lower() + "-")]
    if not cand:
        raise SystemExit(f"no wheel was built from {w['file']}")
    whl = cand[-1]
    if not whl.name.endswith("-none-any.whl"):
        raise SystemExit(f"{whl.name} is not pure Python: it cannot be shipped for every PC; use a wheel from PyPI")
    return whl


def components(lock: dict, base_url: str, out: Path, mirror_base: str = "", work: Optional[Path] = None,
               builder: Optional[Callable[..., Path]] = None) -> Tuple[List[dict], List[dict], dict]:
    """(components, modules, runtime meta) for the manifest."""
    comps: List[dict] = []
    by_group: Dict[str, List[str]] = {}
    for w in lock["wheels"]:
        entry = dict(w)
        if w.get("sdist"):
            whl = (builder or build_sdist_wheel)(w, (work or out / ".work") / w["dist"], out)
            entry = {"dist": w["dist"], "version": w["version"], "file": whl.name, "url": f"{base_url.rstrip('/')}/{whl.name}",
                     "size": whl.stat().st_size, "sha256": sha256_of(whl), "group": w["group"]}
        c = {"id": cid(entry), "file": entry["file"], "url": entry["url"], "size": int(entry["size"]), "sha256": entry["sha256"],
             "kind": "wheel", "role": "runtime", "module": w["group"], "markers": [marker(entry["file"])],
             "unpacked_bytes": int(entry["size"] * (1.6 if w["group"] == "torch" else 3))}
        if mirror_base and not w.get("sdist"):
            c["urls"] = [f"{mirror_base.rstrip('/')}/{entry['file']}"]
        if w.get("flavor"):
            c["flavor"] = w["flavor"]
        comps.append(c)
        by_group.setdefault(w["group"], []).append(c["id"])
    titles = lock.get("group_titles", {})
    mods = []
    for g in sorted(by_group, key=lambda g: ORDER.index(g) if g in ORDER else -1):
        ids = by_group[g]
        in_g = [c for c in comps if c["id"] in ids]
        # PyTorch exists in several flavors; the size shown is that of the largest (the client filters by flavor)
        m = {"id": g, "title": titles.get(g, g), "required": True, "components": ids,
             "size": sum(c["size"] for c in in_g if not c.get("flavor") or c["flavor"] == lock.get("flavors", [""])[0]),
             "unpacked_bytes": sum(c["unpacked_bytes"] for c in in_g if not c.get("flavor") or c["flavor"] == lock.get("flavors", [""])[0])}
        if g == "torch":
            m["reusable"] = "torch"
        mods.append(m)
    meta = {"compat": lock.get("compat", {}), "flavors": lock.get("flavors", []), "torch_version": lock.get("torch_version", ""),
            "python": lock.get("python", ""), "platform": lock.get("platform", ""), "mirror": mirror_base}
    return comps, mods, meta
