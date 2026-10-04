"""Split the installed Python packages of the build environment into the RUNTIME MODULES of the thin installer.

    python tools/make_runtime_modules.py --site .venv/Lib/site-packages --out build/online-release \
        --base-url https://github.com/OWNER/REPO/releases/download/TAG [--limit-mib 1800]

The thin shell (``build_thin.bat``) contains only Qt, numpy, soundfile and the program; everything heavy is a *module*: the
files of a group of installed distributions, copied with their relative path under ``site-packages`` (the files listed in each
distribution's RECORD, so ``*.dist-info`` metadata comes along).  The app extracts a module into ``<app home>/runtime`` and
puts that folder on ``sys.path`` (``infra/modules.py``).  Every module is one or more zips (< 2 GiB each: GitHub's asset
limit) described in the manifest like the normal payload parts (``role: runtime``, ``module: <id>``).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PREFIX = "Voxprint-rt"

#: id, title, required for the narration/training pipeline, distribution names (normalised: lower case, ``-`` -> ``_``)
MODULES: List[Tuple[str, str, bool, List[str]]] = [
    ("torch", "PyTorch (neural network engine)", True,
     ["torch", "torchaudio", "sympy", "mpmath", "networkx", "jinja2", "markupsafe", "filelock", "fsspec", "typing_extensions",
      "triton", "nvidia_*"]),
    ("audio", "Audio and science libraries", True,
     ["scipy", "librosa", "numba", "llvmlite", "scikit_learn", "joblib", "threadpoolctl", "av", "audioread", "soxr", "pooch",
      "platformdirs", "msgpack", "lazy_loader", "decorator", "pydub", "audioop_lts", "sox", "pillow"]),
    ("text", "Text preparation and translation libraries", True,
     ["nagisa", "dynet38", "cython", "six", "num2words", "eng_to_ipa", "ru_normalizr", "rutextnorm", "pymorphy3",
      "pymorphy3_dicts_ru", "dawg2_python", "docopt", "roman", "sentencepiece", "regex"]),
    ("ffmpeg", "ffmpeg (audio encoder)", True, ["imageio_ffmpeg"]),
    ("ml", "Speech models runtime (transformers, qwen-tts, peft ...)", True, ["*"]),     # everything else
]

#: stays in the shell (or is build tooling that never ships)
SHELL = ["pyside6", "pyside6_essentials", "pyside6_addons", "shiboken6", "numpy", "soundfile", "cffi", "pycparser", "certifi",
         "packaging", "psutil"]
TOOLING = ["pip", "setuptools", "wheel", "uv", "pyinstaller", "pyinstaller_hooks_contrib", "altgraph", "pefile", "pywin32_ctypes",
           "pytest", "pluggy", "iniconfig", "pygments", "packaging_*", "build", "colorama"]


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def _match(name: str, patterns: Iterable[str]) -> bool:
    for p in patterns:
        if p == "*" or name == p or (p.endswith("*") and name.startswith(p[:-1])):
            return True
    return False


def module_of(dist_name: str) -> Optional[str]:
    """Module id for a distribution, ``None`` for what stays in the shell or is tooling."""
    n = norm(dist_name)
    if _match(n, SHELL) or _match(n, TOOLING):
        return None
    for mid, _t, _r, pats in MODULES:
        if pats != ["*"] and _match(n, pats):
            return mid
    return "ml"


def dist_files(site: Path) -> Dict[str, List[str]]:
    """``{module id: [relative file, ...]}`` for every distribution installed under ``site`` (no file in two modules)."""
    out: Dict[str, List[str]] = {m[0]: [] for m in MODULES}
    owner: Dict[str, str] = {}
    for d in md.distributions(path=[str(site)]):
        name = d.metadata["Name"]
        mid = module_of(name or "")
        if mid is None or not d.files:
            continue
        for f in d.files:
            rel = str(f).replace("\\", "/")
            if rel.startswith("..") or "__pycache__" in rel or rel.endswith((".pyc", ".pth")) or rel in owner:
                continue
            if (site / rel).is_file():
                owner[rel] = mid
                out[mid].append(rel)
    return {k: sorted(v) for k, v in out.items() if v}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_modules(site: Path, out: Path, base_url: str, limit_mib: int = 1800) -> Tuple[List[dict], List[dict]]:
    """Write ``Voxprint-rt-<module>-NN.zip`` into ``out``; returns ``(components, modules)`` for the manifest."""
    site, out = Path(site), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = base_url.rstrip("/")
    limit = limit_mib * 1024 * 1024
    titles = {m[0]: (m[1], m[2]) for m in MODULES}
    comps: List[dict] = []
    mods: List[dict] = []
    for mid, files in dist_files(site).items():
        files = sorted(files, key=lambda r: ((site / r).stat().st_size, r))      # small files first, the big libraries last
        parts: List[dict] = []
        cur: Optional[zipfile.ZipFile] = None
        cur_path: Optional[Path] = None
        raw, markers = 0, []

        def close() -> None:
            nonlocal cur, cur_path, raw, markers
            if cur is not None and cur_path is not None:
                cur.close()
                parts.append({"path": cur_path, "raw": raw, "markers": markers})
            cur, cur_path, raw, markers = None, None, 0, []

        for rel in files:
            if cur is None:
                cur_path = out / f"{PREFIX}-{mid}-{len(parts) + 1:02d}.zip"
                cur = zipfile.ZipFile(cur_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True)
            cur.write(site / rel, rel)
            size = (site / rel).stat().st_size
            raw += size
            if len(markers) < 3 and size > 0 and rel.endswith((".py", ".pyd", ".so", ".dll", "METADATA")):
                markers.append(rel)
            if cur_path is not None and cur_path.stat().st_size >= limit:
                close()
        close()
        ids = []
        for i, p in enumerate(parts, 1):
            path = p["path"]
            if path.stat().st_size >= 2 * 1024 ** 3:
                raise SystemExit(f"{path.name} is over the 2 GiB asset limit: lower --limit-mib")
            cid = f"rt-{mid}-{i:02d}"
            ids.append(cid)
            comps.append({"id": cid, "file": path.name, "url": f"{base}/{path.name}", "size": path.stat().st_size,
                          "sha256": _sha256(path), "unpacked_bytes": p["raw"], "markers": p["markers"],
                          "role": "runtime", "module": mid})
        title, required = titles[mid]
        mods.append({"id": mid, "title": title, "required": required, "components": ids,
                     "size": sum(c["size"] for c in comps if c["module"] == mid),
                     "unpacked_bytes": sum(c["unpacked_bytes"] for c in comps if c["module"] == mid)})
    return comps, mods


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--site", required=True, help="site-packages folder of the build environment")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--limit-mib", type=int, default=1800)
    ap.add_argument("--list", action="store_true", help="only print the files per module, write nothing")
    a = ap.parse_args(argv)
    if a.list:
        for mid, files in dist_files(Path(a.site)).items():
            total = sum((Path(a.site) / r).stat().st_size for r in files)
            print(f"{mid:8} {len(files):6} files {total / 2**20:9.1f} MiB")
        return 0
    comps, mods = build_modules(Path(a.site), Path(a.out), a.base_url, a.limit_mib)
    print(json.dumps(mods, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
