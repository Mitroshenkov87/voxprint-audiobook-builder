"""Write ``infra/runtime_lock.json``: the pinned list of third-party wheels the thin installer downloads from THEIR upstream sites.

Why: building and uploading ~3 GB of PyTorch & co. on every release took an hour although all of it already lives on PyPI and
download.pytorch.org.  The release now holds only our own small shell and a manifest that points to the original files; every
file is pinned (version, size, SHA-256).  Run this tool when the dependencies change (needs network and ``uv``), review the diff,
commit the lock.  CI never resolves anything: it only reads the committed file.

* pure libraries: ``uv pip compile requirements.txt requirements-verified.txt`` for Windows / the build Python -> the wheel of every
  pin from the PyPI JSON API (``cp311-win_amd64``, ``abi3`` or ``py3-none-any``; an sdist-only package is marked ``sdist`` - CI builds a pure-Python wheel from it);
* ``requirements-nodeps.txt`` (qwen-asr / qwen-tts conflict on transformers): pinned as they are, without their own dependencies;
* PyTorch: ``torch`` + ``torchaudio`` of one version for every flavor (cu128, cu126, cpu) from download.pytorch.org;
* the packages that the PyInstaller shell bundles (Qt, numpy, soundfile ...) are NOT listed as downloads - they are listed under
  ``shell`` so that CI installs exactly those versions into the shell's build environment.

Usage: ``python tools/make_runtime_lock.py [--torch 2.11.0] [--python 3.11]``
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "infra" / "runtime_lock.json"
SCHEMA = 1
UA = "voxprint-lock/1"
FLAVORS = ("cu128", "cu126", "cpu")
#: Provided by the PyInstaller shell (installed from this lock into the build environment, bundled into the exe).
SHELL_PROVIDED = {"pyside6", "pyside6-essentials", "pyside6-addons", "shiboken6", "numpy", "soundfile", "cffi", "pycparser",
                  "certifi", "psutil", "packaging"}
#: Dropped from the downloads: build tools / things the program never needs at run time.
SKIP = {"pyinstaller", "pytest", "pip", "setuptools", "wheel"}
#: PyTorch comes from download.pytorch.org per flavor.
TORCH = ("torch", "torchaudio")
#: Which libraries are downloaded together (one progress bar in the Components window).
GROUPS = [
    ("audio", "Audio and science libraries", ("scipy", "librosa", "numba", "llvmlite", "scikit-learn", "soxr", "audioread", "pooch",
                                              "joblib", "threadpoolctl", "lazy-loader", "decorator", "msgpack", "onnxruntime",
                                              "av", "pillow", "pydub", "imageio-ffmpeg", "sox", "resampy", "audioop-lts")),
    ("text", "Text and language", ("nagisa", "pymorphy3", "pymorphy3-dicts-ru", "ru-normalizr", "rutextnorm", "num2words", "eng-to-ipa",
                                   "sentencepiece", "regex", "dawg2-python", "docopt-ng", "six", "dynet", "cython")),
]
COMPAT = {"torch": ">=2.8,<2.13", "python_minor_must_match": True}


def norm(n: str) -> str:
    return re.sub(r"[-_.]+", "-", n).lower()


def get(url: str, binary: bool = False, method: str = "GET"):
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method=method)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r if method == "HEAD" else (r.read() if binary else r.read().decode("utf-8"))


def compile_pins(py: str) -> Dict[str, str]:
    """name -> version from ``uv pip compile`` (torch excluded)."""
    cmd = ["uv", "pip", "compile", str(ROOT / "requirements.txt"), str(ROOT / "requirements-verified.txt"), "--python-version", py,
           "--python-platform", "windows", "--no-emit-package", "torch", "--no-emit-package", "torchaudio", "--quiet"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    pins = {}
    for line in out.splitlines():
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;#]+)", line)
        if m:
            pins[norm(m.group(1))] = m.group(2)
    for line in (ROOT / "requirements-nodeps.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;#]+)", line.strip())
        if m:
            pins[norm(m.group(1))] = m.group(2)
    return pins


def wheel_score(fn: str, tag: str) -> Optional[int]:
    """Higher = more specific; None = not installable on win_amd64 / CPython ``tag`` (e.g. cp311)."""
    from packaging.utils import parse_wheel_filename

    try:
        _n, _v, _b, tags = parse_wheel_filename(fn)
    except Exception:  # noqa: BLE001
        return None
    best = None
    minor = int(tag[2:])
    for t in tags:
        py, abi, plat = t.interpreter, t.abi, t.platform
        if plat not in ("win_amd64", "any"):
            continue
        if py == tag and abi == tag:
            s = 4
        elif py.startswith("cp3") and abi == "abi3" and int(re.sub(r"\D", "", py[2:]) or 99) <= minor:
            s = 3
        elif py == tag and abi == "none":
            s = 2
        elif py.startswith("py3") or py == "py2.py3":
            s = 1
        else:
            continue
        s += 10 if plat == "win_amd64" else 0
        best = s if best is None or s > best else best
    return best


def pypi_wheel(name: str, version: str, tag: str) -> dict:
    data = json.loads(get(f"https://pypi.org/pypi/{name}/{version}/json"))
    best, best_s = None, -1
    for f in data["urls"]:
        if f["packagetype"] != "bdist_wheel":
            continue
        s = wheel_score(f["filename"], tag)
        if s is not None and s > best_s:
            best, best_s = f, s
    if best is None:           # sdist only: CI builds a (pure Python) wheel from it and ships that small file itself
        for f in data["urls"]:
            if f["packagetype"] == "sdist":
                return {"dist": name, "version": version, "file": f["filename"], "url": f["url"], "size": f["size"],
                        "sha256": f["digests"]["sha256"], "sdist": True}
        raise SystemExit(f"{name}=={version}: neither a wheel for Windows x64 / {tag} nor an sdist on PyPI - pin another version")
    return {"dist": name, "version": version, "file": best["filename"], "url": best["url"], "size": best["size"],
            "sha256": best["digests"]["sha256"]}


def torch_wheel(pkg: str, version: str, flavor: str, tag: str) -> dict:
    base = f"https://download.pytorch.org/whl/{flavor}/{pkg}/"
    html = get(base)
    want = f"{pkg}-{version}+{flavor}-{tag}-{tag}-win_amd64.whl"
    for m in re.finditer(r'href="([^"]+)"', html):
        href = m.group(1)
        url, _, frag = href.partition("#sha256=")
        fn = urllib.parse.unquote(url.rsplit("/", 1)[-1])
        if fn == want:
            full = base.replace(f"/{pkg}/", "/") + urllib.parse.quote(fn)      # canonical address (the index points to a CDN host)
            size = int(get(full, method="HEAD").headers["Content-Length"])
            return {"dist": pkg, "version": f"{version}+{flavor}", "file": fn, "url": full, "size": size, "sha256": frag,
                    "flavor": flavor}
    raise SystemExit(f"{want} not found at {base}")


def group_of(name: str) -> str:
    for gid, _t, members in GROUPS:
        if name in members:
            return gid
    return "libs"


def build(py: str, torch_version: str) -> dict:
    tag = "cp" + py.replace(".", "")
    pins = compile_pins(py)
    shell = {n: v for n, v in sorted(pins.items()) if n in SHELL_PROVIDED}
    todo = {n: v for n, v in sorted(pins.items()) if n not in SHELL_PROVIDED and n not in SKIP and n not in TORCH}
    with cf.ThreadPoolExecutor(8) as ex:
        libs = list(ex.map(lambda kv: pypi_wheel(kv[0], kv[1], tag), todo.items()))
    for w in libs:
        w["group"] = group_of(w["dist"])
    torch = []
    for fl in FLAVORS:
        for pkg in TORCH:
            w = torch_wheel(pkg, torch_version, fl, tag)
            w["group"] = "torch"
            torch.append(w)
    titles = {"torch": "PyTorch (neural networks)", "libs": "Libraries (transformers, PEFT, tokenizers ...)"}
    titles.update({g[0]: g[1] for g in GROUPS})
    return {"schema": SCHEMA, "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"), "python": py,
            "platform": "win_amd64", "compat": COMPAT, "torch_version": torch_version, "flavors": list(FLAVORS),
            "group_titles": titles, "shell": shell, "wheels": libs + torch}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--torch", default="2.11.0")
    ap.add_argument("--python", default="3.11")
    ap.add_argument("--out", default=str(LOCK))
    ap.add_argument("--shell-requirements", action="store_true", help="print the pinned shell packages (for CI) and exit")
    a = ap.parse_args(argv)
    if a.shell_requirements:
        lock = json.loads(Path(a.out).read_text(encoding="utf-8"))
        print("\n".join(f"{n}=={v}" for n, v in lock["shell"].items()))
        return 0
    lock = build(a.python, a.torch)
    Path(a.out).write_text(json.dumps(lock, indent=1) + "\n", encoding="utf-8")
    tot = sum(w["size"] for w in lock["wheels"] if w.get("flavor") in (None, "cu128"))
    print(f"{len(lock['wheels'])} wheels, {tot / 2**20:.0f} MiB for cu128 -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
