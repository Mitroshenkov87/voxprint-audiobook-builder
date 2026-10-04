"""Build the release assets of the (experimental) LINUX package.

    python tools/make_linux_package.py --out build/linux --tag v0.1.0-beta [--repo OWNER/NAME] [--root .]

Linux has no frozen build: the program is plain Python source installed into a venv by ``install-voxprint-linux.sh``.
Written to ``--out``:

* ``Voxprint-linux-app.zip``        the program source (core, infra, ui, workers, locales, licenses, assets ...), ~ a few MB
* ``manifest-linux.json``           the same schema as the Windows online installer's manifest (``tools/online_fetch.py``)
* ``voxprint-fetch.py``             the stdlib downloader (copy of ``tools/online_fetch.py``); verifies SHA-256, resumable
* ``install-voxprint-linux.sh``     the installer script (``@REPO@`` / ``@TAG@`` filled in)
* ``Voxprint-linux-experimental.tar.gz``  everything in one archive: ``voxprint-linux/{install-voxprint-linux.sh, app/, ...}``
* ``SHA256SUMS-linux.txt``          SHA-256 of all the files above
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

SCHEMA = 1
ZIP_NAME = "Voxprint-linux-app.zip"
TAR_NAME = "Voxprint-linux-experimental.tar.gz"
MANIFEST_NAME = "manifest-linux.json"
SCRIPT_NAME = "install-voxprint-linux.sh"
FETCH_NAME = "voxprint-fetch.py"
SUMS_NAME = "SHA256SUMS-linux.txt"
DEFAULT_REPO = "Mitroshenkov87/voxprint-audiobook-builder"

TOP_FILES = ["main.py", "credits.json", "requirements.txt", "requirements-verified.txt", "requirements-nodeps.txt",
             "requirements-torch.txt", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"]
TOP_DIRS = ["core", "infra", "ui", "workers", "locales", "licenses"]
EXTRA = ["assets/voxprint.png", "assets/voxprint.svg", "assets/voxprint.ico", "assets/check.png", "assets/ICON-LICENSE.txt",
         "installer/linux/voxprint.desktop", "installer/linux/voxprint-256.png", "installer/linux/" + SCRIPT_NAME,
         "docs/LINUX-TEST-CHECKLIST.md"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def app_files(root: Path) -> List[str]:
    """Relative POSIX paths of the files that make up the program (sorted; no caches, no tests, no Windows installer)."""
    rels = set()
    for name in TOP_FILES:
        if (root / name).is_file():
            rels.add(name)
    for d in TOP_DIRS:
        for p in (root / d).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo"):
                rels.add(p.relative_to(root).as_posix())
    for name in EXTRA:
        if (root / name).is_file():
            rels.add(name)
    return sorted(rels)


def inline_netroute(fetch_src: str, netroute_src: str) -> str:
    """The single-file downloader: ``tools/online_fetch.py`` with ``infra/netroute.py`` (the interface hopper) inlined."""
    a, b = fetch_src.index("# --- netroute:"), fetch_src.index("# --- end netroute")
    block = ("# --- netroute (inlined from infra/netroute.py by tools/make_linux_package.py)\n"
             "import types as _types\n"
             f"_NETROUTE_SRC = {netroute_src!r}\n"
             "_nr = _types.ModuleType('netroute')\n"
             "sys.modules['netroute'] = _nr\n"
             "exec(compile(_NETROUTE_SRC, 'netroute.py', 'exec'), _nr.__dict__)\n")
    return fetch_src[:a] + block + fetch_src[b:]


def _fix_script(text: str, repo: str, tag: str) -> str:
    return text.replace("@REPO@", repo).replace("@TAG@", tag)


def build(root: Path, out: Path, tag: str, repo: str = DEFAULT_REPO, base_url: str = "", app_version: str = "") -> Path:
    """Write all assets into ``out``; returns the manifest path."""
    root, out = Path(root).resolve(), Path(out)
    if not (root / "main.py").is_file():
        raise SystemExit(f"{root} is not the project root (no main.py)")
    out.mkdir(parents=True, exist_ok=True)
    base = (base_url or f"https://github.com/{repo}/releases/download/{tag}").rstrip("/")
    files = app_files(root)
    if not all(m in files for m in ("main.py", "credits.json", "requirements.txt")):
        raise SystemExit("the project root is incomplete")
    try:
        version = app_version or json.loads((root / "credits.json").read_text(encoding="utf-8"))["app"]["version"]
    except (OSError, KeyError, ValueError):
        version = tag.lstrip("vV")

    # the installer script with the release filled in (also stored inside the zip, so --from-dir / the tarball match)
    script_src = (root / "installer" / "linux" / SCRIPT_NAME).read_text(encoding="utf-8")
    script_text = _fix_script(script_src, repo, tag)
    (out / SCRIPT_NAME).write_text(script_text, encoding="utf-8", newline="\n")
    (out / SCRIPT_NAME).chmod(0o755)
    (out / FETCH_NAME).write_text(inline_netroute((root / "tools" / "online_fetch.py").read_text(encoding="utf-8"),
                                                  (root / "infra" / "netroute.py").read_text(encoding="utf-8")),
                                  encoding="utf-8", newline="\n")

    stamp = (1980, 1, 1, 0, 0, 0)      # fixed timestamps: the zip depends only on the content
    zp = out / ZIP_NAME
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in files:
            data = (script_text.encode("utf-8") if rel == "installer/linux/" + SCRIPT_NAME else (root / rel).read_bytes())
            zi = zipfile.ZipInfo(rel, stamp)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = (0o755 if rel.endswith(".sh") else 0o644) << 16
            z.writestr(zi, data)
    raw = sum((root / r).stat().st_size for r in files)
    manifest = {"schema": SCHEMA, "channel": "linux-experimental", "platform": "linux", "experimental": True,
                "app_version": version, "tag": tag, "repo": repo,
                "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "components": [{"id": "linux-app", "file": ZIP_NAME, "url": f"{base}/{ZIP_NAME}", "size": zp.stat().st_size,
                                "sha256": sha256_of(zp), "unpacked_bytes": raw, "markers": ["main.py", "core/translate.py"]}]}
    mp = out / MANIFEST_NAME
    mp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # the one-file archive: the installer next to the program tree
    top = "voxprint-linux"
    with tarfile.open(out / TAR_NAME, "w:gz", compresslevel=9) as t:
        def add(name: str, data: bytes, mode: int = 0o644) -> None:
            ti = tarfile.TarInfo(f"{top}/{name}")
            ti.size, ti.mode, ti.mtime = len(data), mode, 315532800
            ti.uname = ti.gname = "root"
            t.addfile(ti, io.BytesIO(data))

        add(SCRIPT_NAME, script_text.encode("utf-8"), 0o755)
        add(FETCH_NAME, (out / FETCH_NAME).read_bytes(), 0o755)
        add("README-LINUX.txt", (
            "Voxprint for Linux (EXPERIMENTAL)\n\n  ./install-voxprint-linux.sh --check          what is missing\n"
            "  ./install-voxprint-linux.sh --install-deps   install system libraries with apt, then Voxprint\n"
            "  ./install-voxprint-linux.sh --uninstall\n\nDetails: app/docs/LINUX-TEST-CHECKLIST.md and the README of the project.\n").encode())
        for rel in files:
            data = script_text.encode("utf-8") if rel == "installer/linux/" + SCRIPT_NAME else (root / rel).read_bytes()
            add(f"app/{rel}", data, 0o755 if rel.endswith(".sh") else 0o644)

    names = [ZIP_NAME, MANIFEST_NAME, FETCH_NAME, SCRIPT_NAME, TAR_NAME]
    (out / SUMS_NAME).write_text("".join(f"{sha256_of(out / n)}  {n}\n" for n in names), encoding="utf-8")
    return mp


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--base-url", default="")
    a = ap.parse_args(list(argv) if argv is not None else None)
    build(Path(a.root), Path(a.out), a.tag, a.repo, a.base_url)
    print((Path(a.out) / SUMS_NAME).read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
