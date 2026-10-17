"""Split a PyInstaller ``--onedir`` build into the release assets of the ONLINE installer.

    python tools/make_online_payload.py --dist dist/Voxprint --out build/online --tag v0.1.0-beta \
        --repo Mitroshenkov87/voxprint-audiobook-builder [--channel beta|stable] [--base-url URL] [--limit-mib 1800]

Writes ``Voxprint-payload-01.zip`` ... (every zip stays under ``--limit-mib`` MiB: a GitHub release asset must be below
2 GiB) and ``manifest-<channel>.json`` (schema 1: id, file, url, size, sha256, unpacked_bytes, markers per component).  The
download address of each part is ``<base-url>/<file>``; by default the assets of the release ``--tag`` of ``--repo``.
The channel is ``beta`` for a tag with a pre-release suffix (``v0.1.0-beta``), ``stable`` otherwise, unless given.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import List, Optional

from voxprint_common.hashing import sha256_file

SCHEMA = 1
DEFAULT_LIMIT_MIB = 1800
PREFIX = "Voxprint-payload"
# A release asset that travels beside the setup exe. No scheme, no slash, no parent segment.
_LOCAL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,199}$")


def is_local_asset_name(url: str) -> bool:
    """True when ``url`` is one file name, not an address."""
    return bool(url) and "/" not in url and "\\" not in url and ":" not in url and ".." not in url and _LOCAL_NAME.fullmatch(url) is not None


def publish_url(base: str, filename: str, local_assets: bool) -> str:
    """Download address of one file we ship: the release URL, or the file name when ``local_assets``."""
    if local_assets:
        if not is_local_asset_name(filename):
            raise SystemExit(f"local asset name is not a single file name: {filename}")
        return filename
    return f"{base.rstrip('/')}/{filename}"


def channel_for(tag: str) -> str:
    """``beta`` when the tag carries a pre-release suffix (``-beta``, ``-rc1`` ...), else ``stable``."""
    return "beta" if "-" in tag.lstrip("vV") else "stable"


def sha256_of(path: Path) -> str:
    return sha256_file(path)


def list_files(dist: Path) -> List[Path]:
    """All files, the program itself first (so that part 01 alone is a recognisable core), then the rest sorted."""
    files = sorted(p for p in dist.rglob("*") if p.is_file())
    files.sort(key=lambda p: (0 if p.parent == dist and p.name.lower() == "voxprint.exe" else 1 if p.parent == dist
                              else 2 if "torch" not in p.parts else 3, str(p).lower()))
    return files


def build(dist: Path, out: Path, tag: str, repo: str, channel: Optional[str] = None, base_url: str = "",
          limit_mib: int = DEFAULT_LIMIT_MIB, app_version: str = "", runtime_site: Optional[Path] = None,
          runtime_lock: Optional[Path] = None, mirror_base: str = "", prefix: str = "",
          local_assets: bool = False) -> Path:
    """Write the zips and the manifest into ``out``; returns the manifest path.

    ``runtime_lock`` (THIN installer v2, ``docs/THIN-INSTALLER.md``): ``infra/runtime_lock.json``.  ``dist`` is the thin shell (the only
    thing zipped); the third-party libraries are listed in the manifest with their ORIGINAL addresses (PyPI, download.pytorch.org) and
    SHA-256 - nothing of them is repacked or uploaded.  The manifest is ``manifest-thin-<channel>.json``.

    ``runtime_site`` (THIN installer): the site-packages of the build environment.  ``dist`` is then the thin shell; its parts
    get ``role: core``, the heavy libraries become the runtime modules (``tools/make_runtime_modules.py``, ``role: runtime``)
    and a ``modules.json`` (where the app finds the manifest) is added to the shell.  Without it every part is ``core`` and the
    manifest is exactly as before."""
    dist, out = Path(dist), Path(out)
    prefix = prefix or ("Voxprint-shell" if runtime_lock is not None else PREFIX)      # never the name of an older release asset
    if not dist.is_dir():
        raise SystemExit(f"{dist} is not a folder")
    out.mkdir(parents=True, exist_ok=True)
    channel = channel or channel_for(tag)
    base = (base_url or f"https://github.com/{repo}/releases/download/{tag}").rstrip("/")
    limit = limit_mib * 1024 * 1024
    thin = runtime_site is not None or runtime_lock is not None
    mname = f"manifest-thin-{channel}.json" if runtime_lock is not None else f"manifest-{channel}.json"
    if thin:        # tell the shell where the manifest is (the app's module manager reads it)
        mj = (dist / "_internal" if (dist / "_internal").is_dir() else dist) / "modules.json"
        # manifest_url = pinned (this release); latest_manifest_url = the newest release's (try-latest option and fallback;
        # only for GitHub release assets - a custom base_url has no "latest" alias)
        if local_assets:
            manifest_url, latest = mname, ""
        else:
            latest = "" if base_url else f"https://github.com/{repo}/releases/latest/download/{mname}"
            manifest_url = f"{base}/{mname}"
        mj.write_text(json.dumps({"schema": 1, "manifest_url": manifest_url, "latest_manifest_url": latest, "tag": tag},
                                 indent=1), encoding="utf-8")
    parts: List[dict] = []
    cur: Optional[zipfile.ZipFile] = None
    cur_path: Optional[Path] = None
    cur_raw = 0
    cur_markers: List[str] = []

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
            cur_path = out / f"{prefix}-{len(parts) + 1:02d}.zip"
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
        comps.append({"id": f"{'shell' if runtime_lock is not None else 'payload'}-{i:02d}", "file": path.name,
                      "url": publish_url(base, path.name, local_assets), "size": path.stat().st_size,
                      "sha256": sha256_of(path), "unpacked_bytes": p["raw"], "markers": p["markers"]})
    modules: List[dict] = []
    meta: dict = {}
    if runtime_lock is not None:
        for c in comps:
            c["role"] = "core"
        try:
            from tools import runtime_manifest as rm
        except ImportError:
            import runtime_manifest as rm                 # type: ignore[no-redef]

        rt_comps, modules, meta = rm.components(json.loads(Path(runtime_lock).read_text(encoding="utf-8")), base, out, mirror_base)
        if local_assets:
            prefix_url = base.rstrip("/") + "/"
            for entry in rt_comps:
                url = str(entry.get("url") or "")
                # Wheels we built (sdist) use ``base``. Upstream addresses stay as they are.
                if url.startswith(prefix_url) or is_local_asset_name(url):
                    entry["url"] = entry["file"]
        comps += rt_comps
    elif runtime_site is not None:
        for c in comps:
            c["role"] = "core"
        try:                                              # only the thin build needs it
            from tools import make_runtime_modules as rt
        except ImportError:                               # run as a script: tools/ is sys.path[0]
            import make_runtime_modules as rt             # type: ignore[no-redef]

        rt_comps, modules = rt.build_modules(Path(runtime_site), out, base, limit_mib)
        comps += rt_comps
    manifest = {"schema": SCHEMA, "channel": channel, "app_version": app_version or tag.lstrip("vV"), "tag": tag, "repo": repo,
                "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "components": comps}
    build = os.environ.get("VOXPRINT_BUILD", "").strip()   # CI build number (tools/build_number.py): 665, 666, ...
    if build.isdigit() and int(build) > 0:
        manifest["build"] = int(build)
        codename = os.environ.get("VOXPRINT_CODENAME", "").strip()
        if codename:
            manifest["codename"] = codename
    if os.environ.get("GITHUB_SHA"):
        manifest["commit"] = os.environ["GITHUB_SHA"][:8]
    if thin:
        manifest["thin"] = True
        manifest["modules"] = modules
        if meta:
            manifest["runtime"] = meta
    if local_assets:
        _seal_local_shell(out, mname, manifest)
    mp = out / mname
    text = json.dumps(manifest, indent=1)
    mp.write_text(text, encoding="utf-8")
    if local_assets:
        (out / "embedded-manifest.json").write_text(text, encoding="utf-8")
    return mp


def _zip_contains(path: Path, suffix: str) -> bool:
    with zipfile.ZipFile(path) as archive:
        return any(name.replace("\\", "/").endswith(suffix) for name in archive.namelist())


def _arc_dir(path: Path) -> str:
    """Directory inside ``path`` that holds ``modules.json`` (``_internal`` or the zip root)."""
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            norm = name.replace("\\", "/")
            if norm.endswith("modules.json"):
                parent = str(Path(norm).parent).replace("\\", "/")
                return "" if parent == "." else parent
    return ""


def _append_files(zip_path: Path, arc_dir: str, files: List[Path], names: Optional[dict] = None) -> None:
    """Add ``files`` next to ``modules.json``. ``names`` maps a source path to the archive file name."""
    names = names or {}
    with zipfile.ZipFile(zip_path, "a", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        existing = {name.replace("\\", "/") for name in archive.namelist()}
        for src in files:
            arc_name = names.get(src, src.name)
            arc = f"{arc_dir}/{arc_name}" if arc_dir else arc_name
            if arc not in existing:
                archive.write(src, arc)


def _refresh_component(comp: dict, path: Path) -> None:
    comp["size"] = path.stat().st_size
    comp["sha256"] = sha256_of(path)


def _seal_local_shell(out: Path, mname: str, manifest: dict) -> None:
    """Put local wheels and a manifest copy in the shell zip, then record that zip's final hash.

    The copy inside the zip is written before the hash is refreshed, so the installer uses the
    manifest beside the setup exe (the hash matches the zip). The copy inside the shell is what
    the app reads after extract; its wheel addresses are file names next to that copy.
    """
    holder = None
    for comp in manifest["components"]:
        if comp.get("role", "core") != "core":
            continue
        path = out / comp["file"]
        if path.is_file() and _zip_contains(path, "modules.json"):
            holder = comp
            break
    if holder is None:
        return
    zpath = out / holder["file"]
    arc = _arc_dir(zpath)
    wheels = [out / comp["file"] for comp in manifest["components"]
              if comp.get("kind") == "wheel" and is_local_asset_name(str(comp.get("url") or "")) and (out / comp["file"]).is_file()]
    _append_files(zpath, arc, wheels)
    _refresh_component(holder, zpath)
    snapshot = out / (mname + ".inject")
    snapshot.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    try:
        _append_files(zpath, arc, [snapshot], {snapshot: mname})
    finally:
        snapshot.unlink(missing_ok=True)
    _refresh_component(holder, zpath)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repo", default="Mitroshenkov87/voxprint-audiobook-builder")
    ap.add_argument("--channel", choices=("beta", "stable"))
    ap.add_argument("--base-url", default="")
    ap.add_argument("--limit-mib", type=int, default=DEFAULT_LIMIT_MIB)
    ap.add_argument("--runtime-site", help="THIN installer: site-packages of the build environment (heavy libraries become runtime modules)")
    ap.add_argument("--runtime-lock", help="THIN installer v2: infra/runtime_lock.json (libraries come from their upstream sites)")
    ap.add_argument("--mirror-base", default="", help="optional fallback address prefix for the lock's files (our mirror)")
    ap.add_argument("--local-assets", action="store_true",
                    help="shell and sdist wheels are file names next to the setup exe (no GitHub release URL)")
    a = ap.parse_args(argv)
    mp = build(Path(a.dist), Path(a.out), a.tag, a.repo, a.channel, a.base_url, a.limit_mib,
               runtime_site=Path(a.runtime_site) if a.runtime_site else None,
               runtime_lock=Path(a.runtime_lock) if a.runtime_lock else None, mirror_base=a.mirror_base,
               local_assets=a.local_assets)
    m = json.loads(mp.read_text(encoding="utf-8"))
    for c in m["components"]:
        print(f"{c['file']}  {c['size']:>12}  {c['sha256']}  {c.get('role', 'core')}")
    print(f"manifest: {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
