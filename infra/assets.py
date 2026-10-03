"""Pinned non-pip assets (ffmpeg): download -> sha256 -> staging -> smoke test -> atomic swap -> rollback.

Mirrors the approach Unsloth Studio uses for its prebuilt binaries (read from its installer sources, 2026-10-03):
a pinned URL + sha256 in a JSON file is the trust anchor; the archive is verified before anything is extracted, only
whitelisted members are extracted (no path traversal), the new copy must pass a smoke test *before* it replaces the
old one, the swap is ``os.replace`` and the previous copy is restored if anything fails.  Our own folder only: a
directory is replaced/deleted only if it carries our ownership marker.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from infra import paths

log = logging.getLogger("voxprint.assets")
MANIFEST_PATH = Path(__file__).with_name("assets_manifest.json")
OWNER_MARKER = ".voxprint-owned"
INFO_FILE = ".asset.json"
CHUNK = 1024 * 1024


class AssetError(Exception):
    """``code`` is a stable reason code (never a traceback): download_failed, sha256_mismatch, bad_archive,
    smoke_failed, swap_failed, unsupported_platform."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


Runner = Callable[[List[str]], Tuple[int, str]]
Opener = Callable[[urllib.request.Request, float], Any]


def _open(req: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - pinned https URL, verified by sha256


def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def spec_for(name: str, platform: Optional[str] = None, manifest: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    m = manifest or load_manifest()
    asset = (m.get("assets") or {}).get(name)
    if not asset:
        return None
    s = (asset.get("platforms") or {}).get(platform or sys.platform)
    return {**s, "version": asset.get("version", ""), "name": name} if s else None


def tools_dir() -> Path:
    p = paths.app_home() / "tools"
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(args: List[str]) -> Tuple[int, str]:
    from infra.updater import run_subprocess

    return run_subprocess(args, timeout=60)


def smoke_ok(exe: Path, args: List[str], expect: str, run: Runner = _run) -> bool:
    rc, out = run([str(exe), *args])
    return rc == 0 and expect.lower() in out.lower()


def installed_path(spec: Dict[str, Any], root: Optional[Path] = None) -> Optional[Path]:
    """Path of the installed asset if it is the pinned one (marker sha matches) - no smoke test, no network."""
    d = (root or tools_dir()) / spec["name"]
    try:
        info = json.loads((d / INFO_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    exe = d / spec["exe"]
    return exe if info.get("sha256") == spec["sha256"] and exe.is_file() else None


def _download(spec: Dict[str, Any], dest: Path, opener: Opener, progress: Callable[[float], None]) -> None:
    """Resumable download into ``dest`` (a .part file kept between attempts)."""
    have = dest.stat().st_size if dest.exists() else 0
    total = int(spec.get("size") or 0)
    if total and have > total:
        dest.unlink()
        have = 0
    headers = {"User-Agent": "Voxprint"}
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        with opener(urllib.request.Request(spec["url"], headers=headers), 60.0) as r:
            if have and getattr(r, "status", 200) != 206:
                have = 0
            with open(dest, "ab" if have else "wb") as f:
                while True:
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    if total:
                        progress(min(1.0, f.tell() / total))
    except OSError as exc:
        raise AssetError("download_failed", str(exc)) from exc


def _extract(archive: Path, spec: Dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as z:
            names = set(z.namelist())
            for member, target in spec["extract"].items():
                if member not in names or "/" in target or "\\" in target or target.startswith("."):
                    raise AssetError("bad_archive", f"{member} missing")
                with z.open(member) as src, open(out / target, "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
    except zipfile.BadZipFile as exc:
        raise AssetError("bad_archive", str(exc)) from exc


def install_asset(spec: Dict[str, Any], root: Optional[Path] = None, opener: Opener = _open, run: Runner = _run,
                  progress: Callable[[float], None] = lambda f: None) -> Path:
    """Installs (or re-validates) the pinned asset; returns the path of its executable.  Raises AssetError."""
    root = root or tools_dir()
    final = root / spec["name"]
    exe_name = spec["exe"]
    cur = installed_path(spec, root)
    if cur is not None and smoke_ok(cur, spec["smoke"], spec["smoke_expect"], run):
        log.info("%s %s already installed", spec["name"], spec.get("version"))
        return cur
    stage = root / ".staging" / spec["name"]
    stage.mkdir(parents=True, exist_ok=True)
    archive = stage / "archive.part"          # kept between attempts => resumable
    new = stage / "new"
    shutil.rmtree(new, ignore_errors=True)
    for attempt in (1, 2):
        _download(spec, archive, opener, progress)
        digest = sha256_file(archive)
        if digest == spec["sha256"]:
            break
        archive.unlink(missing_ok=True)       # corrupt: start over once, then give up
        log.warning("sha256 mismatch for %s (attempt %d): %s", spec["name"], attempt, digest)
        if attempt == 2:
            raise AssetError("sha256_mismatch", digest)
    try:
        _extract(archive, spec, new)
        (new / OWNER_MARKER).write_text("voxprint", encoding="utf-8")
        (new / INFO_FILE).write_text(json.dumps({"sha256": spec["sha256"], "version": spec.get("version", "")}),
                                     encoding="utf-8")
        if not smoke_ok(new / exe_name, spec["smoke"], spec["smoke_expect"], run):
            raise AssetError("smoke_failed", exe_name)
        bak = final.with_name(final.name + ".bak")
        if bak.exists() and (bak / OWNER_MARKER).exists():
            shutil.rmtree(bak, ignore_errors=True)
        had_old = final.exists()
        if had_old:
            if not (final / OWNER_MARKER).exists():
                raise AssetError("swap_failed", "target folder is not owned by Voxprint")   # never touch foreign dirs
            os.replace(final, bak)
        try:
            os.replace(new, final)
        except OSError as exc:
            if had_old and not final.exists():
                os.replace(bak, final)            # rollback
            raise AssetError("swap_failed", str(exc)) from exc
        shutil.rmtree(bak, ignore_errors=True)
    finally:
        shutil.rmtree(new, ignore_errors=True)
    archive.unlink(missing_ok=True)
    log.info("%s %s installed", spec["name"], spec.get("version"))
    return final / exe_name


def ensure_ffmpeg_tool(progress=None, opener: Opener = _open, run: Runner = _run,
                       which: Callable[[str], Optional[str]] = shutil.which, platform: Optional[str] = None,
                       root: Optional[Path] = None) -> Optional[Path]:
    """ffmpeg for Voxprint: a system one if it passes the smoke test, else the pinned LGPL build (downloaded,
    verified, staged, swapped atomically).  Returns None when neither is available (the bundled imageio-ffmpeg
    stays as the offline fallback).  Never raises."""
    from core.i18n import tr
    from infra import env_probe

    say = progress or (lambda f, m: None)
    try:
        info = env_probe.probe_ffmpeg(which, run)
        if info and info.ok:
            say(1.0, tr("env.ffmpeg_reused", version=info.version or "?"))
            return Path(info.path)
        spec = spec_for("ffmpeg", platform)
        if spec is None:
            return None
        cur = installed_path(spec, root)
        if cur is not None and smoke_ok(cur, spec["smoke"], spec["smoke_expect"], run):
            return cur
        say(0.0, tr("progress.ffmpeg_downloading"))
        exe = install_asset(spec, root, opener, run, lambda f: say(f, tr("progress.ffmpeg_downloading")))
        say(1.0, tr("progress.ffmpeg_ready"))
        return exe
    except AssetError as exc:
        log.warning("ffmpeg tool not installed (%s) - using the bundled fallback", exc.code)
    except Exception as exc:  # noqa: BLE001 - optional step, never blocks the app
        log.warning("ffmpeg tool step failed: %s", exc)
    return None
