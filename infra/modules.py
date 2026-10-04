"""Runtime modules of the THIN installer: the heavy libraries (PyTorch, transformers, scipy ...) are downloaded by the app itself.

A *thin* build (``build_thin.bat``) ships only Qt, numpy, soundfile and the program.  Its folder contains ``modules.json``
(``{"manifest_url": ...}``); in a normal (full) build that file does not exist, :func:`is_thin` is False and nothing in this
module does anything.  Modules are described by the release manifest (``role: runtime`` components grouped in ``modules``,
see ``tools/make_runtime_modules.py``); they are downloaded with the installer's own downloader (``tools/online_fetch.py``:
resumable, SHA-256 verified, with the network interface fallback) into ``<app home>/runtime`` - a folder the user can write to,
so no administrator rights are needed - and :func:`activate` puts that folder on ``sys.path`` (at start-up, and right after an
install: the heavy libraries are imported lazily, so no restart is needed).

Environment: ``VOXPRINT_MODULES_CONFIG`` = path of another ``modules.json`` (tests, mirrors).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from infra import paths

log = logging.getLogger("voxprint.modules")

CONFIG_NAME = "modules.json"
MANIFEST_CACHE = "modules_manifest.json"
Progress = Callable[[float, str], None]


class ModulesError(Exception):
    """A module download failed or was cancelled; the text is shown to the user."""


class Cancelled(ModulesError):
    pass


@dataclass
class Module:
    id: str
    title: str
    required: bool
    size: int                       # download size, bytes
    unpacked: int                   # bytes on disk
    components: List[str] = field(default_factory=list)
    installed: bool = False


# ------------------------------------------------------------------------------------------------ configuration
def config_path() -> Optional[Path]:
    env = os.environ.get("VOXPRINT_MODULES_CONFIG", "").strip()
    cands = [Path(env)] if env else []
    cands.append(paths.resource_dir() / CONFIG_NAME)
    try:
        cands.append(Path(sys.executable).resolve().parent / CONFIG_NAME)
    except OSError:
        pass
    return next((c for c in cands if c.is_file()), None)


def load_config() -> Optional[Dict[str, Any]]:
    p = config_path()
    if p is None:
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8-sig"))
        return d if isinstance(d, dict) and d.get("manifest_url") else None
    except (OSError, ValueError):
        return None


def is_thin() -> bool:
    """True in a thin build (a ``modules.json`` is shipped)."""
    return load_config() is not None


def runtime_dir() -> Path:
    """Where the modules are unpacked: ``<app home>/runtime`` (created on demand)."""
    p = paths.app_home() / "runtime"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = paths.app_home() / "setup-cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _fetch_module():
    from tools import online_fetch

    return online_fetch


# ------------------------------------------------------------------------------------------------ manifest and state
def load_manifest(source: Optional[str] = None, offline_ok: bool = True) -> Dict[str, Any]:
    """The release manifest (downloaded; the last good copy is kept in ``state/`` for offline use)."""
    of = _fetch_module()
    cfg = load_config() or {}
    src = source or str(cfg.get("manifest_url", ""))
    if not src:
        raise ModulesError("no manifest address (modules.json)")
    cache = paths.state_dir() / MANIFEST_CACHE
    try:
        man = of.read_manifest(src)
        try:
            cache.write_text(json.dumps(man), encoding="utf-8")
        except OSError:
            pass
        return man
    except of.FetchError as exc:
        if offline_ok:
            try:
                return of.validate_manifest(json.loads(cache.read_text(encoding="utf-8")))
            except (OSError, ValueError, of.FetchError):
                pass
        raise ModulesError(str(exc)) from exc


def modules(manifest: Dict[str, Any]) -> List[Module]:
    """The modules of a manifest with their installed state (all components present with the recorded SHA-256)."""
    of = _fetch_module()
    state = of.load_state(runtime_dir())
    by_id = {c["id"]: c for c in manifest.get("components", [])}
    out: List[Module] = []
    for m in manifest.get("modules", []):
        comps = [by_id[i] for i in m.get("components", []) if i in by_id]
        out.append(Module(str(m["id"]), str(m.get("title", m["id"])), bool(m.get("required", True)),
                          sum(int(c["size"]) for c in comps), sum(int(c.get("unpacked_bytes", 0)) for c in comps),
                          [c["id"] for c in comps],
                          bool(comps) and all(of.installed_ok(c, state, runtime_dir()) for c in comps)))
    return out


def installed_without_network() -> bool:
    """Quick start-up check (no network): True when a cached manifest exists and all its required modules are installed,
    or when this is not a thin build."""
    if not is_thin():
        return True
    of = _fetch_module()
    try:
        man = of.validate_manifest(json.loads((paths.state_dir() / MANIFEST_CACHE).read_text(encoding="utf-8")))
    except (OSError, ValueError, of.FetchError):
        return False
    return all(m.installed for m in modules(man) if m.required)


def missing_required(manifest: Optional[Dict[str, Any]] = None) -> List[Module]:
    """Required modules that are not installed (needs the manifest, which is downloaded unless given)."""
    if not is_thin():
        return []
    return [m for m in modules(manifest or load_manifest()) if m.required and not m.installed]


# ------------------------------------------------------------------------------------------------ install
class _Status:
    """Duck-typed ``online_fetch.Status``: forwards progress, can cancel by raising."""

    def __init__(self, progress: Progress, cancelled: Callable[[], bool]) -> None:
        self.progress, self.cancelled, self.was_cancelled = progress, cancelled, False

    def write(self, state: str, fraction: float, text: str, force: bool = False) -> None:
        if self.cancelled():
            self.was_cancelled = True
            raise _fetch_module().FetchError("cancelled")      # a FetchError is not retried by the downloader
        self.progress(max(0.0, min(1.0, fraction)), text)


def install(module_ids: Optional[List[str]] = None, progress: Optional[Progress] = None,
            cancelled: Callable[[], bool] = lambda: False, manifest_source: Optional[str] = None) -> int:
    """Download and unpack the given modules (default: all required ones that are missing); returns the number of
    components fetched.  Resumable: run it again after a failure.  Activates the runtime folder when done."""
    of = _fetch_module()
    cfg = load_config() or {}
    src = manifest_source or str(cfg.get("manifest_url", ""))
    man = load_manifest(src)
    mods = modules(man)
    chosen = [m for m in mods if (m.id in module_ids if module_ids else (m.required and not m.installed))]
    comp_ids = [c for m in chosen for c in m.components]
    if not comp_ids:
        activate()
        return 0
    st = _Status(progress or (lambda f, t: None), cancelled)
    try:
        n = of.run(src, runtime_dir(), cache_dir(), st, only=comp_ids)
    except of.FetchError as exc:
        if st.was_cancelled:
            raise Cancelled("cancelled") from exc
        raise ModulesError(str(exc)) from exc
    activate()
    return n


def activate() -> Optional[Path]:
    """Put the runtime folder on ``sys.path`` (after the updater's ``packages`` overlay).  Safe to call repeatedly."""
    try:
        rd = paths.app_home() / "runtime"
        if rd.is_dir() and any(rd.iterdir()):
            sp = str(rd)
            if sp not in sys.path:
                sys.path.insert(1 if sys.path and sys.path[0] == str(paths.packages_dir()) else 0, sp)
            import importlib

            importlib.invalidate_caches()
            return rd
    except OSError:
        pass
    return None
