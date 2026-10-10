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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from infra import paths
from infra.cuda12_libs import prepare as prepare_cuda12

log = logging.getLogger("voxprint.modules")

CONFIG_NAME = "modules.json"
MANIFEST_CACHE = "modules_manifest.json"
CHANNEL_FILE = "modules_channel.json"      # {"prefer": "pinned" | "latest", "installed_from": manifest URL of the last install}
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
    reused: bool = False            # satisfied by a verified copy that another program installed
    update: bool = False            # installed before, but the manifest now pins other files (an update is available)


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


# ------------------------------------------------------------------------------------------------ pinned / latest
def _channel_state() -> Dict[str, Any]:
    try:
        d = json.loads((paths.state_dir() / CHANNEL_FILE).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_channel_state(d: Dict[str, Any]) -> None:
    f = paths.state_dir() / CHANNEL_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d), encoding="utf-8")


def prefer_latest() -> bool:
    """Components window option "Try the latest versions": the newest release's manifest first, the pinned one as the fallback."""
    return _channel_state().get("prefer") == "latest"


def set_prefer_latest(on: bool) -> None:
    d = _channel_state()
    d["prefer"] = "latest" if on else "pinned"
    _save_channel_state(d)


def latest_url() -> str:
    """Manifest of the newest release (``modules.json`` ``latest_manifest_url``); '' when the build has none."""
    return str((load_config() or {}).get("latest_manifest_url", ""))


def runtime_dir() -> Path:
    """Where the modules are unpacked: ``<app home>/runtime`` (created on demand)."""
    p = paths.app_home() / "runtime"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = paths.app_home() / "setup-cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


_FLAVOR_CACHE: Dict[str, Any] = {}


def driver_cuda() -> Optional[tuple]:
    """CUDA version of the NVIDIA driver (``nvidia-smi``), cached for the process; None = no NVIDIA GPU."""
    if "cuda" not in _FLAVOR_CACHE:
        try:
            from infra import env_probe

            _FLAVOR_CACHE["cuda"] = env_probe.detect_driver_cuda()
        except Exception:  # noqa: BLE001
            _FLAVOR_CACHE["cuda"] = None
    return _FLAVOR_CACHE["cuda"]


def flavor_for(manifest: Dict[str, Any]) -> str:
    """The PyTorch flavor (``cu130``, or ``cpu`` when CI forces it) this PC gets; '' when the manifest has none."""
    flavors = list((manifest.get("runtime") or {}).get("flavors") or [])
    if not flavors:
        return ""
    from infra import runtime_reuse

    return runtime_reuse.choose_flavor(flavors, driver_cuda())


def _components_of(module: Dict[str, Any], by_id: Dict[str, Any], flavor: str) -> List[Dict[str, Any]]:
    """Components of a module; a component with a ``flavor`` belongs to this PC only if it matches."""
    return [by_id[i] for i in module.get("components", []) if i in by_id and (not by_id[i].get("flavor") or by_id[i]["flavor"] == flavor)]


def setup_folder() -> Optional[Path]:
    """The portable setup folder of this PC (``infra/portable.py``; the installer remembers it), or None."""
    try:
        from infra import portable

        return portable.configured_folder()
    except Exception:  # noqa: BLE001 - optional
        return None


def _fetch_module():
    from tools import online_fetch

    return online_fetch


# ------------------------------------------------------------------------------------------------ manifest and state
def load_manifest(source: Optional[str] = None, offline_ok: bool = True) -> Dict[str, Any]:
    """The release manifest (downloaded; the last good copy is kept in ``state/`` for offline use)."""
    of = _fetch_module()
    cfg = load_config() or {}
    # the manifest the runtime was last installed from (the latest one after "try latest" or a fallback), else the pinned one
    src = source or str(_channel_state().get("installed_from") or cfg.get("manifest_url", ""))
    if not src:
        raise ModulesError("no manifest address (modules.json)")
    cache = paths.state_dir() / MANIFEST_CACHE
    try:
        man = of.read_manifest(src, attempts=1 if setup_folder() is not None else 4)
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
            folder = setup_folder()                     # no internet and no cached copy: the manifest of the setup folder
            if folder is not None:
                try:
                    return of.validate_manifest(json.loads((folder / "manifest.json").read_text(encoding="utf-8-sig")))
                except (OSError, ValueError, of.FetchError):
                    pass
        raise ModulesError(str(exc)) from exc


def modules(manifest: Dict[str, Any]) -> List[Module]:
    """The modules of a manifest with their installed state.

    A module is *installed* when every component is (:func:`tools.online_fetch.installed_ok`: a wheel by package name and
    version, a zip part by SHA-256).  It is an *update* when it is not, but every outdated component still has an older
    copy in place (the program works as it is); otherwise it is *missing*.  ``size`` is what a download would really fetch:
    only the components that are not installed."""
    of = _fetch_module()
    rd = runtime_dir()
    state = of.load_state(rd)
    dist_infos = _dist_infos(rd)
    by_id = {c["id"]: c for c in manifest.get("components", [])}
    flavor = flavor_for(manifest)
    installed_v, installed_b = _installed_release()
    available_v, available_b = str(manifest.get("app_version", "")), str(manifest.get("build", ""))
    out: List[Module] = []
    for m in manifest.get("modules", []):
        comps = _components_of(m, by_id, flavor)
        todo = [c for c in comps if not of.installed_ok(c, state, rd)]
        ok = bool(comps) and not todo
        reused = False
        if not ok and m.get("reusable") == "torch":
            from infra import runtime_reuse

            reused = ok = runtime_reuse.reused("torch") is not None           # a verified copy from another program
        if ok:
            todo = []
        mod = Module(str(m["id"]), str(m.get("title", m["id"])), bool(m.get("required", True)),
                     0 if reused else sum(int(c["size"]) for c in (todo or comps)),
                     0 if reused else sum(int(c.get("unpacked_bytes", 0)) for c in (todo or comps)),
                     [c["id"] for c in comps], ok)
        mod.reused = reused
        present = bool(todo) and all(_older_copy(c, state, rd, dist_infos) for c in todo)
        if present and all(c.get("kind") != "wheel" for c in todo) and \
                not is_newer(available_v, installed_v, available_b, installed_b):
            # zip parts have no package version: a same-release rebuild (other bytes, same content) is not an update
            mod.installed = True
            mod.size = mod.unpacked = 0
        else:
            mod.update = present
        out.append(mod)
    return out


def _dist_infos(rd: Path) -> set:
    """Lower-case distribution names of the ``*.dist-info`` folders in the runtime folder."""
    try:
        return {p.name.split("-")[0].lower() for p in rd.iterdir() if p.name.endswith(".dist-info")}
    except OSError:
        return set()


def _older_copy(comp: Dict[str, Any], state: Dict[str, str], rd: Path, dist_infos: set) -> bool:
    """True when another version of this component is installed: a wheel by its distribution name (the marker
    ``<name>-<version>.dist-info``), a zip part by its id (stable across releases) with its marker files."""
    markers = list(comp.get("markers", []))
    if comp.get("kind") == "wheel" and markers:
        return markers[0].split("-")[0].lower() in dist_infos
    return comp["id"] in state and all((rd / mk).exists() for mk in markers)


def _installed_release() -> tuple:
    """``(app_version, build)`` of the manifest the installed runtime components came from (empty strings if unknown)."""
    try:
        d = json.loads((runtime_dir() / _fetch_module().STATE_FILE).read_text(encoding="utf-8"))
        return str(d.get("app_version", "")), str(d.get("build", ""))
    except (OSError, ValueError, AttributeError):
        return "", ""


def _version_key(v: str) -> tuple:
    """Sortable key of ``0.1.1-beta`` / ``v0.2.0`` / ``0.2.0-rc.1``: numbers first; a pre-release sorts before the final."""
    v = v.strip().lstrip("vV")
    core, _, pre = v.partition("-")
    nums = []
    for part in core.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        nums.append(int(digits) if digits else 0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums), (0, pre) if pre else (1, "")


def is_newer(available: str, installed: str, available_build: object = "", installed_build: object = "") -> bool:
    """True only when ``available`` is strictly newer than ``installed``: a newer app version, or the same version with a
    higher build number (CI builds 665, 666 ...).  Unknown installed version = newer; a missing build number on either
    side never makes a same-version build "newer" (that was the false "0.1.0-beta -> 0.1.0-beta" offer)."""
    if not available:
        return False
    if not installed:
        return True
    a, i = _version_key(available), _version_key(installed)
    if a != i:
        return a > i
    ab, ib = str(available_build or ""), str(installed_build or "")
    return ab.isdigit() and ib.isdigit() and int(ab) > int(ib)


def label(version: str, build: object = "") -> str:
    """``0.1.1-beta · build 665`` - the version with its build number when known."""
    return f"{version} \u00b7 build {build}" if version and build else version


def versions(manifest: Dict[str, Any]) -> tuple:
    """``(installed, available)`` labels of the runtime: the manifest the installed components came from vs this one,
    each with its build id when known."""
    installed, build = _installed_release()
    return label(installed, build), label(str(manifest.get("app_version", "")), str(manifest.get("build", "")))


def pending(mods: List[Module]) -> List[Module]:
    """What the Components window downloads: the missing required modules and every module with an update.

    The soundscape model is not a runtime wheel. ``infra.soundscape_model`` downloads it only when the user
    turns the setting on, and :func:`pending` never includes it.
    """
    return [m for m in mods if (m.required and not m.installed) or m.update]


def not_a_runtime_module(name: str) -> bool:
    """True for an optional piece the thin installer must not fetch. ``soundscape`` is ACE-Step."""
    return name == "soundscape"


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
    return all(m.installed or m.update for m in modules(man) if m.required)   # an update waits for the user's click


def missing_required(manifest: Optional[Dict[str, Any]] = None) -> List[Module]:
    """Required modules that are not installed (needs the manifest, which is downloaded unless given)."""
    if not is_thin():
        return []
    return [m for m in modules(manifest or load_manifest()) if m.required and not m.installed and not m.update]


# ------------------------------------------------------------------------------------------------ install
def live_line(phase: str, name: str, done: int = 0, total: int = 0, speed: float = 0.0) -> str:
    """The one translated status line of the Components window for a structured progress event of the downloader
    (``tools/online_fetch.py``): what is being downloaded right now, its percentage and speed, or that it is being verified."""
    from core.i18n import tr
    from infra.download_watch import fmt_bytes

    pct = int(100 * done / total) if total > 0 else 0
    if phase == "download":
        return tr("modules.now_download", name=name, pct=pct, speed=f"{fmt_bytes(speed)}/s" if speed > 0 else "\u2026")
    if phase == "verify":
        return tr("modules.now_verify", name=name)
    if phase == "unpack":
        return tr("modules.now_unpack", name=name, pct=pct)
    if phase == "folder":
        return tr("modules.now_folder", name=name)
    return name


class _Status:
    """Duck-typed ``online_fetch.Status``: forwards progress, can cancel by raising.  Its :meth:`event` receives the structured
    values of the downloader, so the window shows ONE translated live line ("Downloading: PyTorch (torch-...whl) - 42% - 35 MB/s")."""

    MIN_GAP = 0.15                   # seconds between two lines of the same file (a 1 MiB chunk arrives many times a second)

    def __init__(self, progress: Progress, cancelled: Callable[[], bool], titles: Optional[Dict[str, str]] = None) -> None:
        self.progress, self.cancelled, self.was_cancelled = progress, cancelled, False
        self.titles = titles or {}   # component id -> module title ("PyTorch"), so the line names more than a wheel file
        self._last_key, self._last_t = ("", ""), 0.0

    def _check_cancel(self) -> None:
        if self.cancelled():
            self.was_cancelled = True
            raise _fetch_module().FetchError("cancelled")      # a FetchError is not retried by the downloader

    def write(self, state: str, fraction: float, text: str, force: bool = False) -> None:
        self._check_cancel()
        self.progress(max(0.0, min(1.0, fraction)), text)

    def event(self, fraction: float, phase: str, comp_id: str, name: str, done: int, total: int, speed: float) -> None:
        self._check_cancel()
        now, key = time.monotonic(), (phase, comp_id)
        if key == self._last_key and now - self._last_t < self.MIN_GAP and not (total and done >= total):
            return
        self._last_key, self._last_t = key, now
        title = self.titles.get(comp_id, "")
        shown = f"{title} ({name})" if title and title.lower() not in name.lower() else (name or title)
        self.progress(max(0.0, min(1.0, fraction)), live_line(phase, shown, done, total, speed))


def install(module_ids: Optional[List[str]] = None, progress: Optional[Progress] = None,
            cancelled: Callable[[], bool] = lambda: False, manifest_source: Optional[str] = None) -> int:
    """Download and unpack the given modules (default: all required ones that are missing); returns the number of
    components fetched.  Resumable: run it again after a failure.  Activates the runtime folder when done."""
    of = _fetch_module()
    cfg = load_config() or {}
    src = manifest_source or str(cfg.get("manifest_url", ""))
    latest = "" if manifest_source else latest_url()
    try:
        man = load_manifest(src)
    except ModulesError:
        if not latest:
            raise
        man = load_manifest(latest)                    # the pinned manifest is gone: the newest release's one

    mods = modules(man)
    # default: what is missing; an update is installed only when its module is named (the user pressed Download)
    chosen = [m for m in mods if (m.id in module_ids if module_ids else (m.required and not m.installed and not m.update))]
    chosen = _reuse_first(man, chosen, progress)
    comp_ids = [c for m in chosen for c in m.components]
    if not comp_ids:
        activate()
        return 0
    titles = {cid: m.title for m in chosen for cid in m.components}
    st = _Status(progress or (lambda f, t: None), cancelled, titles)
    try:
        # a setup folder (installer option "Keep a portable setup folder") is used first: valid files there need no download,
        # newly downloaded ones are kept in it; without internet its own manifest is used
        n, used = of.run_channel(src, latest, runtime_dir(), cache_dir(), st, "latest" if prefer_latest() else "pinned",
                                 modules=[m.id for m in chosen], portable=setup_folder(), flavor=flavor_for(man) or "auto",
                                 prune=True)
    except of.FetchError as exc:
        if st.was_cancelled:
            raise Cancelled("cancelled") from exc
        raise ModulesError(str(exc)) from exc
    d = _channel_state()
    d["installed_from"] = used if used != src else ""
    _save_channel_state(d)
    if used != src:                                    # the module list must now be read from the manifest actually used
        log.info("runtime modules installed from %s (fallback / latest)", used)
        load_manifest(used)
    activate()
    return n


def _reuse_first(man: Dict[str, Any], chosen: List[Module], progress: Optional[Progress]) -> List[Module]:
    """Before downloading PyTorch look for a working copy on this PC (``infra/runtime_reuse.py``); drop it from the list if found."""
    reusable = {str(m["id"]) for m in man.get("modules", []) if m.get("reusable") == "torch"}
    if not any(m.id in reusable for m in chosen):
        return chosen
    from infra import runtime_reuse

    flavor = flavor_for(man)
    say = (lambda t: progress(0.0, t)) if progress else (lambda t: None)
    try:
        ext = runtime_reuse.try_reuse_torch(man.get("runtime") or {}, driver_cuda(), flavor, say)
    except Exception as exc:  # noqa: BLE001 - never let the search break the download
        log.warning("search for an existing PyTorch failed: %s", exc)
        ext = None
    if ext is None:
        return chosen
    return [m for m in chosen if m.id not in reusable]


def activate() -> Optional[Path]:
    """Put the runtime folder on ``sys.path`` (after the updater's ``packages`` overlay).  Safe to call repeatedly."""
    rd = paths.app_home() / "runtime"
    try:
        # reuse.json and .users.json are bookkeeping. A users file alone must not put this folder on the import path.
        if rd.is_dir() and any(p.name not in ("reuse.json", ".users.json") for p in rd.iterdir()):
            sp = str(rd)
            if sp not in sys.path:
                sys.path.insert(1 if sys.path and sys.path[0] == str(paths.packages_dir()) else 0, sp)
            import importlib

            importlib.invalidate_caches()
    except OSError:
        pass
    try:                                   # a verified foreign PyTorch: LAST on the path, our own pinned libraries win
        from infra import runtime_reuse

        for extra in runtime_reuse.extra_paths():
            if extra not in sys.path:
                sys.path.append(extra)
        import importlib

        importlib.invalidate_caches()
    except Exception:  # noqa: BLE001
        pass
    try:                                   # CTranslate2 is a CUDA 12 build; register its libraries before any import of it
        prepare_cuda12()
    except Exception:  # noqa: BLE001
        pass
    return rd if rd.is_dir() else None
