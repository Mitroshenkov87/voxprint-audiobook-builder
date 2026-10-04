"""Reuse what is already on the PC before anything is downloaded (thin installer, see docs/THIN-INSTALLER.md).

PyTorch is the only big part (2.5 GB), and many people already have a working copy in a system Python, a conda/venv project or a
Pinokio app.  :func:`find_torch` looks for such copies WITHOUT running anything from them (it reads ``torch-*.dist-info`` in the
``site-packages`` folders it finds); :func:`assess` applies the compatibility rules below; :func:`verify` imports the candidate in a
child process of Voxprint itself and runs a tiny computation (on the GPU when there is one).  Only a candidate that passes is used -
its ``site-packages`` is put at the END of ``sys.path`` (``activate()`` in ``infra/modules.py``), so our own pinned libraries always
win and the foreign environment only supplies ``torch`` and ``torchaudio``.  If anything fails the module is downloaded as usual.

Compatibility rules (all must hold; every rejection is logged with its reason):

1. same CPython minor version as the running Voxprint (binary wheels: the ``Tag:`` in the wheel's ``WHEEL`` file, e.g. ``cp311``) and
   64-bit Windows (``win_amd64``);
2. ``torch`` version inside ``compat.torch`` of ``infra/runtime_lock.json`` (a range around the version we tested);
3. ``torchaudio`` is installed in the same ``site-packages`` with the same release number as ``torch`` (they are built as a pair);
4. flavor: a CPU build is accepted only on a PC without an NVIDIA GPU; a CUDA build must not be newer than the driver supports
   (an older CUDA build than the driver offers is fine); a build without a local tag on Windows is a CPU build;
5. ``torch/lib`` and the package folders really exist (a half-deleted environment is skipped);
6. the child-process check passes (``import torch``, a tensor computation, ``.numpy()`` round trip, and a CUDA computation if the
   driver offers a GPU), within the time limit.

Ready copies from an earlier run are in ``<app home>/runtime`` and recorded by the downloader (SHA-256), so a reinstall or an update
that does not change a part downloads nothing.  ``VOXPRINT_NO_REUSE=1`` turns the search off.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from infra import env_probe, paths

log = logging.getLogger("voxprint.reuse")

STATE_NAME = "reuse.json"
VERIFY_TIMEOUT = 150
MAX_ROOTS = 80
OK_MARK = "VXTORCH OK"


# ------------------------------------------------------------------------------------------------ what was found
@dataclass
class ExternalTorch:
    site: str                  # the site-packages folder that holds torch (and torchaudio)
    torch: str                 # '2.11.0+cu128'
    torchaudio: str            # '' when missing
    py_tag: str                # 'cp311' from the wheel's WHEEL file, '' when unknown
    platform: str              # 'win_amd64' ...
    flavor: str                # 'cu128' / 'cpu' / 'unknown'

    def key(self) -> Tuple[Version, str]:
        try:
            return (Version(self.torch.split("+")[0]), self.site)
        except InvalidVersion:
            return (Version("0"), self.site)


def _read_dist(site: Path, name: str) -> Optional[Tuple[str, str, str]]:
    """(version, python tag, platform) of distribution ``name`` in ``site`` from its dist-info, or None."""
    try:
        cands = [d for d in site.iterdir() if d.is_dir() and d.name.lower().startswith(name + "-") and d.name.endswith(".dist-info")]
    except OSError:
        return None
    for d in cands:
        m = re.match(rf"^{name}-(.+)\.dist-info$", d.name, re.I)
        if not m:
            continue
        py = plat = ""
        try:
            for line in (d / "WHEEL").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("Tag:"):
                    t = line.split(":", 1)[1].strip().split("-")
                    if len(t) == 3:
                        py, plat = t[0], t[2]
                    break
        except OSError:
            pass
        return m.group(1), py, plat
    return None


def flavor_of(version: str) -> str:
    """'2.11.0+cu128' -> 'cu128'.  No local tag: on Windows the PyPI wheel is a CPU build."""
    f = env_probe.torch_flavor_of(version)
    if f:
        return f
    return "cpu" if sys.platform == "win32" else "unknown"


def inspect_site(site: Path) -> Optional[ExternalTorch]:
    t = _read_dist(site, "torch")
    if t is None:
        return None
    a = _read_dist(site, "torchaudio")
    return ExternalTorch(str(site), t[0], a[0] if a else "", t[1], t[2], flavor_of(t[0]))


def _site_dirs_of(prefix: Path) -> List[Path]:
    out = []
    for rel in ("Lib/site-packages", "lib/site-packages"):
        out.append(prefix / rel)
    try:
        out += sorted((prefix / "lib").glob("python3.*/site-packages"))
    except OSError:
        pass
    return [p for p in out if p.is_dir()]


def candidate_prefixes(environ: Optional[Dict[str, str]] = None, home: Optional[Path] = None) -> List[Path]:
    """Folders that may be a Python installation / virtual environment (cheap globs, capped; nothing is executed)."""
    env = os.environ if environ is None else environ
    try:
        home = home or Path.home()
    except (RuntimeError, OSError):
        home = Path(".")
    pre: List[Path] = []

    def add(p: Path) -> None:
        try:
            if p not in pre and p.is_dir():
                pre.append(p)
        except OSError:
            pass

    for var in ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONHOME"):
        if env.get(var):
            add(Path(env[var]))
    try:
        for exe, _src in env_probe.candidate_pythons(environ=dict(env)):
            e = Path(exe)
            add(e.parent.parent if e.parent.name.lower() in ("scripts", "bin") else e.parent)
    except Exception:  # noqa: BLE001
        pass
    la, ad = env.get("LOCALAPPDATA"), env.get("APPDATA")
    pf = [env.get("ProgramFiles"), env.get("ProgramFiles(x86)"), env.get("ProgramData"), "C:\\"]
    roots: List[Path] = []
    if la:
        roots.append(Path(la) / "Programs" / "Python")
    for base in pf:
        if base:
            roots.append(Path(base))
    for r in roots:
        try:
            for d in sorted(r.glob("Python3*")):
                add(d)
        except OSError:
            pass
    if ad:
        try:
            for d in sorted((Path(ad) / "Python").glob("Python3*")):
                if (d / "site-packages").is_dir():
                    add(d)                                    # the per-user site: <prefix>/site-packages
        except OSError:
            pass
    conda_roots = [home / n for n in ("miniconda3", "anaconda3", "miniforge3", "mambaforge", ".conda")]
    for base in pf[:3]:
        if base:
            conda_roots += [Path(base) / n for n in ("miniconda3", "anaconda3", "miniforge3")]
    for c in conda_roots:
        add(c)
        try:
            for d in sorted((c / "envs").iterdir())[:40]:
                add(d)
        except OSError:
            pass
    for pk in ([Path(env["PINOKIO_HOME"])] if env.get("PINOKIO_HOME") else []) + [home / "pinokio"]:
        try:
            for app in sorted((pk / "api").iterdir())[:80]:
                for sub in ("env", "venv", ".venv", "app/env", "app/venv", "app/.venv"):
                    add(app / sub)
        except OSError:
            pass
    # virtual environments of the user's projects: <folder>/<project>/<venv> and <folder>/<group>/<project>/<venv>
    def venvs(proj: Path) -> None:
        for v in (".venv", "venv", "env"):
            add(proj / v)
        add(proj)

    for top in (home, home / "Documents", home / "Desktop", home / "source" / "repos", home / "projects", home / "Projects",
                home / "Downloads", home / "dev", Path("C:/AI"), Path("D:/AI")):
        try:
            level1 = sorted(p for p in top.iterdir() if p.is_dir() and not p.name.startswith("."))[:100]
        except OSError:
            continue
        for proj in level1:
            venvs(proj)
            try:
                for sub in sorted(p for p in proj.iterdir() if p.is_dir() and not p.name.startswith("."))[:40]:
                    venvs(sub)
            except OSError:
                continue
    return pre


def find_torch(prefixes: Optional[Iterable[Path]] = None, own_runtime: Optional[Path] = None) -> List[ExternalTorch]:
    """All foreign ``torch`` installations found (newest first).  Only file names / small text files are read."""
    seen, found = set(), []
    rt = None
    try:
        rt = (own_runtime or (paths.app_home() / "runtime")).resolve()
    except OSError:
        pass
    n = 0
    for pre in (prefixes if prefixes is not None else candidate_prefixes()):
        sites = _site_dirs_of(pre)
        if (pre / "site-packages").is_dir():
            sites.append(pre / "site-packages")
        for site in sites:
            try:
                key = site.resolve()
            except OSError:
                continue
            if key in seen or key == rt:
                continue
            seen.add(key)
            n += 1
            if n > MAX_ROOTS:
                break
            ext = inspect_site(site)
            if ext is not None:
                found.append(ext)
    found.sort(key=lambda e: e.key(), reverse=True)
    return found


# ------------------------------------------------------------------------------------------------ rules
def assess(ext: ExternalTorch, lock: Dict, wanted_flavor: str, driver_cuda: Optional[Tuple[int, int]],
           py_tag: Optional[str] = None) -> Tuple[bool, str]:
    """Apply the compatibility rules 1-5 (see the module docstring); returns (ok, reason)."""
    py_tag = py_tag or f"cp{sys.version_info.major}{sys.version_info.minor}"
    if ext.py_tag and ext.py_tag != py_tag:
        return False, f"built for {ext.py_tag}, Voxprint runs {py_tag}"
    if not ext.py_tag:
        return False, "the Python version of the wheel is unknown"
    want_plat = "win_amd64" if sys.platform == "win32" else ext.platform
    if ext.platform and ext.platform != want_plat:
        return False, f"platform {ext.platform}"
    spec = str(lock.get("compat", {}).get("torch", ""))
    try:
        v = Version(ext.torch.split("+")[0])
        if spec and v not in SpecifierSet(spec):
            return False, f"torch {ext.torch} is outside the supported range {spec}"
    except (InvalidVersion, InvalidSpecifier):
        return False, f"unreadable torch version {ext.torch!r}"
    if not ext.torchaudio:
        return False, "torchaudio is not installed next to torch"
    if ext.torchaudio.split("+")[0] != ext.torch.split("+")[0]:
        return False, f"torchaudio {ext.torchaudio} does not match torch {ext.torch}"
    have_gpu = driver_cuda is not None and wanted_flavor != "cpu"
    if ext.flavor == "cpu" and have_gpu:
        return False, "a CPU build, but this PC has an NVIDIA GPU"
    if ext.flavor.startswith("cu"):
        cu = env_probe._cuda_tuple(ext.flavor)
        if have_gpu and cu and driver_cuda and cu > driver_cuda:
            return False, f"{ext.flavor} needs a newer NVIDIA driver than installed (CUDA {driver_cuda[0]}.{driver_cuda[1]})"
    site = Path(ext.site)
    for need in ("torch/lib", "torch/__init__.py", "torchaudio/__init__.py"):
        if not (site / need).exists():
            return False, f"{need} is missing"
    return True, "ok"


def probe_command() -> List[str]:
    """Command line of the Voxprint instance that runs the check (``--probe-torch``)."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--probe-torch"]
    return [sys.executable, str(Path(__file__).resolve().parent.parent / "main.py"), "--probe-torch"]


def verify(ext: ExternalTorch, run: Optional[Callable[[List[str], Dict[str, str], int], Tuple[int, str]]] = None) -> Tuple[bool, str]:
    """Rule 6: import and use the candidate in a child process of Voxprint (its own interpreter / DLLs / numpy)."""
    env = dict(os.environ)
    env["VOXPRINT_EXTRA_SITE"] = ext.site
    run = run or _run_child
    rc, out = run(probe_command(), env, VERIFY_TIMEOUT)
    ok = rc == 0 and OK_MARK in out
    detail = next((l for l in out.splitlines() if OK_MARK in l), (out.strip().splitlines() or [""])[-1])
    return ok, detail[:300]


def _run_child(cmd: List[str], env: Dict[str, str], timeout: int) -> Tuple[int, str]:
    flags = 0x08000000 if sys.platform == "win32" else 0        # CREATE_NO_WINDOW
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout, env=env, creationflags=flags)
        return p.returncode, (p.stdout or "") + (p.stderr or "")[-800:]
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def probe_torch_main() -> int:
    """``Voxprint --probe-torch``: import torch from ``VOXPRINT_EXTRA_SITE`` (added at the END of sys.path) and compute something."""
    extra = os.environ.get("VOXPRINT_EXTRA_SITE", "")
    if extra and extra not in sys.path:
        sys.path.append(extra)
    try:
        import numpy as np
        import torch
        import torchaudio  # noqa: F401

        x = torch.arange(4, dtype=torch.float32) * 2 + 1
        assert float(x.sum()) == 16.0 and np.asarray(x.numpy()).shape == (4,)
        dev = "cpu"
        if torch.cuda.is_available():
            y = (x.cuda() @ x.cuda())
            torch.cuda.synchronize()
            assert float(y) == 1 + 9 + 25 + 49
            dev = "cuda:" + torch.cuda.get_device_name(0)
        print(f"{OK_MARK} torch {torch.__version__} from {Path(torch.__file__).parent.parent} on {dev}", flush=True)
        return 0
    except BaseException as exc:  # noqa: BLE001 - any failure = not usable
        print(f"VXTORCH FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1


# ------------------------------------------------------------------------------------------------ remembered choice
def state_path() -> Path:
    return paths.app_home() / "runtime" / STATE_NAME


def load_state() -> Dict[str, Dict]:
    try:
        d = json.loads(state_path().read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_choice(group: str, ext: ExternalTorch) -> None:
    st = load_state()
    st[group] = dict(asdict(ext), verified=datetime.datetime.now().isoformat(timespec="seconds"))
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def forget(group: str = "torch") -> None:
    st = load_state()
    if st.pop(group, None) is not None:
        p = state_path()
        p.write_text(json.dumps(st, indent=1), encoding="utf-8")


def reused(group: str = "torch") -> Optional[ExternalTorch]:
    """The remembered foreign copy, if it is still there unchanged (same version in the same folder); no import, no subprocess."""
    d = load_state().get(group)
    if not d or os.environ.get("VOXPRINT_NO_REUSE", "") not in ("", "0"):
        return None
    try:
        ext = ExternalTorch(**{k: d[k] for k in ("site", "torch", "torchaudio", "py_tag", "platform", "flavor")})
    except (KeyError, TypeError):
        return None
    now = inspect_site(Path(ext.site))
    if now is None or now.torch != ext.torch or now.torchaudio != ext.torchaudio:
        return None
    for need in ("torch/lib", "torch/__init__.py"):
        if not (Path(ext.site) / need).exists():
            return None
    return ext


def extra_paths() -> List[str]:
    """Folders to append to ``sys.path`` (end) for the reused copies."""
    e = reused("torch")
    return [e.site] if e else []


# ------------------------------------------------------------------------------------------------ the decision
def choose_flavor(flavors: List[str], driver_cuda: Optional[Tuple[int, int]]) -> str:
    """Best flavor of the lock for this PC: the newest CUDA build the driver supports, else ``cpu``.
    ``VOXPRINT_TORCH_FLAVOR`` forces one (tests, support)."""
    forced = os.environ.get("VOXPRINT_TORCH_FLAVOR", "").strip()
    if forced and forced in flavors:
        return forced
    best, best_cu = "cpu", (0, 0)
    for f in flavors:
        cu = env_probe._cuda_tuple(f)
        if cu and driver_cuda and cu <= driver_cuda and cu > best_cu:
            best, best_cu = f, cu
    return best


def try_reuse_torch(lock: Dict, driver_cuda: Optional[Tuple[int, int]], wanted_flavor: str,
                    progress: Callable[[str], None] = lambda t: None, prefixes: Optional[Iterable[Path]] = None,
                    verifier: Optional[Callable[[ExternalTorch], Tuple[bool, str]]] = None) -> Optional[ExternalTorch]:
    """Find, assess and verify a foreign PyTorch; remembers and returns the first that passes, else None (-> download)."""
    if os.environ.get("VOXPRINT_NO_REUSE", "") not in ("", "0"):
        log.info("reuse of existing libraries is switched off (VOXPRINT_NO_REUSE)")
        return None
    t0 = time.monotonic()
    progress("Looking for an existing PyTorch on this PC")
    found = find_torch(prefixes)
    log.info("existing PyTorch copies found: %s (%.1f s)", [f"{e.torch} in {e.site}" for e in found] or "none", time.monotonic() - t0)
    for ext in found:
        ok, why = assess(ext, lock, wanted_flavor, driver_cuda)
        if not ok:
            log.info("not reused: torch %s in %s - %s", ext.torch, ext.site, why)
            continue
        progress(f"Checking PyTorch {ext.torch} found in {ext.site}")
        good, detail = (verifier or verify)(ext)
        if good:
            log.info("PyTorch %s in %s is reused: %s", ext.torch, ext.site, detail)
            save_choice("torch", ext)
            return ext
        log.warning("not reused: torch %s in %s failed the check: %s", ext.torch, ext.site, detail)
    return None
