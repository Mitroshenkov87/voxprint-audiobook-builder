"""Completion manifest, quick health check with stable reason codes, and the uv-based per-user venv plan.

Ideas taken from Unsloth Studio's installer (read from its sources, 2026-10-03; see README):
  * the manifest is DELETED when an install/repair starts and WRITTEN ATOMICALLY as the very last step, so "manifest
    present" == "install finished"; a killed installer can never look ready;
  * it records app version, Python version, sha256 of the requirement files and the PyTorch wheel flavor; "ready" means
    manifest present AND nothing it recorded has changed AND key packages are importable;
  * one venv at a fixed per-user path, always driven as ``uv pip install --python <venv python>`` so nothing can
    silently target another environment; an ownership marker guarantees we never delete a folder we did not create;
  * torch is reinstalled only when the wheel flavor changes; failures give stable reason codes (``health.*`` in the
    locales), never raw tracebacks, and the UI/CLI can offer "Repair".
Nothing here touches any environment other than Voxprint's own venv (``<app home>/venv``).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from core import appinfo
from infra import paths

log = logging.getLogger("voxprint.install")

SCHEMA = 1
MANIFEST_NAME = "install_manifest.json"
OWNER_MARKER = ".voxprint-owned"
#: Modules whose presence proves a usable install (checked without importing them - fast; see ``deep``).
HEALTH_MODULES = ("torch", "transformers", "peft", "accelerate", "safetensors", "qwen_tts", "qwen_asr")
REQUIREMENT_FILES = ("requirements.txt", "requirements-verified.txt", "requirements-nodeps.txt")
PYTHON_VERSION_DEFAULT = "3.11"   # = the Python of the build/dev machine (build.bat: py -3.11); verify_install compares only major.minor with the running interpreter

# Stable reason codes (each has a locale key ``health.<code>``); they never contain tracebacks.
R_NO_MANIFEST = "no_manifest"
R_SCHEMA = "manifest_unreadable"
R_APP = "app_version_changed"
R_PYTHON = "python_changed"
R_REQS = "requirements_changed"
R_TORCH = "torch_flavor_changed"
R_MISSING = "package_missing"
R_IMPORT = "import_failed"
R_DUP = "dist_info_inconsistent"
R_MODEL_PARTIAL = "model_partial"
ALL_CODES = (R_NO_MANIFEST, R_SCHEMA, R_APP, R_PYTHON, R_REQS, R_TORCH, R_MISSING, R_IMPORT, R_DUP, R_MODEL_PARTIAL)


def manifest_path() -> Path:
    """Path of the install-completion manifest: ``<app_home>/install_manifest.json``."""
    return paths.app_home() / MANIFEST_NAME


def venv_dir() -> Path:
    """Fixed per-user location of Voxprint's own virtual environment: ``<app_home>/venv``."""
    return paths.app_home() / "venv"


def venv_python(venv: Optional[Path] = None) -> Path:
    """Path of the Python executable inside the venv (``Scripts/python.exe`` on Windows, ``bin/python`` elsewhere)."""
    v = venv or venv_dir()
    return v / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def requirement_hashes(root: Optional[Path] = None) -> Dict[str, str]:
    """sha256 of every requirements file (and the verified manifest), so a later change can be detected."""
    root = root or paths.resource_dir()
    out = {}
    for name in (*REQUIREMENT_FILES, "infra/verified_manifest.json"):
        p = root / name
        try:
            out[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            continue
    return out


def _atomic_write(path: Path, text: str) -> bool:
    """Write ``text`` to ``path`` through a temp file + fsync + ``os.replace``; returns False (and logs) on OSError."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)               # atomic on the same volume
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return True
    except OSError as exc:
        log.warning("cannot write %s: %s", path.name, exc)
        return False


def begin_install() -> None:
    """Called first: the previous manifest becomes ``.previous.json`` and the real one disappears."""
    p = manifest_path()
    try:
        if p.exists():
            os.replace(p, p.with_suffix(".previous.json"))
    except OSError as exc:
        log.warning("cannot retire manifest: %s", exc)
        try:
            p.unlink()
        except OSError:
            pass


def write_manifest(torch_flavor: str, python: Optional[str] = None, req_root: Optional[Path] = None,
                   now: Callable[[], float] = time.time) -> bool:
    """Called LAST, only after every install step succeeded: records what the environment was built from."""
    data = {
        "schema": SCHEMA,
        "completed_at": int(now()),
        "app": appinfo.APP_NAME,
        "app_version": appinfo.APP_VERSION,
        "python": python or ".".join(str(x) for x in sys.version_info[:3]),
        "platform": sys.platform,
        "expected_torch_tag": torch_flavor,
        "requirement_files": requirement_hashes(req_root),
    }
    return _atomic_write(manifest_path(), json.dumps(data, indent=2, sort_keys=True))


def read_manifest() -> Optional[dict]:
    """Load the install manifest, or None if it is missing / unreadable / not a JSON object."""
    try:
        d = json.loads(manifest_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


@dataclass
class HealthReport:
    """Result of :func:`verify_install`: ``ok`` plus reasons as ``"code"`` or ``"code:detail"`` strings."""
    ok: bool = True
    reasons: List[str] = field(default_factory=list)      # "code" or "code:detail"

    def add(self, code: str, detail: str = "") -> None:
        """Record a failure (marks the report as not ok)."""
        self.ok = False
        self.reasons.append(f"{code}:{detail}" if detail else code)

    @property
    def codes(self) -> List[str]:
        """Only the stable codes, without the details."""
        return [r.split(":", 1)[0] for r in self.reasons]


def _has_module(name: str) -> bool:
    """True if the module can be found without importing it (fast)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def inconsistent_dist_info(site: Path) -> List[str]:
    """Interrupted pip leaves duplicate dist-info folders or ``~``-prefixed backups: flag them."""
    if not site.is_dir():
        return []
    seen: Dict[str, int] = {}
    bad: List[str] = []
    for d in site.iterdir():
        n = d.name
        if n.startswith("~"):
            bad.append(n)
        elif n.endswith(".dist-info"):
            key = n.rsplit("-", 1)[0].lower().replace("_", "-")
            seen[key] = seen.get(key, 0) + 1
    bad += [f"{k}(x{c})" for k, c in seen.items() if c > 1]
    return bad


def verify_install(current_torch_flavor: Optional[str] = None, has_module: Callable[[str], bool] = _has_module,
                   modules: Tuple[str, ...] = HEALTH_MODULES, req_root: Optional[Path] = None,
                   require_manifest: bool = True, deep: bool = False,
                   importer: Callable[[str], object] = importlib.import_module) -> HealthReport:
    """Quick (milliseconds, no imports, no network) check of the install; ``reasons`` hold stable codes."""
    rep = HealthReport()
    m = read_manifest()
    if m is None:
        if manifest_path().exists():
            rep.add(R_SCHEMA)
        elif require_manifest:
            rep.add(R_NO_MANIFEST)
    else:
        if m.get("schema") != SCHEMA:
            rep.add(R_SCHEMA)
        if m.get("app_version") != appinfo.APP_VERSION:
            rep.add(R_APP, str(m.get("app_version")))
        if m.get("python") and tuple(str(m["python"]).split(".")[:2]) != tuple(str(x) for x in sys.version_info[:2]):
            rep.add(R_PYTHON, str(m.get("python")))
        old = m.get("requirement_files") or {}
        now = requirement_hashes(req_root)
        changed = sorted(k for k in now if k in old and old[k] != now[k])
        if changed:
            rep.add(R_REQS, ",".join(changed))
        if current_torch_flavor and m.get("expected_torch_tag") and current_torch_flavor != m["expected_torch_tag"]:
            rep.add(R_TORCH, f"{m['expected_torch_tag']}->{current_torch_flavor}")
    for mod in modules:
        if not has_module(mod):
            rep.add(R_MISSING, mod)
    if deep:    # real imports (slow: torch takes seconds) - used by --verify-install and Repair, not at every start
        for mod in modules:
            if has_module(mod):
                try:
                    importer(mod)
                except Exception as exc:  # noqa: BLE001 - only the class name is reported, never a traceback
                    rep.add(R_IMPORT, f"{mod}/{type(exc).__name__}")
    try:
        for p in (paths.packages_dir(),):
            bad = inconsistent_dist_info(p)
            if bad:
                rep.add(R_DUP, ",".join(bad[:3]))
    except OSError:
        pass
    return rep


def describe_reasons(rep: HealthReport) -> List[str]:
    """Localized text per reason (no tracebacks, no English fallbacks leaking into other locales)."""
    from core.i18n import tr

    out = []
    for r in rep.reasons:
        code, _, detail = r.partition(":")
        out.append(tr(f"health.{code}", detail=detail))
    return out


# ------------------------------------------------------------------------------------- the per-user venv (uv)
def owned(path: Path) -> bool:
    """True if the folder carries Voxprint's ownership marker (so it may be rebuilt/deleted)."""
    return (path / OWNER_MARKER).exists()


def plan_commands(uv: str, torch_flavor: str, python_version: str = PYTHON_VERSION_DEFAULT,
                  req_root: Optional[Path] = None, venv: Optional[Path] = None,
                  skip_torch: bool = False) -> List[List[str]]:
    """The exact command lines of a fresh install / repair.  Every pip step is ``uv pip install --python <venv>``."""
    venv = venv or venv_dir()
    root = req_root or paths.resource_dir()
    py = str(venv_python(venv))
    cmds: List[List[str]] = [[uv, "venv", "--python", python_version, "--seed", str(venv)]]
    if not skip_torch:
        index = f"https://download.pytorch.org/whl/{torch_flavor}"
        cmds.append([uv, "pip", "install", "--python", py, "torch", "torchaudio", "--index-url", index])
    cmds.append([uv, "pip", "install", "--python", py, "-r", str(root / "requirements.txt"),
                 "-r", str(root / "requirements-verified.txt")])
    cmds.append([uv, "pip", "install", "--python", py, "--no-deps", "-r", str(root / "requirements-nodeps.txt")])
    return cmds


@dataclass
class InstallResult:
    """Outcome of :func:`run_install`: success flag, index of the failed step and the tail of its output."""
    ok: bool
    failed_step: int = -1
    log: str = ""


def run_install(uv: str, torch_flavor: str, run: Callable[[List[str]], Tuple[int, str]],
                python_version: str = PYTHON_VERSION_DEFAULT, req_root: Optional[Path] = None,
                venv: Optional[Path] = None, current_flavor: Optional[str] = None,
                progress: Callable[[float, str], None] = lambda f, m: None) -> InstallResult:
    """Install / repair into Voxprint's own venv.  The manifest is removed first and written last.

    ``current_flavor``: torch flavor already in the venv (from the old manifest); equal => torch is not touched."""
    venv = venv or venv_dir()
    if venv.exists() and any(venv.iterdir()) and not owned(venv):
        return InstallResult(False, 0, "venv folder is not owned by Voxprint")     # never rebuild a foreign folder
    skip_torch = bool(current_flavor and current_flavor == torch_flavor and venv_python(venv).exists())
    begin_install()
    cmds = plan_commands(uv, torch_flavor, python_version, req_root, venv, skip_torch)
    if venv_python(venv).exists():
        cmds = cmds[1:]                                  # reuse the existing venv, repair only what differs
    for i, cmd in enumerate(cmds):
        progress(i / max(1, len(cmds)), " ".join(cmd[:4]))
        rc, out = run(cmd)
        if rc != 0:
            log.error("install step %d failed (rc=%s): %s", i, rc, out[-1500:])
            return InstallResult(False, i, out[-1500:])
        if i == 0 and not owned(venv) and venv.exists():
            (venv / OWNER_MARKER).write_text("voxprint", encoding="utf-8")
    if venv.exists() and not owned(venv):
        (venv / OWNER_MARKER).write_text("voxprint", encoding="utf-8")
    ok = write_manifest(torch_flavor, python=python_version)
    progress(1.0, "done")
    return InstallResult(ok)


def cli_verify(deep: bool = True, print_fn: Callable[[str], None] = print) -> int:
    """`main.py --verify-install`: prints localized reasons, returns 0 when healthy."""
    from core.i18n import tr

    rep = verify_install(deep=deep)
    if rep.ok:
        print_fn(tr("health.ok"))
        return 0
    print_fn(tr("health.failed"))
    for line in describe_reasons(rep):
        print_fn("  - " + line)
    print_fn(tr("health.repair_hint"))
    return 1


def repair_install(run: Callable[[List[str]], Tuple[int, str]], which: Callable[[str], Optional[str]],
                   progress: Callable[[float, str], None] = lambda f, m: None,
                   detect_cuda: Optional[Callable[[], object]] = None) -> Tuple[int, str]:
    """Rebuilds only what differs in Voxprint's own venv (needs `uv` on PATH).  Returns (exit code, localized text);
    shared by `--repair` and the GUI "Repair" button."""
    from core.i18n import tr
    from infra import env_probe

    uv = which("uv")
    if not uv:
        return 2, tr("health.no_uv")
    cuda = detect_cuda() if detect_cuda else env_probe.detect_driver_cuda()
    flavor = env_probe.torch_flavor_for_driver(cuda)   # type: ignore[arg-type]
    old = (read_manifest() or {}).get("expected_torch_tag")
    res = run_install(uv, flavor, run, current_flavor=old, progress=progress)
    if res.ok:
        return 0, tr("health.repair_done")
    return 1, tr("health.repair_failed", step=res.failed_step)


def cli_repair(run: Callable[[List[str]], Tuple[int, str]], which: Callable[[str], Optional[str]],
               print_fn: Callable[[str], None] = print, detect_cuda: Optional[Callable[[], object]] = None) -> int:
    """`main.py --repair`."""
    rc, text = repair_install(run, which, lambda f, m: print_fn(f"  [{int(f * 100):3d}%] {m}"), detect_cuda)
    print_fn(text)
    return rc
