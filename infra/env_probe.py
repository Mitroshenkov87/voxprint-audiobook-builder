"""Read-only probe of what is already installed, and the reuse / upgrade / install decision for each component.

Rule (agreed with the user, revised):
  * installed and current (== the newest stable VERIFIED version) or proven compatible -> REUSE;
  * installed in Voxprint's OWN environment (packaged exe overlay / Voxprint venv) but older than the newest verified
    version -> UPGRADE automatically;
  * installed in the USER's environment (Voxprint was started from a system/conda/Pinokio Python) and older ->
    OFFER_UPGRADE: nothing is touched silently; the UI asks once (what changes, which environment).  Accepted -> pip
    upgrades it there; declined and still compatible -> reused as it is; declined and incompatible -> Voxprint's own
    ``packages/`` overlay shadows it (the user's environment stays untouched);
  * missing / incompatible-and-not-outdated -> INSTALL the newest stable verified version into Voxprint's own
    environment (the user's environment is not modified).
Other environments (system Python, Pinokio apps, conda...) are only *looked at*: a subprocess runs
``importlib.metadata`` there (no package is imported, nothing is written).  Their contents inform the report and the
choice of the Python used for ``pip --target`` (matching version), but their packages are never loaded into Voxprint:
mixing site-packages of different environments breaks binary wheels.

Unsloth (checked 2026-10-03): NOT used.  It does not support Qwen3-TTS fine-tuning yet (unslothai/unsloth#3951 open,
needs huggingface/transformers#44517, which is still an open PR), Qwen3-TTS uses a custom training loop, and installing
Unsloth would pull its own pins (trl, xformers, transformers range incl. 5.x) that conflict with the verified set.
Voxprint keeps its own LoRA loop (core/lora_trainer.py).  If found it is only reported (reason ``no_qwen3_tts_training``).

NVIDIA driver -> PyTorch wheel flavor follows the CUDA version printed by ``nvidia-smi`` (same mapping Unsloth's
installer uses: cu118 / cu124 / cu126 / cu128 / cu130, else CPU).  When the header has no ``CUDA Version`` /
``CUDA UMD Version``, the driver version is used instead: 570+ is treated as CUDA 12.8, 560-569 as 12.6, older as CPU.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


from infra.version_manager import (ACTION_IGNORE, ACTION_INSTALL, ACTION_OFFER, ACTION_REUSE, ACTION_UPGRADE,
                                   IGNORED_PACKAGES,
                                   TRACKED_PACKAGES, Decision, decide_package)

log = logging.getLogger("voxprint.env")

Runner = Callable[[List[str]], Tuple[int, str]]
#: What we ask other Pythons about (metadata only).
PROBED_PACKAGES = ("torch", "torchaudio", "transformers", "peft", "accelerate", "bitsandbytes", "huggingface_hub",
                   "qwen-tts", "qwen-asr", "unsloth", "safetensors")
#: Upper bound on the number of other interpreters we ask (each probe is a subprocess).
MAX_PYTHONS = 6

# Tiny script run by *other* Pythons (``python -I -c``): prints the installed versions as one JSON line.
# It uses importlib.metadata only - nothing is imported from the probed environment, nothing is written.
_PROBE_CODE = (
    "import json,sys\n"
    "from importlib import metadata as m\n"
    "out={'python':list(sys.version_info[:3]),'packages':{}}\n"
    "for n in sys.argv[1:]:\n"
    "    try: out['packages'][n]=m.version(n)\n"
    "    except Exception: pass\n"
    "print('VXPROBE '+json.dumps(out))\n"
)


def is_own_environment() -> bool:
    """True if the interpreter running Voxprint belongs to Voxprint: the packaged exe (bundled libraries + our
    ``packages/`` overlay) or Voxprint's own venv.  Anything else (system Python, a conda/Pinokio env the user started
    Voxprint from) is the USER's environment: its outdated components are never changed without asking.
    ``VOXPRINT_OWN_ENV=1/0`` overrides (tests, unusual setups)."""
    v = os.environ.get("VOXPRINT_OWN_ENV", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    if getattr(sys, "frozen", False):
        return True
    try:
        from infra import paths

        return Path(sys.prefix).resolve().is_relative_to((paths.app_home() / "venv").resolve())
    except (OSError, ValueError):
        return False


def _run(args: List[str]) -> Tuple[int, str]:
    """Default runner for subprocess probes (30 s timeout)."""
    from infra.updater import run_subprocess

    return run_subprocess(args, timeout=30)


# ------------------------------------------------------------------------------------- torch / CUDA flavor
#: (minimum CUDA version of the driver, wheel tag), checked top-down.
CUDA_FLAVORS: Tuple[Tuple[Tuple[int, int], str], ...] = (
    ((13, 0), "cu130"), ((12, 8), "cu128"), ((12, 6), "cu126"), ((12, 4), "cu124"), ((11, 8), "cu118"))


_CUDA_IN_HEADER = re.compile(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)")   # 6xx drivers print "CUDA UMD Version"
_DRIVER_MAJOR = re.compile(r"^\s*(\d+)\.")


def parse_nvidia_smi_cuda(text: str) -> Optional[Tuple[int, int]]:
    """Extract the driver's CUDA version ``(major, minor)`` from ``nvidia-smi`` output, or None."""
    m = _CUDA_IN_HEADER.search(text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def cuda_from_driver_version(text: str) -> Optional[Tuple[int, int]]:
    """Map ``nvidia-smi --query-gpu=driver_version`` when the header has no CUDA version.

    The first line ``major.minor...`` decides: >= 570 -> CUDA 12.8, >= 560 -> CUDA 12.6, otherwise None (CPU).
    """
    lines = (text or "").splitlines()
    if not lines:
        return None
    m = _DRIVER_MAJOR.match(lines[0])
    if not m:
        return None
    major = int(m.group(1))
    if major >= 570:
        return (12, 8)
    if major >= 560:
        return (12, 6)
    return None


def torch_flavor_for_driver(cuda: Optional[Tuple[int, int]]) -> str:
    """Wheel flavor for the driver's CUDA version; ``cpu`` without an NVIDIA driver or with a too old one."""
    if cuda:
        for need, tag in CUDA_FLAVORS:
            if cuda >= need:
                return tag
    return "cpu"


def torch_flavor_of(version: Optional[str]) -> Optional[str]:
    """'2.8.0+cu128' -> 'cu128'; '2.8.0+cpu' -> 'cpu'; no local tag -> None (unknown)."""
    if not version or "+" not in version:
        return None
    return version.split("+", 1)[1] or None


def _cuda_tuple(tag: str) -> Optional[Tuple[int, int]]:
    """'cu128' -> (12, 8); 'cu118' -> (11, 8); anything else -> None."""
    m = re.fullmatch(r"cu(\d{2,3})", tag or "")
    if not m:
        return None
    n = m.group(1)
    return (int(n[:-1]), int(n[-1])) if len(n) == 3 else (int(n[:2]), int(n[2:]))


def decide_torch(installed: Optional[str], wanted_flavor: str, external: bool = False) -> Decision:
    """torch has no pin: any reasonably recent build is reusable; the *flavor* must fit the machine.
    CPU build on a machine with an NVIDIA driver, or a CUDA build newer than the driver supports -> replace
    (in Voxprint's own environment); an older CUDA build than the driver offers is fine (drivers are backward compatible)."""
    if installed is None:
        return Decision("torch", None, wanted_flavor, ACTION_INSTALL, "missing")
    have = torch_flavor_of(installed)
    if have is None:
        return Decision("torch", installed, wanted_flavor, ACTION_REUSE, "flavor_unknown")
    if have == wanted_flavor:
        return Decision("torch", installed, wanted_flavor, ACTION_REUSE, "current")
    change = ACTION_OFFER if external else ACTION_UPGRADE      # the user's torch is never swapped silently
    if have == "cpu" and wanted_flavor != "cpu":
        return Decision("torch", installed, wanted_flavor, change, "torch_flavor_changed", outdated=True)
    hv, wv = _cuda_tuple(have), _cuda_tuple(wanted_flavor)
    if hv and wv and hv <= wv:
        return Decision("torch", installed, wanted_flavor, ACTION_REUSE, "compatible_older_cuda")
    if wanted_flavor == "cpu":
        return Decision("torch", installed, wanted_flavor, ACTION_REUSE, "cuda_build_without_gpu")
    return Decision("torch", installed, wanted_flavor, change, "torch_flavor_changed", outdated=True,
                    compatible=False)


def detect_driver_cuda(run: Runner = _run, which: Callable[[str], Optional[str]] = shutil.which) -> Optional[Tuple[int, int]]:
    """CUDA version supported by the installed NVIDIA driver (via ``nvidia-smi``), or None without a driver.

    The header's ``CUDA Version`` / ``CUDA UMD Version`` wins.  Without that line, the first
    ``driver_version`` query line is mapped (>= 570 -> 12.8, >= 560 -> 12.6, else None).  Any error is None.
    """
    try:
        exe = which("nvidia-smi")
        if not exe:
            return None
        rc, out = run([exe])
        found = parse_nvidia_smi_cuda(out) if rc == 0 else None
        if found:
            return found
        rc, out = run([exe, "--query-gpu=driver_version", "--format=csv,noheader"])
        if rc != 0:
            return None
        return cuda_from_driver_version(out)
    except Exception:  # noqa: BLE001 - a missing or broken nvidia-smi means CPU
        return None


# ------------------------------------------------------------------------------------- ffmpeg
@dataclass(frozen=True)
class FfmpegInfo:
    """An ffmpeg found on PATH: location, version string and whether ``ffmpeg -version`` really ran."""
    path: str
    version: str
    ok: bool            # `ffmpeg -version` ran and printed a version


def probe_ffmpeg(which: Callable[[str], Optional[str]] = shutil.which, run: Runner = _run) -> Optional[FfmpegInfo]:
    """The ffmpeg on PATH, smoke-tested (a broken or foreign binary must not be trusted)."""
    exe = which("ffmpeg")
    if not exe:
        return None
    rc, out = run([exe, "-hide_banner", "-version"])
    m = re.search(r"ffmpeg version\s+(\S+)", out or "")
    return FfmpegInfo(exe, m.group(1) if m else "", rc == 0 and bool(m))


# ------------------------------------------------------------------------------------- other Pythons
@dataclass
class PythonEnv:
    """Another Python interpreter we looked at: executable, version, tracked package versions and where it was found."""
    exe: str
    version: Tuple[int, int, int]
    packages: Dict[str, str] = field(default_factory=dict)
    source: str = "path"


def candidate_pythons(which: Callable[[str], Optional[str]] = shutil.which,
                      environ: Optional[Dict[str, str]] = None) -> List[Tuple[str, str]]:
    """(exe, source) of interpreters worth asking: PATH, active venv/conda, Pinokio app envs.  Capped, de-duplicated."""
    env = os.environ if environ is None else environ
    raw: List[Tuple[str, str]] = []
    exe_name = "python.exe" if sys.platform == "win32" else "python"
    sub = "Scripts" if sys.platform == "win32" else "bin"
    for var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        v = env.get(var)
        if v:
            raw.append((str(Path(v) / sub / exe_name) if var == "VIRTUAL_ENV" else
                        str(Path(v) / (exe_name if sys.platform == "win32" else "bin/python")), var.lower()))
    for n in ("python", "python3", "py"):
        w = which(n)
        if w:
            raw.append((w, "path"))
    homes = [Path(env["PINOKIO_HOME"])] if env.get("PINOKIO_HOME") else []
    try:
        homes.append(Path.home() / "pinokio")
    except (RuntimeError, OSError):
        pass
    for h in homes:
        api = h / "api"
        try:
            apps = sorted(d for d in api.iterdir() if d.is_dir())[:60] if api.is_dir() else []
        except OSError:
            apps = []
        for app in apps:
            for venv in ("env", "venv", ".venv"):
                for base in (app, app / "app"):
                    raw.append((str(base / venv / sub / exe_name), "pinokio"))
    seen, out = set(), []
    me = None
    try:
        me = str(Path(sys.executable).resolve())
    except OSError:
        pass
    for exe, src in raw:
        try:
            if not Path(exe).is_file():
                continue
            key = str(Path(exe).resolve())
        except OSError:
            continue
        if key in seen or key == me:
            continue
        seen.add(key)
        out.append((exe, src))
        if len(out) >= MAX_PYTHONS:
            break
    return out


def probe_python(exe: str, source: str = "path", run: Runner = _run,
                 names: Tuple[str, ...] = PROBED_PACKAGES) -> Optional[PythonEnv]:
    """Ask another interpreter for its version and the versions of ``names`` (metadata only); None on any failure."""
    rc, out = run([exe, "-I", "-c", _PROBE_CODE, *names])
    if rc != 0 or "VXPROBE" not in out:
        return None
    try:
        data = json.loads(out.split("VXPROBE", 1)[1].strip().splitlines()[0])
        v = tuple(int(x) for x in data["python"][:3])
        return PythonEnv(exe, v, dict(data.get("packages") or {}), source)  # type: ignore[arg-type]
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def pick_pip_python(envs: List[PythonEnv],
                    want: Tuple[int, int] = (sys.version_info[0], sys.version_info[1])) -> Optional[str]:
    """Python used for ``pip install --target <Voxprint dir>``: same major.minor as Voxprint itself (binary wheels must
    match), newest patch.  pip --target writes only into our folder, the interpreter's environment stays untouched."""
    ok = [e for e in envs if e.version[:2] == tuple(want)]
    ok.sort(key=lambda e: e.version, reverse=True)
    return ok[0].exe if ok else None


# ------------------------------------------------------------------------------------- the report
@dataclass
class EnvReport:
    """Everything the read-only probe found: per-component decisions, other Pythons, ffmpeg, driver CUDA, ignored packages."""
    decisions: List[Decision] = field(default_factory=list)
    pythons: List[PythonEnv] = field(default_factory=list)
    ffmpeg: Optional[FfmpegInfo] = None
    driver_cuda: Optional[Tuple[int, int]] = None
    wanted_torch_flavor: str = "cpu"
    ignored: Dict[str, str] = field(default_factory=dict)       # package -> reason code (e.g. unsloth)

    def counts(self) -> Dict[str, int]:
        """Number of decisions per action (reuse / upgrade / offer_upgrade / install)."""
        c = {ACTION_REUSE: 0, ACTION_UPGRADE: 0, ACTION_OFFER: 0, ACTION_INSTALL: 0}
        for d in self.decisions:
            if d.action in c:
                c[d.action] += 1
        return c

    def by_name(self, name: str) -> Optional[Decision]:
        """The decision for package ``name``, or None."""
        return next((d for d in self.decisions if d.name == name), None)


def probe_environment(manifest_pins: Optional[Dict[str, str]] = None,
                      installed_fn: Optional[Callable[[str], Optional[str]]] = None,
                      run: Runner = _run, which: Callable[[str], Optional[str]] = shutil.which,
                      scan_other_pythons: bool = True, packages: Optional[Dict[str, str]] = None,
                      environ: Optional[Dict[str, str]] = None,
                      external_env: Optional[bool] = None) -> EnvReport:
    """Everything read-only.  ``manifest_pins``: verified versions (None -> bundled manifest).
    ``external_env``: the running environment is the user's own (None -> detect): outdated components then get
    ``offer_upgrade`` instead of ``upgrade``."""
    if external_env is None:
        external_env = not is_own_environment()

    def _installed_here(n: str) -> Optional[str]:
        try:
            return metadata.version(n)
        except metadata.PackageNotFoundError:
            return None

    version_of = installed_fn if installed_fn is not None else _installed_here
    if manifest_pins is None:
        try:
            from infra.verified_manifest import load_bundled

            manifest_pins = dict(load_bundled().packages)
        except Exception:  # noqa: BLE001
            manifest_pins = {}
    rep = EnvReport()
    tracked = packages if packages is not None else TRACKED_PACKAGES
    from infra.version_manager import OPTIONAL_PACKAGES

    for name, constraint in tracked.items():
        inst = version_of(name)
        pin = manifest_pins.get(name)
        if pin is None and inst is None and name in OPTIONAL_PACKAGES:
            continue                                   # optional and absent: nothing to do
        d = decide_package(name, inst, pin, constraint, external_env)
        if d.action != ACTION_IGNORE:
            rep.decisions.append(d)
    rep.driver_cuda = detect_driver_cuda(run, which)
    rep.wanted_torch_flavor = torch_flavor_for_driver(rep.driver_cuda)
    rep.decisions.append(decide_torch(version_of("torch"), rep.wanted_torch_flavor, external_env))
    for name, reason in IGNORED_PACKAGES.items():
        if version_of(name):
            rep.ignored[name] = reason
    rep.ffmpeg = probe_ffmpeg(which, run)
    if scan_other_pythons:
        for exe, src in candidate_pythons(which, environ):
            pe = probe_python(exe, src, run)
            if pe:
                rep.pythons.append(pe)
                for name, reason in IGNORED_PACKAGES.items():
                    if name in pe.packages:
                        rep.ignored.setdefault(name, reason)
    log.info("environment: %s; torch wanted %s (driver CUDA %s); ffmpeg %s; other pythons %d; ignored %s",
             rep.counts(), rep.wanted_torch_flavor, rep.driver_cuda,
             (rep.ffmpeg.version if rep.ffmpeg and rep.ffmpeg.ok else "none"), len(rep.pythons), rep.ignored or "-")
    return rep


def user_messages(rep: EnvReport) -> List[str]:
    """Localized one-liners for the progress/log area (keys env.*)."""
    from core.i18n import tr

    c = rep.counts()
    out = [tr("env.summary", reuse=c[ACTION_REUSE], upgrade=c[ACTION_UPGRADE], offer=c[ACTION_OFFER],
              install=c[ACTION_INSTALL])]
    for d in rep.decisions:
        if d.action == ACTION_OFFER:
            out.append(tr("env.offer", name=d.name, old=d.installed or "-", new=d.target or "-"))
        elif d.action == ACTION_UPGRADE:
            out.append(tr("env.upgrade", name=d.name, old=d.installed or "-", new=d.target or "-"))
        elif d.action == ACTION_INSTALL:
            out.append(tr("env.install", name=d.name, new=d.target or "-"))
    if rep.ffmpeg and rep.ffmpeg.ok:
        out.append(tr("env.ffmpeg_reused", version=rep.ffmpeg.version or "?"))
    for name in sorted(rep.ignored):
        out.append(tr("env.ignored", name=name))
    return out
