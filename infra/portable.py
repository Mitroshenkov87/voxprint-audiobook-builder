"""Portable setup folder: everything the installer downloads, kept in ONE folder for an offline re-install (standard library only).

The wizard option "Keep a portable setup folder" makes the online installer put ALL components into a
folder the user chose, even those the PC already has; the program is then installed from that folder.
Later the installer (or ``voxprint-fetch --from-folder``) finds the folder and installs WITHOUT internet;
when the internet is there it compares the new manifest with the one in the folder and fetches only
what changed.

Models are NOT part of a setup folder any more: VOXPrint MDL owns the shared model store
(:mod:`infra.mdl_models`).  A folder made by an older installer may still carry ``models/<Owner--Name>/``
subfolders; bring them into the store with ``voxprint-mdl repair-store --app audiobook-builder --from <folder>``.

Layout::

    Voxprint Portable/
      manifest.json            the manifest the folder was made from (components: shell, wheels, PyTorch of the chosen flavor)
      SHA256SUMS.txt           <sha256> *components/<id>/<file>
      components/<id>/<file>   one folder per component
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from voxprint_common.hashing import sha256_file as _sha256_file

from infra import env_probe

log = logging.getLogger("voxprint.portable")

MANIFEST, SUMS = "manifest.json", "SHA256SUMS.txt"
DIR_NAME = "Voxprint Portable"
STATE_FILE = "portable_dir.txt"          # <app home>/state/: the folder the installer used (the program installs modules from it)
ENV_DIR = "VOXPRINT_PORTABLE_DIR"


class PortableError(Exception):
    """A problem with the setup folder; the text is shown to the user."""


# ------------------------------------------------------------------------------------------------ where the folder is
def default_folder() -> Path:
    """``Documents\\Voxprint Portable`` (the installer passes the real, possibly redirected, Documents folder itself)."""
    return Path.home() / "Documents" / DIR_NAME


def state_file() -> Path:
    """Path of the small file that remembers the offline setup folder."""
    from infra import paths

    return paths.state_dir() / STATE_FILE


def is_setup_folder(folder: Path) -> bool:
    """True if ``folder`` holds a usable ``manifest.json`` (a setup folder made by the installer)."""
    try:
        d = json.loads((Path(folder) / MANIFEST).read_text(encoding="utf-8-sig"))
        return isinstance(d, dict) and isinstance(d.get("components"), list) and bool(d["components"])
    except (OSError, ValueError):
        return False


def configured_folder() -> Optional[Path]:
    """The setup folder of this PC: ``$VOXPRINT_PORTABLE_DIR`` or the path the installer remembered; None if there is none."""
    env = os.environ.get(ENV_DIR, "").strip().strip('"')
    cands = [env] if env else []
    if not env:
        try:
            cands += [l.strip().strip('"') for l in state_file().read_text(encoding="utf-8-sig").splitlines() if l.strip()]
        except OSError:
            pass
    for c in cands:
        if c and is_setup_folder(Path(c)):
            return Path(c)
    return None


def set_folder(folder: Optional[Path]) -> None:
    """Remember (or forget) the setup folder."""
    f = state_file()
    if folder is None:
        f.unlink(missing_ok=True)
    else:
        f.write_text(str(folder) + "\n", encoding="utf-8")


def find_folder(extra: Iterable[Path] = ()) -> Optional[Path]:
    """First existing setup folder among: ``extra``, the configured one, ``Documents\\Voxprint Portable``, ``Voxprint Portable``
    next to the running program (a folder copied to a USB stick together with the installer)."""
    cands: List[Path] = [Path(p) for p in extra]
    c = configured_folder()
    if c:
        cands.append(c)
    cands.append(default_folder())
    try:
        here = Path(sys.argv[0]).resolve().parent
        cands += [here / DIR_NAME, here]
    except OSError:
        pass
    return next((p for p in cands if is_setup_folder(p)), None)


def read_manifest(folder: Path) -> dict:
    """Read ``manifest.json`` from a setup folder."""
    try:
        return json.loads((Path(folder) / MANIFEST).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise PortableError(f"The setup folder has no readable {MANIFEST}: {exc}") from exc


# ------------------------------------------------------------------------------------------------ this PC
def _run(cmd: List[str], timeout: float = 10.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_cuda(run: Callable[[List[str]], str] = _run) -> Optional[Tuple[int, int]]:
    """CUDA version the NVIDIA driver supports (``nvidia-smi``), or None without an NVIDIA driver.

    Same probe as :func:`infra.env_probe.detect_driver_cuda`: the header, then the driver version
    when the header has no ``CUDA Version`` / ``CUDA UMD Version``.
    """
    def runner(args: List[str]) -> Tuple[int, str]:
        text = run(args)
        return (0, text) if text else (1, "")

    return env_probe.detect_driver_cuda(runner, shutil.which)


def detect_vram_mb(run: Callable[[List[str]], str] = _run) -> int:
    """VRAM of the largest NVIDIA GPU in MB (0 = none)."""
    exe = shutil.which("nvidia-smi")
    vals = [int(x) for x in re.findall(r"^\s*(\d+)\s*$", run([exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"]) if exe else "", re.M)]
    return max(vals) if vals else 0


def choose_flavor(flavors: List[str], cuda: Optional[Tuple[int, int]]) -> str:
    """Newest CUDA flavor the driver supports, else ``""`` (no CPU choice; same rule as ``infra.runtime_reuse.choose_flavor``).

    ``VOXPRINT_TORCH_FLAVOR`` forces one flavor. That variable is internal (CI/tests); ``cpu`` is accepted even when
    it is not a public flavor (CI wheels).
    """
    forced = os.environ.get("VOXPRINT_TORCH_FLAVOR", "").strip()
    if forced == "cpu" or (forced and forced in flavors):
        return forced
    best, best_cu = "", (0, 0)
    for f in flavors:
        m = re.match(r"^cu(\d{2,3})$", f)
        if not m or not cuda:
            continue
        digits = m.group(1)
        cu = (int(digits[:-1]), int(digits[-1]))             # cu128 -> 12.8, cu118 -> 11.8
        if cu <= cuda and cu > best_cu:
            best, best_cu = f, cu
    return best


def flavor_of(man: dict, wanted: str = "auto") -> str:
    """``wanted``: ``auto`` (this PC), ``all`` (every flavor, for another PC) or a flavor name; '' when the manifest has none."""
    flavors = list((man.get("runtime") or {}).get("flavors") or [])
    if not flavors:
        return ""
    if wanted == "all":
        return "all"
    return choose_flavor(flavors, detect_cuda()) if wanted in ("", "auto") else wanted


def matches_flavor(comp: dict, flavor: str) -> bool:
    """A component with no flavor always matches. ``all`` keeps every flavor. ``""`` keeps none of the flavored wheels."""
    if not comp.get("flavor"):
        return True
    if flavor == "all":
        return True
    return bool(flavor) and comp["flavor"] == flavor


# ------------------------------------------------------------------------------------------------ components
def component_path(folder: Path, comp: dict) -> Path:
    """``<folder>/components/<id>/<file>`` (same rule as ``tools/online_fetch.portable_file``)."""
    return Path(folder) / "components" / comp["id"] / (comp.get("file") or f"{comp['sha256']}.zip")


def sha256_file(path: Path) -> str:
    """The SHA-256 hex digest of ``path``."""
    return _sha256_file(path)


@dataclass
class Report:
    """Names that verified, and the names that were missing or the wrong size."""

    ok: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    damaged: List[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """True when nothing was missing or damaged."""
        return not self.missing and not self.damaged

    def text(self) -> str:
        """A short sentence listing up to five missing or damaged files."""
        bad = self.missing + self.damaged
        return f"{len(bad)} file(s) missing or damaged: " + ", ".join(bad[:5]) + (" ..." if len(bad) > 5 else "") if bad else "complete"


def verify(folder: Path, flavor: str = "all", deep: bool = False, only: Optional[Iterable[str]] = None) -> Report:
    """Check the components of a setup folder: present with the right size, with ``deep`` also the SHA-256."""
    folder, rep = Path(folder), Report()
    man = read_manifest(folder)
    want = set(only) if only is not None else None
    for c in man["components"]:
        if (want is not None and c["id"] not in want) or not matches_flavor(c, flavor):
            continue
        p = component_path(folder, c)
        name = p.relative_to(folder).as_posix()
        if not p.is_file():
            rep.missing.append(name)
        elif p.stat().st_size != int(c["size"]) or (deep and sha256_file(p) != c["sha256"]):
            rep.damaged.append(name)
        else:
            rep.ok.append(name)
    return rep


@dataclass
class Diff:
    """What a newer manifest changes compared with the one stored in the setup folder."""
    old_version: str
    new_version: str
    changed: List[str]          # same id, other SHA-256
    added: List[str]
    removed: List[str]
    fetch_bytes: int            # bytes that must be fetched (changed + added, flavor-matching, not already valid in the folder)

    @property
    def newer(self) -> bool:
        """True when the new manifest is not the same set of files as the old one."""
        return bool(self.changed or self.added or self.removed) or self.old_version != self.new_version

    def text(self) -> str:
        """A sentence saying the folder is current, or how much a newer one would fetch."""
        n = len(self.changed) + len(self.added)
        if not self.newer:
            return "The setup folder is up to date"
        return (f"A newer version is available ({self.old_version or '?'} -> {self.new_version or '?'}): "
                f"{n} part(s) to fetch ({self.fetch_bytes >> 20} MB)")


def diff(old: dict, new: dict, flavor: str = "all", folder: Optional[Path] = None) -> Diff:
    """What changed between two setup manifests for ``flavor``."""
    o = {c["id"]: c for c in old.get("components", [])}
    changed, added = [], []
    need = 0
    for c in new.get("components", []):
        if not matches_flavor(c, flavor):
            continue
        prev = o.get(c["id"])
        if prev is None:
            added.append(c["id"])
        elif prev["sha256"] != c["sha256"]:
            changed.append(c["id"])
        else:
            continue
        need += int(c["size"])
    removed = [i for i in o if i not in {c["id"] for c in new.get("components", [])}]
    return Diff(str(old.get("app_version", "")), str(new.get("app_version", "")), changed, added, removed, need)


def prune(folder: Path, man: dict, flavor: str = "all") -> List[str]:
    """Delete component files that the manifest no longer wants (old versions after an update); returns what was removed."""
    folder = Path(folder)
    gone: List[str] = []
    keep = {component_path(folder, c).resolve() for c in man["components"] if matches_flavor(c, flavor)}
    root = folder / "components"
    if not root.is_dir():
        return gone
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.resolve() not in keep and not p.name.endswith(".part"):
            p.unlink(missing_ok=True)
            gone.append(p.relative_to(folder).as_posix())
    for d in sorted(root.glob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    return gone


def write_sums(folder: Path, man: dict, flavor: str = "all") -> None:
    """``SHA256SUMS.txt`` for the components of the manifest."""
    folder = Path(folder)
    lines = [f"{c['sha256']} *{component_path(folder, c).relative_to(folder).as_posix()}"
             for c in man["components"] if matches_flavor(c, flavor) and component_path(folder, c).is_file()]
    (folder / SUMS).write_text("\n".join(lines) + "\n", encoding="ascii")
