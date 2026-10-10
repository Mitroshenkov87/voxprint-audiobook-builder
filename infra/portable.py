"""Portable setup folder: everything the installer downloads, kept in ONE folder for an offline re-install (standard library only).

The wizard option "Keep a portable setup folder" makes the online installer put ALL components (and the models this PC needs)
into a folder the user chose, even those the PC already has; the program is then installed from that folder.  Later the
installer (or ``voxprint-fetch --from-folder``) finds the folder and installs WITHOUT internet; when the internet is there it
compares the new manifest with the one in the folder and fetches only what changed.

Layout::

    Voxprint Portable/
      manifest.json            the manifest the folder was made from (components: shell, wheels, PyTorch of the chosen flavor)
      portable.json            what else is in the folder (app version, flavor, models)
      SHA256SUMS.txt           <sha256> *components/<id>/<file>   and   <sha256> *models/<Owner--Name>/<file>
      components/<id>/<file>   one folder per component
      models/<Owner--Name>/    models exactly as Voxprint's own models folder (+ .revision): usable as an "existing models folder"

Models come from the same sources, in the same order, with the same stall watchdog as in the program
(:func:`infra.model_downloader.ensure_model`): GitHub release (small models) -> original Hugging Face -> our Hugging Face
mirror -> ModelScope (ModelScope first when Hugging Face is slow); every file is checked against the SHA-256 pinned in
``infra/model_mirrors.json``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from infra import download_watch, env_probe, model_mirrors, model_release, modelscope_mirror, vc_model

log = logging.getLogger("voxprint.portable")

SCHEMA = 1
MANIFEST, INDEX, SUMS = "manifest.json", "portable.json", "SHA256SUMS.txt"
DIR_NAME = "Voxprint Portable"
STATE_FILE = "portable_dir.txt"          # <app home>/state/: the folder the installer used (the program installs modules from it)
ENV_DIR = "VOXPRINT_PORTABLE_DIR"

ALIGNER, ASR = "Qwen/Qwen3-ForcedAligner-0.6B", "Qwen/Qwen3-ASR-0.6B"
ASR_LARGE = "Qwen/Qwen3-ASR-1.7B"
#: Qwen3-ASR-1.7B from an "8 GB" card on (nvidia-smi reports ~8188 MB; the program's rule: infra/asr_choice.py), else 0.6B.
ASR_LARGE_MIN_VRAM_MB = 7500
TTS_LARGE, TTS_SMALL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
SAGE = "ai-forever/sage-fredt5-distilled-95m"
#: The 1.7B base model from this much VRAM on (the planner of the program: >= 10 GB), else the 0.6B one.
LARGE_MIN_VRAM_MB = 9500


class PortableError(Exception):
    """A problem with the setup folder; the text is shown to the user."""


# ------------------------------------------------------------------------------------------------ where the folder is
def default_folder() -> Path:
    """``Documents\\Voxprint Portable`` (the installer passes the real, possibly redirected, Documents folder itself)."""
    return Path.home() / "Documents" / DIR_NAME


def state_file() -> Path:
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
    try:
        return json.loads((Path(folder) / MANIFEST).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise PortableError(f"The setup folder has no readable {MANIFEST}: {exc}") from exc


def read_index(folder: Path) -> dict:
    try:
        return json.loads((Path(folder) / INDEX).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {"schema": SCHEMA, "models": {}}


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
    """Newest CUDA flavor the driver supports, else ``cpu`` (the same rule as ``infra.runtime_reuse.choose_flavor``)."""
    forced = os.environ.get("VOXPRINT_TORCH_FLAVOR", "").strip()
    if forced and forced in flavors:
        return forced
    best, best_cu = "cpu", (0, 0)
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
    return not comp.get("flavor") or flavor in ("", "all") or comp["flavor"] == flavor


# ------------------------------------------------------------------------------------------------ components
def component_path(folder: Path, comp: dict) -> Path:
    """``<folder>/components/<id>/<file>`` (same rule as ``tools/online_fetch.portable_file``)."""
    return Path(folder) / "components" / comp["id"] / (comp.get("file") or f"{comp['sha256']}.zip")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


@dataclass
class Report:
    ok: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    damaged: List[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing and not self.damaged

    def text(self) -> str:
        bad = self.missing + self.damaged
        return f"{len(bad)} file(s) missing or damaged: " + ", ".join(bad[:5]) + (" ..." if len(bad) > 5 else "") if bad else "complete"


def verify(folder: Path, flavor: str = "all", deep: bool = False, only: Optional[Iterable[str]] = None) -> Report:
    """Check the components (and models) of a setup folder: present with the right size, with ``deep`` also the SHA-256."""
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
        return bool(self.changed or self.added or self.removed) or self.old_version != self.new_version

    def text(self) -> str:
        n = len(self.changed) + len(self.added)
        if not self.newer:
            return "The setup folder is up to date"
        return (f"A newer version is available ({self.old_version or '?'} -> {self.new_version or '?'}): "
                f"{n} part(s) to fetch ({self.fetch_bytes >> 20} MB)")


def diff(old: dict, new: dict, flavor: str = "all", folder: Optional[Path] = None) -> Diff:
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
    folder, gone = Path(folder), []
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


# ------------------------------------------------------------------------------------------------ models
def model_dir(folder: Path, repo: str) -> Path:
    return Path(folder) / "models" / repo.replace("/", "--")


def models_for(mode: str, vram_mb: int, entries: Optional[Dict[str, Any]] = None) -> List[str]:
    """Models for the setup folder: ``none``; ``auto`` = what this PC needs (aligner, the TTS base model and the speech
    recognition model its VRAM allows, SAGE text clean-up); ``all`` = every model with a pinned file list (a folder for other PCs)."""
    if mode == "none":
        return []
    if mode == "all":
        # The voice converter is optional and lives in models/openvoice-v2 (the Re-voice window downloads it).
        # The generic Owner--Name folder would not be the copy the app loads.
        names = sorted(entries if entries is not None else model_mirrors.load())
        return [r for r in names if r != vc_model.REPO]
    return [ALIGNER, TTS_LARGE if vram_mb >= LARGE_MIN_VRAM_MB else TTS_SMALL,
            ASR_LARGE if vram_mb >= ASR_LARGE_MIN_VRAM_MB else ASR, SAGE]


_EN = {
    "progress.downloading": "{short}: {pct}%",
    "progress.detail": "{short}: {pct}% - {done} of {total} - {speed} - from {source}",
    "progress.detail_unknown": "{short}: {done} downloaded - {speed} - from {source}",
    "progress.eta": " - about {eta} left",
}


def _tr(key: str, **kw: Any) -> str:
    return _EN[key].format(**kw)


def _complete(d: Path, entry: model_mirrors.MirrorEntry) -> bool:
    try:
        if (d / ".revision").read_text(encoding="utf-8").strip() != entry.source_revision:
            return False
        return all((d / Path(*n.split("/"))).stat().st_size == int(m["size"]) for n, m in entry.downloadable().items())
    except OSError:
        return False


def verify_model(d: Path, entry: model_mirrors.MirrorEntry) -> List[str]:
    """Names of files of the model folder that are missing or have a wrong SHA-256 (the bad ones are deleted)."""
    bad = []
    for n, m in entry.downloadable().items():
        p = d / Path(*n.split("/"))
        if not model_release.file_ok(p, m):
            bad.append(n)
            p.unlink(missing_ok=True)
    return bad


def fetch_models(folder: Path, repos: List[str], progress: Callable[[float, str], None] = lambda f, t: None,
                 mirror_manifest: Optional[Path] = None, release_manifest: Optional[Path] = None,
                 opener: Optional[Callable[..., Any]] = None, hf_fast: Optional[bool] = None,
                 stall: Optional[float] = None, poll: Optional[float] = None) -> Dict[str, str]:
    """Download ``repos`` into ``<folder>/models/<Owner--Name>`` (resuming; models already complete are skipped).  Returns
    ``{repo: source}`` (``cached`` when nothing had to be fetched).  Raises :class:`PortableError` if a model cannot be had."""
    folder = Path(folder)
    entries: Dict[str, model_mirrors.MirrorEntry] = {}
    for r in repos:
        e = model_mirrors.load(mirror_manifest).get(r)
        if e is None:
            raise PortableError(f"No pinned file list for {r}; it cannot be put into the setup folder.")
        entries[r] = e
    root = folder / "models"
    root.mkdir(parents=True, exist_ok=True)
    total = sum(int(m["size"]) for e in entries.values() for m in e.downloadable().values())
    free = shutil.disk_usage(root).free
    have = download_watch.dir_bytes(root)
    if free < (total - have) * 1.05:
        raise PortableError(f"Not enough free disk space for the models: about {(total - have) >> 20} MB needed, {free >> 20} MB free.")
    meter = download_watch.Meter("")
    meter.set_total(total)
    used: Dict[str, str] = {}
    for repo, entry in entries.items():
        d = model_dir(folder, repo)
        short = repo.split("/")[-1]
        if _complete(d, entry) and not verify_model(d, entry):
            used[repo] = "cached"
            continue
        rel = model_release.entry_for(repo, entry.source_revision, release_manifest)
        errors: List[str] = []
        order = (["gh"] if rel is not None else [])
        fast = hf_fast if hf_fast is not None else modelscope_mirror.hf_is_fast(repo)
        order += ["hf", "hfm", "ms"] if fast else ["ms", "hf", "hfm"]
        if not entry.has_mirror:                     # hashes-only entry: there is no backup mirror to try
            order = [o for o in order if o != "hfm"]
        names = {"gh": "GitHub", "hf": "Hugging Face", "hfm": "Hugging Face mirror", "ms": "ModelScope"}
        done_src = ""
        for src in order:
            cancel = threading.Event()
            meter.set_source(names[src])

            def ping(n: int = 0, _c=cancel) -> None:
                if _c.is_set():
                    raise download_watch.Stalled("download abandoned (no data)")

            # Loop variables are bound as defaults: an abandoned (stalled) download thread must keep writing into ITS
            # model folder, never into the folder of the model the loop moved on to.
            def one_file(base: str, rev: str, name: str, meta: Dict[str, Any], d: Path = d) -> None:
                url = f"{modelscope_mirror.hf_endpoint()}/{base}/resolve/{rev}/{urllib.parse.quote(name)}"
                t = d / Path(*name.split("/"))
                if not model_release.file_ok(t, meta):
                    t.unlink(missing_ok=True)
                    model_release.fetch_file(url, t, meta, ping, opener or model_release._open, 30.0)

            def fn(src=src, rel=rel, entry=entry, repo=repo, d=d, one_file=one_file) -> None:
                if src == "gh":
                    model_release.download(rel, d, lambda f: ping(), None, opener)
                elif src in ("hf", "hfm"):
                    base, rev = (repo, entry.source_revision) if src == "hf" else (entry.mirror_repo, entry.mirror_revision)
                    for n, m in sorted(entry.downloadable().items()):
                        one_file(base, rev, n, m)
                else:
                    modelscope_mirror.download_repo(repo, d, lambda f: ping())
                bad = verify_model(d, entry)
                if bad:
                    raise PortableError(f"{src}: wrong SHA-256 / missing: {', '.join(bad[:3])}")

            def tick(m: "download_watch.Meter", short=short) -> None:
                pct = int(min(0.99, m.done / total) * 100) if total else 0
                progress(min(0.99, m.done / total) if total else 0.0, m.text(_tr, short, pct))

            try:
                download_watch.run_watched(fn, root, meter, cancel, on_tick=tick, stall=stall, poll=poll,
                                           idle_ok=lambda e=entry, d=d: _complete(d, e))
                done_src = src
                break
            except Exception as exc:  # noqa: BLE001 - the next source takes over, the partial files stay
                errors.append(f"{src}: {type(exc).__name__}: {exc}")
                log.warning("setup folder: %s from %s failed: %s", repo, src, exc)
        if not done_src:
            raise PortableError(f"Could not download {short}: " + " | ".join(errors))
        (d / ".revision").write_text(entry.source_revision, encoding="utf-8")
        used[repo] = done_src
    idx = read_index(folder)
    idx.setdefault("models", {}).update({r: {"revision": entries[r].source_revision, "dir": model_dir(folder, r).name} for r in repos})
    idx["schema"] = SCHEMA
    (folder / INDEX).write_text(json.dumps(idx, indent=1), encoding="utf-8")
    progress(1.0, "Models are in the setup folder")
    return used


def write_sums(folder: Path, man: dict, flavor: str = "all") -> None:
    """``SHA256SUMS.txt`` for the components of the manifest and the models listed in ``portable.json``."""
    folder = Path(folder)
    lines = [f"{c['sha256']} *{component_path(folder, c).relative_to(folder).as_posix()}"
             for c in man["components"] if matches_flavor(c, flavor) and component_path(folder, c).is_file()]
    entries = model_mirrors.load()
    for repo in read_index(folder).get("models", {}):
        e, d = entries.get(repo), model_dir(folder, repo)
        if e is not None:
            lines += [f"{m['sha256']} *models/{d.name}/{n}" for n, m in sorted(e.downloadable().items()) if (d / Path(*n.split('/'))).is_file()]
    (folder / SUMS).write_text("\n".join(lines) + "\n", encoding="ascii")
