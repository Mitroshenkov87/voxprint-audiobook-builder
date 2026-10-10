"""Backup and restore of the models, the ffmpeg tool and (optionally) the voice library to any folder or drive.

What goes into a backup (:func:`collect_items`)
-----------------------------------------------
* ``models/<Owner>--<Name>/``  every complete model the app uses: Voxprint's own models folder (TTS base, forced aligner,
  clean-up models ...) and, for the repositories the caller names, complete copies from the Hugging Face cache of other
  programs (symlinks are followed, so the backup holds real files; the revision is stored in a ``.revision`` file);
* ``tools/``  the pinned ffmpeg build, if it was downloaded;
* ``voices/<id>/``  the voice library (only if asked: ``include_voices``).

Layout on the target: ``<chosen folder>/Voxprint-backup/`` with ``voxprint-backup.json`` (the manifest: relative path, size,
SHA-256 and modification time of every file, one entry per item, ``complete`` once all files of the item are in place)
and the folders above.  The backup is a plain copy: it can be browsed, zipped or restored by hand.

Properties
----------
* **Resumable**: every file is written as ``<name>.part`` and renamed when complete; finished files are skipped on the
  next run, so Cancel / a crash / pulling the cable loses at most the file that was being written.
* **Skips identical files**: same size and the same SHA-256 (a preserved modification time is not enough).
  ``verify=True`` still reads the copy back after writing.
* **Clear free-space check before anything is written** (:func:`check_space`): only bytes that really need copying count.
* **Hash-verified**: the SHA-256 is computed while the source is read and stored in the manifest; restoring and importing
  compare the copy with it (read-back), a mismatch removes the copy and raises ``BackupError(code="hash")``.
* Restoring never overwrites a voice that exists with different content (it is reported as a conflict); a model folder that
  differs is replaced only after the new copy is complete and verified.

Everything that touches the disk or the clock is injectable so the tests run without large files.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from core.errors import BackupError
from core.events import CancelToken
from core.i18n import tr

log = logging.getLogger("voxprint.backup")

MANIFEST_NAME = "voxprint-backup.json"
BACKUP_DIRNAME = "Voxprint-backup"
SCHEMA = 1
CHUNK = 4 * 1024 * 1024
MTIME_TOLERANCE = 2.0                       # FAT / exFAT keep modification times with 2 s resolution
SPACE_MARGIN = 64 * 1024 * 1024             # free space we insist on besides the bytes to copy
SKIP_SUFFIXES = (".part", ".partial", ".incomplete", ".lock", ".tmp")
KIND_MODEL, KIND_TOOLS, KIND_VOICES = "model", "tools", "voices"
#: Trees under the models folder that are not ``Owner--Name`` Hugging Face snapshots: the Gemma GGUF and the
#: unpacked llama.cpp runtime live in ``llm/``, DeepFilterNet in ``deepfilternet/``, DNSMOS in ``dnsmos/``.
HEAVY_DIRS = ("llm", "deepfilternet", "dnsmos")

#: progress(fraction 0..1, message)
Progress = Callable[[float, str], None]


def _noop(_f: float, _m: str = "") -> None:   # pragma: no cover - default
    return None


def format_size(n: float) -> str:
    """Human readable size (binary units, e.g. ``3.4 GB``)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"           # pragma: no cover


# ----------------------------------------------------------------------------------------------------- data
@dataclass
class FileEntry:
    """One file of an item: path relative to the item folder, size, SHA-256 (``""`` = not known yet), mtime."""
    p: str
    size: int
    sha256: str = ""
    mtime: float = 0.0


@dataclass
class Item:
    """A folder that goes into the backup (a model, the tools folder, a voice)."""
    kind: str
    name: str                    # folder name below the kind's root, e.g. ``Qwen--Qwen3-TTS-12Hz-1.7B-Base``
    src: Path
    files: List[FileEntry] = field(default_factory=list)
    revision: str = ""
    complete: bool = False

    @property
    def rel(self) -> str:
        """Folder of the item inside the backup (posix style)."""
        if self.kind == KIND_TOOLS:
            return "tools"
        return f"{'models' if self.kind == KIND_MODEL else 'voices'}/{self.name}"

    @property
    def size(self) -> int:
        """Total size of the item's files."""
        return sum(f.size for f in self.files)


@dataclass
class Plan:
    """What a run would do: files to copy (``todo``) and files that are already in place (``skip``)."""
    items: List[Item]
    todo: List[Tuple[Item, FileEntry]] = field(default_factory=list)
    skip: List[Tuple[Item, FileEntry]] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        """Size of everything the backup contains."""
        return sum(i.size for i in self.items)

    @property
    def bytes_to_copy(self) -> int:
        """Bytes that really have to be written."""
        return sum(f.size for _i, f in self.todo)


@dataclass
class Report:
    """Result of a backup / restore run."""
    target: Path
    copied_files: int = 0
    skipped_files: int = 0
    copied_bytes: int = 0
    items: int = 0
    conflicts: List[str] = field(default_factory=list)      # voices that exist with other content and were left alone
    problems: List[str] = field(default_factory=list)       # relative paths that were corrupt or missing (not installed)
    external: str = ""                                      # set when models are used in place instead of copied
    seconds: float = 0.0


# ----------------------------------------------------------------------------------------------------- hashing / copying
def sha256_file(p: Path, cancel: Optional[CancelToken] = None, on_bytes: Optional[Callable[[int], None]] = None) -> str:
    """SHA-256 of a file (read in chunks; ``on_bytes`` gets the chunk sizes, ``cancel`` is checked per chunk)."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            if cancel:
                cancel.check()
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
            if on_bytes:
                on_bytes(len(b))
    return h.hexdigest()


def copy_file(src: Path, dst: Path, cancel: Optional[CancelToken] = None, on_bytes: Optional[Callable[[int], None]] = None,
              verify: bool = False) -> str:
    """Copy ``src`` to ``dst`` through ``dst.part`` and return the SHA-256 of the data read from ``src``.

    The modification time is preserved (it is part of the "identical" test).  With ``verify`` the written file is read
    back and compared (``BackupError(code="hash")`` on mismatch; the copy is removed)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    h = hashlib.sha256()
    try:
        with open(src, "rb") as fi, open(part, "wb") as fo:
            while True:
                if cancel:
                    cancel.check()
                b = fi.read(CHUNK)
                if not b:
                    break
                h.update(b)
                fo.write(b)
                if on_bytes:
                    on_bytes(len(b))
            fo.flush()
            try:
                os.fsync(fo.fileno())
            except OSError:        # some network / removable file systems refuse it
                pass
        try:
            st = os.stat(src)
            os.utime(part, ns=(st.st_atime_ns, st.st_mtime_ns))
        except OSError:
            pass
        digest = h.hexdigest()
        if verify and sha256_file(part, cancel) != digest:
            raise BackupError(tr("backup.err_hash", name=dst.name), code="hash", details=str(dst))
        os.replace(part, dst)
        return digest
    except BaseException:
        # a cancelled / failed file leaves no ".part" behind that could be mistaken for data
        try:
            part.unlink()
        except OSError:
            pass
        raise


def _mtime(p: Path) -> float:
    return os.stat(p).st_mtime


def _is_identical(src_size: int, _src_mtime: float, src_sha: Callable[[], str], dst: Path, _verify: bool) -> bool:
    """Same size and the same SHA-256. Timestamp and ``verify`` do not skip the hash."""
    try:
        st = os.stat(dst)
    except OSError:
        return False
    if st.st_size != src_size:
        return False
    try:
        return sha256_file(dst) == src_sha()
    except OSError:
        return False


# ----------------------------------------------------------------------------------------------------- collecting
def list_files(root: Path) -> List[FileEntry]:
    """All regular files below ``root`` (symlinks followed, temporary download leftovers skipped), sorted."""
    out: List[FileEntry] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames.sort()
        for fn in sorted(filenames):
            if fn.endswith(SKIP_SUFFIXES):
                continue
            full = Path(dirpath) / fn
            try:
                st = os.stat(full)
            except OSError:
                continue            # dangling link
            if not full.is_file():
                continue
            rel = full.relative_to(root).as_posix()
            out.append(FileEntry(rel, st.st_size, "", st.st_mtime))
    return out


def _is_model_dir(p: Path) -> bool:
    return p.is_dir() and (p / "config.json").exists() and (any(p.glob("*.safetensors")) or any(p.glob("*.bin")))


def _heavy_dir_names(models_root: Path) -> List[str]:
    """Folder names of the small / non-snapshot models (``llm``, ``deepfilternet``, ``dnsmos``, a loose ``llama.cpp*``)."""
    names = list(HEAVY_DIRS)
    try:
        for d in models_root.iterdir():
            if d.is_dir() and (d.name.startswith("llama.cpp") or d.name.startswith("llama-cpp")):
                names.append(d.name)
    except OSError:
        pass
    return names


def collect_items(include_voices: bool = True, repos: Iterable[str] = (), models_root: Optional[Path] = None,
                  voices_root: Optional[Path] = None, tools_root: Optional[Path] = None,
                  locate: Optional[Callable[[str], Optional[Tuple[Path, str]]]] = None,
                  include_models: bool = True) -> List[Item]:
    """The folders a backup contains.

    ``repos`` are model repositories the app uses; one that is not in Voxprint's own models folder is taken from the
    place ``locate(repo) -> (folder, revision)`` finds (default: the model locator, i.e. caches of other programs).
    Every complete ``Owner--Name`` folder in the own models folder is included whether it is listed or not."""
    from infra import assets, paths

    models_root = models_root or paths.models_dir()
    voices_root = voices_root or paths.voices_dir()
    items: List[Item] = []
    seen = set()
    try:
        own = sorted(d for d in models_root.iterdir() if d.is_dir() and "--" in d.name
                     and not d.name.endswith((".partial", ".importing", ".restoring", ".old")))
    except OSError:
        own = []
    if not include_models:
        own = []
    for d in own:
        if _is_model_dir(d):
            rev = ""
            try:
                rev = (d / ".revision").read_text(encoding="utf-8").strip()
            except OSError:
                pass
            items.append(Item(KIND_MODEL, d.name, d, revision=rev))
            seen.add(d.name)
    if include_models:
        for repo in repos:
            name = repo.replace("/", "--")
            if name in seen:
                continue
            found = (locate or _default_locate)(repo)
            if found and _is_model_dir(found[0]):
                items.append(Item(KIND_MODEL, name, found[0], revision=found[1] or ""))
                seen.add(name)
    if include_models:
        for name in _heavy_dir_names(models_root):
            if name in seen:
                continue
            d = models_root / name
            try:
                if d.is_dir():
                    items.append(Item(KIND_MODEL, name, d))
                    seen.add(name)
            except OSError:
                pass
    tools = tools_root or assets.tools_dir()
    try:
        if tools.is_dir() and any(tools.iterdir()):
            items.append(Item(KIND_TOOLS, "tools", tools))
    except OSError:
        pass
    if include_voices:
        try:
            for d in sorted(x for x in voices_root.iterdir() if x.is_dir() and not x.name.startswith(".")):
                items.append(Item(KIND_VOICES, d.name, d))
        except OSError:
            pass
    for it in items:
        it.files = list_files(it.src)
        if it.kind == KIND_MODEL and it.revision and not any(f.p == ".revision" for f in it.files):
            it.files.append(FileEntry(".revision", len(it.revision.encode("utf-8")), hashlib.sha256(it.revision.encode("utf-8")).hexdigest(), 0.0))
    return [i for i in items if i.files]


def _default_locate(repo: str) -> Optional[Tuple[Path, str]]:
    """Complete copy of ``repo`` in another program's cache (read-only), with its revision."""
    from infra import model_downloader as md

    f = md.external_model(repo)
    return (f.path, f.revision or "") if f is not None else None


# ----------------------------------------------------------------------------------------------------- manifest
def backup_root(target: Path) -> Path:
    """``<target>/Voxprint-backup`` (``target`` itself if it already is that folder or holds a manifest)."""
    target = Path(target)
    if target.name == BACKUP_DIRNAME or (target / MANIFEST_NAME).is_file():
        return target
    return target / BACKUP_DIRNAME


def find_backup(folder: Path) -> Optional[Path]:
    """The backup folder (the one with the manifest) for a user-chosen folder, or ``None``."""
    for cand in (Path(folder), Path(folder) / BACKUP_DIRNAME):
        if (cand / MANIFEST_NAME).is_file():
            return cand
    return None


def read_manifest(root: Path) -> dict:
    """Parse ``voxprint-backup.json``; ``BackupError(code="no_manifest")`` if it is missing or unreadable."""
    p = Path(root) / MANIFEST_NAME
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            raise ValueError("bad manifest")
        return data
    except (OSError, ValueError) as exc:
        raise BackupError(tr("backup.err_no_manifest", path=str(root)), code="no_manifest", details=str(exc)) from exc


def app_version_build() -> Tuple[str, int]:
    """``("0.1.1-beta", 665)`` from :mod:`core.appinfo` (build ``0`` on a developer checkout)."""
    from core import appinfo

    version = appinfo.release_version()
    return version, int(appinfo.APP_BUILD)


def write_manifest(root: Path, items: List[Item]) -> None:
    """Atomically write the manifest of the given items.

    ``files`` is the flat list (path relative to the backup root, size, SHA-256). ``items`` keeps the folder grouping
    older readers use. ``version`` / ``build`` are the app that wrote the backup."""
    version, build = app_version_build()
    data = {"schema": SCHEMA, "app": "Voxprint", "version": version, "build": build,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": [{"path": f"{i.rel}/{f.p}", "size": f.size, "sha256": f.sha256} for i in items for f in i.files],
            "items": [{"kind": i.kind, "name": i.name, "rel": i.rel, "revision": i.revision, "complete": i.complete,
                       "files": [{"p": f.p, "size": f.size, "sha256": f.sha256, "mtime": f.mtime} for f in i.files]}
                      for i in items]}
    tmp = Path(root) / (MANIFEST_NAME + ".part")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, Path(root) / MANIFEST_NAME)


def items_from_manifest(data: dict) -> List[Item]:
    """:class:`Item` objects of a parsed manifest (``src`` is filled in by the caller)."""
    out = []
    for d in data.get("items", []):
        out.append(Item(str(d.get("kind", "")), str(d.get("name", "")), Path("."),
                        [FileEntry(str(f["p"]), int(f["size"]), str(f.get("sha256", "")), float(f.get("mtime", 0.0)))
                         for f in d.get("files", [])], str(d.get("revision", "")), bool(d.get("complete"))))
    return out


def _safe_rel(p: str) -> bool:
    """A manifest path must stay inside its item folder (no absolute paths, no ``..``)."""
    parts = Path(p.replace("\\", "/")).parts
    return bool(parts) and not Path(p).is_absolute() and ".." not in parts and not p.startswith(("/", "\\"))


def _contained(root: Path, rel: str) -> Optional[Path]:
    """``root/rel`` when ``rel`` cannot leave the backup folder, else ``None`` (a manifest is untrusted input)."""
    if not _safe_rel(rel):
        return None
    path = root.joinpath(*rel.replace("\\", "/").split("/"))
    try:
        path.resolve().relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return None
    return path


# ----------------------------------------------------------------------------------------------------- space
DiskUsage = Callable[[Path], "shutil._ntuple_diskusage"]


def free_bytes(path: Path, usage: Optional[Callable[[Path], object]] = None) -> int:
    """Free bytes on the drive that holds ``path`` (the nearest existing parent is used)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return int((usage or shutil.disk_usage)(p).free)


def check_space(need: int, target: Path, usage: Optional[Callable[[Path], object]] = None) -> int:
    """Raise ``BackupError(code="space")`` with a clear message if ``need`` bytes (+ a margin) do not fit; returns the free bytes."""
    free = free_bytes(target, usage)
    if free < need + SPACE_MARGIN:
        raise BackupError(tr("backup.err_space", need=format_size(need + SPACE_MARGIN), free=format_size(free),
                             drive=str(Path(target).anchor or target)), code="space", details=f"need={need} free={free}")
    return free


# ----------------------------------------------------------------------------------------------------- backup
def plan_backup(items: List[Item], target: Path, verify: bool = False, cancel: Optional[CancelToken] = None,
                progress: Progress = _noop) -> Plan:
    """Decide per file whether it is already in the backup (and fill in known hashes from the previous manifest)."""
    root = backup_root(target)
    prev: Dict[str, FileEntry] = {}
    try:
        for it in items_from_manifest(read_manifest(root)):
            for f in it.files:
                prev[f"{it.rel}/{f.p}"] = f
    except BackupError:
        pass
    plan = Plan(items)
    total = max(1, sum(len(i.files) for i in items))
    n = 0
    for it in items:
        for f in it.files:
            n += 1
            if cancel:
                cancel.check()
            progress(n / total * 0.05, tr("backup.scanning"))
            dst = root / it.rel / f.p
            if f.p == ".revision" and not (it.src / f.p).exists():       # synthetic entry: always rewritten (tiny)
                plan.todo.append((it, f))
                continue
            src = it.src / f.p
            old = prev.get(f"{it.rel}/{f.p}")
            cached = {"sha": old.sha256 if old and old.size == f.size and abs(old.mtime - f.mtime) <= MTIME_TOLERANCE else ""}

            def src_sha(src=src, cached=cached) -> str:
                if not cached["sha"]:
                    cached["sha"] = sha256_file(src, cancel)
                return cached["sha"]

            if _is_identical(f.size, f.mtime, src_sha, dst, verify):
                if old and old.sha256 and old.size == f.size and abs(old.mtime - f.mtime) <= MTIME_TOLERANCE:
                    f.sha256 = old.sha256
                plan.skip.append((it, f))
            else:
                plan.todo.append((it, f))
    return plan


def run_backup(items: List[Item], target: Path, progress: Progress = _noop, cancel: Optional[CancelToken] = None,
               verify: bool = False, usage: Optional[Callable[[Path], object]] = None,
               plan: Optional[Plan] = None) -> Report:
    """Copy ``items`` to ``<target>/Voxprint-backup``.  Resumable; see the module docstring.

    Raises ``BackupError(code="space")`` before writing anything if the drive is too small."""
    t0 = time.time()
    cancel = cancel or CancelToken()
    root = backup_root(target)
    root.mkdir(parents=True, exist_ok=True)
    plan = plan or plan_backup(items, target, verify, cancel, progress)
    check_space(plan.bytes_to_copy, root, usage)
    report = Report(root, items=len(items), skipped_files=len(plan.skip))
    total = max(1, plan.bytes_to_copy)
    done = [0]
    todo_by_item: Dict[int, List[FileEntry]] = {}
    for it, f in plan.todo:
        todo_by_item.setdefault(id(it), []).append(f)
    previous: Dict[Tuple[str, str], Item] = {}
    try:
        previous = {(i.kind, i.name): i for i in items_from_manifest(read_manifest(root)) if i.complete}
    except BackupError:
        pass

    def save_manifest() -> None:
        """Completed items of this run + everything the earlier manifest already vouched for (an interrupted run
        never makes an existing backup look smaller)."""
        mine = {(i.kind, i.name): i for i in items}
        merged = [i for i in items if i.complete]
        merged += [old for key, old in previous.items() if key not in mine or not mine[key].complete]
        write_manifest(root, merged)

    for it in items:
        it.complete = False
    for it in items:
        for f in todo_by_item.get(id(it), []):
            dst = root / it.rel / f.p
            msg = tr("backup.copying", name=it.name, done=format_size(done[0]), total=format_size(total))
            if f.p == ".revision" and not (it.src / f.p).exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(it.revision, encoding="utf-8")
                f.sha256 = hashlib.sha256(it.revision.encode("utf-8")).hexdigest()
                continue
            progress(0.05 + 0.95 * done[0] / total, msg)

            def on_bytes(n: int, it=it) -> None:
                done[0] += n
                progress(0.05 + 0.95 * min(1.0, done[0] / total),
                         tr("backup.copying", name=it.name, done=format_size(done[0]), total=format_size(total)))

            try:
                f.sha256 = copy_file(it.src / f.p, dst, cancel, on_bytes, verify=verify)
            except OSError as exc:
                raise BackupError(tr("backup.err_io", path=str(dst), error=str(exc)), code="io", details=repr(exc)) from exc
            report.copied_files += 1
            report.copied_bytes += f.size
        # files that were skipped still need a hash for the manifest
        for _it, f in plan.skip:
            if _it is it and not f.sha256:
                try:
                    f.sha256 = sha256_file(root / it.rel / f.p, cancel)
                except OSError as exc:
                    raise BackupError(tr("backup.err_io", path=str(root / it.rel / f.p), error=str(exc)), code="io") from exc
        it.complete = True
        save_manifest()
    progress(1.0, tr("backup.done_msg"))
    report.seconds = time.time() - t0
    return report


# ----------------------------------------------------------------------------------------------------- restore
def _restore_dest(it: Item, models_root: Path, voices_root: Path, tools_root: Path) -> Path:
    if it.kind == KIND_MODEL:
        return models_root / it.name
    if it.kind == KIND_VOICES:
        return voices_root / it.name
    return tools_root


def _file_ok(p: Path, f: FileEntry, cancel: Optional[CancelToken] = None) -> bool:
    """``p`` has the manifest's size and (same modification time, or the manifest's SHA-256)."""
    try:
        st = os.stat(p)
    except OSError:
        return False
    if st.st_size != f.size:
        return False
    if f.mtime and abs(st.st_mtime - f.mtime) <= MTIME_TOLERANCE:
        return True
    try:
        return bool(f.sha256) and sha256_file(p, cancel) == f.sha256
    except OSError:
        return False


def _dir_matches(it: Item, dest: Path, cancel: Optional[CancelToken] = None) -> bool:
    """Every file of the item is in ``dest`` as the manifest describes it."""
    return all(_file_ok(dest / f.p, f, cancel) for f in it.files)


@dataclass
class RestorePlan:
    """What :func:`run_restore` would do."""
    root: Path
    todo: List[Item] = field(default_factory=list)          # items with files to copy
    conflicts: List[str] = field(default_factory=list)      # voices that exist with other content (left alone)
    up_to_date: int = 0
    bytes_to_copy: int = 0


def plan_restore(folder: Path, include_voices: bool = True, models_root: Optional[Path] = None,
                 voices_root: Optional[Path] = None, tools_root: Optional[Path] = None,
                 cancel: Optional[CancelToken] = None, kinds: Optional[Iterable[str]] = None) -> RestorePlan:
    """Read the backup in ``folder`` and decide what has to be copied.  ``BackupError(code="no_manifest")`` if it is no backup."""
    from infra import assets, paths

    root = find_backup(folder)
    if root is None:
        raise BackupError(tr("backup.err_no_manifest", path=str(folder)), code="no_manifest")
    models_root = models_root or paths.models_dir()
    voices_root = voices_root or paths.voices_dir()
    tools_root = tools_root or assets.tools_dir()
    allowed = set(kinds) if kinds is not None else {KIND_MODEL, KIND_TOOLS, KIND_VOICES}
    items = [i for i in items_from_manifest(read_manifest(root)) if i.complete and i.kind in allowed]
    if not include_voices:
        items = [i for i in items if i.kind != KIND_VOICES]
    plan = RestorePlan(root)
    for i in items:
        i.src = root / i.rel
        if not _safe_rel(i.name) or any(not _safe_rel(f.p) for f in i.files):
            raise BackupError(tr("backup.err_no_manifest", path=str(root)), code="no_manifest", details=f"unsafe path in {i.name}")
        dest = _restore_dest(i, models_root, voices_root, tools_root)
        if i.kind == KIND_TOOLS:
            todo_files = [f for f in i.files if not _file_ok(dest / f.p, f, cancel)]
            if not todo_files:
                plan.up_to_date += 1
                continue
            plan.bytes_to_copy += sum(f.size for f in todo_files)
        else:
            if dest.is_dir() and _dir_matches(i, dest, cancel):
                plan.up_to_date += 1
                continue
            if i.kind == KIND_VOICES and dest.exists():
                plan.conflicts.append(i.name)
                continue
            plan.bytes_to_copy += i.size
        plan.todo.append(i)
    return plan


def run_restore(folder: Path, include_voices: bool = True, progress: Progress = _noop, cancel: Optional[CancelToken] = None,
                models_root: Optional[Path] = None, voices_root: Optional[Path] = None, tools_root: Optional[Path] = None,
                usage: Optional[Callable[[Path], object]] = None, plan: Optional[RestorePlan] = None,
                strict: bool = True, pins: Optional[Dict[str, Tuple[int, str]]] = None,
                kinds: Optional[Iterable[str]] = None) -> Report:
    """Copy the backup in ``folder`` back into the app's folders (hash-verified, resumable, nothing valuable is overwritten)."""
    from infra import assets, paths

    t0 = time.time()
    cancel = cancel or CancelToken()
    models_root = models_root or paths.models_dir()
    voices_root = voices_root or paths.voices_dir()
    tools_root = tools_root or assets.tools_dir()
    pins = default_pins() if pins is None else pins
    plan = plan or plan_restore(folder, include_voices, models_root, voices_root, tools_root, cancel, kinds)
    check_space(plan.bytes_to_copy, models_root, usage)
    report = Report(plan.root, items=len(plan.todo), conflicts=list(plan.conflicts))
    total = max(1, plan.bytes_to_copy)
    done = [0]
    for it in plan.todo:
        dest = _restore_dest(it, models_root, voices_root, tools_root)

        def on_bytes(n: int, it=it) -> None:
            done[0] += n
            progress(min(1.0, done[0] / total), tr("backup.restoring", name=it.name, done=format_size(done[0]), total=format_size(total)))

        if it.kind == KIND_TOOLS:
            try:
                _copy_item(it, dest, cancel, on_bytes, report, pins)
            except BackupError as exc:
                if strict or exc.code not in ("hash", "missing"):
                    raise
                report.problems.append(exc.details or it.name)
            continue
        stage = dest.with_name(dest.name + ".restoring")         # a half-restored folder is never mistaken for a model / voice
        try:
            _copy_item(it, stage, cancel, on_bytes, report, pins)
            old = dest.with_name(dest.name + ".old")
            if dest.exists():
                shutil.rmtree(old, ignore_errors=True)
                os.replace(dest, old)
            os.replace(stage, dest)
            shutil.rmtree(old, ignore_errors=True)
        except BackupError as exc:
            shutil.rmtree(stage, ignore_errors=True)
            if strict or exc.code not in ("hash", "missing"):
                raise
            report.problems.append(exc.details or it.name)
    progress(1.0, tr("backup.restored_msg"))
    report.seconds = time.time() - t0
    return report


def _copy_item(it: Item, dest: Path, cancel: CancelToken, on_bytes: Callable[[int], None], report: Report,
               pins: Optional[Dict[str, Tuple[int, str]]] = None) -> None:
    """Copy all files of ``it`` from the backup to ``dest``, verifying each against the manifest hash and any pin."""
    pins = pins or {}
    for f in it.files:
        if cancel:
            cancel.check()
        rel = f"{it.rel}/{f.p}"
        src, dst = it.src / f.p, dest / f.p
        if not src.is_file():
            raise BackupError(tr("backup.err_missing", name=f.p), code="missing", details=rel)
        if _pinned_bad(rel, f.size, f.sha256, pins):
            raise BackupError(tr("backup.err_hash", name=f.p), code="hash", details=rel)
        if dst.is_file() and dst.stat().st_size == f.size and f.sha256 and sha256_file(dst, cancel) == f.sha256:
            report.skipped_files += 1                    # copied before an interruption (or already identical)
            on_bytes(f.size)
            continue
        try:
            digest = copy_file(src, dst, cancel, on_bytes, verify=True)
        except OSError as exc:
            raise BackupError(tr("backup.err_io", path=str(src), error=str(exc)), code="io", details=repr(exc)) from exc
        if f.sha256 and digest != f.sha256:
            try:
                dst.unlink()
            except OSError:
                pass
            raise BackupError(tr("backup.err_hash", name=f.p), code="hash", details=rel)
        if _pinned_bad(rel, f.size, digest, pins):
            try:
                dst.unlink()
            except OSError:
                pass
            raise BackupError(tr("backup.err_hash", name=f.p), code="hash", details=rel)
        report.copied_files += 1
        report.copied_bytes += f.size


def _pinned_bad(rel: str, size: int, digest: str, pins: Dict[str, Tuple[int, str]]) -> bool:
    """True when ``rel`` is a pinned file and the size or SHA-256 is not the pinned one."""
    pin = pins.get(rel)
    if not pin:
        return False
    want_size, want_sha = pin
    if size != want_size:
        return True
    return bool(want_sha) and digest != want_sha


def default_pins() -> Dict[str, Tuple[int, str]]:
    """``{path inside the backup: (size, sha256)}`` for files with a published pin.

    Paths look like ``models/Qwen--Qwen3-TTS-.../model.safetensors``. An empty hash means "size only"
    (a ``KNOWN_SIZES`` entry that the mirror manifest does not list). Card files are skipped."""
    from core.model_locator import KNOWN_SIZES
    from infra import model_mirrors
    from infra.denoise_tool import ASSETS as DENOISE_ASSETS
    from infra.llm_tool import MODEL as LLM_MODEL
    from infra.quality_models import DNSMOS_FILE, DNSMOS_META

    out: Dict[str, Tuple[int, str]] = {}
    for repo, entry in model_mirrors.load().items():
        folder = "models/" + repo.replace("/", "--")
        for rel, meta in entry.files.items():
            if meta.get("card"):
                continue
            sha = str(meta.get("sha256") or "")
            if len(sha) == 64:
                out[f"{folder}/{rel}"] = (int(meta["size"]), sha)
    for repo, (_rev, sizes) in KNOWN_SIZES.items():
        folder = "models/" + repo.replace("/", "--")
        for rel, size in sizes.items():
            out.setdefault(f"{folder}/{rel}", (int(size), ""))
    out[f"models/dnsmos/{DNSMOS_FILE}"] = (int(DNSMOS_META["size"]), str(DNSMOS_META["sha256"]))
    for asset in DENOISE_ASSETS.values():
        out[f"models/deepfilternet/{asset['name']}"] = (int(asset["size"]), str(asset["sha256"]))
    out[f"models/llm/{LLM_MODEL['file']}"] = (int(LLM_MODEL["size"]), str(LLM_MODEL["sha256"]))
    return out


def manifest_index(root: Path) -> Dict[str, Tuple[int, str]]:
    """``{path relative to the backup: (size, sha256)}`` from ``files``, or from ``items`` on an older manifest."""
    data = read_manifest(root)
    out: Dict[str, Tuple[int, str]] = {}
    for f in data.get("files") or []:
        if isinstance(f, dict) and f.get("path") and _safe_rel(str(f["path"])):
            out[str(f["path"])] = (int(f["size"]), str(f.get("sha256") or ""))
    if out:
        return out
    for it in data.get("items") or []:
        base = str(it.get("rel") or "")
        if not _safe_rel(base):
            continue
        for f in it.get("files") or []:
            rel = f"{base}/{f['p']}"
            if _safe_rel(rel) and _safe_rel(str(f.get("p", ""))):
                out[rel] = (int(f["size"]), str(f.get("sha256") or ""))
    return out


def _backup_root_of_file(path: Path) -> Optional[Path]:
    """The backup folder that contains ``path`` (the one with the manifest), or ``None``."""
    p = path if path.is_dir() else path.parent
    for _ in range(8):
        try:
            if (p / MANIFEST_NAME).is_file():
                return p
        except OSError:
            return None
        if p == p.parent:
            break
        p = p.parent
    return None


def linked_model_problem(model_dir: Path) -> Optional[str]:
    """Why ``model_dir`` must not be used in place, or ``None`` when every listed file matches the manifest and the pins."""
    root = _backup_root_of_file(model_dir)
    if root is None:
        return "no manifest"
    try:
        idx = manifest_index(root)
        prefix = model_dir.resolve().relative_to(root.resolve()).as_posix()
    except (BackupError, OSError, ValueError):
        return "manifest"
    listed = [k for k in idx if k == prefix or k.startswith(prefix + "/")]
    if not listed:
        return "not in manifest"
    pins = default_pins()
    for rel in listed:
        size, sha = idx[rel]
        path = _contained(root, rel)
        try:
            if path is None or not path.is_file() or path.stat().st_size != size:
                return rel
            digest = sha256_file(path) if sha or pins.get(rel, (0, ""))[1] else ""
            if sha and digest != sha:
                return rel
        except OSError:
            return rel
        if _pinned_bad(rel, size, digest if sha else (sha256_file(path) if pins.get(rel, (0, ""))[1] else ""), pins):
            return rel
    return None


def _scan_model_problems(root: Path, pins: Dict[str, Tuple[int, str]], cancel: Optional[CancelToken],
                         progress: Progress) -> List[str]:
    """Relative paths under ``models/`` that are missing, the wrong size, or the wrong hash."""
    idx = manifest_index(root)
    rels = sorted(p for p in idx if p.startswith("models/"))
    problems: List[str] = []
    total = max(1, len(rels))
    for n, rel in enumerate(rels):
        if cancel:
            cancel.check()
        progress(min(0.2, n / total * 0.2), tr("backup.scanning"))
        size, sha = idx[rel]
        path = _contained(root, rel)
        try:
            if path is None or not path.is_file() or path.stat().st_size != size:
                problems.append(rel)
                continue
            digest = sha256_file(path, cancel) if (sha or pins.get(rel, (0, ""))[1]) else ""
            if sha and digest != sha:
                problems.append(rel)
                continue
        except OSError:
            problems.append(rel)
            continue
        if _pinned_bad(rel, size, digest, pins):
            problems.append(rel)
    return problems


def run_link(folder: Path, include_voices: bool = True, progress: Progress = _noop, cancel: Optional[CancelToken] = None,
             models_root: Optional[Path] = None, voices_root: Optional[Path] = None, tools_root: Optional[Path] = None,
             usage: Optional[Callable[[Path], object]] = None,
             pins: Optional[Dict[str, Tuple[int, str]]] = None) -> Report:
    """Use the backup's models folder in place (no model copy) and copy voices and tools into the normal folders.

    The chosen path is remembered (:mod:`infra.external_models`). Files that do not match the manifest or a pinned
    hash are listed on :attr:`Report.problems` and are not treated as installed; the caller downloads those."""
    from infra import external_models

    root = find_backup(folder)
    if root is None:
        raise BackupError(tr("backup.err_no_manifest", path=str(folder)), code="no_manifest")
    pins = default_pins() if pins is None else pins
    cancel = cancel or CancelToken()
    problems = _scan_model_problems(root, pins, cancel, progress)
    dest = root / "models" if (root / "models").is_dir() else root
    external_models.set_folder(dest)
    rep = run_restore(folder, include_voices, progress, cancel, models_root, voices_root, tools_root, usage,
                      strict=False, pins=pins, kinds=(KIND_VOICES, KIND_TOOLS))
    rep.problems = list(dict.fromkeys([*problems, *rep.problems]))
    rep.external = str(dest)
    return rep
