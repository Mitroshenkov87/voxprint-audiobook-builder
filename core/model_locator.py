"""Finding Qwen models that other apps have already downloaded (read-only reuse).

Voxprint needs Qwen3-TTS (Base 1.7B / 0.6B) and Qwen3-ForcedAligner-0.6B.  They are several GB each, so before
downloading anything we look for a complete copy that another program left on this computer and use it IN PLACE.
Nothing is ever written, moved, copied or deleted inside another app's folders: this module only stats files and
reads a few header bytes.  Anything unclear (incomplete, unknown revision, other revision than the verified one)
means "not found", and the normal download path runs.

Where such copies live (sources; confidence in brackets)
--------------------------------------------------------
* Standard Hugging Face hub cache (every ``from_pretrained("Qwen/...")`` / ``huggingface-cli download`` without
  ``--local-dir``).  Root = ``$HF_HUB_CACHE`` > ``$HUGGINGFACE_HUB_CACHE`` (deprecated) > ``$HF_HOME/hub`` >
  ``$XDG_CACHE_HOME/huggingface/hub`` > ``~/.cache/huggingface/hub`` (Windows: ``%USERPROFILE%\\.cache\\huggingface\\hub``);
  legacy ``$TRANSFORMERS_CACHE``.  Layout: ``models--Owner--Name/{blobs/, refs/main, snapshots/<commit sha>/...}``;
  snapshot files are symlinks into ``blobs/`` (Windows without developer mode / with ``HF_HUB_DISABLE_SYMLINKS``
  holds real files instead - both are handled).  [high: documented by huggingface_hub]
* Alexandria audiobook app (Finrandojin/alexandria-audiobook, "Finity's Alexandria"), Qwen3-TTS engine.  Its
  ``TTSEngine._load_model`` calls ``huggingface_hub.try_to_load_from_cache(model_id, "config.json")`` and then
  ``Qwen3TTSModel.from_pretrained(local_path or model_id)`` with ``Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice``,
  ``...-1.7B-Base`` and ``...-1.7B-VoiceDesign``; it has no model folder of its own, so the models are in the
  standard HF hub cache above.  [high: verified in its source code]
  - Pinokio install (the route its README recommends): Pinokio's per-app environment sets
    ``HF_HOME=./cache/HF_HOME`` (and ``XDG_CACHE_HOME=./cache/XDG_CACHE_HOME``), so the models are in
    ``<PINOKIO_HOME>/api/alexandria-audiobook.git/cache/HF_HOME/hub`` (or, when that line is absent, in the shared
    ``<PINOKIO_HOME>/cache/HF_HOME/hub``).  ``<PINOKIO_HOME>`` is ``~/pinokio`` (``C:\\Users\\<user>\\pinokio``) by
    default but can be moved.  We scan ``api/*/cache/...`` generically instead of hard-coding the folder name.
    [medium: derived from the Pinokio docs and launcher scripts, not checked on a real installation]
  - Docker install: the volume ``hf_cache:/root/.cache/huggingface`` lives inside Docker - not reachable from the
    host, so it cannot be reused.  Users can still point ``VOXPRINT_MODEL_DIRS`` to a bind mount.  [high]
  Alexandria uses only the TTS models; the ForcedAligner comes from other tools or from our own download.
* Pinokio launchers of other Qwen/TTS apps: same ``cache/HF_HOME/hub`` convention; some set
  ``HF_HUB_CACHE=../cache/HF_HOME/hub`` and ``HF_HUB_DISABLE_SYMLINKS=1`` (=> real files, no symlinks).  [medium]
* ModelScope (``modelscope download`` / ``snapshot_download``): ``$MODELSCOPE_CACHE`` or ``~/.cache/modelscope`` with
  the models under ``hub/models/<Owner>/<Name>`` (older versions: ``hub/<Owner>/<Name>``), dots in the name are
  written as ``___`` (``Qwen3-TTS-12Hz-1___7B-Base``).  ModelScope stores no commit sha, so the revision can only be
  confirmed by file sizes (see below).  [medium: from the ModelScope SDK/CLI documentation]
* Manual downloads following the Qwen READMEs / community guides: ``huggingface-cli download Qwen/<Name>
  --local_dir ./<Name>`` (also ``models/<Name>``, ``models/Qwen/<Name>``).  We look in a few fixed places
  (``~/models``, ``~/Models``, ``./models``, the working directory) for ``<Name>``, ``<Owner>/<Name>``,
  ``<Owner>--<Name>`` - never a full-disk search.  [medium]
* An earlier Voxprint installation (``infra.paths.previous_homes()``: ``~/.voxprint``, ``%APPDATA%\\Voxprint``,
  ``VOXPRINT_PREVIOUS_HOMES``): its ``models/Owner--Name`` folders.  [medium: our own layout]
* ``VOXPRINT_MODEL_DIRS`` (os.pathsep-separated): extra roots, searched first; each may be an HF cache, a ModelScope
  cache or a folder of model folders.  ``VOXPRINT_NO_EXTERNAL_MODELS=1`` switches the whole feature off.  Not a GUI
  setting on purpose.

What counts as a valid copy
---------------------------
``config.json`` (valid JSON); every file that the repository is known to contain (for the three Voxprint models,
including ``speech_tokenizer/`` of the TTS Base models); every ``*.safetensors`` has a consistent header and is not
shorter than its header promises (truncated/incomplete download); with ``model.safetensors.index.json`` every shard
of the ``weight_map`` exists and the sum of sizes covers ``total_size``; symlinks must resolve to existing,
non-empty files; no ``*.incomplete`` / ``*.partial`` leftovers.

Revision rules (``verified_manifest.json`` pins a commit per model)
-------------------------------------------------------------------
* HF cache: the snapshot folder name is the commit sha.  Equal to the pin => reuse; different => not reused
  (WARNING log, the verified revision gets downloaded).  Several snapshots: the pinned one wins, then ``refs/main``.
* Folders without a sha (Alexandria-style plain folders, ModelScope): reused only if the sizes of the weight files
  are identical to the verified revision (:data:`KNOWN_SIZES`, taken from the Hugging Face API for the pinned
  commits); a ``.revision`` file (written by Voxprint itself) counts as a sha.  Otherwise not reused.
* No pin for the model => any valid copy is accepted.
"""
from __future__ import annotations

import json
import logging
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

log = logging.getLogger("voxprint.models")

ENV_EXTRA_DIRS = "VOXPRINT_MODEL_DIRS"
ENV_DISABLE = "VOXPRINT_NO_EXTERNAL_MODELS"

#: File sizes (bytes) of the verified (pinned) commits - from ``GET /api/models/<repo>/tree/<sha>`` on Hugging Face.
#: Used to recognise the verified revision in folders that carry no commit sha.  If the manifest pins another
#: commit, the entry is simply not used (such copies are not reused: safe default).
KNOWN_SIZES: Dict[str, Tuple[str, Dict[str, int]]] = {
    "Qwen/Qwen3-ForcedAligner-0.6B": ("c7cbfc2048c462b0d63a45797104fc9db3ad62b7", {
        "chat_template.json": 1161, "config.json": 5982, "generation_config.json": 115, "merges.txt": 1671853,
        "model.safetensors": 1835544544, "preprocessor_config.json": 330, "tokenizer_config.json": 12666,
        "vocab.json": 2776833}),
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": ("fd4b254389122332181a7c3db7f27e918eec64e3", {
        "config.json": 4494, "generation_config.json": 245, "merges.txt": 1671839,
        "model.safetensors": 3857413744, "preprocessor_config.json": 127,
        "speech_tokenizer/config.json": 2336, "speech_tokenizer/configuration.json": 76,
        "speech_tokenizer/model.safetensors": 682293092, "speech_tokenizer/preprocessor_config.json": 234,
        "tokenizer_config.json": 7344, "vocab.json": 2776833}),
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": ("5d83992436eae1d760afd27aff78a71d676296fc", {
        "config.json": 4494, "generation_config.json": 245, "merges.txt": 1671839,
        "model.safetensors": 1829344272, "preprocessor_config.json": 127,
        "speech_tokenizer/config.json": 2336, "speech_tokenizer/configuration.json": 76,
        "speech_tokenizer/model.safetensors": 682293092, "speech_tokenizer/preprocessor_config.json": 234,
        "tokenizer_config.json": 7344, "vocab.json": 2776833}),
}
#: Files that may be absent (ModelScope copies drop the model card / tooling files).
_OPTIONAL_FILES = {"speech_tokenizer/configuration.json"}
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_API_DIRS = 200          # Pinokio ``api/*`` entries to look at
_MAX_HEADER = 256 * 1024 * 1024


@dataclass(frozen=True)
class FoundModel:
    """A reusable model copy found on this computer.

    ``path`` is the folder to pass to ``from_pretrained`` (it is only ever read); ``location`` is the cache/parent folder it
    was found in (used in messages); ``kind`` says where it came from; ``match`` says how the revision was confirmed.
    """
    repo_id: str
    path: Path               # folder to pass to from_pretrained (never written to)
    location: Path           # the cache / parent folder it was found in (for messages)
    kind: str                # "hf" | "modelscope" | "pinokio" | "voxprint" | "folder" | "env"
    revision: Optional[str]  # commit sha if known
    match: str               # "pinned" | "sizes" | "unpinned"


# ------------------------------------------------------------------------------------------- roots
def disabled() -> bool:
    """True if the user switched external-model reuse off with ``VOXPRINT_NO_EXTERNAL_MODELS``."""
    return os.environ.get(ENV_DISABLE, "").strip().lower() in ("1", "true", "yes", "on")


def _env_path(name: str) -> Optional[Path]:
    """Path from an environment variable with ``~`` and ``%VARS%`` expanded, or ``None`` if it is unset/empty."""
    v = os.environ.get(name, "").strip().strip('"')
    return Path(os.path.expandvars(os.path.expanduser(v))) if v else None


def _home() -> Optional[Path]:
    """The user's home folder, or ``None`` if it cannot be determined."""
    try:
        return Path.home()
    except (RuntimeError, KeyError, OSError):
        return None


def _pinokio_homes() -> List[Path]:
    """Possible Pinokio home folders (``PINOKIO_HOME`` and the default ``~/pinokio``) that exist on disk."""
    homes: List[Path] = []
    p = _env_path("PINOKIO_HOME")
    if p:
        homes.append(p)
    h = _home()
    if h:
        homes.append(h / "pinokio")
    if sys.platform == "win32":
        homes.append(Path("C:\\pinokio"))
    return homes


def _is_dir(p: Path) -> bool:
    """``Path.is_dir`` that never raises (permission errors count as 'no')."""
    try:
        return p.is_dir()
    except OSError:
        return False


def candidate_roots() -> List[Tuple[str, Path]]:
    """(kind, folder) pairs to search, most specific first, existing and de-duplicated."""
    raw: List[Tuple[str, Path]] = []
    try:                       # the folder the user (or the installer) named explicitly goes first
        from infra import existing_models

        raw.extend(existing_models.roots())
    except Exception:  # noqa: BLE001 - optional
        pass
    for part in os.environ.get(ENV_EXTRA_DIRS, "").split(os.pathsep):
        part = part.strip().strip('"')
        if part:
            raw.append(("env", Path(os.path.expandvars(os.path.expanduser(part)))))
    # standard Hugging Face hub cache, in huggingface_hub's own precedence order
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        p = _env_path(name)
        if p:
            raw.append(("hf", p))
    p = _env_path("HF_HOME")
    if p:
        raw.append(("hf", p / "hub"))
    p = _env_path("XDG_CACHE_HOME")
    if p:
        raw.append(("hf", p / "huggingface" / "hub"))
    home = _home()
    if home:
        raw.append(("hf", home / ".cache" / "huggingface" / "hub"))
    # Pinokio (Alexandria's recommended install): per-app and shared caches
    for ph in _pinokio_homes():
        for sub in (("cache", "HF_HOME", "hub"), ("cache", "XDG_CACHE_HOME", "huggingface", "hub")):
            raw.append(("pinokio", ph.joinpath(*sub)))
        api = ph / "api"
        try:
            apps = sorted(d for d in api.iterdir() if d.is_dir())[:_MAX_API_DIRS] if api.is_dir() else []
        except OSError:
            apps = []
        for app in apps:
            for base in (app, app / "app"):
                for sub in (("cache", "HF_HOME", "hub"), ("cache", "XDG_CACHE_HOME", "huggingface", "hub"),
                            ("cache", "huggingface", "hub")):
                    raw.append(("pinokio", base.joinpath(*sub)))
    # ModelScope
    ms = _env_path("MODELSCOPE_CACHE")
    for base in ([ms] if ms else []) + ([home / ".cache" / "modelscope"] if home else []):
        raw.append(("modelscope", base / "hub"))
        raw.append(("modelscope", base))
    # models already downloaded by an earlier Voxprint installation (its own folder layout: Owner--Name)
    try:
        from infra import paths as _paths

        for h in _paths.previous_homes():
            raw.append(("voxprint", h / "models"))
    except Exception:  # noqa: BLE001
        pass
    # conventional places for manual downloads (--local_dir ./models/...)
    if home:
        raw.append(("folder", home / "models"))
        raw.append(("folder", home / "Models"))
    try:
        cwd = Path.cwd()
        raw.append(("folder", cwd / "models"))
        raw.append(("folder", cwd))
    except OSError:
        pass

    own: Optional[Path] = None
    try:
        from infra import paths
        own = paths.models_dir().resolve()
    except Exception:  # noqa: BLE001 - app dirs are optional for the locator
        own = None
    out: List[Tuple[str, Path]] = []
    seen = set()
    for kind, path in raw:
        if not _is_dir(path):
            continue
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen or (own is not None and key == own):
            continue
        seen.add(key)
        out.append((kind, path))
    return out


# ------------------------------------------------------------------------------------------- validation
def _real_size(p: Path) -> Optional[int]:
    """Size of the file behind ``p`` (symlinks followed); None if missing/broken/not a file."""
    try:
        if not p.is_file():
            return None
        return os.stat(p).st_size
    except OSError:
        return None


def check_safetensors(p: Path) -> Optional[str]:
    """None if the file looks complete (header consistent with file size), else a short reason."""
    size = _real_size(p)
    if not size:
        return f"{p.name}: missing or empty"
    try:
        with open(p, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return f"{p.name}: too short"
            (n,) = struct.unpack("<Q", raw)
            if n <= 0 or n > _MAX_HEADER or 8 + n > size:
                return f"{p.name}: bad safetensors header"
            header = json.loads(f.read(n).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return f"{p.name}: unreadable safetensors header"
    if not isinstance(header, dict):
        return f"{p.name}: bad safetensors header"
    end = 0
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        try:
            end = max(end, int(meta["data_offsets"][1]))
        except (KeyError, TypeError, ValueError, IndexError):
            return f"{p.name}: bad tensor entry"
    if size < 8 + n + end:
        return f"{p.name}: truncated ({size} < {8 + n + end} bytes)"
    return None


def _read_json(p: Path) -> Optional[dict]:
    """Parse a JSON object from a non-empty file; ``None`` on any problem or if it is not an object."""
    if not _real_size(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _weight_files(d: Path) -> List[Path]:
    """The ``*.safetensors`` files directly inside ``d``, sorted by name."""
    try:
        return sorted(x for x in d.iterdir() if x.name.endswith(".safetensors"))
    except OSError:
        return []


def check_model_dir(path: Path, repo_id: str) -> Optional[str]:
    """None if ``path`` holds a complete copy of ``repo_id``; otherwise the reason it does not.  Read-only."""
    try:
        if not path.is_dir():
            return "not a folder"
        if path.name.endswith((".partial", ".incomplete", ".tmp")):
            return "unfinished download folder"
        if _read_json(path / "config.json") is None:
            return "config.json missing or invalid"
        known = KNOWN_SIZES.get(repo_id)
        if known:
            for rel in known[1]:
                if rel not in _OPTIONAL_FILES and not _real_size(path / rel):
                    return f"{rel} missing"
        for sub in (path, path / "speech_tokenizer"):   # model folders are tiny trees; no deep walk
            for x in (sorted(sub.iterdir()) if sub.is_dir() else []):
                if x.name.endswith((".incomplete", ".partial")):
                    return f"unfinished download leftover {x.name}"
        # sharded checkpoint
        index = path / "model.safetensors.index.json"
        if _real_size(index):
            idx = _read_json(index)
            wm = idx.get("weight_map") if idx else None
            if not isinstance(wm, dict) or not wm:
                return "model.safetensors.index.json invalid"
            total = 0
            for shard in sorted(set(wm.values())):
                if not isinstance(shard, str) or "/" in shard or "\\" in shard or shard.startswith("."):
                    return "model.safetensors.index.json invalid"
                why = check_safetensors(path / shard)
                if why:
                    return f"shard incomplete: {why}"
                total += _real_size(path / shard) or 0
            need = (idx.get("metadata") or {}).get("total_size") if isinstance(idx.get("metadata"), dict) else None
            if isinstance(need, int) and total < need:
                return f"shards too small ({total} < {need})"
        else:
            files = _weight_files(path)
            if not files:
                return "no safetensors weights"
            m = [re.search(r"-of-(\d+)\.safetensors$", f.name) for f in files]
            if any(m):   # shards without an index: all of them must be present
                want = max(int(x.group(1)) for x in m if x)
                if len([x for x in m if x]) != want:
                    return "shards missing"
        for f in _weight_files(path) + _weight_files(path / "speech_tokenizer"):
            why = check_safetensors(f)
            if why:
                return f"incomplete weights: {why}"
        if known is None and not _weight_files(path):
            return "no safetensors weights"
    except OSError as exc:
        return f"unreadable ({exc.__class__.__name__})"
    return None


def _sizes_match_verified(path: Path, repo_id: str, pinned: str) -> bool:
    """True if the weight files in ``path`` have exactly the sizes of the verified (pinned) revision in :data:`KNOWN_SIZES`."""
    known = KNOWN_SIZES.get(repo_id)
    if not known or known[0] != pinned:
        return False
    for rel, size in known[1].items():
        if rel.endswith(".safetensors") and _real_size(path / rel) != size:
            return False
    return True


# ------------------------------------------------------------------------------------------- layouts
def _name_variants(repo_id: str) -> List[str]:
    """Folder names a copy of ``repo_id`` may have: plain name, ModelScope ``___`` spelling, ``Owner--Name``, ``Owner/Name``, ``models/...``."""
    owner, _, name = repo_id.rpartition("/")
    ms = name.replace(".", "___")
    out = [name, ms, f"{owner}--{name}", f"{owner}/{name}", f"{owner}/{ms}",
           f"models/{owner}/{name}", f"models/{owner}/{ms}", f"models/{name}"] if owner else [name, ms]
    return list(dict.fromkeys(out))


def _hf_snapshots(root: Path, repo_id: str, pinned: Optional[str]) -> Iterator[Tuple[Path, str]]:
    """Snapshot folders of one repo in an HF cache, best first: pinned sha, ``refs/main``, newest others."""
    repo_dir = root / ("models--" + repo_id.replace("/", "--"))
    snaps = repo_dir / "snapshots"
    if not _is_dir(snaps):
        return
    try:
        names = [d.name for d in snaps.iterdir() if d.is_dir()]
    except OSError:
        return
    order: List[str] = []
    if pinned and pinned in names:
        order.append(pinned)
    try:
        main = (repo_dir / "refs" / "main").read_text(encoding="utf-8").strip()
        if main in names and main not in order:
            order.append(main)
    except OSError:
        pass

    def _mtime(n: str) -> float:
        """Modification time of a snapshot folder (0 if unreadable); newest snapshots are tried first."""
        try:
            return (snaps / n).stat().st_mtime
        except OSError:
            return 0.0

    order += sorted((n for n in names if n not in order), key=_mtime, reverse=True)
    for n in order:
        yield snaps / n, n


def _iter_candidates(root: Path, repo_id: str, pinned: Optional[str]) -> Iterator[Tuple[Path, Optional[str], str]]:
    """(model folder, sha if known, layout) in one root."""
    for snap, sha in _hf_snapshots(root, repo_id, pinned):
        yield snap, (sha if _SHA_RE.match(sha) else None), "hf"
    for v in _name_variants(repo_id):
        p = root.joinpath(*v.split("/"))
        if _is_dir(p):
            sha = None
            try:
                sha = (p / ".revision").read_text(encoding="utf-8").strip() or None
            except OSError:
                pass
            yield p, sha, "folder"


def find_model(repo_id: str, pinned_revision: Optional[str] = None,
               roots: Optional[Iterable[Tuple[str, Path]]] = None, ignore_disabled: bool = False) -> Optional[FoundModel]:
    """Best complete copy of ``repo_id`` made by another app, or None (=> download as usual).

    ``pinned_revision`` is the verified commit from the manifest (None = no pin).  ``ignore_disabled`` is for roots the
    user chose explicitly (the "existing models folder"): ``VOXPRINT_NO_EXTERNAL_MODELS`` only switches off the guessing."""
    if disabled() and not ignore_disabled:
        return None
    roots = list(roots) if roots is not None else candidate_roots()
    rejected_other_rev: List[Tuple[Path, str]] = []
    for kind, root in roots:
        try:
            for path, sha, layout in _iter_candidates(root, repo_id, pinned_revision):
                if pinned_revision:
                    if sha and sha != pinned_revision:
                        rejected_other_rev.append((path, sha))
                        continue
                    match = "pinned" if sha else "sizes"
                else:
                    match = "unpinned"
                why = check_model_dir(path, repo_id)
                if why:
                    log.info("ignoring %s at %s: %s", repo_id, path, why)
                    continue
                if match == "sizes" and not _sizes_match_verified(path, repo_id, pinned_revision or ""):
                    log.info("ignoring %s at %s: revision unknown and file sizes differ from the verified revision",
                             repo_id, path)
                    continue
                location = root if layout == "hf" else path.parent
                found = FoundModel(repo_id, path, location, kind, sha, match)
                log.info("reusing %s from another app: %s (revision %s, %s)", repo_id, path, sha or "unknown", match)
                return found
        except OSError as exc:   # a broken network drive etc. must never break the app
            log.info("cannot scan %s: %s", root, exc)
    for path, sha in rejected_other_rev[:3]:
        log.warning("found %s at %s but its revision %s differs from the verified %s - not reusing it, "
                    "the verified revision will be downloaded", repo_id, path, sha[:8], (pinned_revision or "")[:8])
    return None


def locate_models(repo_ids: Iterable[str], pins: Optional[Dict[str, str]] = None) -> Dict[str, Optional[Path]]:
    """Local path per repo (or None where a download is needed)."""
    pins = pins or {}
    roots = candidate_roots()
    out: Dict[str, Optional[Path]] = {}
    for r in repo_ids:
        f = find_model(r, pins.get(r), roots)
        out[r] = f.path if f else None
    return out


def has_unfinished_download(repo_id: str, roots: Optional[Iterable[Tuple[str, Path]]] = None) -> bool:
    """True if another app's HF cache holds an unfinished download (``blobs/*.incomplete``) of this repo.
    Only reported (per-model state "partial"); we never resume or delete someone else's partial files."""
    if disabled():
        return False
    for _kind, root in (list(roots) if roots is not None else candidate_roots()):
        blobs = root / ("models--" + repo_id.replace("/", "--")) / "blobs"
        try:
            if blobs.is_dir() and any(x.name.endswith(".incomplete") for x in blobs.iterdir()):
                return True
        except OSError:
            continue
    return False
