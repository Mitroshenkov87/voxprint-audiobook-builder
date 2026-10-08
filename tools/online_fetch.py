"""voxprint-fetch: the downloader inside the ONLINE installer (``Voxprint-Setup-online.exe``).  Standard library only.

The installer (Inno Setup) hands over a *manifest* (JSON, see ``tools/make_online_payload.py``) and the install folder.
For every component of the manifest this program

* **reuses** it when ``<dest>/voxprint-components.json`` already records the same SHA-256 (a repair / re-run installs nothing twice),
* otherwise **downloads** the zip from its release-asset URL into the cache folder - resumable (HTTP ``Range`` on a
  ``<sha256>.part`` file, retries with back-off), then **verifies** the SHA-256 and the size against the manifest,
* **extracts** it into the install folder (zip-slip protected) and records the component, then deletes the zip.

Progress goes to a small status file for the installer (three lines: ``running|done|error``, permille 0-1000, a short English
message).  Exit codes: 0 ok, 1 failed (message in the status file), 2 bad arguments.  Only ``https`` URLs are accepted
(``http`` only for loopback, used by the tests).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

# --- netroute: the "interface hopper" (infra/netroute.py).  Frozen exe: bundled with --paths infra; the single-file Linux copy
# --- (voxprint-fetch.py) has the module inlined here by tools/make_linux_package.py; without it plain urllib is used.
try:
    import netroute as _nr
except ImportError:
    try:
        from infra import netroute as _nr
    except ImportError:
        _nr = None
# --- end netroute

SCHEMA = 1
STATE_FILE = "voxprint-components.json"
UA = "voxprint-fetch/1"
CHUNK = 1 << 20
RETRIES = 8
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class FetchError(Exception):
    """A failure with a message that is shown to the user."""


# --------------------------------------------------------------------------- status file
class Status:
    """Writes the status file the installer polls (atomic replace).  Line 4 is a heartbeat counter: a background thread
    rewrites the file every ~1.5 s while the job runs, so the installer can tell a busy downloader from a dead one."""

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self._last = 0.0
        self._cur = ("running", 0, "")
        self._beat = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _flush(self) -> None:
        if self.path is None:
            return
        with self._lock:
            self._beat += 1
            state, permille, text = self._cur
            tmp = self.path.with_name(self.path.name + ".tmp")
            try:
                tmp.write_text(f"{state}\n{permille}\n{text.replace(chr(10), ' ')[:300]}\n{self._beat}\n",
                               encoding="ascii", errors="replace")
                os.replace(tmp, self.path)
            except OSError:
                pass

    def write(self, state: str, fraction: float, text: str, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last < 0.2:
            return
        self._last = now
        self._cur = (state, max(0, min(1000, int(fraction * 1000))), text)
        self._flush()

    def start_heartbeat(self) -> None:
        if self.path is None or self._thread is not None:
            return

        def loop() -> None:
            while not self._stop.wait(1.5):
                self._flush()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------- manifest
def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


def check_url(url: str) -> None:
    """Only https (http for loopback) is accepted."""
    p = urllib.parse.urlparse(url)
    if p.scheme == "https" or (p.scheme == "http" and _is_loopback(p.hostname or "")):
        return
    raise FetchError(f"Refusing a non-HTTPS address: {url}")


def _open(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0):
    check_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    if _nr is not None:       # short connect timeout, then the other network interfaces (VPN / odd adapters)
        return _nr.urlopen(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)  # nosec B310 - check_url above allows https only


def read_manifest(source: str, attempts: int = 4) -> dict:
    """Load and validate the manifest from a URL or a local file."""
    try:
        if re.match(r"^https?://", source):
            last: Optional[Exception] = None
            for attempt in range(attempts):
                try:
                    with _open(source) as r:
                        raw = r.read(8 << 20)
                    break
                except FetchError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    if attempt + 1 < attempts:
                        time.sleep(1 + 2 * attempt)
            else:
                raise FetchError(f"Could not download the manifest: {last}")
        else:
            raw = Path(source).read_bytes()
        data = json.loads(raw.decode("utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise FetchError(f"The manifest is unreadable: {exc}") from exc
    return validate_manifest(data)


def validate_manifest(data: dict) -> dict:
    """Check the structure; returns the manifest."""
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise FetchError("The manifest has an unsupported format (update the installer).")
    comps = data.get("components")
    if not isinstance(comps, list) or not comps:
        raise FetchError("The manifest lists no components.")
    seen = set()
    for c in comps:
        try:
            cid, url, size, sha = str(c["id"]), str(c["url"]), int(c["size"]), str(c["sha256"]).lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise FetchError("A component entry of the manifest is incomplete.") from exc
        if not re.match(r"^[A-Za-z0-9._-]{1,64}$", cid) or cid in seen:
            raise FetchError(f"Bad or duplicate component id: {cid!r}")
        seen.add(cid)
        if size <= 0 or not SHA_RE.match(sha):
            raise FetchError(f"Bad size or SHA-256 for {cid}.")
        check_url(url)
        for u in c.get("urls", []):                 # fallback sources (our mirror), tried after ``url`` (the upstream site)
            check_url(str(u))
        if c.get("kind", "zip") not in ("zip", "wheel"):
            raise FetchError(f"Unknown kind for {cid}.")
        c["sha256"] = sha
    return data


# --------------------------------------------------------------------------- state
def load_state(dest: Path) -> Dict[str, str]:
    try:
        d = json.loads((dest / STATE_FILE).read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in d.get("components", {}).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def save_state(dest: Path, state: Dict[str, str], version: str = "", build: str = "") -> None:
    """Write the installed components with the app version and build id of the manifest they came from."""
    tmp = dest / (STATE_FILE + ".tmp")
    tmp.write_text(json.dumps({"schema": SCHEMA, "app_version": version, "build": build, "components": state}, indent=1),
                   encoding="utf-8")
    os.replace(tmp, dest / STATE_FILE)


def installed_ok(comp: dict, state: Dict[str, str], dest: Path) -> bool:
    """True when this component is already installed and its marker files still exist.

    A wheel is identified by its id, ``whl-<dist>-<version>``: the same package version is the same content, whatever the
    bytes of the file (wheels built from sdists differ byte for byte between two release builds).  Other components (zip
    parts) must carry the recorded SHA-256."""
    if comp.get("kind") == "wheel":
        if comp["id"] not in state:
            return False
    elif state.get(comp["id"]) != comp["sha256"]:
        return False
    return all((dest / m).exists() for m in comp.get("markers", []))


# --------------------------------------------------------------------------- download
def sha256_file(path: Path, progress: Optional[Callable[[int], None]] = None) -> str:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
            n += len(b)
            if progress:
                progress(n)
    return h.hexdigest()


def sources(comp: dict) -> List[str]:
    """Addresses of a component, best first: the upstream site (``url``), then the fallbacks (``urls``, e.g. our mirror)."""
    out: List[str] = []
    for u in [comp["url"], *comp.get("urls", [])]:
        if u not in out:
            out.append(str(u))
    return out


_SPEED: Dict[str, float] = {}       # host -> bytes/s of its probe (0 = no answer); measured once per process
PROBE_BYTES = 256 * 1024


def _host_speed(url: str, timeout: float = 5.0) -> float:
    host = urllib.parse.urlparse(url).hostname or ""
    if host not in _SPEED:
        t = time.monotonic()
        try:
            with _open(url, {"Range": f"bytes=0-{PROBE_BYTES - 1}"}, timeout=timeout) as r:
                n = len(r.read(PROBE_BYTES))
            _SPEED[host] = n / max(1e-3, time.monotonic() - t)
        except Exception:  # noqa: BLE001 - an unreachable source just goes last
            _SPEED[host] = 0.0
    return _SPEED[host]


def ranked(urls: List[str]) -> List[str]:
    """Sources fastest first: a short ranged GET per host (once per run).  The sort is stable, so equally fast (or all
    failing) sources keep the manifest order - the upstream site first."""
    if len(urls) < 2:
        return urls
    return sorted(urls, key=lambda u: -_host_speed(u))


def download(comp: dict, cache: Path, progress: Callable[[int, int], None], sleep: Callable[[float], None] = time.sleep,
             verifying: Optional[Callable[[], None]] = None) -> Path:
    """Download one component into ``cache`` (resumable) and return the verified file path.  Every source is tried in turn (the
    partial file is kept: all sources serve the same bytes, the SHA-256 decides).  ``verifying()`` is called right before the
    SHA-256 of a finished file is computed (it takes a few seconds for a big wheel), so the caller can say so."""
    srcs = ranked(sources(comp))
    last: Optional[FetchError] = None
    for i, url in enumerate(srcs):
        try:
            return _download_from(comp, url, cache, progress, sleep, RETRIES if i == len(srcs) - 1 else 3, verifying)
        except FetchError as exc:
            if str(exc) == "cancelled":
                raise
            last = exc
            if i + 1 < len(srcs):
                print(f"{exc} - trying {srcs[i + 1].split('/')[2]}", file=sys.stderr, flush=True)
    assert last is not None
    raise last


def _download_from(comp: dict, url: str, cache: Path, progress: Callable[[int, int], None], sleep: Callable[[float], None],
                   retries: int, verifying: Optional[Callable[[], None]] = None) -> Path:
    """Download one component from one address into ``cache`` (resumable) and return the verified file path."""
    size, sha = int(comp["size"]), comp["sha256"]
    say_verify = verifying or (lambda: None)
    cache.mkdir(parents=True, exist_ok=True)
    final, part = cache / f"{sha}.zip", cache / f"{sha}.part"
    if final.is_file() and final.stat().st_size == size:
        say_verify()
        if sha256_file(final) == sha:
            progress(size, size)
            return final
    final.unlink(missing_ok=True)
    last_err = ""
    for attempt in range(retries):
        have = part.stat().st_size if part.is_file() else 0
        if have > size:
            part.unlink()
            have = 0
        try:
            if have < size:
                headers = {"Range": f"bytes={have}-"} if have else {}
                with _open(url, headers) as r:
                    code = getattr(r, "status", 200)
                    if have and code != 206:          # the server ignored Range: start over
                        have = 0
                    with open(part, "ab" if have else "wb") as f:
                        done = have
                        while True:
                            b = r.read(CHUNK)
                            if not b:
                                break
                            f.write(b)
                            done += len(b)
                            progress(done, size)
            got = part.stat().st_size if part.is_file() else 0
            if got != size:
                raise OSError(f"incomplete download ({got} of {size} bytes)")
            say_verify()
            if sha256_file(part) != sha:
                part.unlink(missing_ok=True)       # corrupt: never resume on top of a bad file
                last_err = "the SHA-256 of the download does not match the manifest"
                if attempt >= 2:
                    break
                continue
            os.replace(part, final)
            return final
        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code}"
            if exc.code == 416:                     # our partial file is not usable
                part.unlink(missing_ok=True)
            elif exc.code in (400, 401, 403, 404, 410):
                break
        except FetchError:
            raise
        except Exception as exc:  # noqa: BLE001 - network errors of any kind: retry, the partial file is kept
            last_err = f"{type(exc).__name__}: {exc}"
        sleep(min(30.0, 1.5 * 2 ** attempt))
    raise FetchError(f"Download of {comp.get('file') or comp['id']} failed: {last_err}")


# --------------------------------------------------------------------------- extract
def _safe_target(dest: Path, name: str) -> Path:
    if name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name):
        raise FetchError(f"Unsafe path in the archive: {name}")
    target = (dest / name).resolve()
    if dest.resolve() not in target.parents and target != dest.resolve():
        raise FetchError(f"Unsafe path in the archive: {name}")
    return target


def extract(zip_path: Path, dest: Path, progress: Callable[[int, int], None], files: Optional[List[str]] = None) -> int:
    """Extract ``zip_path`` into ``dest``; returns the number of files (their paths are appended to ``files``)."""
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        infos = z.infolist()
        total = sum(i.file_size for i in infos) or 1
        done = 0
        for info in infos:
            target = _safe_target(dest, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                while True:
                    b = src.read(CHUNK)
                    if not b:
                        break
                    out.write(b)
                    done += len(b)
                    progress(done, total)
            n += 1
            if files is not None:
                files.append(info.filename)
    return n


def extract_wheel(whl: Path, dest: Path, progress: Callable[[int, int], None], files: Optional[List[str]] = None) -> int:
    """Install a wheel into ``dest`` (a site-packages-like folder): files as they are, ``*.data/purelib|platlib`` merged into the root,
    scripts / headers / data skipped.  Zip-slip protected.  The ``.dist-info`` stays, so ``importlib.metadata`` sees the package."""
    n = 0
    with zipfile.ZipFile(whl) as z:
        infos = z.infolist()
        total = sum(i.file_size for i in infos) or 1
        done = 0
        for info in infos:
            name = info.filename
            m = re.match(r"^[^/]+\.data/([^/]+)/(.*)$", name)
            if m:
                if m.group(1) not in ("purelib", "platlib"):
                    continue
                name = m.group(2)
            if not name or info.is_dir():
                continue
            target = _safe_target(dest, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                while True:
                    b = src.read(CHUNK)
                    if not b:
                        break
                    out.write(b)
                    done += len(b)
                    progress(done, total)
            n += 1
            if files is not None:
                files.append(name)
    return n


FILES_INDEX = "voxprint-files.json"      # component id -> files it unpacked (only where run(prune=True) installs)


def _prune(dest: Path, index: Dict[str, List[str]], stale_ids: List[str], replaced: Dict[str, List[str]]) -> int:
    """Delete the files of the old version: those of a re-installed component that its new archive no longer has, and those of
    components the manifest no longer lists.  Only files recorded in the index, inside ``dest``, and owned by no other
    component are removed.  Returns the number of files deleted."""
    gone: set = set()
    for cid, old in replaced.items():
        gone |= set(old) - set(index.get(cid, []))
    for cid in stale_ids:
        gone |= set(index.pop(cid, []))
    owned = {f for fl in index.values() for f in fl}
    n = 0
    for rel in sorted(gone - owned):
        try:
            _safe_target(dest, rel).unlink()
            n += 1
        except (OSError, FetchError):
            pass
    return n


# --------------------------------------------------------------------------- live progress
class _Rate:
    """Smoothed download speed of ONE file (bytes per second).  An exponential moving average over ~3 s, so the number on the
    live line does not jump with every 1 MiB chunk; a resumed file starts counting from the bytes it already had."""

    TAU = 3.0

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.speed = 0.0
        self._t: Optional[float] = None
        self._done = 0

    def update(self, done: int) -> float:
        now = self.clock()
        if self._t is None or done < self._done:
            self._t, self._done = now, done
            return self.speed
        dt = now - self._t
        if dt >= 0.25:
            inst = (done - self._done) / dt
            alpha = min(1.0, dt / self.TAU)
            self.speed = inst if self.speed == 0 else self.speed + alpha * (inst - self.speed)
            self._t, self._done = now, done
        return self.speed


def _mb_s(speed: float) -> str:
    return f"{speed / (1 << 20):.1f} MB/s" if speed > 0 else "..."


def _say(status, fraction: float, text: str, phase: str, comp: dict, name: str, done: int = 0, total: int = 0,
         speed: float = 0.0, force: bool = False) -> None:
    """One progress step.  A status object that has an ``event`` method (the app's Components window, ``infra/modules.py``)
    gets the structured values and builds its own translated line; the installer's status file gets the short English ``text``."""
    event = getattr(status, "event", None)
    if callable(event):
        event(fraction, phase, str(comp.get("id", "")), name, done, total, speed)
    else:
        status.write("running", fraction, text, force=force)      # force: a phase change must not be lost to the 0.2 s throttle


# --------------------------------------------------------------------------- the whole job
def portable_file(root: Path, comp: dict) -> Path:
    """Where a component lives in a portable setup folder: ``<root>/components/<id>/<file>`` (see docs/THIN-INSTALLER.md)."""
    return root / "components" / comp["id"] / (comp.get("file") or f"{comp['sha256']}.zip")


def _portable_valid(path: Path, comp: dict) -> bool:
    return path.is_file() and path.stat().st_size == int(comp["size"]) and sha256_file(path) == comp["sha256"]


def write_portable_index(root: Path, man: dict, comps: List[dict]) -> None:
    """``manifest.json`` (the manifest the folder was made from) and ``SHA256SUMS.txt`` next to ``components/``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    lines = [f"{c['sha256']} *{portable_file(root, c).relative_to(root).as_posix()}" for c in comps if portable_file(root, c).is_file()]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="ascii")


def _portable_mod():
    """``infra.portable`` (flavor choice, diff, models) or None where it is not bundled (the single-file Linux downloader)."""
    try:
        from infra import portable as mod
        return mod
    except ImportError:
        return None


def _have_size(path: Path, comp: dict) -> bool:
    try:
        return path.is_file() and path.stat().st_size == int(comp["size"])
    except OSError:
        return False


def run(manifest_source: str, dest: Path, cache: Path, status: Status, only: Optional[List[str]] = None,
        sleep: Callable[[float], None] = time.sleep, roles: Optional[List[str]] = None,
        portable: Optional[Path] = None, keep_all: bool = False, offline: bool = False, flavor: str = "auto",
        modules: Optional[List[str]] = None, prune: bool = False) -> int:
    """Install the components of the manifest into ``dest``; returns the number of components fetched.

    ``portable``: a *portable setup folder* (docs/THIN-INSTALLER.md).  Every selected component is also kept there
    (``components/<id>/<file>``), even if it is already installed; a file that is already in the folder and passes the SHA-256
    check is used instead of the network, so the folder works for an offline re-install.
    ``keep_all``: keep ALL components of the manifest (those of the PyTorch ``flavor`` this PC gets: ``auto``, a flavor name or
    ``all``) in the folder, not only the ones installed now (``roles`` / ``only``); the installer uses it to keep the runtime too.
    ``offline``: never touch the network; the manifest is ``<portable>/manifest.json`` and every needed file must be in the folder.
    ``modules``: only the components of these runtime modules (the manifest's ``modules`` lists; of the ``flavor`` when one is
    named) - unlike
    ``only`` this selects the same modules in any release's manifest (the number of parts may differ).
    ``prune`` (the app's runtime folder): record the files of every component (``voxprint-files.json``) and, after an update
    or a switch to another release's manifest, delete the files the new version no longer has (see :func:`_prune`).
    Without internet a folder that has a manifest is used automatically.  When the folder was made from an older manifest, only the
    parts whose SHA-256 changed are fetched and the files of the old version are removed at the end."""
    pm = _portable_mod() if portable is not None else None
    status.write("running", 0.0, "Reading the manifest", force=True)
    local_manifest = portable / "manifest.json" if portable is not None else None
    old_man: Optional[dict] = None
    if local_manifest is not None and local_manifest.is_file():
        try:
            old_man = validate_manifest(json.loads(local_manifest.read_text(encoding="utf-8-sig")))
        except (OSError, ValueError, FetchError):
            old_man = None
    if offline:
        if old_man is None:
            raise FetchError("The setup folder has no usable manifest.json.")
        man = old_man
    else:
        try:
            man = read_manifest(manifest_source, attempts=4 if old_man is None else 1)     # a setup folder is the fallback: no long retries
        except FetchError as exc:
            if old_man is None:
                raise
            man, offline = old_man, True
            status.write("running", 0.0, "No internet connection - installing from the setup folder", force=True)
            print(f"{exc} - using the setup folder", file=sys.stderr, flush=True)
    fl = ""
    if portable is not None and pm is not None:
        fl = pm.flavor_of(man, flavor)
    elif portable is not None and flavor not in ("auto", "all", ""):
        fl = flavor
    flavored = (lambda c: pm.matches_flavor(c, fl)) if (pm is not None and fl) else (lambda c: True)
    comps = [c for c in man["components"] if (not only or c["id"] in only) and (not roles or c.get("role", "core") in roles)
             and (portable is None or flavored(c))]
    if modules is not None:
        named = flavor if flavor not in ("auto", "all", "") else ""
        ids = {i for m in man.get("modules", []) if m.get("id") in modules for i in m.get("components", [])}
        comps = [c for c in comps if (c["id"] in ids or c.get("module") in modules)
                 and (not named or not c.get("flavor") or c["flavor"] == named)]
    keep = [c for c in man["components"] if flavored(c)] if (portable is not None and keep_all) else (comps if portable is not None else [])
    if old_man is not None and not offline and pm is not None and old_man["components"] != man["components"]:
        d = pm.diff(old_man, man, fl or "all")
        if d.newer:
            status.write("running", 0.0, d.text(), force=True)
    dest.mkdir(parents=True, exist_ok=True)
    state = load_state(dest)
    index: Dict[str, List[str]] = {}
    replaced: Dict[str, List[str]] = {}
    if prune:
        try:
            index = {str(k): list(v) for k, v in json.loads((dest / FILES_INDEX).read_text(encoding="utf-8")).items()}
        except (OSError, ValueError, AttributeError):
            index = {}
    todo = [c for c in comps if not installed_ok(c, state, dest)]
    fetch = keep if portable is not None else todo
    for c in todo:                                               # not yet in the install folder: it must come from somewhere
        if c not in fetch:
            fetch.append(c)
    reused = len(comps) - len(todo)
    if offline:
        lost = [portable_file(portable, c).name for c in todo if not _portable_valid(portable_file(portable, c), c)]
        if lost:
            raise FetchError("The setup folder is incomplete or damaged (" + ", ".join(lost[:4]) + (" ..." if len(lost) > 4 else "")
                             + "). Run the installer with an internet connection to repair it.")
        fetch = [c for c in fetch if c in todo]               # offline: only what is installed now
    missing = [c for c in fetch if not (portable is not None and _have_size(portable_file(portable, c), c))]
    need = sum(int(c.get("unpacked_bytes", c["size"] * 2)) for c in todo) + sum(int(c["size"]) for c in missing)
    try:
        free = shutil.disk_usage(dest).free
    except OSError:
        free = None
    if missing and free is not None and free < need * 1.05:
        raise FetchError(f"Not enough free disk space: about {need >> 20} MB needed, {free >> 20} MB free.")
    need_net = [c for c in fetch if not (portable is not None and _have_size(portable_file(portable, c), c))]
    if need_net and not offline and _nr is not None and hasattr(_nr, "pick_fastest"):
        try:                                                     # the fastest network interface for the downloads below
            _nr.pick_fastest(sources(need_net[0])[0])
        except Exception as exc:  # noqa: BLE001 - the probe is an optimisation only
            print(f"network probe failed: {exc}", file=sys.stderr, flush=True)
    weights = [int(c["size"]) * 2 for c in fetch] or [1]      # download + verify/extract
    total_w, base, fetched = float(sum(weights)), 0.0, 0
    if reused:
        status.write("running", 0.0, f"{reused} component(s) already installed, skipped", force=True)
    for c, w in zip(fetch, weights):
        name = c.get("file") or c["id"]
        install = c in todo
        rate = _Rate()

        def dl(done: int, total: int, c=c, name=name, w=w, base=base, rate=rate) -> None:
            pct, speed = int(100 * done / max(1, total)), rate.update(done)
            _say(status, (base + w / 2 * done / max(1, total)) / total_w,
                 f"Downloading {name}: {pct}% - {done >> 20} of {total >> 20} MB - {_mb_s(speed)}", "download", c, name, done, total, speed)

        def verifying(c=c, name=name, w=w, base=base) -> None:
            _say(status, (base + w / 2) / total_w, f"Verifying {name} (SHA-256)", "verify", c, name, force=True)

        keepfile = portable_file(portable, c) if portable is not None else None
        if keepfile is not None and _portable_valid(keepfile, c):
            zpath = keepfile                                      # the folder already has it (no network)
            _say(status, base / total_w, f"Using {name} from the setup folder", "folder", c, name, force=True)
        else:
            zpath = download(c, cache, dl, sleep, verifying)
            if keepfile is not None:
                keepfile.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(zpath, keepfile)
                zpath.unlink(missing_ok=True)
                zpath = keepfile
        base += w / 2

        if install:
            def ex(done: int, total: int, c=c, name=name, w=w, base=base) -> None:
                _say(status, (base + w / 2 * done / max(1, total)) / total_w, f"Unpacking {name}", "unpack", c, name, done, total)

            files: List[str] = []
            (extract_wheel if c.get("kind") == "wheel" else extract)(zpath, dest, ex, files)
            if prune:
                if c["id"] in index:
                    replaced[c["id"]] = index[c["id"]]
                index[c["id"]] = files
                (dest / FILES_INDEX).write_text(json.dumps(index), encoding="utf-8")
            state[c["id"]] = c["sha256"]
            save_state(dest, state, str(man.get("app_version", "")), str(man.get("build", "")))
            fetched += 1
        base += w / 2
        if portable is None:
            zpath.unlink(missing_ok=True)
    if prune:
        listed = {c["id"] for c in man["components"]}
        stale = [cid for cid in index if cid not in listed]
        removed = _prune(dest, index, stale, replaced)
        for cid in stale:
            state.pop(cid, None)
        (dest / FILES_INDEX).write_text(json.dumps(index), encoding="utf-8")
        save_state(dest, state, str(man.get("app_version", "")), str(man.get("build", "")))
        if removed:
            print(f"removed {removed} file(s) of the previous version", file=sys.stderr, flush=True)
    if not todo:
        save_state(dest, state, str(man.get("app_version", "")), str(man.get("build", "")))
    if portable is not None and not offline:
        if keep_all or old_man is None:
            (portable / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
            (portable / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
        if keep_all and pm is not None and not only:
            pm.prune(portable, man, fl or "all")                  # the files of the old version are no longer needed
        if pm is not None:
            pm.write_sums(portable, json.loads((portable / "manifest.json").read_text(encoding="utf-8")), fl or "all")
        else:
            write_portable_index(portable, man, keep)
    status.write("done", 1.0, "Done", force=True)
    return fetched


def run_channel(pinned: str, latest: str, dest: Path, cache: Path, status: Status, prefer: str = "pinned", **kw) -> tuple:
    """:func:`run` with the *pinned* manifest (the one this release was built and tested with) or the *latest* one (the newest
    release's manifest), falling back to the other when a download fails or a SHA-256 does not match.  The default is pinned;
    its fallback to latest keeps an online install working when an upstream file of the pinned set disappears.  A manifest is
    always used as a whole (the parts of two releases must not be mixed).  Returns ``(components fetched, manifest used)``."""
    order = [src for src in ((latest, pinned) if prefer == "latest" else (pinned, latest)) if src]
    order = list(dict.fromkeys(order))
    first: Optional[FetchError] = None
    for i, src in enumerate(order):
        try:
            return run(src, dest, cache, status, **kw), src
        except FetchError as exc:
            if str(exc) == "cancelled":
                raise
            first = first or exc
            if i + 1 == len(order):
                raise first from exc        # the error of the first choice is the one worth showing
            which = "latest" if order[i + 1] == latest else "pinned (verified)"
            status.write("running", 0.0, f"{exc} - trying the {which} components", force=True)
            print(f"{exc} - trying the {which} manifest {order[i + 1]}", file=sys.stderr, flush=True)
    raise FetchError("no manifest")


class _Sub:
    """A slice of the overall progress (components first, then the models) on top of a :class:`Status`."""

    def __init__(self, status: Status, base: float, span: float) -> None:
        self.status, self.base, self.span = status, base, span

    def write(self, state: str, fraction: float, text: str, force: bool = False) -> None:
        if state == "error":
            self.status.write(state, fraction, text, force=True)
        else:
            self.status.write("running", self.base + self.span * fraction, text, force=force and state != "done")


def run_portable(manifest_source: str, dest: Path, cache: Path, status: Status, portable: Path, roles: Optional[List[str]] = None,
                 models: str = "auto", flavor: str = "auto", keep_all: bool = True, offline: bool = False,
                 sleep: Callable[[float], None] = time.sleep) -> int:
    """Components (+ models) into the setup folder and the install of ``roles``; see :func:`run`.  Models are a bonus: if they cannot
    be had the install still succeeds and the final message says so."""
    pm = _portable_mod()
    if pm is None:
        raise FetchError("This build cannot keep a setup folder.")
    repos: List[str] = []
    if models != "none" and not offline:
        repos = pm.models_for(models, pm.detect_vram_mb())
    mbytes = 0
    if repos:
        ents = pm.model_mirrors.load()
        mbytes = sum(int(m["size"]) for r in repos if r in ents for m in ents[r].downloadable().values())
    man_bytes = 1
    try:
        man = read_manifest(manifest_source) if not offline else json.loads((portable / "manifest.json").read_text(encoding="utf-8-sig"))
        fl = pm.flavor_of(man, flavor)
        man_bytes = sum(int(c["size"]) for c in man["components"] if pm.matches_flavor(c, fl)) or 1
    except (FetchError, OSError, ValueError):
        pass
    share = man_bytes / float(man_bytes + mbytes) if mbytes else 1.0
    n = run(manifest_source, dest, cache, _Sub(status, 0.0, share), roles=roles, portable=portable, keep_all=keep_all,
            offline=offline, flavor=flavor, sleep=sleep)
    note = ""
    if repos:
        try:
            pm.fetch_models(portable, repos, lambda f, t: status.write("running", share + (1 - share) * f, t))
            pm.write_sums(portable, json.loads((portable / "manifest.json").read_text(encoding="utf-8")), pm.flavor_of(man, flavor) or "all")
        except Exception as exc:  # noqa: BLE001 - a bonus: the program downloads missing models itself
            note = f" (models are incomplete: {exc})"
            print(f"WARNING: {exc}", file=sys.stderr, flush=True)
    status.write("done", 1.0, "Done" + note, force=True)
    return n


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="voxprint-fetch", description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", help="URL (https) or local path of the manifest JSON (not needed with --from-folder)")
    ap.add_argument("--dest", required=True, help="install folder")
    ap.add_argument("--cache", required=True, help="folder for partial / verified downloads")
    ap.add_argument("--status", help="status file polled by the installer")
    ap.add_argument("--only", action="append", help="install only this component id (repeatable)")
    ap.add_argument("--role", action="append", help="install only components of this role (repeatable; manifest field 'role', "
                                                    "default 'core'); without --role every component is installed")
    ap.add_argument("--portable", help="also keep every selected component in this setup folder (components/<id>/<file>, manifest.json, "
                                       "SHA256SUMS.txt); files already there are used instead of the network (offline re-install)")
    ap.add_argument("--portable-all", action="store_true", help="with --portable: keep ALL components (and the models) in the folder, "
                                                                "not only what is installed now")
    ap.add_argument("--from-folder", help="install from this setup folder without any network access (its manifest.json is used)")
    ap.add_argument("--flavor", default="auto", help="PyTorch flavor kept in the setup folder: auto (this PC), cu128 / cu126 / cpu, or all")
    ap.add_argument("--models", choices=("none", "auto", "all"), default=None,
                    help="models to put into the setup folder (default: auto with --portable-all, else none)")
    ap.add_argument("--latest-manifest", help="manifest of the newest release: the fallback of the pinned --manifest, or the first "
                                               "choice with --prefer latest (falls back to the pinned one)")
    ap.add_argument("--prefer", choices=("pinned", "latest"), default="pinned", help="which manifest to try first (default pinned)")
    ap.add_argument("--iface", help="network interface / local IP to use, 'auto' (default) or 'default' (never hop); "
                                    "same as the VOXPRINT_NET_IFACE variable")
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        return 2
    if a.iface:
        os.environ["VOXPRINT_NET_IFACE"] = a.iface
    if _nr is not None:
        _nr.on_event = lambda msg: print(msg, file=sys.stderr, flush=True)
    st = Status(Path(a.status) if a.status else None)
    st.start_heartbeat()
    try:
        if a.from_folder:
            n = run_portable("", Path(a.dest), Path(a.cache), st, Path(a.from_folder), roles=a.role, models="none", flavor=a.flavor,
                             keep_all=False, offline=True)
        elif not a.manifest:
            raise FetchError("--manifest or --from-folder is required")
        elif a.portable and a.portable_all:
            n = run_portable(a.manifest, Path(a.dest), Path(a.cache), st, Path(a.portable), roles=a.role,
                             models=a.models or "auto", flavor=a.flavor)
        else:
            n, used = run_channel(a.manifest, a.latest_manifest or "", Path(a.dest), Path(a.cache), st, a.prefer, only=a.only,
                                  roles=a.role, portable=Path(a.portable) if a.portable else None, flavor=a.flavor)
            if used != a.manifest:
                print(f"installed from {used}")
    except FetchError as exc:
        st.write("error", 0.0, str(exc), force=True)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        st.write("error", 0.0, f"{type(exc).__name__}: {exc}", force=True)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        st.stop()
    print(f"OK: {n} component(s) installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
