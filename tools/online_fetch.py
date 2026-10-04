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
    return urllib.request.urlopen(req, timeout=timeout)


def read_manifest(source: str) -> dict:
    """Load and validate the manifest from a URL or a local file."""
    try:
        if re.match(r"^https?://", source):
            last: Optional[Exception] = None
            for attempt in range(4):
                try:
                    with _open(source) as r:
                        raw = r.read(8 << 20)
                    break
                except FetchError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    last = exc
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


def save_state(dest: Path, state: Dict[str, str], version: str = "") -> None:
    tmp = dest / (STATE_FILE + ".tmp")
    tmp.write_text(json.dumps({"schema": SCHEMA, "app_version": version, "components": state}, indent=1), encoding="utf-8")
    os.replace(tmp, dest / STATE_FILE)


def installed_ok(comp: dict, state: Dict[str, str], dest: Path) -> bool:
    """True when this exact component (same SHA-256) is already installed and its marker files still exist."""
    if state.get(comp["id"]) != comp["sha256"]:
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


def download(comp: dict, cache: Path, progress: Callable[[int, int], None], sleep: Callable[[float], None] = time.sleep) -> Path:
    """Download one component into ``cache`` (resumable) and return the verified file path.  Every source is tried in turn (the
    partial file is kept: all sources serve the same bytes, the SHA-256 decides)."""
    srcs = sources(comp)
    last: Optional[FetchError] = None
    for i, url in enumerate(srcs):
        try:
            return _download_from(comp, url, cache, progress, sleep, RETRIES if i == len(srcs) - 1 else 3)
        except FetchError as exc:
            if str(exc) == "cancelled":
                raise
            last = exc
            if i + 1 < len(srcs):
                print(f"{exc} - trying {srcs[i + 1].split('/')[2]}", file=sys.stderr, flush=True)
    assert last is not None
    raise last


def _download_from(comp: dict, url: str, cache: Path, progress: Callable[[int, int], None], sleep: Callable[[float], None],
                   retries: int) -> Path:
    """Download one component from one address into ``cache`` (resumable) and return the verified file path."""
    size, sha = int(comp["size"]), comp["sha256"]
    cache.mkdir(parents=True, exist_ok=True)
    final, part = cache / f"{sha}.zip", cache / f"{sha}.part"
    if final.is_file() and final.stat().st_size == size and sha256_file(final) == sha:
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


def extract(zip_path: Path, dest: Path, progress: Callable[[int, int], None]) -> int:
    """Extract ``zip_path`` into ``dest``; returns the number of files."""
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
    return n


def extract_wheel(whl: Path, dest: Path, progress: Callable[[int, int], None]) -> int:
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
    return n


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


def run(manifest_source: str, dest: Path, cache: Path, status: Status, only: Optional[List[str]] = None,
        sleep: Callable[[float], None] = time.sleep, roles: Optional[List[str]] = None,
        portable: Optional[Path] = None) -> int:
    """Install every component of the manifest into ``dest``; returns the number of components fetched.

    ``portable``: a *portable setup folder* (foundation, see docs/THIN-INSTALLER.md): EVERY selected component is also kept there
    (``components/<id>/<file>`` + ``manifest.json`` + ``SHA256SUMS.txt``), even if it is already installed; a file that is already in
    the folder and passes the SHA-256 check is used instead of the network, so the folder works for an offline re-install."""
    status.write("running", 0.0, "Reading the manifest", force=True)
    man = read_manifest(manifest_source)
    comps = [c for c in man["components"] if (not only or c["id"] in only) and (not roles or c.get("role", "core") in roles)]
    dest.mkdir(parents=True, exist_ok=True)
    state = load_state(dest)
    todo = [c for c in comps if not installed_ok(c, state, dest)]
    fetch = comps if portable is not None else todo           # a portable folder wants all of them
    reused = len(comps) - len(todo)
    need = sum(int(c.get("unpacked_bytes", c["size"] * 2)) for c in todo) + sum(int(c["size"]) for c in fetch)
    try:
        free = shutil.disk_usage(dest).free
    except OSError:
        free = None
    if fetch and free is not None and free < need * 1.05:
        raise FetchError(f"Not enough free disk space: about {need >> 20} MB needed, {free >> 20} MB free.")
    weights = [int(c["size"]) * 2 for c in fetch] or [1]      # download + verify/extract
    total_w, base, fetched = float(sum(weights)), 0.0, 0
    if reused:
        status.write("running", 0.0, f"{reused} component(s) already installed, skipped", force=True)
    for c, w in zip(fetch, weights):
        name = c.get("file") or c["id"]
        size = int(c["size"])
        install = c in todo

        def dl(done: int, total: int, c=c, name=name, w=w, base=base) -> None:
            status.write("running", (base + w / 2 * done / max(1, total)) / total_w,
                         f"Downloading {name}: {done >> 20} of {total >> 20} MB")

        keep = portable_file(portable, c) if portable is not None else None
        if keep is not None and _portable_valid(keep, c):
            zpath = keep                                          # offline: the folder already has it
            status.write("running", base / total_w, f"Using {name} from the setup folder")
        else:
            zpath = download(c, cache, dl, sleep)
            if keep is not None:
                keep.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(zpath, keep)
                zpath.unlink(missing_ok=True)
                zpath = keep
        base += w / 2

        if install:
            def ex(done: int, total: int, name=name, w=w, base=base) -> None:
                status.write("running", (base + w / 2 * done / max(1, total)) / total_w, f"Unpacking {name}")

            (extract_wheel if c.get("kind") == "wheel" else extract)(zpath, dest, ex)
            state[c["id"]] = c["sha256"]
            save_state(dest, state, str(man.get("app_version", "")))
            fetched += 1
        base += w / 2
        if portable is None:
            zpath.unlink(missing_ok=True)
    if not todo:
        save_state(dest, state, str(man.get("app_version", "")))
    if portable is not None:
        write_portable_index(portable, man, comps)
    status.write("done", 1.0, "Done", force=True)
    return fetched


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="voxprint-fetch", description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True, help="URL (https) or local path of the manifest JSON")
    ap.add_argument("--dest", required=True, help="install folder")
    ap.add_argument("--cache", required=True, help="folder for partial / verified downloads")
    ap.add_argument("--status", help="status file polled by the installer")
    ap.add_argument("--only", action="append", help="install only this component id (repeatable)")
    ap.add_argument("--role", action="append", help="install only components of this role (repeatable; manifest field 'role', "
                                                    "default 'core'); without --role every component is installed")
    ap.add_argument("--portable", help="also keep every selected component in this setup folder (components/<id>/<file>, manifest.json, "
                                       "SHA256SUMS.txt); files already there are used instead of the network (offline re-install)")
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
        n = run(a.manifest, Path(a.dest), Path(a.cache), st, a.only, roles=a.role, portable=Path(a.portable) if a.portable else None)
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
