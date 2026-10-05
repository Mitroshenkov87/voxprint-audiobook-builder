"""Online voice repository: a small JSON index plus downloadable voice archives, verified by SHA-256.

The index (``index.json``) is a static file anyone can host (e.g. in a GitHub repository)::

    {"schema": 1, "voices": [{"id": "anna-ru", "name": "Anna", "language": "russian", "author": "...",
                              "license": "CC-BY-4.0", "license_url": "", "description": "...", "voice_type": "female",
                              "base_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "url": "https://.../anna-ru.zip",
                              "sha256": "<64 hex>", "size_bytes": 61000000}]}

The URL is configurable (environment variable ``VOXPRINT_VOICES_INDEX``, or ``state/voice_index_url.txt`` written by the
UI).  Until the real repository exists the default points to a placeholder (``OWNER``) and the UI says "not configured
yet".  An empty index, a missing file or no network are normal situations and never raise to the UI as crashes:
:func:`fetch_index` reports them in :class:`IndexResult`.

The last good index is cached (``state/voice_index_cache.json``): offline, :func:`fetch_index` returns the cached voices with
``offline=True``.  Partly downloaded archives are kept (``state/voice_downloads/<sha256>.part``) and resumed with an HTTP
``Range`` request, so a dropped connection never restarts a 60 MB download.  Every voice of the index is shown next to the local
ones (:mod:`infra.voice_catalog`); ``voice.json`` of a downloaded voice carries ``repo_id`` to tell that it is installed.

Downloads: HTTPS only, streamed to a temporary file with a size cap, the SHA-256 must match the index entry before the
archive is imported into the :class:`core.voice_library.VoiceLibrary`.  The licence shown to the user is the one
declared in the index (it overrides whatever the archive's own ``voice.json`` says).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

from core import voice_info
from core.errors import CancelledByUser, VoiceRepositoryError
from core.events import CancelToken
from core.i18n import tr
from core.voice_library import MAX_TOTAL_BYTES, VoiceLibrary, VoiceRecord
from infra import net, paths

log = logging.getLogger("voxprint.voicerepo")

DEFAULT_INDEX_URL = "https://raw.githubusercontent.com/Mitroshenkov87/voxprint-audiobook-builder/main/voices/index.json"
ENV_INDEX_URL = "VOXPRINT_VOICES_INDEX"
INDEX_SCHEMA = 1
MAX_INDEX_BYTES = 2_000_000
Opener = Callable[..., Any]


def _url_file() -> Path:
    """State file holding the user-configured index URL."""
    return paths.state_dir() / "voice_index_url.txt"


def index_url() -> str:
    """The index URL in effect: environment variable, else the saved one, else the default placeholder."""
    env = os.environ.get(ENV_INDEX_URL, "").strip()
    if env:
        return env
    try:
        saved = _url_file().read_text(encoding="utf-8").strip()
    except OSError:
        saved = ""
    return saved or DEFAULT_INDEX_URL


def set_index_url(url: str) -> None:
    """Save the index URL chosen in the UI (an empty string restores the default)."""
    f = _url_file()
    if url.strip():
        f.write_text(url.strip(), encoding="utf-8")
    elif f.exists():
        f.unlink()


def is_configured(url: Optional[str] = None) -> bool:
    """True if the URL is a real https address (not empty and not the ``OWNER`` placeholder)."""
    u = (index_url() if url is None else url).strip()
    return u.lower().startswith("https://") and "OWNER" not in u


@dataclass
class RepoVoice:
    """One downloadable voice of the index."""
    id: str
    name: str
    url: str
    sha256: str
    language: str = ""
    author: str = ""
    license: str = voice_info.DEFAULT_LICENSE
    license_url: str = ""
    description: str = ""
    voice_type: str = ""
    base_model: str = ""
    size_bytes: int = 0
    names: dict = field(default_factory=dict)       # optional localized display names {"ru": "...", "en": "..."}
    descriptions: dict = field(default_factory=dict)   # optional localized descriptions

    @property
    def display_name(self) -> str:
        """The name in the UI language if the index gives one."""
        try:
            from core.i18n import get_language

            return str(self.names.get(get_language()) or self.name)
        except Exception:  # noqa: BLE001
            return self.name

    @property
    def display_description(self) -> str:
        """The description in the UI language if the index gives one."""
        try:
            from core.i18n import get_language

            return str(self.descriptions.get(get_language()) or self.description)
        except Exception:  # noqa: BLE001
            return self.description

    @property
    def commercial_use(self) -> bool:
        """Derived from the declared licence."""
        return voice_info.license_allows_commercial(self.license)


@dataclass
class IndexResult:
    """What :func:`fetch_index` found.  ``error`` is ``""``, ``"not_configured"``, ``"unreachable"`` or ``"invalid"``.

    ``offline`` is True when the voices come from the cache because the index could not be fetched."""
    voices: List[RepoVoice] = field(default_factory=list)
    error: str = ""
    detail: str = ""
    offline: bool = False


def cache_path() -> Path:
    """The cached copy of the last good index."""
    return paths.state_dir() / "voice_index_cache.json"


_cache_lock = threading.Lock()


def _save_cache(raw: bytes) -> None:
    """Atomically replace the cached index.

    Several windows refresh the index in parallel, and on Windows a reader (or an antivirus scan) can hold the target for
    a moment: a shared ``.tmp`` name then failed with WinError 32.  Each write now uses its own temp file, writers are
    serialised, and the final rename is retried briefly before giving up (the cache is only a convenience).
    """
    tmp: Optional[Path] = None
    try:
        p = cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with _cache_lock:
            fd, name = tempfile.mkstemp(prefix=p.stem + ".", suffix=".tmp", dir=p.parent)
            tmp = Path(name)
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            for attempt in range(6):
                try:
                    os.replace(tmp, p)
                    tmp = None
                    return
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(0.1 * (attempt + 1))
    except OSError as exc:
        log.warning("cannot cache the voice index: %s", exc)
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass


def load_cache() -> List["RepoVoice"]:
    """Voices of the cached index ([] if there is none or it is damaged)."""
    try:
        return parse_index(json.loads(cache_path().read_bytes().decode("utf-8-sig")))
    except (OSError, ValueError):
        return []


def _with_cache(res: "IndexResult") -> "IndexResult":
    """An unreachable/invalid result enriched by the cached voices (the error stays visible to the UI)."""
    cached = load_cache()
    if cached:
        res.voices, res.offline = cached, True
    return res


def parse_index(data: Any) -> List[RepoVoice]:
    """Entries of a parsed index; malformed entries (no id/url/sha256, non-https url) are skipped."""
    if not isinstance(data, dict) or not isinstance(data.get("voices"), list):
        raise ValueError("not a voices index")
    out: List[RepoVoice] = []
    seen = set()
    for e in data["voices"]:
        if not isinstance(e, dict):
            continue
        vid, url, sha = str(e.get("id", "")).strip(), str(e.get("url", "")).strip(), str(e.get("sha256", "")).strip().lower()
        if not vid or vid in seen or not url.lower().startswith("https://") or len(sha) != 64 \
                or any(c not in "0123456789abcdef" for c in sha):
            continue
        seen.add(vid)
        lic = str(e.get("license") or voice_info.DEFAULT_LICENSE).strip()
        try:
            size = max(0, int(e.get("size_bytes", 0)))
        except (TypeError, ValueError):
            size = 0
        out.append(RepoVoice(
            id=vid, name=voice_info.clean_line(str(e.get("name") or vid)), url=url, sha256=sha,
            language=str(e.get("language", "")), author=voice_info.clean_line(str(e.get("author", ""))),
            license=lic, license_url=voice_info.clean_license_url(str(e.get("license_url", ""))) or
            voice_info.license_url_for(lic),
            description=voice_info.clean_description(str(e.get("description", ""))),
            voice_type=voice_info.normalize_voice_type(str(e.get("voice_type", ""))),
            base_model=str(e.get("base_model", "")), size_bytes=size, names=voice_info.clean_names(e.get("names")),
            descriptions=voice_info.clean_names(e.get("descriptions"), voice_info.clean_description)))
    return out


def fetch_index(url: Optional[str] = None, opener: Opener = net.urlopen, timeout: float = 15.0) -> IndexResult:
    """Download and parse the index.  Never raises: problems are reported in ``IndexResult.error``."""
    u = (index_url() if url is None else url).strip()
    if not is_configured(u):
        return IndexResult(error="not_configured", detail=u)
    try:
        with opener(urllib.request.Request(u, headers={"User-Agent": "Voxprint"}), timeout=timeout) as r:
            raw = r.read(MAX_INDEX_BYTES + 1)
    except Exception as exc:  # noqa: BLE001 - offline, DNS, 404 ... all mean "cannot reach"
        log.info("voice index unreachable: %s", exc)
        return _with_cache(IndexResult(error="unreachable", detail=str(exc)))
    if len(raw) > MAX_INDEX_BYTES:
        return _with_cache(IndexResult(error="invalid", detail="index too large"))
    try:
        voices = parse_index(json.loads(raw.decode("utf-8-sig")))
    except (ValueError, UnicodeDecodeError) as exc:
        return _with_cache(IndexResult(error="invalid", detail=str(exc)))
    _save_cache(raw)
    return IndexResult(voices=voices)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_voice(entry: RepoVoice, library: VoiceLibrary, opener: Opener = net.urlopen,
                   progress: Optional[Callable[[float, str], None]] = None, cancel: Optional[CancelToken] = None,
                   timeout: float = 30.0) -> VoiceRecord:
    """Download ``entry``'s archive, check its SHA-256 and import it into ``library``.

    Raises :class:`VoiceRepositoryError` on network problems, size overrun or a hash mismatch (nothing is imported then).
    """
    if not entry.url.lower().startswith("https://"):
        raise VoiceRepositoryError(tr("err.voice_repo_download"), details="non-https url")
    cap = min(MAX_TOTAL_BYTES, int(entry.size_bytes * 1.05) + (1 << 20)) if entry.size_bytes else MAX_TOTAL_BYTES
    part_dir = paths.state_dir() / "voice_downloads"
    part_dir.mkdir(parents=True, exist_ok=True)
    tmp = part_dir / f"{entry.sha256}.part"            # named by the expected hash: a resumable download of exactly this archive
    try:
        have = tmp.stat().st_size if tmp.exists() else 0
        if have > cap:
            tmp.unlink()
            have = 0
        headers = {"User-Agent": "Voxprint"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with opener(urllib.request.Request(entry.url, headers=headers), timeout=timeout) as r:
                if have and getattr(r, "status", getattr(r, "code", 200)) != 206:
                    have = 0                          # the server ignored the range: start over
                total, got = entry.size_bytes, have
                with open(tmp, "ab" if have else "wb") as out:
                    while True:
                        if cancel is not None:
                            cancel.check()
                        block = r.read(1 << 20)
                        if not block:
                            break
                        got += len(block)
                        if got > cap:
                            tmp.unlink(missing_ok=True)
                            raise VoiceRepositoryError(tr("err.voice_repo_download"), details="download too large")
                        out.write(block)
                        if progress:
                            progress(min(1.0, got / total) if total else 0.0, entry.name)
        except (VoiceRepositoryError, CancelledByUser):
            raise
        except Exception as exc:  # noqa: BLE001 - the .part file stays for the next try
            raise VoiceRepositoryError(tr("err.voice_repo_download"), details=str(exc)) from exc
        digest = _sha256_file(tmp)
        if digest != entry.sha256:
            tmp.unlink(missing_ok=True)               # a wrong/corrupt partial must not poison the next attempt
            raise VoiceRepositoryError(tr("err.voice_repo_hash"), details=f"{digest} != {entry.sha256}")
        overrides = {"name": entry.name, "author": entry.author, "license": entry.license,
                     "license_url": entry.license_url, "voice_type": entry.voice_type,
                     "description": entry.description, "language": entry.language, "base_model": entry.base_model,
                     "names": entry.names or None, "descriptions": entry.descriptions or None, "repo_id": entry.id}
        rec = library.import_zip(tmp, overrides=overrides)
        tmp.unlink(missing_ok=True)
        return rec
    except BaseException:
        raise
