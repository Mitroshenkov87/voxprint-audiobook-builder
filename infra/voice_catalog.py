"""The voices the user sees: the local library plus every voice of the online index that is not installed yet.

Qt-free.  :func:`build` merges the two lists (a downloaded voice carries ``repo_id`` in its ``voice.json``, so it is shown once, as
local); a remote voice has the key ``repo:<id>`` and :func:`ensure_local` downloads it on first use (SHA-256 checked, resumable, see
:mod:`infra.voice_repository`).  Any voice that is added to the index later appears automatically after the next refresh
(start of the window, the *Refresh* button); offline the cached index is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from core import consent, voice_info
from core.voice_library import VoiceLibrary, VoiceRecord
from infra import voice_repository as repo

REMOTE_PREFIX = "repo:"


@dataclass
class CatalogItem:
    """One voice in the list, local or remote."""
    key: str                                   # local voice id, or ``repo:<index id>``
    name: str
    description: str
    language: str
    license: str
    license_url: str
    scope: str
    commercial: bool
    installed: bool
    record: Optional[VoiceRecord] = None
    entry: Optional[repo.RepoVoice] = None
    size_bytes: int = 0
    author: str = ""

    @property
    def remote(self) -> bool:
        """True while the voice is only in the online index."""
        return not self.installed


def scope_for_license(license_id: str) -> str:
    """Usage scope shown for a voice that has no consent block yet (a remote one): derived from the licence like for local voices."""
    lic = (license_id or "").strip()
    if voice_info.license_allows_commercial(lic):
        return consent.COMMERCIAL
    return consent.PUBLIC_NC if "-NC" in lic else consent.PRIVATE


def remote_key(entry_id: str) -> str:
    """List key of a not-yet-downloaded index voice."""
    return REMOTE_PREFIX + entry_id


def is_remote_key(key: str) -> bool:
    """True for a key made by :func:`remote_key`."""
    return str(key or "").startswith(REMOTE_PREFIX)


def build(library: VoiceLibrary, entries: List[repo.RepoVoice]) -> List[CatalogItem]:
    """Local voices first (as :meth:`VoiceLibrary.list_voices` orders them), then the index voices that are not installed."""
    items: List[CatalogItem] = []
    installed_ids = set()
    for rec in library.list_voices():
        rid = str(rec.info.get("repo_id") or "")
        if rid:
            installed_ids.add(rid)
        items.append(CatalogItem(rec.id, rec.name, rec.description, rec.language, rec.license, str(rec.info.get("license_url", "")),
                                 rec.scope, rec.commercial_use, True, record=rec, size_bytes=0, author=str(rec.info.get("author", ""))))
    for e in entries:
        if e.id in installed_ids:
            continue
        items.append(CatalogItem(remote_key(e.id), e.display_name, e.display_description, e.language, e.license, e.license_url,
                                 scope_for_license(e.license), e.commercial_use, False, entry=e, size_bytes=e.size_bytes, author=e.author))
    return items


def find_entry(entries: List[repo.RepoVoice], key: str) -> Optional[repo.RepoVoice]:
    """The index entry behind a remote key."""
    rid = str(key)[len(REMOTE_PREFIX):] if is_remote_key(key) else str(key)
    return next((e for e in entries if e.id == rid), None)


def ensure_local(key: str, library: VoiceLibrary, entries: List[repo.RepoVoice],
                 download: Callable[..., VoiceRecord] = repo.download_voice, **kw) -> Optional[VoiceRecord]:
    """The local record for ``key``; a remote voice is downloaded first.  ``None`` if the key is unknown."""
    if not is_remote_key(key):
        return library.get(key)
    entry = find_entry(entries, key)
    if entry is None:
        return None
    return download(entry, library, **kw)


def size_text(n: int) -> str:
    """"57.9 MB" style size for the list (empty when unknown)."""
    if n <= 0:
        return ""
    return f"{n / 1e6:.0f} MB" if n < 1e9 else f"{n / 1e9:.1f} GB"
