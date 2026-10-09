"""Open voices that come with Voxprint: *Tirzah* (female) and *Gideon* (male), Russian, CC0-1.0.

Trained only from public-domain LibriVox recordings (reader, source and hash: ``voices/BUNDLED.md``).  The weights are not
in git: each voice is one zip on the ``voices-v1`` release, SHA-256 pinned here, and is part of the standard model download
(:func:`workers.pipeline_runner.prefetch_models`: the Full setup downloads them upfront, Quick with the first-launch download).
They are imported into the voices library read-only (``"bundled": true`` in ``voice.json``: no edit, no delete) with the licence
shown in the Voices window.  The entries mirror ``voices/index.json``; fields left out here come from the zip's ``voice.json``.

*Boaz* shipped with 0.1.3 and is retired. Gideon replaces it and is used as recorded, with no retraining. Optional catalog
voices (Asher, Noa, Eitan) live in ``voices/index.json`` and are not listed here, so they are never part of this download.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

from core.voice_library import VoiceLibrary
from infra import voice_repository as repo

log = logging.getLogger(__name__)

_URL = "https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/voices-v1/"
_CC0 = ("CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/")

VOICES = (
    {"id": "tirzah", "name": "Tirzah", "language": "ru", "url": _URL + "tirzah.zip", "size_bytes": 53274433,
     "sha256": "aef7c849298ec6e3f2aac1114b993ca111b6297bfd9fef0bf1634527be7ab95f"},
    {"id": "gideon", "name": "Gideon", "language": "ru", "url": _URL + "gideon.zip", "size_bytes": 53031093,
     "sha256": "6ca0e6824080ee86c21a4b34472bd013bd182ef15bb393c6e494fd554e1c2266"},
)


#: Retired voices (index ids). A copy may still be in a user's library from an older build; it stays usable as a
#: narrator, but the multi-voice role lists of the Narrate window do not offer it (the user asked for Boaz to go).
RETIRED = ("boaz",)


def is_retired(record) -> bool:
    """True for a library record of a retired voice (by its index id, else by its name)."""
    info = getattr(record, "info", None) or {}
    rid = str(info.get("repo_id") or "").strip().lower()
    name = str(getattr(record, "name", "") or info.get("name") or "").strip().lower()
    return (rid or name) in RETIRED or name in RETIRED


def offered_for_roles(records) -> list:
    """Library records offered in the multi-voice role lists: every voice except the retired ones."""
    return [r for r in records if not is_retired(r)]


def entries() -> List[repo.RepoVoice]:
    """The bundled voices as index entries."""
    return repo.parse_index({"schema": repo.INDEX_SCHEMA, "voices": [
        {**v, "license": _CC0[0], "license_url": _CC0[1], "bundled": True} for v in VOICES]})


def total_bytes() -> int:
    """Download size of all bundled voices."""
    return sum(int(v["size_bytes"]) for v in VOICES)


def missing(library: Optional[VoiceLibrary] = None) -> List[repo.RepoVoice]:
    """Bundled voices not in the library yet (a downloaded voice carries its index id as ``repo_id``; a locally trained or
    imported voice with the same id or name counts as present too, so it is not downloaded a second time)."""
    all_entries = entries()
    have = repo.installed_entry_ids(all_entries, (library or VoiceLibrary()).list_voices())
    return [e for e in all_entries if e.id not in have]


def ensure(progress: Optional[Callable[[float, str], None]] = None, library: Optional[VoiceLibrary] = None,
           download: Callable[..., object] = repo.download_voice) -> List[str]:
    """Download and import the missing bundled voices (hash-checked, resumable).  Best effort per voice; returns the ids added."""
    library = library or VoiceLibrary()
    todo = missing(library)
    done: List[str] = []
    for i, e in enumerate(todo):
        try:
            download(e, library, progress=(lambda f, n, i=i: progress((i + float(f)) / len(todo), n)) if progress else None)
            done.append(e.id)
        except Exception as exc:  # noqa: BLE001 - a voice is optional; the next start tries again
            log.warning("bundled voice %s download failed: %s", e.id, exc)
    return done
