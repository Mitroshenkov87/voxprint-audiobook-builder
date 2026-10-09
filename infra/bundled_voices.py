"""Open voices that come with Voxprint: *Levi* (male) and *Miriam* (female), Russian, CC0-1.0.

Trained only from public-domain LibriVox recordings (reader, source and hash: ``voices/BUNDLED.md``).  The weights are not
in git: each voice is one zip on the ``voices-v2`` release, SHA-256 pinned here, and is part of the standard model download
(:func:`workers.pipeline_runner.prefetch_models`: the Full setup downloads them upfront, Quick with the first-launch download).
They are imported into the voices library read-only (``"bundled": true`` in ``voice.json``: no edit, no delete) with the licence
shown in the Voices window.  The entries mirror ``voices/index.json``; fields left out here come from the zip's ``voice.json``.

Build 703 replaced the earlier bundled pair *Gideon* / *Tirzah* (``voices-v1``) with Levi / Miriam. An existing install keeps
the old voices it already has (nothing is deleted); they are just no longer downloaded. *Boaz* shipped with 0.1.3 and is
retired. Optional catalog voices (Natan, Shimon, Rivka, Noa) live in ``voices/index.json`` and are not listed here, so they are
never part of this download.

:data:`DEFAULT_CAST` is the preferred voice per role for multi-voice narration (narrator Levi, men Natan and Shimon, woman
Miriam). A role whose preferred voice is not installed falls back to the generic choice (:func:`core.speakers.default_role_picks`).
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

from core.voice_library import VoiceLibrary
from infra import voice_repository as repo

log = logging.getLogger(__name__)

_URL = "https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/voices-v2/"
_CC0 = ("CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/")

VOICES = (
    {"id": "levi", "name": "Levi", "language": "ru", "url": _URL + "levi.zip", "size_bytes": 53227088,
     "sha256": "45d649f2665a0a98a7a69e6694e279745e2d9469245470f6681464fd02744ea3"},
    {"id": "miriam", "name": "Miriam", "language": "ru", "url": _URL + "miriam.zip", "size_bytes": 53215075,
     "sha256": "a43a0e7e834603086e3aac623760e28271990a69074db6d968d2b1d2f1e9ef0c"},
)

#: Preferred voice (index id) per multi-voice role. Used only when that voice is in the library; see :func:`preferred_ids`.
DEFAULT_CAST = {"narrator": "levi", "male": "natan", "male2": "shimon", "female": "miriam"}


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


def _key_of(record) -> set:
    """The names a library record answers to: its index id (``repo_id``), its folder id and its display name (lower case)."""
    info = getattr(record, "info", None) or {}
    keys = {str(info.get("repo_id") or ""), str(getattr(record, "id", "") or ""), str(getattr(record, "name", "") or "")}
    return {k.strip().lower() for k in keys if k and k.strip()}


def find_record(records, voice_id: str):
    """The library record of catalog voice ``voice_id`` (by ``repo_id``, folder id or name), or ``None``."""
    want = (voice_id or "").strip().lower()
    if not want:
        return None
    by_repo = [r for r in records if str((getattr(r, "info", None) or {}).get("repo_id") or "").strip().lower() == want]
    if by_repo:
        return by_repo[0]
    return next((r for r in records if want in _key_of(r)), None)


def preferred_ids(records) -> dict:
    """:data:`DEFAULT_CAST` resolved against ``records``: role -> library id; a role whose voice is missing is left out."""
    out = {}
    for role, vid in DEFAULT_CAST.items():
        rec = find_record(records, vid)
        if rec is not None:
            out[role] = str(rec.id)
    return out


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
