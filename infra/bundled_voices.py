"""Open voices that come with Voxprint: *Tirzah* (female), Russian, CC0-1.0.

Trained only from a public-domain LibriVox recording (reader, source and hash: ``voices/BUNDLED.md``).  The weights are not
in git: the voice is one zip on the ``voices-v1`` release, SHA-256 pinned here, and is part of the standard model download
(:func:`workers.pipeline_runner.prefetch_models`: the Full setup downloads it upfront, Quick with the first-launch download).
It is imported into the voices library read-only (``"bundled": true`` in ``voice.json``: no edit, no delete) with its licence
shown in the Voices window.  The entry mirrors ``voices/index.json``; fields left out here come from the zip's ``voice.json``.

*Boaz* shipped with 0.1.3 and is no longer installed. A new male bundled voice will be trained later. Optional catalog
voices (Asher, Noa) live in ``voices/index.json`` and are not listed here, so they are never part of this download.
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
)


def entries() -> List[repo.RepoVoice]:
    """The bundled voices as index entries."""
    return repo.parse_index({"schema": repo.INDEX_SCHEMA, "voices": [
        {**v, "license": _CC0[0], "license_url": _CC0[1], "bundled": True} for v in VOICES]})


def total_bytes() -> int:
    """Download size of all bundled voices."""
    return sum(int(v["size_bytes"]) for v in VOICES)


def missing(library: Optional[VoiceLibrary] = None) -> List[repo.RepoVoice]:
    """Bundled voices not in the library yet (a downloaded voice carries its index id as ``repo_id``)."""
    have = {str(r.info.get("repo_id") or "") for r in (library or VoiceLibrary()).list_voices()}
    return [e for e in entries() if e.id not in have]


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
