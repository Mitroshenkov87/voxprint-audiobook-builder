"""``voice.json`` - the human- and machine-readable description of a voice (schema 2).

The file lives next to the adapter files, both in the voice library (``%LOCALAPPDATA%\\Voxprint\\voices\\<id>\\``) and in the
training output folder, so any application that picks up the adapter can show a friendly name, language, author and the
**licence** of the voice without opening the training metadata.

Fields (all always present)::

    schema, id, name, language, created, duration, epochs, base_model,
    author, license, license_url, voice_type, description, commercial_use

* ``license`` is an SPDX-like identifier (``CC0-1.0``, ``CC-BY-4.0``, ``CC-BY-NC-4.0`` ...) or ``custom/personal-only``;
* ``commercial_use`` is **derived** from ``license`` (see :func:`license_allows_commercial`) and recomputed on every
  read, so a hand-edited file cannot claim more than the licence allows.  Unknown licences are treated as *not* allowing
  commercial use.  The licence covers the trained model; it does not replace the consent of the person whose voice it is;
* ``duration`` is the amount of cleaned speech (seconds) the voice was trained on;
* ``voice_type`` is ``male | female | child | other`` or empty; ``description`` is free text (<= 500 characters).

Schema 1 (``voice_name``/``speech_seconds``, no licence) files are still read: :func:`normalize_info` upgrades them.
The module is pure Python (no Qt, no torch).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#: File name of the description, stored next to ``adapter_model.safetensors``.
VOICE_FILENAME = "voice.json"
#: Format version of the file; bump when fields change incompatibly.
VOICE_SCHEMA = 2
#: Allowed values of the optional ``voice_type`` field ("" means "not set").
VOICE_TYPES = ("male", "female", "child", "other")
#: Descriptions are free text; this keeps the file small and the UI sane.
MAX_DESCRIPTION_CHARS = 500
MAX_NAME_CHARS = 80

#: Licence used for a freshly trained voice: the safe default is "only for me".
DEFAULT_LICENSE = "custom/personal-only"
#: Licence for voices meant for testing only (distributed separately from the program).
LICENSE_TEST_ONLY = "custom/test-use-only"
#: Human-readable text of the custom licences (English; the UI shows a localized reminder).
LICENSE_TEXTS = {
    LICENSE_TEST_ONLY: "Test use only - no public release of generated audio, no commercial use",
}

#: Known licences: id -> (commercial use allowed, canonical URL).  Order = order in the UI.
LICENSES: Dict[str, Tuple[bool, str]] = {
    "CC0-1.0": (True, "https://creativecommons.org/publicdomain/zero/1.0/"),
    "CC-BY-4.0": (True, "https://creativecommons.org/licenses/by/4.0/"),
    "CC-BY-SA-4.0": (True, "https://creativecommons.org/licenses/by-sa/4.0/"),
    "CC-BY-NC-4.0": (False, "https://creativecommons.org/licenses/by-nc/4.0/"),
    "CC-BY-NC-SA-4.0": (False, "https://creativecommons.org/licenses/by-nc-sa/4.0/"),
    DEFAULT_LICENSE: (False, ""),
    LICENSE_TEST_ONLY: (False, ""),
}


def license_allows_commercial(license_id: Optional[str]) -> bool:
    """True only for licences known to allow commercial use; anything unknown (or ``custom/...``) is "personal only"."""
    entry = LICENSES.get((license_id or "").strip())
    return bool(entry and entry[0])


def license_url_for(license_id: Optional[str]) -> str:
    """The canonical URL of a known licence, or ``""``."""
    entry = LICENSES.get((license_id or "").strip())
    return entry[1] if entry else ""


def clean_names(value: Any, clean=None) -> Dict[str, str]:
    """Localized texts ``{"ru": "...", "en": "..."}`` (names by default, descriptions with ``clean=clean_description``):
    two-letter lower-case codes, cleaned text, empty ones dropped."""
    if not isinstance(value, dict):
        return {}
    clean = clean or clean_line
    out = {}
    for k, v in value.items():
        code, name = str(k).strip().lower(), clean(str(v or ""))
        if len(code) == 2 and code.isalpha() and name:
            out[code] = name
    return out


#: Word added to the names of trained voices (library folder, result folder, package zip): ``anna_male``, ``anna_female`` ...
#: A voice whose type is not set gets ``_unspecified`` (older voices have no type and are shown as "not specified").
TYPE_SUFFIXES = {"male": "male", "female": "female", "child": "child", "other": "other", "": "unspecified"}


def type_suffix(voice_type: Optional[str]) -> str:
    """``male`` / ``female`` / ``child`` / ``other`` / ``unspecified`` for any input."""
    return TYPE_SUFFIXES[normalize_voice_type(voice_type)]


def with_type_suffix(base: str, voice_type: Optional[str], sep: str = "_") -> str:
    """``base`` + ``_<type>``; a type word that is already at the end of ``base`` is replaced, never doubled."""
    base = (base or "").strip()
    low = base.lower()
    for word in TYPE_SUFFIXES.values():
        for s in ("_", "-", " "):
            if low.endswith(s + word) and len(base) > len(word) + 1:
                base = base[: -(len(word) + 1)].rstrip("_- ")
                break
        else:
            continue
        break
    return f"{base or 'voice'}{sep}{type_suffix(voice_type)}"


def normalize_voice_type(value: Optional[str]) -> str:
    """Return a valid voice type (lower-case) or ``""`` for anything unknown or empty."""
    v = (value or "").strip().lower()
    return v if v in VOICE_TYPES else ""


def clean_description(value: Optional[str]) -> str:
    """Collapse whitespace/newlines and cut the description to :data:`MAX_DESCRIPTION_CHARS`."""
    return " ".join((value or "").split())[:MAX_DESCRIPTION_CHARS]


def clean_line(value: Optional[str], limit: int = MAX_NAME_CHARS) -> str:
    """A single-line free text field (name, author): whitespace collapsed, cut to ``limit`` characters."""
    return " ".join((value or "").split())[:limit]


def clean_license_url(value: Optional[str]) -> str:
    """Keep only http(s) URLs (anything else, e.g. ``javascript:``, becomes ``""``)."""
    v = (value or "").strip()
    return v if v.lower().startswith(("https://", "http://")) and len(v) <= 500 else ""


def build_voice_info(name: str, language: str, duration: float, epochs: int, base_model: str,
                     voice_type: Optional[str] = "", description: Optional[str] = "",
                     now: Optional[datetime] = None, *, voice_id: str = "", author: str = "",
                     license: str = DEFAULT_LICENSE, license_url: str = "",  # noqa: A002
                     consent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the ``voice.json`` dictionary.

    ``language`` is the language of the recording (as detected from the text), ``duration`` the amount of cleaned
    speech used for training (seconds), ``base_model`` the Hugging Face repo id of the base TTS model.  An empty
    ``license_url`` is filled in for the well-known licences.
    """
    created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    lic = (license or DEFAULT_LICENSE).strip()
    info = {
        "schema": VOICE_SCHEMA,
        "id": voice_id,
        "name": clean_line(name),
        "language": language,
        "created": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration": round(float(duration), 1),
        "epochs": int(epochs),
        "base_model": base_model,
        "author": clean_line(author),
        "license": lic,
        "license_url": clean_license_url(license_url) or license_url_for(lic),
        "voice_type": normalize_voice_type(voice_type),
        "description": clean_description(description),
        "commercial_use": license_allows_commercial(lic),
    }
    cons = _clean_consent(consent)
    if cons:
        info["consent"] = cons
    return info


def _clean_consent(data: Any) -> Optional[Dict[str, Any]]:
    from core.consent import clean_consent    # local import: consent.py has no dependency back on this module

    return clean_consent(data)


def normalize_info(data: Dict[str, Any], fallback_id: str = "") -> Dict[str, Any]:
    """Return a complete schema-2 dictionary from whatever was read (older schema, missing or hand-edited fields).

    ``commercial_use`` is always recomputed from ``license``.
    """
    def num(v: Any, cast, default):
        try:
            return cast(v)
        except (TypeError, ValueError):
            return default

    lic = str(data.get("license") or DEFAULT_LICENSE).strip()
    out = {
        "schema": VOICE_SCHEMA,
        "id": str(data.get("id") or fallback_id),
        "name": clean_line(str(data.get("name") or data.get("voice_name") or fallback_id)),
        "language": str(data.get("language") or ""),
        "created": str(data.get("created") or ""),
        "duration": num(data.get("duration", data.get("speech_seconds", 0.0)), float, 0.0),
        "epochs": num(data.get("epochs", 0), int, 0),
        "base_model": str(data.get("base_model") or ""),
        "author": clean_line(str(data.get("author") or "")),
        "license": lic,
        "license_url": clean_license_url(str(data.get("license_url") or "")) or license_url_for(lic),
        "voice_type": normalize_voice_type(str(data.get("voice_type") or "")),
        "description": clean_description(str(data.get("description") or "")),
        "commercial_use": license_allows_commercial(lic),
    }
    if data.get("repo_id"):
        out["repo_id"] = clean_line(str(data["repo_id"]))   # id in the online voices index: the voice was downloaded from there
    names = clean_names(data.get("names"))
    if names:
        out["names"] = names       # shown instead of ``name`` when the UI language has an entry (e.g. a different name per UI language)
    descs = clean_names(data.get("descriptions"), clean_description)
    if descs:
        out["descriptions"] = descs     # localized description, like ``names``
    cons = _clean_consent(data.get("consent"))
    if cons:
        out["consent"] = cons      # who allowed what (see core/consent.py); the scope never lifts the licence limits above
    return out


def write_voice_json(adapter_dir: Path, info: Dict[str, Any]) -> Path:
    """Write ``info`` as UTF-8 JSON into ``adapter_dir/voice.json`` and return the path."""
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    path = Path(adapter_dir) / VOICE_FILENAME
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def read_voice_json(adapter_dir: Path) -> Optional[Dict[str, Any]]:
    """Read ``voice.json`` from an adapter folder (as stored); ``None`` if it is missing or unreadable."""
    try:
        data = json.loads((Path(adapter_dir) / VOICE_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
