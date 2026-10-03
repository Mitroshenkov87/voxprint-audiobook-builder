"""``voice.json`` - a small human- and machine-readable description of a trained voice.

The file is written next to the adapter (``<output>/<voice name>/voice.json``) so that any application that picks up
the adapter can show a friendly name, language and notes without opening the training metadata.

Automatic fields are filled by the pipeline; ``voice_type`` and ``description`` are optional and come from the two
optional fields in the Voxprint window.  The module is pure Python (no Qt, no torch) and never raises on I/O problems
that should not abort a finished training run - callers decide how to handle ``OSError``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

#: File name of the description, stored next to ``adapter_model.safetensors``.
VOICE_FILENAME = "voice.json"
#: Format version of the file; bump when fields change incompatibly.
VOICE_SCHEMA = 1
#: Allowed values of the optional ``voice_type`` field ("" means "not set").
VOICE_TYPES = ("male", "female", "child", "other")
#: Descriptions are free text; this keeps the file small and the UI sane.
MAX_DESCRIPTION_CHARS = 500


def normalize_voice_type(value: Optional[str]) -> str:
    """Return a valid voice type (lower-case) or ``""`` for anything unknown or empty."""
    v = (value or "").strip().lower()
    return v if v in VOICE_TYPES else ""


def clean_description(value: Optional[str]) -> str:
    """Collapse whitespace/newlines and cut the description to :data:`MAX_DESCRIPTION_CHARS`."""
    return " ".join((value or "").split())[:MAX_DESCRIPTION_CHARS]


def build_voice_info(voice_name: str, language: str, speech_seconds: float, epochs: int, base_model: str,
                     voice_type: Optional[str] = "", description: Optional[str] = "",
                     now: Optional[datetime] = None) -> Dict[str, Any]:
    """Assemble the ``voice.json`` dictionary.

    ``language`` is the language of the recording (as detected from the text), ``speech_seconds`` the amount of
    cleaned speech used for training, ``base_model`` the Hugging Face repo id of the base TTS model.
    """
    created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return {
        "schema": VOICE_SCHEMA,
        "voice_name": voice_name,
        "language": language,
        "created": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "speech_seconds": round(float(speech_seconds), 1),
        "epochs": int(epochs),
        "base_model": base_model,
        "voice_type": normalize_voice_type(voice_type),
        "description": clean_description(description),
    }


def write_voice_json(adapter_dir: Path, info: Dict[str, Any]) -> Path:
    """Write ``info`` as UTF-8 JSON into ``adapter_dir/voice.json`` and return the path."""
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    path = Path(adapter_dir) / VOICE_FILENAME
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def read_voice_json(adapter_dir: Path) -> Optional[Dict[str, Any]]:
    """Read ``voice.json`` from an adapter folder; ``None`` if it is missing or unreadable."""
    try:
        data = json.loads((Path(adapter_dir) / VOICE_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
