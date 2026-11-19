"""The installer's setup type (``state/install_mode.txt``, written by installer/Voxprint.iss; decided 2026-10-08).

* ``full`` (the default in the wizard): on the first start the app downloads EVERYTHING without a further click - the
  runtime components and every pinned model including the optional ones (both TTS and both speech-recognition sizes, the
  translators, OpenVoice V2, DeepFilterNet, DNSMOS, the Gemma text model with llama.cpp) and ffmpeg, all through
  VOXPrint MDL.
* ``quick``: only the app is installed; on the first start the Components window OFFERS the same complete download and
  waits for the Download click.

Nothing is cut in either mode: after the download both installs are identical.  No file (a developer run, an install
made by an older setup) = the earlier behaviour: the models this PC needs are fetched automatically, the big optional
ones on demand.

:func:`full_sizes` is the honest total behind the wizard's "about N GB": it adds up the pinned sizes (every file of
every ``voxprint-app-models.json`` row, shared blobs counted once, the runtime wheels of the CUDA flavour in
runtime_lock.json, and the pinned ffmpeg build)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from infra import paths

FULL, QUICK = "full", "quick"
_FILE = "install_mode.txt"


def mode(state_dir: Optional[Path] = None) -> str:
    """``full``, ``quick`` or ``""`` (no setup type recorded)."""
    try:
        text = ((state_dir or paths.state_dir()) / _FILE).read_text(encoding="utf-8-sig").strip().lower()
    except OSError:
        return ""
    return text if text in (FULL, QUICK) else ""


def set_mode(value: str, state_dir: Optional[Path] = None) -> None:
    """Record the setup type (the installer does this; tests and the CLI use it too)."""
    d = state_dir or paths.state_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / _FILE).write_text(value + "\n", encoding="utf-8")


def everything(state_dir: Optional[Path] = None) -> bool:
    """True when the complete set is the goal (both setup types; they differ only in who presses Download)."""
    return mode(state_dir) in (FULL, QUICK)


def auto_download(state_dir: Optional[Path] = None) -> bool:
    """False only for Quick: the download waits for the click in the Components window."""
    return mode(state_dir) != QUICK


def full_sizes(flavor: str = "cu130", platform: str = "win-x64") -> Dict[str, int]:
    """Bytes of the complete download from the pinned sizes: ``models``, ``runtime`` and ``total``.

    The model bytes are the files of every manifest row the Full / Quick first run actually fetches through
    VOXPrint MDL - every required row plus the on-demand rows the prefetch pulls (Gemma) - with shared blobs
    counted once (the two TTS rows and the speech-tokenizer row share their tokenizer files). ACE-Step and the
    optional catalog voices are not part of the first run (the user enables them later), so they stay out."""
    from voxprint_common.mdl.catalog import load_manifest

    from infra.mdl_gate import manifest_path

    seen: Dict[str, int] = {}
    for row in load_manifest(manifest_path()).models:
        if row.id == "ace-step" or (row.id.startswith("voice-") and row.on_demand):
            continue
        for f in row.files:
            seen.setdefault(f.sha256, f.size)
    models = sum(seen.values())
    here = Path(__file__).resolve().parent
    assets = json.loads((here / "assets_manifest.json").read_text(encoding="utf-8"))
    models += int(assets["assets"]["ffmpeg"]["platforms"].get("win32" if platform.startswith("win") else "linux", {}).get("size", 0))
    lock = json.loads((here / "runtime_lock.json").read_text(encoding="utf-8"))
    runtime = sum(int(w["size"]) for w in lock["wheels"] if w.get("flavor") in (None, "", flavor))
    return {"models": models, "runtime": runtime, "total": models + runtime}
