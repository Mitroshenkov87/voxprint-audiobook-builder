"""The installer's setup type (``state/install_mode.txt``, written by installer/Voxprint.iss; decided 2026-10-08).

* ``full`` (the default in the wizard): on the first start the app downloads EVERYTHING without a further click - the runtime
  components, every pinned model including the optional ones (both TTS and both speech-recognition sizes, the translators,
  OpenVoice V2, DeepFilterNet, DNSMOS, the Gemma text model with llama.cpp) and ffmpeg.
* ``quick``: only the app is installed; on the first start the Components window OFFERS the same complete download and waits for
  the Download click.

Nothing is cut in either mode: after the download both installs are identical.  No file (a developer run, an install made by
an older setup) = the earlier behaviour: the models this PC needs are fetched automatically, the big optional ones on demand.

:func:`full_sizes` is the honest total behind the wizard's "about N GB": it adds up the pinned sizes (model_mirrors.json, the
translator, text-model, clean-up and MOS pins, the runtime wheels of the CUDA flavour in runtime_lock.json)."""
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


def full_sizes(flavor: str = "cu128", platform: str = "win-x64") -> Dict[str, int]:
    """Bytes of the complete download from the pinned sizes: ``models``, ``runtime`` and ``total``."""
    from infra import denoise_tool, llm_tool, model_mirrors, quality_models, text_models

    models = sum(int(m["size"]) for e in model_mirrors.load().values() for m in e.downloadable().values())
    mirrored = set(model_mirrors.load())
    models += sum(m.size_mb * 1024 ** 2 for m in text_models.REGISTRY
                  if m.integrated and m.repo and m.repo not in mirrored)
    pinned = (llm_tool.MODEL["size"], llm_tool.SERVER_ASSETS[platform]["size"], denoise_tool.ASSETS[platform]["size"],
              quality_models.DNSMOS_META["size"])
    models += sum(int(str(n)) for n in pinned)
    here = Path(__file__).resolve().parent
    assets = json.loads((here / "assets_manifest.json").read_text(encoding="utf-8"))
    models += int(assets["assets"]["ffmpeg"]["platforms"].get("win32" if platform.startswith("win") else "linux", {}).get("size", 0))
    lock = json.loads((here / "runtime_lock.json").read_text(encoding="utf-8"))
    runtime = sum(int(w["size"]) for w in lock["wheels"] if w.get("flavor") in (None, "", flavor))
    return {"models": models, "runtime": runtime, "total": models + runtime}
