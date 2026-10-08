"""The optional direct-conversion model: OpenVoice V2 tone-colour converter (MIT).

Not part of the first-run download. The Re-voice window fetches it only when the user presses *Download model*.
The files, revision and SHA-256 live in ``infra/model_mirrors.json`` (hashes only: there is no backup mirror yet).
The checkpoint is about 125 MB and runs on the NVIDIA GPU when CUDA is available, otherwise on the CPU.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from core.i18n import tr
from infra import model_mirrors, model_release, paths

log = logging.getLogger("voxprint.revoice")

REPO = "myshell-ai/OpenVoiceV2"
LABEL = "OpenVoice V2"
#: Only the tone-colour converter. Base-speaker TTS checkpoints are not used and not downloaded.
FILES = ("converter/config.json", "converter/checkpoint.pth")


def entry(manifest: Optional[Path] = None) -> model_mirrors.MirrorEntry:
    """The manifest entry (revision, sizes, SHA-256). Raises if the bundled manifest has no such model."""
    found = model_mirrors.load(manifest).get(REPO)
    if found is None:
        raise model_release.ReleaseError(f"{REPO} is not in the model manifest")
    return found


def model_dir(models_dir: Optional[Path] = None) -> Path:
    """``<models folder>/openvoice-v2`` (not the generic ``Org--Name`` folder, so Check & repair does not fetch it)."""
    if models_dir is None:
        models_dir = paths.models_dir()
    return Path(models_dir) / "openvoice-v2"


def download_bytes(manifest: Optional[Path] = None) -> int:
    """Bytes of the converter download."""
    e = entry(manifest)
    return sum(int(e.files[name]["size"]) for name in FILES)


def download_mb(manifest: Optional[Path] = None) -> int:
    """The size shown on the button, in megabytes."""
    return max(1, int(round(download_bytes(manifest) / 1e6)))


def ready(models_dir: Optional[Path] = None, manifest: Optional[Path] = None) -> bool:
    """True when both pinned files are present at the expected size (the hash was checked on download)."""
    try:
        e = entry(manifest)
    except model_release.ReleaseError:
        return False
    folder = model_dir(models_dir)
    for name in FILES:
        path = folder.joinpath(*name.split("/"))
        try:
            if not path.is_file() or path.stat().st_size != int(e.files[name]["size"]):
                return False
        except OSError:
            return False
    return True


def ensure(progress: Callable[[float, str], None] = lambda f, m="": None, models_dir: Optional[Path] = None,
           opener=None, timeout: float = 30.0, manifest: Optional[Path] = None) -> Path:
    """Download the converter if it is missing (size + SHA-256, resumable). Only on the user's explicit request.

    Returns the model folder. Raises :class:`infra.model_release.ReleaseError`.
    """
    e = entry(manifest)
    folder = model_dir(models_dir)
    opener = opener or model_release._open
    total = float(download_bytes(manifest)) or 1.0
    done = 0

    def _report(so_far: int) -> None:
        frac = min(1.0, so_far / total)
        progress(min(0.99, frac), tr("progress.downloading", short=LABEL, pct=int(100 * frac)))

    for name in FILES:
        meta = e.files[name]
        target = folder.joinpath(*name.split("/"))
        size = int(meta["size"])
        if model_release.file_ok(target, meta):
            done += size
            _report(done)
            continue
        if target.exists():
            target.unlink()
        base = done
        seen = [0]

        def on_bytes(n: int, _seen: list = seen, _base: int = base) -> None:
            _seen[0] += n
            _report(_base + _seen[0])

        url = f"https://huggingface.co/{REPO}/resolve/{e.source_revision}/{name}"
        model_release.fetch_file(url, target, meta, on_bytes, opener, timeout)
        done += size
    progress(1.0, tr("progress.model_verifying", short=LABEL))
    log.info("OpenVoice V2 converter ready in %s", folder)
    return folder
