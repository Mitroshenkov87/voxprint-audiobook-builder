"""Optional noise clean-up of a training recording (DeepFilterNet3, :mod:`infra.denoise_tool`) and the hint that offers it.

Never on by default: artefacts of a clean-up are learnt as part of the voice, so it is only *suggested* when the recording is
noisy, judged by the DNSMOS P.835 background score (BAK, 1-5; :mod:`core.mos`) of a few excerpts.  The user decides.

The original file is never modified: the cleaned copy (48 kHz WAV) goes into the job folder and is used instead of it.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

log = logging.getLogger("voxprint.denoise")

#: Suggest the clean-up below this background score.  First guess: clean home recordings usually score ~3.5-4.2, recordings
#: with audible fan / street noise ~2-3 (DNS-Challenge data).  NEEDS VALIDATION with real recordings.
BAK_SUGGEST_BELOW = 3.0
EXCERPTS = 6                    # excerpts scored per file (evenly spread), 10 s each
EXCERPT_SECONDS = 10.0
TOOL_RATE = 48000               # deep-filter accepts 48 kHz WAV only
#: Maximum attenuation in dB (deep-filter ``-a``): the noisy signal is mixed back in beyond it, which leaves a little natural
#: background instead of the "underwater" artefacts of a full suppression (100 dB).  First guess, NEEDS VALIDATION.
ATTEN_LIMIT_DB = 30


def background_score(files: Sequence[Path], mos, excerpts: int = EXCERPTS) -> Optional[float]:
    """Mean DNSMOS background score (BAK) over evenly spread excerpts of ``files``; None when nothing could be scored.

    Many files (a folder of clips): one excerpt from each of the first ``excerpts`` files."""
    files = list(files)[:excerpts]
    per_file = max(1, excerpts // max(1, len(files)))
    scores = []
    for path in files:
        try:
            x, sr = _read_mono(Path(path))
            n = min(len(x), int(EXCERPT_SECONDS * sr))
            k = max(1, min(per_file, len(x) // max(1, n)))
            for i in range(k):
                a = int((len(x) - n) * (i + 0.5) / k)
                s = mos.score(x[a:a + n], sr)
                if s is not None:
                    scores.append(float(s["bak"]))
        except Exception as exc:  # noqa: BLE001 - a hint must never get in the way
            log.info("background score of %s failed: %s", path, exc)
    return round(float(np.mean(scores)), 2) if scores else None


def _read_mono(path: Path):
    """Mono float audio at its own rate (soundfile), or at 16 kHz through ffmpeg for formats soundfile cannot read."""
    import soundfile as sf

    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return x.mean(axis=1), sr
    except Exception:  # noqa: BLE001
        from core.audio_utils import load_audio

        return load_audio(path, 16000)


def should_suggest(bak: Optional[float], threshold: float = BAK_SUGGEST_BELOW) -> bool:
    """True when the clean-up should be offered."""
    return bak is not None and bak < threshold


def denoise_file(src: Path, dst: Path, tool: Path, run: Optional[Callable[[list], int]] = None,
                 atten_db: int = ATTEN_LIMIT_DB) -> Path:
    """Write a cleaned copy of ``src`` to ``dst`` (48 kHz mono WAV, same length) with the ``deep-filter`` program ``tool``.

    ``run(cmd) -> return code`` is injectable for tests.  Raises RuntimeError when the program fails."""
    import soundfile as sf

    from core.audio_utils import load_audio

    x, _ = load_audio(src, TOOL_RATE)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vx_dn_") as tmp:
        inp = Path(tmp) / "input.wav"
        out_dir = Path(tmp) / "out"
        sf.write(str(inp), x, TOOL_RATE, subtype="FLOAT")
        cmd = [str(tool), "-D", "-a", str(int(atten_db)), "-o", str(out_dir), str(inp)]   # -D: no STFT delay in the output
        if run is None:
            def run(c):
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                return subprocess.run(c, capture_output=True, creationflags=flags).returncode
        code = run(cmd)
        res = out_dir / inp.name
        if code != 0 or not res.is_file():
            raise RuntimeError(f"deep-filter failed (exit code {code})")
        y, sr = sf.read(str(res), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    # the program drops a few frames of look-ahead at the end: pad / trim so every timestamp matches the original
    y = np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]
    sf.write(str(dst), y, TOOL_RATE, subtype="PCM_24")
    log.info("noise clean-up: %s -> %s", src, dst)
    return dst
