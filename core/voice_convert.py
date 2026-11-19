"""Direct voice conversion: the source's timing and intonation, the target voice's timbre.

:class:`VoiceConverter` is the whole contract. Callers (:func:`convert_file`, the Re-voice window) never import a
particular model. OpenVoice V2 is the current implementation (:class:`core.vc_openvoice.OpenVoiceConverter`); a newer
model is another class with the same methods, returned by :func:`make_converter`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Protocol, Tuple

import numpy as np
import soundfile as sf

from core import audio_utils as au
from core import revoice
from core.audiobook_export import safe_filename
from core.events import CancelToken

log = logging.getLogger("voxprint.revoice")


class VoiceConverter(Protocol):
    """One local voice-conversion model. ``convert`` keeps the source length's meaning: same timeline, new timbre."""

    key: str

    def convert(self, source: np.ndarray, source_sr: int, reference: np.ndarray, reference_sr: int,
                cancel: Optional[CancelToken] = None, progress: Optional[Callable[[float], None]] = None
                ) -> Tuple[np.ndarray, int]:
        """``(mono float32, sample rate)``. The result's duration matches ``source``."""

    def unload(self) -> None:
        """Free the model (GPU memory) after a conversion."""


def make_converter(factory: Optional[Callable[[], VoiceConverter]] = None) -> VoiceConverter:
    """The converter to use. Tests pass a factory; the app uses OpenVoice V2."""
    if factory is not None:
        return factory()
    # Late import: vc_openvoice imports fit_length from this module.
    from core.vc_openvoice import OpenVoiceConverter

    return OpenVoiceConverter()


def fit_length(samples: np.ndarray, n: int) -> np.ndarray:
    """``samples`` stretched or squeezed to exactly ``n`` samples (the timeline stays put)."""
    y = np.asarray(samples, dtype=np.float32).reshape(-1)
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if y.size == n:
        return y
    if y.size == 0:
        return np.zeros(n, dtype=np.float32)
    if y.size == 1:
        return np.full(n, float(y[0]), dtype=np.float32)
    pos = np.linspace(0.0, 1.0, n, dtype=np.float64)
    src = np.linspace(0.0, 1.0, y.size, dtype=np.float64)
    return np.interp(pos, src, y.astype(np.float64)).astype(np.float32)


def read_pcm(path: Path) -> Tuple[np.ndarray, int]:
    """The file as mono float32 PCM in memory, at its own rate (48 kHz when the container needs ffmpeg)."""
    path = Path(path)
    if path.suffix.lower() in (".wav", ".flac", ".ogg", ".opus"):
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
            x = au.to_mono_float(np.asarray(data))
            if x.size and int(sr) > 0:
                return np.clip(x, -1.0, 1.0), int(sr)
        except (OSError, RuntimeError, ValueError):
            pass
    return au.load_audio(path, 48000)


def convert_file(source: Path, reference: Path, folder: Path, converter: VoiceConverter,
                 ffmpeg: Optional[str] = None, run: Optional[Callable[..., object]] = None,
                 cancel: Optional[CancelToken] = None, progress: Optional[Callable[[float], None]] = None) -> Path:
    """Convert ``source`` towards ``reference`` and store Ogg Opus (PCM only while the model runs).

    A FLAC file is the intermediate, same as a dictaphone recording (:func:`core.revoice.to_opus`); it is removed when
    the Opus file is written. Returns the file to play.
    """
    cancel = cancel or CancelToken()
    source, reference, folder = Path(source), Path(reference), Path(folder)
    src, src_sr = read_pcm(source)
    ref, ref_sr = read_pcm(reference)
    cancel.check()
    audio, sr = converter.convert(src, src_sr, ref, ref_sr, cancel=cancel, progress=progress)
    audio = fit_length(np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0), max(1, int(round(len(src) * sr / src_sr))))
    folder.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(f"{source.stem} re-voiced", 80, fallback="re-voiced")
    flac = folder / f"{stem}.flac"
    sf.write(str(flac), audio, int(sr), format="FLAC")
    out = revoice.to_opus(flac, ffmpeg if ffmpeg is not None else au.ensure_ffmpeg(), run=run)
    log.info("re-voice: %s -> %s (%d samples at %d Hz)", source.name, out.name, len(audio), sr)
    return out


class FakeVoiceConverter:
    """Test double: half the source amplitude, same length and rate. Never loads a model."""

    key = "fake"

    def __init__(self) -> None:
        """Start with an empty call log."""
        self.calls: list = []
        self.unloaded = False

    def convert(self, source: np.ndarray, source_sr: int, reference: np.ndarray, reference_sr: int,
                cancel: Optional[CancelToken] = None, progress: Optional[Callable[[float], None]] = None
                ) -> Tuple[np.ndarray, int]:
        """Record the call and return a quieter copy of ``source``. No model runs."""
        if cancel is not None:
            cancel.check()
        self.calls.append((int(len(source)), int(source_sr), int(len(reference)), int(reference_sr)))
        if progress is not None:
            progress(1.0)
        return np.clip(np.asarray(source, dtype=np.float32) * 0.5, -1.0, 1.0), int(source_sr)

    def unload(self) -> None:
        """Mark the fake converter as unloaded."""
        self.unloaded = True
