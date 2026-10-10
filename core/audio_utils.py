"""Audio helpers: reading any format (ffmpeg / pydub, with soundfile as the primary reader for WAV/FLAC),
resampling, writing WAV, frame energy, voiced/silence detection, SNR and fades.

All arrays are mono ``float32`` in ``[-1, 1]``.  Heavy imports (pydub, scipy, soundfile) are done lazily so that
importing this module stays cheap.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from core.errors import AudioReadError
from core.i18n import tr

_FFMPEG_READY = False


def ensure_ffmpeg() -> Optional[str]:
    """Locate ffmpeg, point pydub at it and return its path (``None`` if nothing was found).

    Search order: next to the executable; ``ffmpeg`` on ``PATH`` *only if* ``ffmpeg -version`` really works (so a broken
    shim is skipped); the pinned LGPL build Voxprint downloaded itself (``infra/assets.py``); the ``imageio-ffmpeg``
    wheel (a GPL build, kept as an offline fallback).
    """
    global _FFMPEG_READY
    candidates: List[str] = []
    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    base = Path(getattr(sys, "_MEIPASS", Path(sys.argv[0]).resolve().parent))
    for d in (base, Path(sys.executable).parent):   # no resolve(): a venv's python links to /usr/bin, whose ffmpeg is not 'next to the program'
        p = d / exe_name
        if p.exists():
            candidates.append(str(p))
    path = candidates[0] if candidates else None
    if path is None:
        try:
            from infra import env_probe

            info = env_probe.probe_ffmpeg()
            if info and info.ok:
                path = info.path
        except Exception:  # noqa: BLE001
            path = None
    if path is None:
        try:
            from infra import assets

            spec = assets.spec_for("ffmpeg")
            managed = assets.installed_path(spec) if spec else None
            path = str(managed) if managed else None
        except Exception:  # noqa: BLE001
            path = None
    if path is None:
        try:
            import imageio_ffmpeg

            path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            path = None
    if path:
        folder = str(Path(path).parent)       # libraries that look for ffmpeg on PATH (pydub's own probe) find it too
        if folder not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
        try:
            import warnings

            with warnings.catch_warnings():   # pydub warns at import when PATH had no ffmpeg; it is set right below
                warnings.simplefilter("ignore", RuntimeWarning)
                from pydub import AudioSegment

            AudioSegment.converter = path
            # pydub looks for ffprobe separately; it is only needed for mediainfo, so we do not require it
        except Exception:  # noqa: BLE001
            pass
        _FFMPEG_READY = True
    return path


def to_mono_float(x: np.ndarray) -> np.ndarray:
    """Convert to a mono float32 array (stereo is averaged)."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """High-quality polyphase resampling (``scipy.signal.resample_poly``), clipped to ``[-1, 1]``."""
    if sr_from == sr_to or x.size == 0:
        return x.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = math.gcd(int(sr_from), int(sr_to))
    y = resample_poly(x.astype(np.float64), int(sr_to) // g, int(sr_from) // g)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def _read_with_ffmpeg(path: Path) -> Tuple[np.ndarray, int]:
    """Decode through ffmpeg directly into a temporary WAV and read that.

    This does not need ``ffprobe`` (the ``imageio-ffmpeg`` wheel ships no ffprobe, and pydub calls it for m4a/aac/mp4,
    which failed with ``[WinError 2]`` on Windows).
    """
    import subprocess
    import tempfile

    exe = ensure_ffmpeg()
    if not exe:
        raise FileNotFoundError("ffmpeg not found")
    with tempfile.TemporaryDirectory(prefix="voxprint_") as tmp:
        out = Path(tmp) / "decoded.wav"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run([exe, "-v", "error", "-nostdin", "-y", "-i", str(path), "-vn", "-acodec", "pcm_f32le",
                               str(out)], capture_output=True, creationflags=flags)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip()[-300:] or f"ffmpeg rc={proc.returncode}")
        return _read_with_soundfile(out)


def _read_with_pydub(path: Path) -> Tuple[np.ndarray, int]:
    """Decode with pydub (needs ffmpeg); returns ``(mono float32, sample_rate)``."""
    ensure_ffmpeg()
    from pydub import AudioSegment

    seg = AudioSegment.from_file(str(path))
    sr = seg.frame_rate
    ch = seg.channels
    arr = np.array(seg.get_array_of_samples())
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)
    scale = float(1 << (8 * seg.sample_width - 1))
    return (arr.astype(np.float64) / scale).astype(np.float32), sr


def _read_with_soundfile(path: Path) -> Tuple[np.ndarray, int]:
    """Read WAV/FLAC/OGG with libsndfile; returns ``(mono float32, sample_rate)``."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono_float(data), int(sr)


def load_audio(path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Read any audio file and return ``(mono float32 in [-1, 1], target_sr)``.

    WAV/FLAC/OGG try soundfile first; every other container goes through ffmpeg, then pydub, then soundfile.  Raises
    ``AudioReadError`` with the per-reader errors in ``details`` if nothing works.
    """
    p = Path(path)
    if not p.exists():
        raise AudioReadError(tr("err.audio_missing", path=p))
    errors: List[str] = []
    order = [_read_with_soundfile, _read_with_pydub] if p.suffix.lower() in (".wav", ".flac", ".ogg") else [
        _read_with_ffmpeg, _read_with_pydub, _read_with_soundfile]
    for reader in order:
        try:
            x, sr = reader(p)
            if x.size == 0:
                raise ValueError("empty audio")
            return resample(x, sr, target_sr), target_sr
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{reader.__name__}: {exc}")
    raise AudioReadError(
        tr("err.audio_unreadable"),
        details="; ".join(errors),
    )


def load_audio_multi(path, rates: Tuple[int, ...]) -> dict:
    """Decode the file once and return ``{rate: samples}`` for each requested rate (avoids double resampling)."""
    p = Path(path)
    if not p.exists():
        raise AudioReadError(tr("err.audio_missing", path=p))
    x, sr = None, 0
    errors: List[str] = []
    order = [_read_with_soundfile, _read_with_pydub] if p.suffix.lower() in (".wav", ".flac", ".ogg") else [
        _read_with_ffmpeg, _read_with_pydub, _read_with_soundfile]
    for reader in order:
        try:
            x, sr = reader(p)
            if x.size:
                break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{reader.__name__}: {exc}")
            x = None
    if x is None or x.size == 0:
        raise AudioReadError(
            tr("err.audio_unreadable"),
            details="; ".join(errors),
        )
    return {r: resample(x, sr, r) for r in rates}


def write_wav(path, samples: np.ndarray, sr: int) -> None:
    """Write 16-bit PCM WAV (creates parent folders, clips to ``[-1, 1]``)."""
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(samples, -1.0, 1.0), sr, subtype="PCM_16")


def duration(samples: np.ndarray, sr: int) -> float:
    """Length of ``samples`` in seconds."""
    return len(samples) / float(sr)


def rms(x: np.ndarray) -> float:
    """Root-mean-square level of the signal (0 for empty input)."""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))


def frame_rms(x: np.ndarray, sr: int, frame_ms: float = 20.0, hop_ms: float = 10.0) -> np.ndarray:
    """RMS per frame (default 20 ms frames every 10 ms); frame ``i`` starts at ``i * hop``. Uses a cumulative sum, so it is fast."""
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if x.size < frame:
        return np.array([rms(x)], dtype=np.float32)
    n = 1 + (x.size - frame) // hop
    sq = np.square(x.astype(np.float64))
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    starts = np.arange(n) * hop
    means = (csum[starts + frame] - csum[starts]) / frame
    return np.sqrt(np.maximum(means, 0.0)).astype(np.float32)


def voiced_mask(x: np.ndarray, sr: int, hop_ms: float = 10.0) -> np.ndarray:
    """Boolean per-frame mask: True where speech is present.

    The threshold is adaptive: noise floor (10th percentile) + 12 dB, but never lower than peak - 40 dB and never higher
    than peak - 6 dB.
    """
    fr = frame_rms(x, sr, hop_ms=hop_ms)
    if fr.size == 0 or float(fr.max()) <= 1e-6:
        return np.zeros(fr.size, dtype=bool)
    db = 20 * np.log10(np.maximum(fr, 1e-7))
    noise = np.percentile(db, 10)
    peak = np.percentile(db, 98)
    thr = max(noise + 12.0, peak - 40.0)
    thr = min(thr, peak - 6.0)
    return db > thr


def voiced_bounds(x: np.ndarray, sr: int, hop_ms: float = 10.0) -> Tuple[float, float]:
    """``(start of the first speech, end of the last speech)`` in seconds inside ``x``."""
    m = voiced_mask(x, sr, hop_ms=hop_ms)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return 0.0, len(x) / float(sr)
    return idx[0] * hop_ms / 1000.0, idx[-1] * hop_ms / 1000.0 + 0.02


def voiced_seconds(x: np.ndarray, sr: int, hop_ms: float = 10.0) -> float:
    """Total duration of voiced frames in seconds."""
    return float(voiced_mask(x, sr, hop_ms=hop_ms).sum()) * hop_ms / 1000.0


def find_silences(x: np.ndarray, sr: int, min_len_s: float = 0.25, hop_ms: float = 10.0) -> List[Tuple[float, float]]:
    """Silence intervals ``(start, end)`` in seconds that are at least ``min_len_s`` long."""
    m = voiced_mask(x, sr, hop_ms=hop_ms)
    hop = hop_ms / 1000.0
    out: List[Tuple[float, float]] = []
    i, n = 0, m.size
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            if (j - i) * hop >= min_len_s:
                out.append((i * hop, j * hop))
            i = j
        else:
            i += 1
    return out


def noise_floor_rms(x: np.ndarray, sr: int) -> float:
    """Estimated noise floor: the 10th percentile of frame RMS."""
    fr = frame_rms(x, sr)
    return float(np.percentile(fr, 10)) if fr.size else 0.0


def snr_db(x: np.ndarray, sr: int) -> float:
    """Rough signal-to-noise estimate: RMS of the voiced frames against the noise floor (10th percentile), in dB."""
    fr = frame_rms(x, sr)
    if fr.size == 0:
        return 0.0
    m = voiced_mask(x, sr)
    speech = float(np.sqrt(np.mean(np.square(fr[m])))) if m.any() else float(fr.max())
    floor = max(float(np.percentile(fr, 10)), 1e-6)
    return 20.0 * math.log10(max(speech, 1e-9) / floor)


def peak_abs(x: np.ndarray) -> float:
    """Largest absolute sample value (0 for empty input)."""
    return float(np.max(np.abs(x))) if x.size else 0.0


def apply_fade(x: np.ndarray, sr: int, ms: float = 8.0) -> np.ndarray:
    """Short fade-in/out (default 8 ms) so cuts do not click."""
    n = min(int(sr * ms / 1000), x.size // 2)
    if n <= 0:
        return x
    y = x.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def trim_silence(x: np.ndarray, sr: int, threshold: float = 0.01, keep_ms: float = 30.0, hop_ms: float = 10.0) -> np.ndarray:
    """``x`` without its own leading / trailing silence (frames below ``threshold`` RMS, about -40 dBFS), keeping
    ``keep_ms`` around the voice so no consonant is cut.  An all-quiet piece is returned unchanged."""
    if x.size == 0 or sr <= 0:
        return x
    hop = max(1, int(sr * hop_ms / 1000))
    n = x.size // hop
    if n == 0:
        return x
    rms = np.sqrt(np.mean(np.square(x[: n * hop].astype(np.float64)).reshape(n, hop), axis=1))
    loud = np.flatnonzero(rms >= threshold)
    if loud.size == 0:
        return x
    keep = int(sr * keep_ms / 1000)
    start = max(0, loud[0] * hop - keep)
    end = min(x.size, (loud[-1] + 1) * hop + keep)
    return x[start:end]


def time_stretch(x: np.ndarray, sr: int, rate: float, frame_ms: float = 30.0) -> np.ndarray:
    """Change the speed by ``rate`` (> 1 faster, < 1 slower) without changing the pitch (WSOLA overlap-add).

    Good for the small factors of :mod:`core.pace` (0.8 ... 1.2); ``rate`` close to 1 returns ``x`` unchanged."""
    if x.size == 0 or abs(rate - 1.0) < 0.01:
        return x
    x = x.astype(np.float32, copy=False)
    n = max(64, int(sr * frame_ms / 1000)) // 2 * 2
    hop_out = n // 2
    hop_in = hop_out * rate
    tol = n // 4
    win = np.hanning(n).astype(np.float32)
    if x.size < 2 * n:
        return x
    out_len = int(x.size / rate) + n
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)
    pad = np.concatenate([np.zeros(tol, np.float32), x, np.zeros(n + tol, np.float32)])
    prev = pad[tol: tol + n]
    pos_out, k = 0, 0
    while True:
        center = int(round(k * hop_in)) + tol
        if center + n + tol > pad.size or pos_out + n > out_len:
            break
        if k == 0:
            best = center
        else:   # the input frame (within +-tol) that continues the previous output frame best
            target = prev[hop_out:]
            seg = pad[center - tol: center + tol + hop_out]
            corr = np.correlate(seg, target, mode="valid")
            best = center - tol + int(np.argmax(corr))
        frame = pad[best: best + n]
        out[pos_out: pos_out + n] += frame * win
        norm[pos_out: pos_out + n] += win
        prev = pad[best: best + n]
        pos_out += hop_out
        k += 1
    norm[norm < 1e-3] = 1.0
    y = out / norm
    return np.clip(y[: int(x.size / rate)], -1.0, 1.0).astype(np.float32)
