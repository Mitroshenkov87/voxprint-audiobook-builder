"""Аудио: чтение через pydub/ffmpeg (с запасным soundfile), ресемплинг, запись WAV, энергия, тишина."""
from __future__ import annotations

from core.i18n import tr
import math
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from core.errors import AudioReadError

_FFMPEG_READY = False


def ensure_ffmpeg() -> Optional[str]:
    """Finds ffmpeg and configures pydub.  Order: next to the exe; ffmpeg on PATH ONLY if `ffmpeg -version` works;
    the pinned LGPL build that Voxprint downloaded itself (infra/assets.py); the imageio-ffmpeg wheel (GPL build,
    offline fallback)."""
    global _FFMPEG_READY
    candidates: List[str] = []
    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    base = Path(getattr(sys, "_MEIPASS", Path(sys.argv[0]).resolve().parent))
    for d in (base, Path(sys.executable).resolve().parent):
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
            import imageio_ffmpeg  # type: ignore

            path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            path = None
    if path:
        try:
            from pydub import AudioSegment

            AudioSegment.converter = path
            # pydub ищет ffprobe отдельно; он нужен лишь для mediainfo - не требуем
        except Exception:  # noqa: BLE001
            pass
        _FFMPEG_READY = True
    return path


def to_mono_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Качественный полифазный ресемплинг (scipy)."""
    if sr_from == sr_to or x.size == 0:
        return x.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = math.gcd(int(sr_from), int(sr_to))
    y = resample_poly(x.astype(np.float64), int(sr_to) // g, int(sr_from) // g)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def _read_with_ffmpeg(path: Path) -> Tuple[np.ndarray, int]:
    """Декодирует через ffmpeg напрямую во временный WAV. Не требует ffprobe (его нет в колесе imageio-ffmpeg, а pydub
    для m4a/aac/mp4 его вызывает - иначе [WinError 2])."""
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
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono_float(data), int(sr)


def load_audio(path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Читает любой аудиофайл -> (mono float32 [-1,1], sr). Сначала soundfile для wav/flac, иначе pydub."""
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
    """Читает файл один раз и возвращает {sr: samples} для нужных частот (без двойного ресемплинга)."""
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
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(samples, -1.0, 1.0), sr, subtype="PCM_16")


def duration(samples: np.ndarray, sr: int) -> float:
    return len(samples) / float(sr)


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))


def frame_rms(x: np.ndarray, sr: int, frame_ms: float = 20.0, hop_ms: float = 10.0) -> np.ndarray:
    """RMS по кадрам. Возвращает массив длиной n_frames; кадр i начинается в i*hop."""
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
    """Булев массив по кадрам: есть речь. Порог адаптивный (шум +12 дБ или пик -40 дБ)."""
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
    """(начало первой речи, конец последней речи) в секундах внутри x."""
    m = voiced_mask(x, sr, hop_ms=hop_ms)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return 0.0, len(x) / float(sr)
    return idx[0] * hop_ms / 1000.0, idx[-1] * hop_ms / 1000.0 + 0.02


def voiced_seconds(x: np.ndarray, sr: int, hop_ms: float = 10.0) -> float:
    return float(voiced_mask(x, sr, hop_ms=hop_ms).sum()) * hop_ms / 1000.0


def find_silences(x: np.ndarray, sr: int, min_len_s: float = 0.25, hop_ms: float = 10.0) -> List[Tuple[float, float]]:
    """Интервалы тишины (start, end) в секундах длиной >= min_len_s."""
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
    fr = frame_rms(x, sr)
    return float(np.percentile(fr, 10)) if fr.size else 0.0


def snr_db(x: np.ndarray, sr: int) -> float:
    """Оценка отношения сигнал/шум: RMS речевых кадров против пола шума (10-й перцентиль)."""
    fr = frame_rms(x, sr)
    if fr.size == 0:
        return 0.0
    m = voiced_mask(x, sr)
    speech = float(np.sqrt(np.mean(np.square(fr[m])))) if m.any() else float(fr.max())
    floor = max(float(np.percentile(fr, 10)), 1e-6)
    return 20.0 * math.log10(max(speech, 1e-9) / floor)


def peak_abs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if x.size else 0.0


def apply_fade(x: np.ndarray, sr: int, ms: float = 8.0) -> np.ndarray:
    """Короткие fade-in/out, чтобы не было щелчков на границах вырезки."""
    n = min(int(sr * ms / 1000), x.size // 2)
    if n <= 0:
        return x
    y = x.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y
