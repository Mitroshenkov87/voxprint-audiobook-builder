"""Suggest a voice type (male / female) from the pitch of a recording.  Qt-free; the answer is only a hint."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

#: Median F0 (Hz) below which a voice is suggested as male, and from which as female; in between (and above ~300 Hz, which
#: may be a child or an octave error) nothing is suggested.
MALE_BELOW_HZ = 160.0
FEMALE_FROM_HZ = 180.0
FEMALE_UP_TO_HZ = 330.0
ANALYSE_SECONDS = 8.0


def type_from_f0(f0: float) -> str:
    """``male`` / ``female`` or ``""`` (unknown / ambiguous) for a median pitch in Hz."""
    if f0 <= 0:
        return ""
    if f0 < MALE_BELOW_HZ:
        return "male"
    if FEMALE_FROM_HZ <= f0 <= FEMALE_UP_TO_HZ:
        return "female"
    return ""


def suggest_voice_type(audio: Path, seconds: float = ANALYSE_SECONDS) -> Tuple[str, float]:
    """``(type, median F0 in Hz)`` for up to ``seconds`` of speech from the middle of the file; ``("", 0.0)`` if unreadable."""
    try:
        import numpy as np
        import soundfile as sf

        from core.voice_check import f0_median

        with sf.SoundFile(str(audio)) as f:
            sr = f.samplerate
            total = f.frames
            n = min(total, int(seconds * sr))
            f.seek(max(0, total // 2 - n // 2))
            x = f.read(n, dtype="float32", always_2d=True).mean(axis=1)
        if sr > 16000:                                   # autocorrelation is slow: 16 kHz is plenty for the pitch
            step = sr // 16000
            x, sr = np.ascontiguousarray(x[::step]), sr // step
        f0 = f0_median(x, sr)
        return type_from_f0(f0), float(f0)
    except Exception:  # noqa: BLE001 - a hint must never get in the way
        return "", 0.0
