"""Reading the speaker's spoken consent from the end of a recording (Qt-free; the recogniser is injected)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from core import audio_utils as au
from core import consent
from core.asr import BaseASR, split_at_pauses

log = logging.getLogger("voxprint.consent")

TAIL_SEC = 45.0     # the statement is read at the very end; this much of the recording is recognised and (optionally) kept


def tail_of(audio_path, seconds: float = TAIL_SEC) -> Tuple[np.ndarray, int]:
    """The last ``seconds`` of a recording as mono float32 at 16 kHz."""
    x = au.load_audio_multi(audio_path, (16000,))[16000]
    return x[-int(seconds * 16000):], 16000


def detect_from_recording(audio_path, asr: BaseASR, language: Optional[str] = None,
                          seconds: float = TAIL_SEC) -> Tuple[consent.Parsed, np.ndarray]:
    """Recognise the end of ``audio_path`` and parse the consent statement; returns (parsed, the tail audio)."""
    x, sr = tail_of(audio_path, seconds)
    texts: List[str] = []
    asr.load()
    try:
        for a, b in split_at_pauses(x, sr, max_s=20.0, min_s=4.0):
            if (b - a) / sr >= 1.0:
                texts.append(asr.transcribe(x[a:b], sr, language).text.strip())
    finally:
        asr.unload()
    return consent.parse_statement(" ".join(t for t in texts if t)), x


def save_clip(folder, samples: np.ndarray, sr: int = 16000) -> str:
    """Write the statement next to the adapter; returns the file name."""
    au.write_wav(Path(folder) / consent.CLIP_NAME, samples, sr)
    return consent.CLIP_NAME
