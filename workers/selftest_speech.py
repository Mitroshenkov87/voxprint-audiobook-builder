"""``voxprint --selftest-speech``: a model-free speech check (no GPU, no downloaded model).

A sine-wave stand-in speaks one sentence, the clip is written and read back as WAV, a fake recogniser
returns the sentence, and a fake aligner places the words on the clip. Used by CI.
Exit code 0 = all steps passed.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import soundfile as sf

from core.aligner import FakeAligner
from core.asr import AsrResult, FakeASR


class _SineEngine:
    """One second of a 440 Hz tone for any text. Implements the narrator's engine protocol."""

    sample_rate = 16000
    tag = "sine-smoke"

    def synthesize(self, text: str) -> np.ndarray:
        if not text.strip():
            raise AssertionError("empty text")
        t = np.linspace(0, 1.0, self.sample_rate, endpoint=False)
        return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def close(self) -> None:
        return None


def _step_speak() -> str:
    text = "The lighthouse stood on the cliff"
    engine = _SineEngine()
    samples = engine.synthesize(text)
    engine.close()
    if samples.ndim != 1 or samples.size != engine.sample_rate:
        raise AssertionError(f"unexpected samples: shape={samples.shape}")
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "speech.wav"
        sf.write(str(wav), samples, engine.sample_rate)
        read, sr = sf.read(str(wav), always_2d=False)
        if sr != engine.sample_rate or len(read) != samples.size:
            raise AssertionError(f"wav round trip failed: sr={sr} n={len(read)}")
    heard = FakeASR([AsrResult(text)]).transcribe(samples, engine.sample_rate, "English")
    if heard.text != text:
        raise AssertionError(f"recogniser returned {heard.text!r}")
    words = FakeAligner().align(samples, engine.sample_rate, text, "English")
    if [w.word for w in words] != text.split() or words[-1].end <= words[0].start:
        raise AssertionError(f"alignment failed: {words!r}")
    return f"{len(words)} words, {samples.size} samples"


STEPS: List[Tuple[str, Callable[[], str]]] = [("speech (sine TTS, fake ASR, fake aligner)", _step_speak)]


def run() -> int:
    bad = 0
    lines = []
    for name, fn in STEPS:
        try:
            lines.append(f"OK    {name}: {fn()}")
        except Exception as exc:  # noqa: BLE001
            bad += 1
            lines.append(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    lines.append("SELFTEST_SPEECH " + ("FAILED" if bad else "OK"))
    text = "\n".join(lines)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"))
    except (OSError, ValueError):
        pass
    return 1 if bad else 0
