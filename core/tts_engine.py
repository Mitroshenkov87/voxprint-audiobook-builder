"""The real speech engine: Qwen3-TTS Base + the voice's LoRA adapter, cloning from the voice's reference clip.

Loads the base model that the adapter was trained on (``voice.base_model``), puts the adapter on the talker exactly like
the consumer in Alexandria does (``PeftModel.from_pretrained(model.model.talker, adapter_dir)``), merges it for speed
and synthesizes with ``generate_voice_clone`` using the voice's ``ref_sample.wav`` + its text.

NOT VERIFIED ON A REAL GPU: this module follows the public ``qwen-tts`` API and the adapter contract used by training,
but could only be reviewed, not run, in the Linux test environment (it needs the 1.7B/0.6B weights, qwen-tts and CUDA).
Unit tests use a fake engine behind :class:`core.narration.TTSEngine`.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from core.errors import NarrationError
from core.events import ProgressCallback, Stage, noop_progress
from core.i18n import tr
from core.voice_library import VoiceRecord

log = logging.getLogger("voxprint.tts")


#: The 12 Hz speech tokenizer produces ~12.5 codec frames per second of audio.
FRAMES_PER_SECOND = 12.5
#: Narration speed is ~11-16 characters per second; allow generous slack, but never let a chunk run away.
MAX_SECONDS_PER_CHAR = 0.19
MIN_TOKENS = 48
MAX_TOKENS = 2048


def max_tokens_for(text: str) -> int:
    """Upper bound of codec frames for ``text``.

    Without a bound the model may fail to emit its end-of-speech token (typical for very short chunks such as a one-word
    chapter title) and keep generating up to the library default of 2048 frames (~160 s of babble, ~9 minutes of GPU
    time).  Found in the first real narration test on an RTX 4090.
    """
    seconds = 3.0 + MAX_SECONDS_PER_CHAR * len(text.strip())
    return max(MIN_TOKENS, min(MAX_TOKENS, int(round(seconds * FRAMES_PER_SECOND))))


def engine_tag(voice: VoiceRecord) -> str:
    """Identity of voice + adapter weights + base model: a retrained or replaced adapter invalidates cached audio."""
    adapter = voice.path / "adapter_model.safetensors"
    try:
        st = adapter.stat()
        fingerprint = f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        fingerprint = "?"
    raw = f"{voice.id}|{voice.base_model}|{fingerprint}|clone-v1"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class Qwen3AdapterEngine:
    """Synthesizes with a LoRA voice.  Create it lazily (loading takes a while and ~4-7 GB of VRAM)."""

    def __init__(self, voice: VoiceRecord, base_dir: Path, language: str = "", device: str = "auto") -> None:
        """Load the model, apply the adapter and prepare the voice-clone prompt."""
        import torch
        from peft import PeftModel
        from qwen_tts import Qwen3TTSModel  # type: ignore

        self.tag = engine_tag(voice)
        self.language = (language or voice.language or "auto").strip().capitalize()
        use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
        dtype = torch.bfloat16 if use_cuda else torch.float32
        self._q = Qwen3TTSModel.from_pretrained(str(base_dir), device_map="cuda:0" if use_cuda else None, dtype=dtype,
                                                attn_implementation="eager")
        talker = PeftModel.from_pretrained(self._q.model.talker, str(voice.path))
        try:
            talker = talker.merge_and_unload()          # faster inference; fall back to the unmerged adapter
        except Exception:  # noqa: BLE001
            log.warning("adapter merge failed; using the unmerged adapter", exc_info=True)
        self._q.model.talker = talker
        self._q.model.eval()
        ref_audio = voice.path / "ref_sample.wav"
        if not ref_audio.is_file() or not voice.ref_text:
            raise NarrationError(tr("err.voice_invalid"), details="reference clip or its text is missing")
        self._prompt = self._q.create_voice_clone_prompt(ref_audio=str(ref_audio), ref_text=voice.ref_text)
        self.sample_rate = 24000

    def synthesize(self, text: str) -> np.ndarray:
        """Mono float32 samples for ``text``."""
        import torch

        with torch.inference_mode():
            wavs, sr = self._q.generate_voice_clone(text=text, language=self.language, voice_clone_prompt=self._prompt,
                                                    max_new_tokens=max_tokens_for(text))
        self.sample_rate = int(sr)
        return np.asarray(wavs[0], dtype=np.float32).reshape(-1)

    def close(self) -> None:
        """Free the model and the GPU cache."""
        self._q = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def make_engine_factory(voice: VoiceRecord, language: str = "", progress: ProgressCallback = noop_progress,
                        ensure_model: Optional[Callable[..., Path]] = None) -> Callable[[], Qwen3AdapterEngine]:
    """A factory the narrator calls when the first chunk needs synthesis: finds/downloads the base model, then loads."""
    def factory() -> Qwen3AdapterEngine:
        from infra import model_downloader as md

        repo = voice.base_model
        if not repo:
            from infra.vram_optimizer import detect_gpu, plan_training

            repo = plan_training(detect_gpu(), 100).base_model
        base = (ensure_model or md.ensure_model)(repo, lambda s, f, m: progress(Stage.MODEL, f, m))
        return Qwen3AdapterEngine(voice, Path(base), language)
    return factory
