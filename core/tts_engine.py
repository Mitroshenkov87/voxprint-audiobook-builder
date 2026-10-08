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
from typing import Callable, List, Optional, Sequence

import numpy as np

from core import adapter_strength, cpu_budget, model_cache
from core.errors import NarrationError
from core.events import ProgressCallback, Stage, noop_progress
from core.i18n import tr
from core.languages import AUTO, qwen_language
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


#: Batched synthesis: at most this many chunks per generate call; VRAM per extra sequence (KV cache + activations; measured 0.5-0.6 GB
#: for chunks up to ~150 characters on an RTX 4090, rounded up for longer ones) and a reserve that is never planned.
MAX_BATCH = 12
VRAM_PER_ITEM_GB = 0.9
VRAM_RESERVE_GB = 2.0


def _attn_candidates(attn: str, use_cuda: bool) -> List[str]:
    """Attention implementations to try, best first.  ``attn`` forces one ("eager" / "sdpa" / "flash_attention_2"); "auto" picks."""
    if attn and attn != "auto":
        return [attn] + [a for a in ("sdpa", "eager") if a != attn]
    out: List[str] = []
    if use_cuda:
        try:
            import flash_attn  # noqa: F401  (only usable when installed; not available for Windows from PyPI)

            out.append("flash_attention_2")
        except Exception:  # noqa: BLE001
            pass
    return out + ["sdpa", "eager"]


def engine_tag(voice: VoiceRecord, language: str = "", scale: Optional[float] = None) -> str:
    """Identity of voice + adapter weights + base model (+ language token, adapter strength): a retrained or replaced adapter,
    a different language token or another strength invalidates cached audio.  ``scale`` defaults to the voice's own."""
    adapter = voice.path / "adapter_model.safetensors"
    try:
        st = adapter.stat()
        fingerprint = f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        fingerprint = "?"
    raw = f"{voice.id}|{voice.base_model}|{fingerprint}|clone-v1"
    # Older versions always spoke with the voice's own language and did not put it into the tag.  The token joins the tag
    # only when it differs from that, so chunks cached by older versions stay valid whenever they were made the same way.
    token = qwen_language(language)
    if token and token != (qwen_language(voice.language) or AUTO):
        raw += f"|lang={token}"
    strength = voice.adapter_scale if scale is None else adapter_strength.clamp(scale, 1.0)
    if not adapter_strength.is_full(strength):           # 1.0 adds nothing: caches of older voices stay valid
        raw += f"|scale={strength:.2f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class Qwen3AdapterEngine:
    """Synthesizes with a LoRA voice.  Create it lazily (loading takes a while and ~4-7 GB of VRAM)."""

    def __init__(self, voice: VoiceRecord, base_dir: Path, language: str = "", device: str = "auto", attn: str = "auto",
                 adapter_scale: Optional[float] = None, merge: bool = True) -> None:
        """Load the model, apply the adapter and prepare the voice-clone prompt.

        ``adapter_scale`` overrides the voice's strength (:mod:`core.adapter_strength`).  ``merge=False`` keeps the adapter
        separate (a little slower) so :meth:`set_adapter_scale` / :meth:`switch_adapter` can change it without reloading the
        base model - used by the quick preview and the automatic checkpoint pick."""
        import torch
        from peft import PeftModel
        from qwen_tts import Qwen3TTSModel  # type: ignore

        self.adapter_scale = voice.adapter_scale if adapter_scale is None else adapter_strength.clamp(adapter_scale, 1.0)
        self.tag = engine_tag(voice, language, self.adapter_scale)
        # Exact Qwen3-TTS name ("English", "Russian" ... or "Auto"): voice.json stores codes like "ru" since schema 3
        self.language = qwen_language(language) or qwen_language(voice.language) or AUTO
        use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
        dtype = torch.bfloat16 if use_cuda else torch.float32
        # torch's CPU thread pool would otherwise take every core and fight the narration pool (FLAC writes, chapter
        # joins, ffmpeg encodes) - on a GPU it only needs a few; on the CPU it gets nearly all (core/cpu_budget.py).
        self._threads_before = torch.get_num_threads()
        torch.set_num_threads(cpu_budget.plan(use_cuda).torch_threads)
        self._q = None
        self.attn = ""
        # "Preload models at startup" (infra/preload.py) may hold this base model in RAM already: take it over instead of
        # reading it from disk again (it was loaded on the CPU in the right dtype; only the move to the GPU is left).
        pre = model_cache.take(model_cache.key("tts", base_dir, use_cuda)) if attn == "auto" else None
        if pre is not None:
            try:
                self._q, self.attn = (model_cache.to_device(pre[0], "cuda:0") if use_cuda else pre[0]), pre[1]
                log.info("using the preloaded base model (%s)", self.attn)
            except Exception:  # noqa: BLE001 - e.g. out of VRAM while moving: load normally below
                log.warning("the preloaded model could not be used; loading it again", exc_info=True)
                self._q = None
            pre = None
            if self._q is None and use_cuda:
                torch.cuda.empty_cache()         # whatever had already been moved before the failure
        # Attention backend: flash-attn 2 (if installed) > SDPA (PyTorch fused kernels) > eager.  Every step falls back to the
        # next one if the model refuses to load with it, so an unusual GPU/driver never blocks narration.
        for impl in ([] if self._q is not None else _attn_candidates(attn, use_cuda)):
            try:
                self._q = Qwen3TTSModel.from_pretrained(str(base_dir), device_map="cuda:0" if use_cuda else None, dtype=dtype,
                                                        attn_implementation=impl)
                self.attn = impl
                break
            except Exception:  # noqa: BLE001
                log.warning("attention backend %s is not available; trying the next one", impl, exc_info=True)
                self._q = None
        if self._q is None:
            raise NarrationError(tr("err.voice_invalid"), details="the model could not be loaded")
        peft_talker = PeftModel.from_pretrained(self._q.model.talker, str(voice.path))
        if not adapter_strength.is_full(self.adapter_scale):       # 1.0 = as loaded; older voices take exactly the old path
            adapter_strength.apply(peft_talker, self.adapter_scale)  # before the merge: the merged delta carries the scale
        talker, self._peft, self._adapter_no = peft_talker, peft_talker, 0
        if merge:
            try:
                talker, self._peft = peft_talker.merge_and_unload(), None   # faster inference; fall back to the unmerged adapter
            except Exception:  # noqa: BLE001
                log.warning("adapter merge failed; using the unmerged adapter", exc_info=True)
        self._q.model.talker = talker
        self._q.model.eval()
        ref_audio = voice.path / "ref_sample.wav"
        if not ref_audio.is_file() or not voice.ref_text:
            raise NarrationError(tr("err.voice_invalid"), details="reference clip or its text is missing")
        self._prompt = self._q.create_voice_clone_prompt(ref_audio=str(ref_audio), ref_text=voice.ref_text)
        self.sample_rate = 24000

    def set_adapter_scale(self, scale: float) -> None:
        """Change the adapter strength (only for an engine created with ``merge=False``)."""
        if self._peft is None:
            raise RuntimeError("the adapter is merged; create the engine with merge=False")
        self.adapter_scale = adapter_strength.clamp(scale, 1.0)
        adapter_strength.apply(self._peft, self.adapter_scale)

    def switch_adapter(self, folder: Path) -> None:
        """Replace the adapter weights by another checkpoint of the same voice (``merge=False`` engines); keeps the strength."""
        if self._peft is None:
            raise RuntimeError("the adapter is merged; create the engine with merge=False")
        old = self._peft.active_adapter
        self._adapter_no += 1
        name = f"candidate_{self._adapter_no}"
        self._peft.load_adapter(str(folder), adapter_name=name)
        self._peft.set_adapter(name)
        self._peft.delete_adapter(old)            # only one candidate in memory at a time
        adapter_strength.apply(self._peft, self.adapter_scale)

    def speaker_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Embedding of ``audio`` by the model's own speaker encoder (used for the voice-similarity check)."""
        import torch

        from core.audio_utils import resample

        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sr != 24000:
            x = resample(x, sr, 24000)
        with torch.inference_mode():
            emb = self._q.model.extract_speaker_embedding(audio=x, sr=24000)
        return emb.detach().float().cpu().numpy().reshape(-1)

    def synthesize(self, text: str) -> np.ndarray:
        """Mono float32 samples for ``text``."""
        import torch

        with torch.inference_mode():
            wavs, sr = self._q.generate_voice_clone(text=text, language=self.language, voice_clone_prompt=self._prompt,
                                                    max_new_tokens=max_tokens_for(text))
        self.sample_rate = int(sr)
        return np.asarray(wavs[0], dtype=np.float32).reshape(-1)

    def synthesize_batch(self, texts: Sequence[str]) -> List[np.ndarray]:
        """Several chunks in ONE generate call (the GPU is mostly idle with a single sequence, so a batch is almost free).

        Texts of similar length should be batched together (the narrator sorts them): the batch runs until its longest item ends.
        """
        import torch

        n = len(texts)
        with torch.inference_mode():
            wavs, sr = self._q.generate_voice_clone(text=list(texts), language=[self.language] * n, voice_clone_prompt=self._prompt,
                                                    max_new_tokens=max(max_tokens_for(t) for t in texts))
        self.sample_rate = int(sr)
        return [np.asarray(w, dtype=np.float32).reshape(-1) for w in wavs]

    def max_batch(self) -> int:
        """How many chunks fit into the free VRAM at once (``VRAM_PER_ITEM_GB`` each, capped); 1 on CPU or if unknown."""
        try:
            import torch

            if not torch.cuda.is_available():
                return 1
            free = torch.cuda.mem_get_info()[0] / 1024 ** 3
            return int(max(1, min(MAX_BATCH, (free - VRAM_RESERVE_GB) // VRAM_PER_ITEM_GB)))
        except Exception:  # noqa: BLE001
            return 1

    def close(self) -> None:
        """Free the model and the GPU cache; give torch its CPU threads back."""
        self._q = None
        self._peft = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            torch.set_num_threads(getattr(self, "_threads_before", torch.get_num_threads()))
        except Exception:  # noqa: BLE001
            pass


def make_engine_factory(voice: VoiceRecord, language: str = "", progress: ProgressCallback = noop_progress,
                        ensure_model: Optional[Callable[..., Path]] = None, **engine_kw) -> Callable[[], Qwen3AdapterEngine]:
    """A factory the narrator calls when the first chunk needs synthesis: finds/downloads the base model, then loads."""
    def factory() -> Qwen3AdapterEngine:
        from infra import model_downloader as md

        repo = voice.base_model
        if not repo:
            from infra.vram_optimizer import detect_gpu, plan_training

            repo = plan_training(detect_gpu(), 100).base_model
        base = (ensure_model or md.ensure_model)(repo, lambda s, f, m: progress(Stage.MODEL, f, m))
        return Qwen3AdapterEngine(voice, Path(base), language, **engine_kw)
    return factory
