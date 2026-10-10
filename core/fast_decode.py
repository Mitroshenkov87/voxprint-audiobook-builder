"""Optional fast decode path: Qwen3-TTS through faster-qwen3-tts (CUDA Graphs), one chunk at a time.

Off by default (setting ``gpu.fast_decode``, :mod:`infra.gpu_prefs`) until ``voxprint bench`` has compared it with the
batched path on a real GPU.  The Movie Dubber measured 0.6x -> 3.0x realtime with it on an RTX 4090 Laptop (one sequence);
our batched path already reaches ~2-3x realtime with 8-12 chunks per generate call, so which one wins depends on the card.

Package: ``faster-qwen3-tts`` 0.3.x (MIT, Andres Marafioti, https://github.com/andimarafioti/faster-qwen3-tts).  0.3.x is the
last line written for ``qwen-tts`` 0.1.1 + ``transformers`` 4.57 (our pins); 0.4+ needs ``qwen-tts-hf`` + transformers 5 and
must not be installed.  Pure Python (``torch.cuda.CUDAGraph``; no Triton, no flash-attn), so it works with torch 2.11 cu130 on
Windows.  It is an optional module: when it is missing or the graphs cannot be captured, narration silently keeps the batched
path.

The LoRA adapter is merged into the talker BEFORE the graphs are built (a graph replays the weights it captured).
The code predictor graph samples with the model's defaults (top-k 50, temperature 0.9); the per-chunk check's regeneration
with other sampling settings therefore uses the normal path.

:func:`install_rope_compat` makes the "default" RoPE initialiser tolerant of configs without ``rope_theta`` (the codec's
``MimiConfig`` under transformers >= 5.18, found by the Movie Dubber).  Harmless on transformers 4.57.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, TypedDict

import numpy as np

log = logging.getLogger("voxprint.fast_decode")

PACKAGE = "faster-qwen3-tts"
#: Versions written for qwen-tts 0.1.1 + transformers 4.x (0.4.0 moved to qwen-tts-hf + transformers 5).
SUPPORTED = ">=0.3.0,<0.4"
#: Static KV cache of the talker graph (prompt = reference codes + texts, plus the generated frames).
MAX_SEQ_LEN = 3072
#: Room kept for the prompt when the number of new frames is capped to the static cache.
PROMPT_ROOM = 768
DEFAULT_ROPE_THETA = 10000.0


# ------------------------------------------------------------------------------------------------- RoPE compatibility
class _RopeConfigView:
    """A config seen through a fallback ``rope_theta`` (from ``rope_parameters`` / ``rope_scaling``, else 10000)."""

    def __init__(self, config: Any) -> None:
        object.__setattr__(self, "_cfg", config)

    def __getattr__(self, name: str) -> Any:
        cfg = object.__getattribute__(self, "_cfg")
        if name == "rope_theta":
            for holder in ("rope_parameters", "rope_scaling"):
                d = getattr(cfg, holder, None)
                if isinstance(d, dict) and d.get("rope_theta") is not None:
                    return float(d["rope_theta"])
            return DEFAULT_ROPE_THETA
        return getattr(cfg, name)


def _has_rope_theta(config: Any) -> bool:
    try:
        return getattr(config, "rope_theta", None) is not None
    except Exception:  # noqa: BLE001
        return False


def tolerant(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a "default" RoPE initialiser so a config without ``rope_theta`` gets the fallback instead of an AttributeError."""
    if getattr(fn, "_voxprint_tolerant", False):
        return fn

    def default_rope(config: Any = None, *args: Any, **kwargs: Any) -> Any:
        if config is not None and not _has_rope_theta(config):
            config = _RopeConfigView(config)
        return fn(config, *args, **kwargs)

    default_rope._voxprint_tolerant = True  # type: ignore[attr-defined]
    default_rope.__wrapped__ = fn  # type: ignore[attr-defined]
    return default_rope


def install_rope_compat() -> List[str]:
    """Register the tolerant initialiser in ``transformers``' ``ROPE_INIT_FUNCTIONS["default"]`` and in
    ``qwen_tts._transformers_compat._default_rope_parameters`` (qwen-tts-hf) - before any model is loaded.  Idempotent;
    returns where it was installed."""
    done: List[str] = []
    try:
        from transformers import modeling_rope_utils as mru

        table = getattr(mru, "ROPE_INIT_FUNCTIONS", None)
        if isinstance(table, dict) and callable(table.get("default")):
            table["default"] = tolerant(table["default"])
            done.append("transformers.ROPE_INIT_FUNCTIONS")
    except Exception:  # noqa: BLE001
        pass
    try:
        import importlib

        compat = importlib.import_module("qwen_tts._transformers_compat")
        fn = getattr(compat, "_default_rope_parameters", None)
        if callable(fn):
            setattr(compat, "_default_rope_parameters", tolerant(fn))
            done.append("qwen_tts._transformers_compat")
    except Exception:  # noqa: BLE001 - qwen-tts 0.1.1 has no such module
        pass
    return done


# ------------------------------------------------------------------------------------------------- availability
def installed_version() -> Optional[str]:
    """The installed ``faster-qwen3-tts`` version, or None when the package is missing."""
    try:
        from importlib.metadata import version

        return version(PACKAGE)
    except Exception:  # noqa: BLE001
        return None


def check() -> Tuple[bool, str]:
    """``(usable, reason)`` without loading a model: package present in a supported version, CUDA available."""
    ver = installed_version()
    if ver is None:
        return False, f"{PACKAGE} is not installed"
    try:
        from packaging.specifiers import SpecifierSet

        if ver not in SpecifierSet(SUPPORTED):
            return False, f"{PACKAGE} {ver} is not supported (need {SUPPORTED})"
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA Graphs need an NVIDIA GPU"
    except Exception as exc:  # noqa: BLE001
        return False, f"PyTorch: {exc}"
    return True, f"{PACKAGE} {ver}"


def cap_new_tokens(n: int, max_seq_len: int = MAX_SEQ_LEN) -> int:
    """New frames that still fit into the static cache next to the prompt."""
    return max(1, min(int(n), max_seq_len - PROMPT_ROOM))


# ------------------------------------------------------------------------------------------------- the decoder
class GraphDecoder:
    """CUDA Graph decoder around an already loaded (and adapter-merged) ``qwen_tts.Qwen3TTSModel``."""

    def __init__(self, q: Any, device: str = "cuda:0", dtype: Any = None, max_seq_len: int = MAX_SEQ_LEN,
                 warmup: bool = True) -> None:
        """Wrap the loaded model ``q`` in talker and predictor CUDA graphs."""
        import torch
        from faster_qwen3_tts import FasterQwen3TTS
        from faster_qwen3_tts.predictor_graph import PredictorGraph
        from faster_qwen3_tts.talker_graph import TalkerGraph

        install_rope_compat()
        dtype = dtype or torch.bfloat16
        talker = q.model.talker
        talker_config = q.model.config.talker_config
        predictor = talker.code_predictor
        pred = PredictorGraph(predictor, predictor.model.config, talker_config.hidden_size, device=device, dtype=dtype,
                              do_sample=True, top_k=50, temperature=0.9)
        tg = TalkerGraph(talker.model, talker_config, device=device, dtype=dtype, max_seq_len=max_seq_len)
        self.max_seq_len = max_seq_len
        self.fast = FasterQwen3TTS(base_model=q, predictor_graph=pred, talker_graph=tg, device=device, dtype=dtype,
                                   max_seq_len=max_seq_len)
        if warmup:
            self.fast.warmup(prefill_len=100)

    def synthesize(self, text: str, language: str, prompt: Any, max_new_tokens: int) -> Tuple[np.ndarray, int]:
        """One chunk; ``prompt`` = the voice-clone prompt items of ``create_voice_clone_prompt`` (ICL with the reference)."""
        import torch

        ref_text = ""
        try:
            ref_text = prompt[0].ref_text or ""
        except Exception:  # noqa: BLE001
            pass
        with torch.inference_mode():
            wavs, sr = self.fast.generate_voice_clone(text=text, language=language, ref_text=ref_text,
                                                      voice_clone_prompt=list(prompt),
                                                      max_new_tokens=cap_new_tokens(max_new_tokens, self.max_seq_len))
        return np.asarray(wavs[0], dtype=np.float32).reshape(-1), int(sr)


# ------------------------------------------------------------------------------------------------- optional install
#: The pinned wheel (pure Python, MIT).  Not part of the installer's locked runtime: it is fetched only on request
#: (``voxprint bench --install-graphs``) into the packages folder that is put on ``sys.path`` at start.
class _WheelSpec(TypedDict):
    version: str
    file: str
    url: str
    sha256: str
    size: int


WHEEL: _WheelSpec = {
    "version": "0.3.2",
    "file": "faster_qwen3_tts-0.3.2-py3-none-any.whl",
    "url": "https://files.pythonhosted.org/packages/ea/c7/49525c63af1644d6706d415c61558c8013c675fbd5ef50bc5e4b7c2641dc/"
           "faster_qwen3_tts-0.3.2-py3-none-any.whl",
    "sha256": "b17b2839d56cb178f5c382d85214ae62852b5728830bfeda066e6a15c832a364",
    "size": 42924,
}


def install_wheel(target: Optional[Path] = None, fetch: Optional[Callable[[str], bytes]] = None) -> Path:
    """Download the pinned wheel, check size + SHA-256 and unpack it into ``target`` (default: the packages folder).

    The wheel only holds ``faster_qwen3_tts/`` and its ``dist-info``; nothing else is replaced, no dependency is installed
    (they are already part of Voxprint).  ``ValueError`` on a wrong size or hash (nothing is unpacked then)."""
    import hashlib
    import io
    import zipfile

    if target is None:
        from infra import paths

        target = paths.packages_dir()
    target = Path(target)
    if fetch is None:
        def fetch(url: str) -> bytes:
            from infra.net import urlopen

            with urlopen(url, timeout=60) as r:
                return r.read()
    data = fetch(WHEEL["url"])
    if len(data) != WHEEL["size"] or hashlib.sha256(data).hexdigest() != WHEEL["sha256"]:
        raise ValueError(f"{WHEEL['file']}: size or SHA-256 does not match; nothing installed")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        if any(not (n.startswith("faster_qwen3_tts/") or n.startswith("faster_qwen3_tts-")) or ".." in n for n in names):
            raise ValueError(f"{WHEEL['file']}: unexpected content; nothing installed")
        target.mkdir(parents=True, exist_ok=True)
        zf.extractall(target)
    return target / "faster_qwen3_tts"
