"""Optional "Preload models into memory at startup" (Settings; OFF by default).

When enabled, a low-priority background thread loads the heavy models that are installed - the TTS base model of the
current voice, the ASR model and SAGE - into RAM a few seconds after the start, so pressing Narrate starts at once.  The
engines take the loaded objects over from :mod:`core.model_cache` instead of reading them from disk again.

Memory rules (all from the real model files, see :func:`find_targets` / :func:`availability`):

* need = weights as they will sit in RAM (+ a small per-model overhead) + :data:`HEADROOM` for the OS, the app and a
  narration job; the option is offered only if the PC has that much RAM in total;
* loading waits until that much is actually free, and everything preloaded is dropped again when free RAM falls below the
  low-water mark (:func:`low_water`) or the user unticks the option.

Qt-free; the Studio window calls :meth:`Preloader.tick` from its one-second timer.
"""
from __future__ import annotations

import json
import logging
import math
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from core import model_cache
from infra import paths, sysinfo

log = logging.getLogger("voxprint.preload")

GB = 1024 ** 3
SETTINGS_FILE = "preload.json"
#: RAM that must stay usable next to the preloaded models: OS, the app itself, a narration job (chunk audio, encoders).
HEADROOM = 4 * GB
#: Tokenizers, Python objects and buffers of one model on top of its weights.
OVERHEAD_PER_MODEL = GB // 4
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
#: How often :meth:`Preloader.tick` really looks at memory and models (it is called every second).
CHECK_EVERY_S = 10.0
#: After the start the window and the voice list come first.
START_DELAY_S = 5.0
#: After freeing because of low memory, wait this long before warming again.
COOLDOWN_S = 120.0

OK, TOO_SMALL, LOW_NOW, NOTHING, UNKNOWN = "ok", "too_small", "low_now", "nothing", "unknown"


# --------------------------------------------------------------------------- the setting
def enabled() -> bool:
    """The user's choice (``False`` until they tick the box)."""
    try:
        data = json.loads((paths.state_dir() / SETTINGS_FILE).read_text(encoding="utf-8"))
        return bool(isinstance(data, dict) and data.get("enabled") is True)
    except (OSError, ValueError):
        return False


def set_enabled(on: bool) -> None:
    """Remember the choice."""
    f = paths.state_dir() / SETTINGS_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"enabled": bool(on)}), encoding="utf-8")


# --------------------------------------------------------------------------- what would be loaded, and how much RAM that takes
@dataclass(frozen=True)
class Target:
    """One installed model: ``kind`` is "tts", "asr" or "sage"; ``ram`` is what its weights take in RAM (bytes)."""
    kind: str
    path: Path
    ram: int


def weights_bytes(folder: Path) -> int:
    """Size of the weight files under ``folder`` (recursive: the TTS keeps its speech tokenizer in a sub-folder)."""
    total = 0
    try:
        for p in Path(folder).rglob("*"):
            if p.suffix.lower() in WEIGHT_SUFFIXES and p.is_file():
                total += p.stat().st_size
    except OSError:
        pass
    return total


def _stored_16bit(folder: Path) -> bool:
    """True if the checkpoint stores 16-bit floats (read from the safetensors header; assumed when there is none)."""
    try:
        f = next(Path(folder).glob("*.safetensors"))
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(min(n, 16 * 1024 * 1024)))
        dtypes = {v.get("dtype") for k, v in header.items() if k != "__metadata__" and isinstance(v, dict)}
        return bool(dtypes) and dtypes <= {"BF16", "F16"}
    except (StopIteration, OSError, ValueError, struct.error):
        return True


def ram_for(folder: Path, cuda: bool) -> int:
    """Bytes the model takes once loaded: as stored for CUDA (loaded in bfloat16), doubled on a CPU-only PC where a 16-bit
    checkpoint is loaded as float32 (that is how the engines run on the CPU)."""
    size = weights_bytes(folder)
    return size if cuda or not _stored_16bit(folder) else 2 * size


def find_targets(base_repo: str, cuda: bool, ready: Optional[Callable[[str], Optional[Path]]] = None,
                 sage_dir: Optional[Callable[[], Optional[Path]]] = None) -> List[Target]:
    """The installed models worth preloading: TTS base of the current voice (``base_repo``, "" = none), ASR, SAGE."""
    from infra import model_downloader as md

    ready = ready or md.ready_model_path
    sage_dir = sage_dir or _sage_dir
    found = []
    for kind, repo in (("tts", base_repo), ("asr", md.ASR_REPO)):
        path = ready(repo) if repo else None
        if path is not None:
            found.append((kind, Path(path)))
    sage = sage_dir()
    if sage is not None:
        found.append(("sage", Path(sage)))
    return [Target(kind, path, ram_for(path, cuda)) for kind, path in found]


def _sage_dir() -> Optional[Path]:
    from infra import text_models as tm

    m = tm.get("sage-ru")
    return m.local_dir if tm.state(m) == tm.STATE_READY else None


def low_water(total: int) -> int:
    """Free RAM below which preloaded models are dropped (8 % of the RAM, at least 1 GB)."""
    return max(GB, int(total * 0.08))


@dataclass
class Availability:
    """Can the models be preloaded on this PC?  ``status`` is one of OK / TOO_SMALL / LOW_NOW / NOTHING / UNKNOWN."""
    status: str
    need: int = 0          # total RAM the PC should have (models + headroom)
    models: int = 0        # RAM of the models themselves
    total: int = 0
    available: int = 0
    targets: List[Target] = field(default_factory=list)

    @property
    def offered(self) -> bool:
        """The checkbox can be ticked (it may still wait for free RAM)."""
        return self.status in (OK, LOW_NOW, UNKNOWN)


def availability(targets: Sequence[Target], total: int, available: int, held: int = 0) -> Availability:
    """Compare what ``targets`` need with the RAM of this PC.  ``held`` = bytes already preloaded (they count as free)."""
    models = sum(t.ram for t in targets) + OVERHEAD_PER_MODEL * len(targets)
    need = models + HEADROOM
    av = Availability(UNKNOWN, need, models, total, available, list(targets))
    if not targets:
        av.status = NOTHING
    elif total <= 0:
        av.status = UNKNOWN
    elif total < need:
        av.status = TOO_SMALL
    elif available + held < models + low_water(total):
        av.status = LOW_NOW
    else:
        av.status = OK
    return av


def gb(n: int, up: bool = False) -> int:
    """Whole gigabytes for the UI (rounded up for a requirement)."""
    return int(math.ceil(n / GB)) if up else int(round(n / GB))


def cuda_likely() -> bool:
    """Cheap guess for the UI thread (see :func:`core.cpu_budget.cuda_likely`); the preloader itself asks torch."""
    from core.cpu_budget import cuda_likely as likely

    return likely()


# --------------------------------------------------------------------------- loading
def _torch_cuda() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def load_tts(path: Path, cuda: bool) -> Any:
    """Qwen3-TTS base on the CPU, in the dtype the engine uses; returns ``(model, attention)`` (SDPA: flash-attn 2 cannot
    be selected while the weights are on the CPU, and it is not available for Windows anyway)."""
    import torch
    from qwen_tts import Qwen3TTSModel  # type: ignore

    dtype = torch.bfloat16 if cuda else torch.float32
    for impl in ("sdpa", "eager"):
        try:
            return Qwen3TTSModel.from_pretrained(str(path), device_map=None, dtype=dtype, attn_implementation=impl), impl
        except Exception:  # noqa: BLE001
            log.warning("preload: attention %s is not available", impl, exc_info=True)
    raise RuntimeError("the TTS model could not be loaded")


def load_asr(path: Path, cuda: bool) -> Any:
    """Qwen3-ASR on the CPU (moved to the GPU by :meth:`core.asr.Qwen3ASR.load`)."""
    import torch
    from qwen_asr import Qwen3ASRModel

    return Qwen3ASRModel.from_pretrained(str(path), dtype=torch.bfloat16 if cuda else torch.float32, device_map="cpu")


def load_sage(path: Path, cuda: bool) -> Any:
    """SAGE tokenizer + model on the CPU (as :meth:`core.text_cleanup.SageEngine._load` loads them)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path)), AutoModelForSeq2SeqLM.from_pretrained(str(path))


LOADERS: Dict[str, Callable[[Path, bool], Any]] = {"tts": load_tts, "asr": load_asr, "sage": load_sage}


def touch_files(folder: Optional[Path]) -> None:
    """Read the voice's adapter and reference clip once (a few MB): the OS keeps them cached, so opening the voice is
    instant too.  Applying the adapter itself cannot be done ahead - it is merged into the base weights."""
    if folder is None:
        return
    for name in ("adapter_model.safetensors", "adapter_config.json", "ref_sample.wav"):
        try:
            with open(Path(folder) / name, "rb") as f:
                while f.read(1 << 22):
                    pass
        except OSError:
            pass


class Preloader:
    """Keeps the installed models warm in RAM while the option is on (see the module docstring)."""

    def __init__(self, voice_fn: Callable[[], Any], *, loaders: Optional[Dict[str, Callable[[Path, bool], Any]]] = None,
                 memory: Callable[[], tuple] = sysinfo.memory, cuda: Callable[[], bool] = _torch_cuda,
                 targets: Optional[Callable[[str, bool], List[Target]]] = None, is_enabled: Callable[[], bool] = enabled,
                 clock: Callable[[], float] = time.monotonic, start_delay: float = START_DELAY_S) -> None:
        """``voice_fn()`` returns the current :class:`core.voice_library.VoiceRecord` (or ``None``); the rest is injectable."""
        self.voice_fn = voice_fn
        self.loaders = loaders or LOADERS
        self.memory = memory
        self.cuda = cuda
        self.targets = targets or find_targets
        self.is_enabled = is_enabled
        self.clock = clock
        self._not_before = clock() + start_delay
        self._next_check = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._held: Dict[tuple, int] = {}            # registry key -> RAM bytes, for the keys this preloader put there
        self._lock = threading.Lock()
        self.last_error = ""

    # ------------------------------------------------------------------ state
    @property
    def loading(self) -> bool:
        """True while the background thread runs."""
        return self._thread is not None and self._thread.is_alive()

    def held_bytes(self) -> int:
        """RAM of the models that are preloaded now and not taken by an engine yet."""
        present = set(model_cache.keys())
        with self._lock:
            self._held = {k: v for k, v in self._held.items() if k in present}
            return sum(self._held.values())

    def set_enabled(self, on: bool) -> None:
        """The user (un)ticked the box: remember it; unticking frees the models at once, ticking warms them soon."""
        set_enabled(on)
        if on:
            self._next_check = 0.0
        else:
            self.free("switched off")

    def free(self, reason: str) -> int:
        """Stop a load in progress and drop everything this preloader holds; returns the number of models dropped."""
        self._stop.set()
        with self._lock:
            mine, self._held = list(self._held), {}
        n = sum(1 for k in mine if model_cache.drop(k))
        if n:
            log.info("preloaded models freed (%s)", reason)
        return n

    # ------------------------------------------------------------------ the timer
    def tick(self, busy: bool = False) -> None:
        """Called every second by the Studio window; ``busy`` = a narration / training / download job runs."""
        now = self.clock()
        if now < self._next_check:
            return
        self._next_check = now + CHECK_EVERY_S
        if not self.is_enabled():
            if self._held or self.loading:
                self.free("switched off")
            return
        total, available = self.memory()
        if total and available < low_water(total) and (self.held_bytes() or self.loading):
            self.free("low memory")
            self._not_before = now + COOLDOWN_S
            return
        if busy or self.loading or now < self._not_before:
            return
        self.start()

    def start(self) -> Optional[threading.Thread]:
        """Warm whatever is missing in a background thread (no-op if nothing is missing or the RAM is not there)."""
        if self.loading:
            return self._thread
        voice = self.voice_fn()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._warm, args=(voice, self._stop), name="model-preload", daemon=True)
        self._thread.start()
        return self._thread

    # ------------------------------------------------------------------ the thread
    def _warm(self, voice: Any, stop: threading.Event) -> None:
        sysinfo.lower_thread_priority()
        try:
            cuda = self.cuda()
            targets = self.targets(getattr(voice, "base_model", "") or "", cuda)
            wanted = {model_cache.key(t.kind, t.path, cuda): t for t in targets}
            with self._lock:
                stale = [k for k in self._held if k not in wanted]       # e.g. another voice with another base model
            for k in stale:
                model_cache.drop(k)
                with self._lock:
                    self._held.pop(k, None)
            total, available = self.memory()
            if availability(targets, total, available, self.held_bytes()).status not in (OK, UNKNOWN):
                return
            present = set(model_cache.keys())
            for k, t in wanted.items():
                if stop.is_set():
                    return
                if k in present:
                    continue
                total, available = self.memory()
                if total and available - t.ram < low_water(total) + OVERHEAD_PER_MODEL:
                    log.info("preload: not enough free RAM for %s", t.kind)
                    return
                with model_cache.loading(k):
                    obj = self.loaders[t.kind](t.path, cuda)
                    if stop.is_set():
                        return                    # switched off / low memory while loading: drop it
                    model_cache.put(k, obj)
                    with self._lock:
                        self._held[k] = t.ram
                log.info("preloaded %s from %s", t.kind, t.path)
            if "tts" in {t.kind for t in targets}:
                touch_files(getattr(voice, "path", None))
        except Exception as exc:  # noqa: BLE001 - a failed preload only means a normal load later
            self.last_error = str(exc)
            log.warning("model preload failed", exc_info=True)
