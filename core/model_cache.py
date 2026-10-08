"""Models preloaded into RAM, handed over to the engine that needs them (Qt-free; filled by :mod:`infra.preload`).

The registry holds at most one object per key, e.g. ``("tts", "<base dir>", "bfloat16")``.  :func:`take` *moves* the object
out (ownership transfer): the TTS engine merges the voice adapter into the base weights and the recognisers move to the
GPU, so a preloaded object can only be used once.  The preloader warms it again when the app is idle.

A key that is being loaded right now is announced with :func:`loading`; :func:`take` then waits for that load instead of
reading the same gigabytes from disk a second time (pressing Narrate a few seconds after the start must not double the RAM).
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Hashable, Iterator, List, Optional

log = logging.getLogger("voxprint.model_cache")

_lock = threading.Lock()
_items: Dict[Hashable, Any] = {}
_loading: Dict[Hashable, threading.Event] = {}
#: How long :func:`take` waits for a load in progress (loading the same model again would take about as long).
WAIT_S = 600.0


def put(key: Hashable, obj: Any) -> None:
    """Store a loaded object (replaces an older one with the same key)."""
    with _lock:
        _items[key] = obj


@contextmanager
def loading(key: Hashable) -> Iterator[None]:
    """Mark ``key`` as being loaded for the duration of the ``with`` block (``take`` waits for it)."""
    ev = threading.Event()
    with _lock:
        _loading[key] = ev
    try:
        yield
    finally:
        with _lock:
            if _loading.get(key) is ev:
                del _loading[key]
        ev.set()


def take(key: Hashable, wait_s: float = WAIT_S) -> Optional[Any]:
    """Remove and return the object stored under ``key`` (``None`` if there is none); waits for a load in progress."""
    with _lock:
        ev = _loading.get(key) if key not in _items else None
    if ev is not None:
        log.info("waiting for the preload of %s", key)
        ev.wait(wait_s)
    with _lock:
        return _items.pop(key, None)


def keys() -> List[Hashable]:
    """Keys of the objects held now."""
    with _lock:
        return list(_items)


def busy() -> bool:
    """True while some key is being loaded."""
    with _lock:
        return bool(_loading)


def drop(key: Hashable) -> bool:
    """Forget one object (the memory is freed once nobody else references it)."""
    with _lock:
        found = _items.pop(key, None) is not None
    if found:
        _collect()
    return found


def clear() -> int:
    """Forget every preloaded object; returns how many there were."""
    with _lock:
        n = len(_items)
        _items.clear()
    if n:
        _collect()
    return n


def _collect() -> None:
    import gc

    gc.collect()


def to_device(wrapper: Any, device: str) -> Any:
    """Move a qwen-tts / qwen-asr wrapper loaded on the CPU to ``device`` (e.g. ``"cuda:0"``) and return it.

    The wrappers cache the device they were loaded on (``wrapper.device``, and the speech tokenizer inside the TTS model has
    its own ``device``); ``nn.Module.to`` alone would leave those stale and the inputs would be built on the CPU."""
    import torch

    dev = torch.device(device)
    inner = getattr(wrapper, "model", None)
    if inner is not None:
        inner.to(dev)
        tok = getattr(inner, "speech_tokenizer", None)
        if tok is not None and getattr(tok, "model", None) is not None:
            tok.model.to(dev)
            tok.device = dev
    if hasattr(wrapper, "device"):
        wrapper.device = dev
    return wrapper


def key(kind: str, path: Any, cuda: bool) -> tuple:
    """Registry key shared by the preloader and the engines: model kind, its folder, and whether it is meant for CUDA
    (the dtype follows: bfloat16 for CUDA, float32 on the CPU)."""
    import os
    from pathlib import Path

    try:
        folder = str(Path(path).resolve())
    except OSError:
        folder = str(path)
    return kind, os.path.normcase(folder), "cuda" if cuda else "cpu"
