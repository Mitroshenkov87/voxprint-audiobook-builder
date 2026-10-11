"""Let official qwen-asr 0.0.6 import on transformers 5.16.

qwen-asr 0.0.6 was written for transformers 4.57.6. Two things break on 5.15 and newer:

* ``@check_model_inputs()`` calls the helper with no arguments. transformers 5 defines
  ``check_model_inputs(func)`` as a plain decorator.
* ``AutoConfig.register("qwen3_asr", ...)`` raises because transformers 5 already ships
  that model type. The official classes have to replace the native entry so the verified
  ``Qwen/Qwen3-ASR`` checkpoints keep using qwen-asr.

Call :func:`install` before the first ``import qwen_asr`` or ``import qwen_tts``.
``transformers`` is imported inside :func:`install`: the thin installer and the unit tests
import ``infra`` before that package exists.

Example::

    from infra.qwen_tf_compat import install
    install()
    import qwen_asr
"""
from __future__ import annotations

import importlib
from typing import Any

_INSTALLED = False


def _optional_call(original: Any) -> Any:
    """A decorator that still works when the caller writes ``@decorator()``."""
    if getattr(original, "_voxprint_optional_call", False):
        return original

    def check_model_inputs(func: Any = None) -> Any:
        if func is None:
            return lambda fn: original(fn)
        return original(func)

    check_model_inputs._voxprint_optional_call = True  # type: ignore[attr-defined]
    return check_model_inputs


def _force_exist_ok(cls: type, name: str) -> None:
    """Make ``cls.name(..., exist_ok=True)`` so a native model type can be replaced."""
    current = cls.__dict__.get(name)
    if current is None:
        return
    func = getattr(current, "__func__", current)
    if getattr(func, "_voxprint_exist_ok", False):
        return

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        kwargs["exist_ok"] = True
        return func(*args, **kwargs)

    wrapped._voxprint_exist_ok = True  # type: ignore[attr-defined]
    if isinstance(current, staticmethod):
        setattr(cls, name, staticmethod(wrapped))
    elif isinstance(current, classmethod):
        setattr(cls, name, classmethod(wrapped))
    else:
        setattr(cls, name, wrapped)


def _patch_tied_weight_keys() -> None:
    """transformers 5 reads ``_tied_weights_keys`` as a dict. qwen-tts-hf still stores a list of names."""
    modeling = importlib.import_module("transformers.modeling_utils")
    original = getattr(modeling, "_get_tied_weight_keys", None)
    if not callable(original) or getattr(original, "_voxprint_list_ok", False):
        return

    def _get_tied_weight_keys(module: Any) -> Any:
        changed = []
        for _name, submodule in module.named_modules():
            tied = getattr(submodule, "_tied_weights_keys", None)
            if isinstance(tied, (list, tuple)):
                changed.append((submodule, tied))
                submodule._tied_weights_keys = {key: key for key in tied}
        try:
            return original(module)
        finally:
            for submodule, tied in changed:
                submodule._tied_weights_keys = tied

    _get_tied_weight_keys._voxprint_list_ok = True  # type: ignore[attr-defined]
    setattr(modeling, "_get_tied_weight_keys", _get_tied_weight_keys)


def install() -> bool:
    """Patch transformers so qwen-asr 0.0.6 can be imported.

    Returns:
        True when the patches are in place. False when transformers is not installed.
        Calling it again does nothing.
    """
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        generic = importlib.import_module("transformers.utils.generic")
        auto_cfg = importlib.import_module("transformers.models.auto.configuration_auto")
        auto_factory = importlib.import_module("transformers.models.auto.auto_factory")
        processing = importlib.import_module("transformers.models.auto.processing_auto")
    except ImportError:
        return False
    original = getattr(generic, "check_model_inputs", None)
    if not callable(original):
        return False
    setattr(generic, "check_model_inputs", _optional_call(original))
    _patch_tied_weight_keys()
    _force_exist_ok(auto_cfg.AutoConfig, "register")
    _force_exist_ok(auto_factory._BaseAutoModelClass, "register")
    _force_exist_ok(processing.AutoProcessor, "register")
    _INSTALLED = True
    return True
