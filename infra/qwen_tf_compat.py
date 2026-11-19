"""Let official qwen-asr 0.0.6 import and run on transformers 5.16.

qwen-asr 0.0.6 was written for transformers 4.57.6. Several things break on 5.15 and newer:

Import time:
* ``@check_model_inputs()`` calls the helper with no arguments. transformers 5 defines
  ``check_model_inputs(func)`` as a plain decorator.
* ``AutoConfig.register("qwen3_asr", ...)`` raises because transformers 5 already ships
  that model type. The official classes have to replace the native entry so the verified
  ``Qwen/Qwen3-ASR`` checkpoints keep using qwen-asr.

Load/run time:
* transformers 5 validates special-token ids inside ``PretrainedConfig.__init__`` (strict
  dataclasses). The validation calls ``Qwen3ASRConfig.get_text_config`` before that class
  has built ``thinker_config``.
* transformers 5 removed the ``"default"`` entry of ``ROPE_INIT_FUNCTIONS`` and its generic
  weight init probes rope modules for ``compute_default_rope_parameters``; qwen-asr indexes
  the dict directly and does not define that method.
* Special-token ids (``pad_token_id`` ...) moved from model configs to ``GenerationConfig``;
  the qwen-asr configs and model still read ``config.pad_token_id``.
* ``AutoProcessor.from_pretrained`` resolves ``processor_class`` by name and finds the
  native same-named transformers 5 classes before the ones qwen-asr registered.
* ``create_causal_mask`` renamed ``input_embeds`` to ``inputs_embeds`` and dropped
  ``cache_position``.
* from_pretrained builds models on the meta device; non-persistent buffers that are not in
  the checkpoint (the audio positional table) stay uninitialized.
* generate() no longer passes ``cache_position`` into ``prepare_inputs_for_generation``;
  without it the thinker recomputes full-length rope positions during decode.

Call :func:`install` before the first ``import qwen_asr`` or ``import qwen_tts``.
``transformers`` is imported inside :func:`install`: the thin installer and the unit tests
import ``infra`` before that package exists. The qwen-asr class patches are applied when
``install`` first sees qwen-asr importable, and are retried on later calls until then.

Example::

    from infra.qwen_tf_compat import install
    install()
    import qwen_asr
"""
from __future__ import annotations

import importlib
import math
from typing import Any

_INSTALLED = False
_QWEN_ASR_PATCHED = False

#: token ids that transformers 5 moved from model configs to GenerationConfig; qwen-asr 0.0.6
#: still reads them from the model configs (as None-able attributes).
_TOKEN_ID_DEFAULTS = ("pad_token_id", "bos_token_id", "eos_token_id", "decoder_start_token_id")


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


def _compute_default_rope_parameters(config: Any = None, device: Any = None, seq_len: Any = None,
                                     layer_type: Any = None, **kwargs: Any) -> Any:
    """The ``"default"`` RoPE entry of transformers 4.57 (removed in 5), layer_type-tolerant.

    transformers 5.16 calls rope functions as ``fn(config, device, seq_len=..., layer_type=...)``
    and, during weight init, as ``module.compute_default_rope_parameters(module.config)``.
    """
    import torch  # local import: install() may run where torch is not importable yet

    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def _patch_rope_init_functions() -> None:
    """Re-add the 'default' entry transformers 5 removed from ROPE_INIT_FUNCTIONS."""
    rope_utils = importlib.import_module("transformers.modeling_rope_utils")
    if "default" not in rope_utils.ROPE_INIT_FUNCTIONS:
        rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _patch_processor_lookup() -> None:
    """Registered processors must win over transformers 5's native same-named classes.

    ``AutoProcessor.from_pretrained`` resolves the ``processor_class`` stored in
    preprocessor_config.json by name; the native mapping is scanned before the classes
    registered via ``AutoProcessor.register`` (``_extra_content``), so the native
    ``Qwen3ASRProcessor`` would shadow the one from qwen-asr.
    """
    processing = importlib.import_module("transformers.models.auto.processing_auto")
    original = processing.processor_class_from_name
    if getattr(original, "_voxprint_registered_first", False):
        return

    def processor_class_from_name(class_name: str) -> Any:
        for processor in processing.PROCESSOR_MAPPING._extra_content.values():
            if getattr(processor, "__name__", None) == class_name:
                return processor
        return original(class_name)

    processor_class_from_name._voxprint_registered_first = True  # type: ignore[attr-defined]
    setattr(processing, "processor_class_from_name", processor_class_from_name)


def _wrap_causal_mask(original: Any) -> Any:
    """Accept the transformers 4.57 kwargs of create_causal_mask.

    transformers 5 renamed ``input_embeds`` to ``inputs_embeds`` and removed
    ``cache_position`` from the signature.
    """

    def create_causal_mask(*args: Any, **kwargs: Any) -> Any:
        if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        kwargs.pop("cache_position", None)
        return original(*args, **kwargs)

    create_causal_mask._voxprint_alias = True  # type: ignore[attr-defined]
    return create_causal_mask


def _patch_transformers_causal_mask() -> None:
    masking = importlib.import_module("transformers.masking_utils")
    if not getattr(masking.create_causal_mask, "_voxprint_alias", False):
        setattr(masking, "create_causal_mask", _wrap_causal_mask(masking.create_causal_mask))


def _patch_qwen_asr() -> None:
    """Patch the qwen-asr 0.0.6 classes. Requires qwen-asr (and torch) to be importable."""
    global _QWEN_ASR_PATCHED
    if _QWEN_ASR_PATCHED:
        return
    qcfg = importlib.import_module("qwen_asr.core.transformers_backend.configuration_qwen3_asr")
    qmodel = importlib.import_module("qwen_asr.core.transformers_backend.modeling_qwen3_asr")

    # 1) transformers 5 strict config validation calls get_text_config during
    #    PretrainedConfig.__init__, before thinker_config exists.
    if not getattr(qcfg.Qwen3ASRConfig.get_text_config, "_voxprint_tolerant", False):
        def get_text_config(self: Any, decoder: Any = None, encoder: Any = None) -> Any:
            thinker = self.__dict__.get("thinker_config")
            if thinker is None:
                return self
            return thinker.get_text_config()

        get_text_config._voxprint_tolerant = True  # type: ignore[attr-defined]
        qcfg.Qwen3ASRConfig.get_text_config = get_text_config

    # 2) token ids moved to GenerationConfig in transformers 5: restore the None-able
    #    class attributes qwen-asr expects on its configs.
    for cls in (qcfg.Qwen3ASRConfig, qcfg.Qwen3ASRThinkerConfig, qcfg.Qwen3ASRTextConfig,
                qcfg.Qwen3ASRAudioEncoderConfig):
        for name in _TOKEN_ID_DEFAULTS:
            if name not in cls.__dict__:
                setattr(cls, name, None)

    # 3) transformers 5 weight init recomputes rope buffers through
    #    module.compute_default_rope_parameters; qwen-asr only has rope_init_fn.
    if not hasattr(qmodel.Qwen3ASRThinkerTextRotaryEmbedding, "compute_default_rope_parameters"):
        qmodel.Qwen3ASRThinkerTextRotaryEmbedding.compute_default_rope_parameters = staticmethod(
            _compute_default_rope_parameters
        )

    # 4) create_causal_mask was imported into the qwen-asr namespace by value.
    if not getattr(qmodel.create_causal_mask, "_voxprint_alias", False):
        setattr(qmodel, "create_causal_mask", _wrap_causal_mask(qmodel.create_causal_mask))

    # 5) transformers 5 builds models on the meta device and materializes non-persistent
    #    buffers that are not in the checkpoint with torch.empty_like: the audio positional
    #    table ends up uninitialized. Recompute it on first use if it is broken.
    if not getattr(qmodel.Qwen3ASRAudioEncoder.forward, "_voxprint_selfheal", False):
        original_forward = qmodel.Qwen3ASRAudioEncoder.forward

        def audio_encoder_forward(self: Any, input_features: Any, feature_lens: Any = None, aftercnn_lens: Any = None) -> Any:
            pe = self.positional_embedding.positional_embedding
            if pe.is_meta or bool(pe.isnan().any()):
                import torch  # local import: see _compute_default_rope_parameters

                channels = pe.shape[1]
                length = pe.shape[0]
                log_timescale_increment = math.log(10000) / (channels // 2 - 1)
                inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2, dtype=torch.float32))
                scaled_time = torch.arange(length, dtype=torch.float32)[:, None] * inv_timescales[None, :]
                value = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1).to(pe.device)
                self.positional_embedding.positional_embedding = value
            return original_forward(self, input_features, feature_lens=feature_lens, aftercnn_lens=aftercnn_lens)

        audio_encoder_forward._voxprint_selfheal = True  # type: ignore[attr-defined]
        qmodel.Qwen3ASRAudioEncoder.forward = audio_encoder_forward

    # 6) transformers 5 generate() no longer passes cache_position. Without it the thinker
    #    recomputes full-length rope positions during decode and the rotary broadcast
    #    explodes the query. Recreate it like transformers 5 does for remote code.
    if not getattr(qmodel.Qwen3ASRThinkerForConditionalGeneration.prepare_inputs_for_generation,
                   "_voxprint_none_safe", False):
        original_prepare = qmodel.Qwen3ASRThinkerForConditionalGeneration.prepare_inputs_for_generation

        def prepare_inputs_for_generation(self: Any, input_ids: Any, past_key_values: Any = None,
                                          attention_mask: Any = None, inputs_embeds: Any = None,
                                          cache_position: Any = None, position_ids: Any = None,
                                          use_cache: Any = True, input_features: Any = None,
                                          feature_attention_mask: Any = None, **kwargs: Any) -> Any:
            if cache_position is None:
                import torch  # local import: see _compute_default_rope_parameters

                past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(input_ids.shape[1], device=input_ids.device) + past_seen
            model_inputs = original_prepare(
                self, input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
                inputs_embeds=inputs_embeds, cache_position=cache_position, position_ids=position_ids,
                use_cache=use_cache, input_features=input_features,
                feature_attention_mask=feature_attention_mask, **kwargs)
            model_inputs["position_ids"] = None
            if int(cache_position[0]) != 0:
                model_inputs["input_features"] = None
            return model_inputs

        prepare_inputs_for_generation._voxprint_none_safe = True  # type: ignore[attr-defined]
        qmodel.Qwen3ASRThinkerForConditionalGeneration.prepare_inputs_for_generation = prepare_inputs_for_generation

    _QWEN_ASR_PATCHED = True


def _try_patch_qwen_asr() -> None:
    """Apply the qwen-asr class patches once qwen-asr is importable; otherwise try again later."""
    try:
        _patch_qwen_asr()
    except ImportError:
        pass


def install() -> bool:
    """Patch transformers so qwen-asr 0.0.6 can be imported and used.

    Returns:
        True when the patches are in place. False when transformers is not installed.
        Calling it again does nothing.
    """
    global _INSTALLED
    if not _INSTALLED:
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
        _patch_rope_init_functions()
        _patch_processor_lookup()
        _patch_transformers_causal_mask()
        _INSTALLED = True
    _try_patch_qwen_asr()
    return True
