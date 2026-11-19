"""The transformers 5 import shim, without loading a model."""
import importlib

import pytest

from infra.qwen_tf_compat import _force_exist_ok, _optional_call, install


def test_check_model_inputs_accepts_a_call_with_no_arguments():
    seen = []

    def original(func):
        seen.append(func)
        return func

    patched = _optional_call(original)

    @patched()
    def forward(x):
        return x

    assert forward(3) == 3 and seen == [forward]
    assert _optional_call(patched) is patched

    def plain(x):
        return x

    assert patched(plain) is plain


class _Config:
    @staticmethod
    def register(model_type, config, exist_ok=False):
        return (model_type, config, exist_ok)


class _Model:
    @classmethod
    def register(cls, config_class, model_class, exist_ok=False):
        return (cls, config_class, model_class, exist_ok)


def test_register_is_forced_to_exist_ok():
    _force_exist_ok(_Config, "register")
    _force_exist_ok(_Model, "register")
    assert _Config.register("qwen3_asr", "Cfg") == ("qwen3_asr", "Cfg", True)
    assert _Model.register("Cfg", "Model")[0] is _Model
    assert _Model.register("Cfg", "Model")[-1] is True
    _force_exist_ok(_Config, "register")
    assert _Config.register("qwen3_asr", "Cfg")[-1] is True


def test_install_is_false_without_transformers(monkeypatch):
    import infra.qwen_tf_compat as compat

    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(compat, "_INSTALLED", False)
    monkeypatch.setattr(compat.importlib, "import_module", missing)
    assert install() is False


def test_install_patches_the_real_symbols_when_transformers_is_present():
    pytest.importorskip("transformers")
    assert install() is True
    generic = importlib.import_module("transformers.utils.generic")
    assert getattr(generic.check_model_inputs, "_voxprint_optional_call", False)
    auto_cfg = importlib.import_module("transformers.models.auto.configuration_auto")
    assert callable(auto_cfg.AutoConfig.register)
    assert install() is True


def test_rope_init_functions_get_the_default_entry_back():
    torch = pytest.importorskip("torch")
    install()
    rope_utils = importlib.import_module("transformers.modeling_rope_utils")
    fn = rope_utils.ROPE_INIT_FUNCTIONS.get("default")
    assert fn is not None
    cfg = type("Cfg", (), {"rope_theta": 1000000.0, "head_dim": 128})()
    inv_freq, scaling = fn(cfg)
    assert inv_freq.shape == (64,) and scaling == 1.0
    # transformers 5 calls rope fns with seq_len/layer_type kwargs
    inv_freq2, _ = fn(cfg, torch.device("cpu"), seq_len=512, layer_type=None)
    assert torch.equal(inv_freq, inv_freq2)
    # idempotent: a second install() keeps the same entry
    assert install() is True
    assert rope_utils.ROPE_INIT_FUNCTIONS["default"] is fn


def test_create_causal_mask_wrapper_accepts_the_old_kwargs():
    from infra.qwen_tf_compat import _wrap_causal_mask
    seen = {}

    def original(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = dict(kwargs)
        return "mask"

    wrapped = _wrap_causal_mask(original)
    assert wrapped(1, 2, input_embeds="emb", cache_position="pos", attention_mask=None) == "mask"
    assert seen["kwargs"] == {"inputs_embeds": "emb", "attention_mask": None}
    # new spelling and unknown kwargs pass through untouched
    wrapped(inputs_embeds="e", other=3)
    assert seen["kwargs"] == {"inputs_embeds": "e", "other": 3}


def _tiny_asr_config():
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
    return Qwen3ASRConfig(thinker_config={
        "text_config": {
            "vocab_size": 100, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
            "head_dim": 16, "max_position_embeddings": 64,
        },
        "audio_config": {
            "encoder_layers": 1, "d_model": 32, "encoder_attention_heads": 2,
            "encoder_ffn_dim": 64, "num_mel_bins": 32, "output_dim": 32, "max_source_positions": 64,
        },
    })


def test_composite_config_constructs_and_reads_like_on_transformers4():
    pytest.importorskip("torch")
    qwen_asr = pytest.importorskip("qwen_asr")
    assert install() is True
    cfg = _tiny_asr_config()
    # get_text_config works once thinker_config exists ...
    assert cfg.get_text_config().vocab_size == 100
    # ... and stays tolerant while the config is still under construction (the state
    # transformers 5 strict validation used to trip over before thinker_config was set)
    thinker = cfg.thinker_config
    delattr(cfg, "thinker_config")
    try:
        assert cfg.get_text_config() is cfg
    finally:
        cfg.thinker_config = thinker
    assert cfg.get_text_config().vocab_size == 100
    # token ids moved to GenerationConfig in transformers 5: qwen-asr configs expose None
    for name in ("pad_token_id", "bos_token_id", "eos_token_id", "decoder_start_token_id"):
        assert getattr(cfg.thinker_config, name, "missing") is None
        assert getattr(cfg.thinker_config.text_config, name, "missing") is None
    assert qwen_asr is not None


def test_registered_processor_wins_over_the_native_same_named_class():
    pytest.importorskip("qwen_asr")
    install()
    processing = importlib.import_module("transformers.models.auto.processing_auto")
    cls = processing.processor_class_from_name("Qwen3ASRProcessor")
    assert cls is not None and cls.__module__.startswith("qwen_asr")


def test_audio_encoder_positional_table_self_heals():
    torch = pytest.importorskip("torch")
    pytest.importorskip("qwen_asr")
    install()
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig

    cfg = Qwen3ASRAudioEncoderConfig(encoder_layers=1, d_model=32, encoder_attention_heads=2,
                                     encoder_ffn_dim=64, num_mel_bins=32, output_dim=32,
                                     max_source_positions=64, n_window=50)
    enc = Qwen3ASRAudioEncoder(cfg)
    healthy = enc.positional_embedding.positional_embedding.clone()
    enc.positional_embedding.positional_embedding = torch.full_like(healthy, float("nan"))
    features = torch.zeros(32, 100)  # one n_window*2 chunk, like the real checkpoints use
    out = enc(features, feature_lens=torch.tensor([100]))
    assert not torch.isnan(out.last_hidden_state).any()
    healed = enc.positional_embedding.positional_embedding
    assert not torch.isnan(healed).any()
    assert torch.allclose(healed, healthy)


def test_generation_helpers_are_patched():
    pytest.importorskip("torch")
    pytest.importorskip("qwen_asr")
    install()
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRThinkerForConditionalGeneration,
        Qwen3ASRThinkerTextRotaryEmbedding,
        create_causal_mask,
    )
    assert getattr(Qwen3ASRThinkerForConditionalGeneration.prepare_inputs_for_generation, "_voxprint_none_safe")
    assert hasattr(Qwen3ASRThinkerTextRotaryEmbedding, "compute_default_rope_parameters")
    assert getattr(create_causal_mask, "_voxprint_alias")
