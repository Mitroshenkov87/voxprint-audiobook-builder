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
