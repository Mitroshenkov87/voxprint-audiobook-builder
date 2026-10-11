"""Tokenizer and config paths on the transformers 5.16 / huggingface_hub 1.32 pin. No GPU and no network."""
import json

import pytest

transformers = pytest.importorskip("transformers")
hub = pytest.importorskip("huggingface_hub")


def test_the_security_pins_are_what_is_installed():
    from packaging.version import Version

    assert Version("5.16.1") <= Version(transformers.__version__) < Version("5.17")
    assert Version("1.32.0") <= Version(hub.__version__) < Version("2")
    assert hasattr(hub, "set_client_factory")
    assert not hasattr(hub, "configure_http_backend")


def test_local_config_and_tokenizer_load_without_a_download(tmp_path):
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer

    config = {
        "model_type": "bert",
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "vocab_size": 100,
        "max_position_embeddings": 32,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "BertTokenizer", "model_max_length": 32}), encoding="utf-8")
    loaded = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
    assert loaded.model_type == "bert"
    tok = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    encoded = tok("hello world")
    assert encoded["input_ids"]
    assert MarianTokenizer and MarianMTModel and AutoModelForSeq2SeqLM


def test_qwen_packages_import_on_this_transformers():
    pytest.importorskip("torch")
    from infra.qwen_tf_compat import install

    assert install()
    qwen_asr = pytest.importorskip("qwen_asr")
    qwen_tts = pytest.importorskip("qwen_tts")
    assert qwen_asr.Qwen3ASRModel and qwen_asr.Qwen3ForcedAligner
    assert qwen_tts.Qwen3TTSModel
