import json
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

import core.audio_utils as au
from core.errors import OutOfMemoryError_
from core.events import CancelToken, Stage
from infra.vram_optimizer import GpuInfo, plan_training


def tiny_model():
    qt = pytest.importorskip("qwen_tts")
    from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
    cp = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
              num_key_value_heads=1, head_dim=16, num_code_groups=4, vocab_size=64)
    t = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
             head_dim=16, num_code_groups=4, text_hidden_size=32, vocab_size=128, text_vocab_size=100,
             code_predictor_config=cp, spk_id={}, codec_language_id={'russian': 9},
             codec_pad_id=1, codec_bos_id=2, codec_eos_token_id=3, codec_think_id=4, codec_nothink_id=5,
             codec_think_bos_id=6, codec_think_eos_id=7,
             rope_scaling={'interleaved': True, 'mrope_section': [4, 2, 2], 'rope_type': 'default', 'type': 'default'})
    sp = dict(enc_dim=32, enc_channels=[16, 16, 16, 16, 48], enc_attention_channels=8, enc_se_channels=8)
    cfg = Qwen3TTSConfig(talker_config=t, speaker_encoder_config=sp, tts_model_type="base",
                         tts_pad_token_id=90, tts_bos_token_id=91, tts_eos_token_id=92)
    torch.manual_seed(0)
    return Qwen3TTSForConditionalGeneration(cfg)


G = 4  # кодовых групп в крошечной модели


def _dataset(tmp_path, n=6):
    from core.dataset_builder import make_rows, write_metadata_jsonl
    from core.types import Segment
    d = tmp_path / "ds"
    segs = []
    for i in range(1, n + 1):
        x = (np.sin(2 * np.pi * 150 * np.arange(24000 * 3) / 24000) * 0.2).astype(np.float32)
        au.write_wav(d / f"segment_{i:03d}.wav", x, 24000)
        segs.append(Segment(i, 0, 3, f"Текст номер {i}."))
    au.write_wav(d / "ref.wav", x, 24000)
    (d / "ref_text.txt").write_text("Образец голоса.\n", encoding="utf-8")
    write_metadata_jsonl(d / "metadata.jsonl", make_rows(segs))
    return d


def _tok(text):
    return torch.randint(1, 80, (1, 14))


def _encode(audio, sr):
    return torch.randint(0, 50, (int(np.random.randint(8, 14)), G))


def _plan(**kw):
    from dataclasses import replace
    return replace(plan_training(GpuInfo(False), 6, language="russian"), epochs=2, grad_accum=2, lr=1e-3,
                   lora_r=4, lora_alpha=16, **kw)


def test_load_training_rows_follows_alexandria_contract(tmp_path):
    from core.lora_trainer import load_training_rows
    d = _dataset(tmp_path)
    data = load_training_rows(d)
    assert len(data["rows"]) == 6 and data["ref_audio"].endswith("ref.wav")
    assert data["ref_text"] == "Образец голоса."
    # audio_filepath тоже допустим (как в train_lora.py); ref_text.txt нет -> текст первого примера
    (d / "ref_text.txt").unlink()
    assert load_training_rows(d)["ref_text"] == "Текст номер 1."


def test_teacher_forcing_input_layout_matches_alexandria():
    from core.teacher_forcing import build_teacher_forcing_input
    m = tiny_model()
    tc = m.config.talker_config
    T, L = 7, 14
    sample = {"codec_ids": torch.randint(0, 50, (T, G)), "spk_embedding": torch.randn(1, 32),
              "text_ids": torch.randint(1, 80, (1, L))}
    full, labels, codes, prefill = build_teacher_forcing_input(sample, m, m.talker, torch.device("cpu"), "russian")
    # prefill = 3 роли + (4 codec-префикс + spk + pad + bos - 1) + (L-8 текста + eos) + 1 конец
    assert prefill == 3 + 6 + (L - 8 + 1) + 1
    assert full.shape == (1, prefill + T, tc.hidden_size) and labels.shape == (1, prefill + T)
    assert (labels[0, :prefill] == -100).all() and (labels[0, prefill:] == sample["codec_ids"][:, 0]).all()
    # язык не найден -> nothink-вариант: на один токен короче
    full2, _, _, prefill2 = build_teacher_forcing_input(sample, m, m.talker, torch.device("cpu"), "klingon")
    assert prefill2 == prefill - 1


def test_train_on_tiny_model_end_to_end_writes_alexandria_adapter_folder(tmp_path):
    from safetensors import safe_open
    from core.lora_trainer import ADAPTER_FILES, load_training_rows, train_on_model
    d = _dataset(tmp_path)
    data = load_training_rows(d)
    m = tiny_model()
    events, warns = [], []
    out_dir = train_on_model(m, _tok, _encode, data, tmp_path / "output", _plan(), "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                             lambda s, f, msg: events.append((s, f, msg)), CancelToken(), warns)
    out = tmp_path / "output"
    assert out_dir == out
    for name in ADAPTER_FILES:
        assert (out / name).exists(), name
    assert not (out / "speaker_embedding.safetensors").exists() and not (out / "voice_meta.json").exists()
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["r"] == 4 and cfg["lora_alpha"] == 16
    assert sorted(cfg["target_modules"]) == ["k_proj", "o_proj", "q_proj", "v_proj"]
    # ключи относительны к talker (не к всей модели): base_model.model.model.layers.N.self_attn.q_proj...
    with safe_open(str(out / "adapter_model.safetensors"), "pt") as f:
        keys = list(f.keys())
    assert keys and all("lora_" in k for k in keys)
    assert any(k.startswith("base_model.model.model.layers.0.self_attn.q_proj.lora_A") for k in keys)
    assert not any(".talker." in k or k.startswith("base_model.model.talker") for k in keys)
    assert any("code_predictor" in k for k in keys)      # как у Alexandria: суффиксное совпадение q/k/v/o_proj
    # training_meta.json - те же ключи, что у Alexandria
    meta = json.loads((out / "training_meta.json").read_text(encoding="utf-8"))
    assert set(meta) == {"model_name", "epochs", "lr", "lora_r", "lora_alpha", "gradient_accumulation_steps",
                         "batch_size", "num_samples", "final_loss", "best_loss", "training_time_seconds",
                         "language", "ref_sample_audio", "ref_sample_text"}
    assert meta["language"] == "russian" and meta["ref_sample_text"] == "Образец голоса."
    assert meta["batch_size"] == 1 and meta["num_samples"] == 6 and meta["epochs"] == 2
    # чекпойнты по эпохам
    for e in (1, 2):
        assert (out / "checkpoints" / f"epoch_{e:02d}" / "adapter_model.safetensors").exists()
    # потребитель (Alexandria): PeftModel.from_pretrained(talker, adapter_path) на свежей модели
    from peft import PeftModel
    fresh = tiny_model()
    reloaded = PeftModel.from_pretrained(fresh.talker, str(out))
    assert any("lora_A" in n for n, _ in reloaded.named_parameters())
    assert any(s is Stage.TRAIN for s, _, _ in events) and any(s is Stage.SAVE for s, _, _ in events)


def test_lora_applied_to_talker_only_and_trains(tmp_path):
    from core.lora_trainer import build_lora_config, compute_sample_loss
    from peft import get_peft_model
    m = tiny_model()
    for p in m.parameters():
        p.requires_grad_(False)
    pt = get_peft_model(m.talker, build_lora_config(4, 16))
    m.talker = pt
    base = pt.base_model.model
    trainable = [n for n, p in pt.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    assert not any(p.requires_grad for n, p in m.named_parameters() if n.startswith("speaker_encoder"))
    sample = {"codec_ids": torch.randint(0, 50, (10, G)), "spk_embedding": torch.randn(1, 32),
              "text_ids": torch.randint(1, 80, (1, 14))}
    opt = torch.optim.AdamW([p for p in pt.parameters() if p.requires_grad], lr=5e-3)
    pt.train()
    first = last = None
    for _ in range(30):
        loss, _, _ = compute_sample_loss(sample, m, base, torch.device("cpu"), "russian")
        loss.backward(); opt.step(); opt.zero_grad()
        first = float(loss.detach()) if first is None else first
        last = float(loss.detach())
    assert last < first


def test_loss_warnings():
    from core.lora_trainer import loss_warnings
    assert any("низкая" in w for w in loss_warnings(2.9, 6.0))
    assert any("не уменьшилась" in w for w in loss_warnings(5.0, 5.0))
    assert loss_warnings(4.5, 7.0) == []


def test_make_optimizer_falls_back_without_bitsandbytes(monkeypatch):
    import core.lora_trainer as lt
    monkeypatch.setattr(lt, "check_bitsandbytes", lambda device: False)
    w = []
    opt = lt.make_optimizer([torch.nn.Parameter(torch.zeros(2))], 1e-6, True, "cuda:0", w)
    assert isinstance(opt, torch.optim.AdamW) and w
    assert lt.check_bitsandbytes("cpu") is False


def test_cancel_during_training(tmp_path):
    from core.errors import CancelledByUser
    from core.lora_trainer import load_training_rows, train_on_model
    d = _dataset(tmp_path, 4)
    tok = CancelToken(); tok.cancel()
    with pytest.raises(CancelledByUser):
        train_on_model(tiny_model(), _tok, _encode, load_training_rows(d), tmp_path / "o", _plan(), "x",
                       lambda *a: None, tok)


def test_oom_retry_chain_and_final_failure(tmp_path):
    from core.lora_trainer import train_lora_from_dataset
    d = _dataset(tmp_path, 3)
    plans = []

    def run(dataset_dir, out, plan, progress, cancel, warnings_out):
        plans.append(plan)
        raise OutOfMemoryError_(details="cuda oom")

    gpu_plan = plan_training(GpuInfo(True, "x", 16.0), 3)
    with pytest.raises(OutOfMemoryError_):
        train_lora_from_dataset(d, tmp_path / "o", plan=gpu_plan, _run=run)
    assert [(p.use_8bit_adam, p.base_model.endswith("0.6B-Base")) for p in plans] == [
        (False, False), (True, False), (True, True)]
    assert all(p.batch_size == 1 for p in plans)

    n = {"c": 0}

    def run2(dataset_dir, out, plan, progress, cancel, warnings_out):
        n["c"] += 1
        if n["c"] == 1:
            raise OutOfMemoryError_()
        return out
    assert train_lora_from_dataset(d, tmp_path / "o", plan=gpu_plan, _run=run2) == tmp_path / "o"


def test_language_is_taken_from_dataset_report(tmp_path):
    from core.lora_trainer import train_lora_from_dataset
    d = _dataset(tmp_path, 3)
    (d / "report.json").write_text(json.dumps({"training_language": "russian"}), encoding="utf-8")
    seen = {}

    def run(dataset_dir, out, plan, progress, cancel, warnings_out):
        seen["lang"] = plan.language
        return out
    train_lora_from_dataset(d, tmp_path / "o", force_cpu=True, _run=run)
    assert seen["lang"] == "russian"
