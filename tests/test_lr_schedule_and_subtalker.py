"""Warmup + cosine learning-rate schedule (default) and the opt-in sub-talker label-shift fix."""
import json
import math

import pytest

from core.lora_trainer import fix_sub_talker_shift_of, lr_factor, lr_schedule_of


def test_lr_factor_warmup_then_cosine_to_the_floor():
    f = [lr_factor(s, 100, 0.1, 0.1) for s in range(100)]
    assert f[0] == pytest.approx(0.1) and f[9] == pytest.approx(1.0)              # linear warmup over 10 steps
    assert all(a < b for a, b in zip(f[:9], f[1:10]))
    assert all(a >= b for a, b in zip(f[10:], f[11:]))                            # then never rises
    assert f[-1] == pytest.approx(0.1) and min(f) == pytest.approx(0.1)
    assert lr_factor(0, 10, 0.0, 0.1) == pytest.approx(1.0)                        # no warmup: starts at the peak
    assert lr_factor(0, 1, 0.1, 0.1) == pytest.approx(1.0)                         # a single step is not starved


def test_switches(monkeypatch):
    from infra.vram_optimizer import GpuInfo, plan_training
    plan = plan_training(GpuInfo(False), 50)
    assert plan.lr_schedule == "cosine" and not plan.fix_sub_talker_shift          # defaults: cosine on, fix off
    monkeypatch.setenv("VOXPRINT_LR_SCHEDULE", "constant")
    assert lr_schedule_of(plan) == "constant"
    monkeypatch.setenv("VOXPRINT_LR_SCHEDULE", "junk")
    assert lr_schedule_of(plan) == "cosine"
    monkeypatch.setenv("VOXPRINT_FIX_SUBTALKER_SHIFT", "1")
    assert fix_sub_talker_shift_of(plan)


torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from tests.test_lora_trainer import G, _dataset, _encode, _plan, _tok, tiny_model  # noqa: E402


def _train(tmp_path, **kw):
    from core.events import CancelToken
    from core.lora_trainer import load_training_rows, train_on_model
    out = tmp_path / "out"
    train_on_model(tiny_model(), _tok, _encode, load_training_rows(_dataset(tmp_path)), out, _plan(**kw), "Qwen/B",
                   lambda *a: None, CancelToken(), [])
    return json.loads((out / "checkpoints" / "losses.json").read_text())


def test_training_follows_the_schedule(tmp_path):
    cos = _train(tmp_path / "a", warmup_fraction=0.5)["lr_schedule"]     # 6 samples, accum 2, 2 epochs -> 6 steps
    assert cos["kind"] == "cosine" and cos["optimizer_steps"] == 6
    assert cos["lr_first"] == pytest.approx(1e-3 / 3) and cos["lr_peak"] == pytest.approx(1e-3)
    assert cos["lr_last"] == pytest.approx(1e-4)
    const = _train(tmp_path / "b", lr_schedule="constant")["lr_schedule"]
    assert const["kind"] == "constant" and const["lr_first"] == const["lr_last"] == pytest.approx(1e-3)


def test_sub_talker_loss_is_aligned_and_the_library_loss_is_shifted_twice():
    import torch.nn.functional as F
    from core.lora_trainer import sub_talker_loss
    m = tiny_model().eval()
    T = 6
    codes = torch.randint(0, 50, (T, G))
    hidden = torch.randn(T, 32)
    with torch.no_grad():
        logits, lib_loss = m.talker.forward_sub_talker_finetune(codes, hidden)
    assert logits.shape[:2] == (T, G - 1)
    V = logits.shape[-1]
    aligned = sum(F.cross_entropy(logits[:, k].float(), codes[:, k + 1]) for k in range(G - 1)) / (G - 1)
    assert sub_talker_loss(logits, codes) == pytest.approx(float(aligned), rel=1e-5)
    # what the library computes: logit k against group k+2, the last group untrained (the double shift)
    shifted = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), codes[:, 2:].reshape(-1))
    assert float(lib_loss) == pytest.approx(float(shifted), rel=1e-5)
    assert not math.isclose(float(lib_loss), float(aligned), rel_tol=1e-3)


def test_fix_flag_reaches_the_loss_and_is_recorded(tmp_path):
    from core.lora_trainer import compute_sample_loss, sub_talker_loss
    m = tiny_model().eval()
    sample = {"codec_ids": torch.randint(0, 50, (8, G)), "spk_embedding": torch.randn(1, 32),
              "text_ids": torch.randint(1, 80, (1, 14))}
    with torch.no_grad():
        _, _, plain = compute_sample_loss(sample, m, m.talker, torch.device("cpu"), "russian")
        total, talker, fixed = compute_sample_loss(sample, m, m.talker, torch.device("cpu"), "russian", fix_sub_shift=True)
    assert not math.isclose(float(plain), float(fixed), rel_tol=1e-4)
    assert float(total) == pytest.approx(float(talker + 0.3 * fixed), rel=1e-5)
    assert _train(tmp_path, fix_sub_talker_shift=True)["fix_sub_talker_shift"] is True
