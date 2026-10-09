"""VRAM policy (suite rule): plan from the memory free right now, never above 70-80 % of the card (default 75 %)."""

import pytest

from core import vram_policy as vp


def test_fraction_is_clamped_and_accepts_percent():
    assert vp.clamp_fraction(0.75) == 0.75
    assert vp.clamp_fraction(75) == 0.75
    assert vp.clamp_fraction(0.5) == vp.MIN_FRACTION and vp.clamp_fraction(0.95) == vp.MAX_FRACTION
    for bad in (None, "x", float("nan"), 0, -1):
        assert vp.clamp_fraction(bad) == vp.DEFAULT_FRACTION


def test_budget_keeps_the_card_below_the_cap():
    # 16 GB card, 5 GB in use (model): 75 % = 12 GB -> 7 GB may still be taken
    assert vp.budget_gb(11.0, 16.0, 0.75) == pytest.approx(7.0)
    # another program already holds most of the card: nothing left to plan
    assert vp.budget_gb(3.0, 16.0, 0.75) == 0.0
    # the absolute reserve wins when it is stricter
    assert vp.budget_gb(4.0, 4.0, 0.80, reserve_gb=2.0) == pytest.approx(2.0)   # 4 - 0.8 = 3.2 > 4 - 2


@pytest.mark.parametrize("free,total,frac,expected", [
    (11.0, 16.0, 0.75, 7),      # RTX 4090 Laptop with the model loaded
    (20.0, 24.0, 0.75, 12),     # 24 GB card: capped by MAX_BATCH
    (3.0, 16.0, 0.75, 1),       # Movie Dubber holds the card: still one at a time, never zero
    (11.0, 16.0, 0.80, 8),
    (11.0, 16.0, 0.70, 6),
])
def test_plan_batch(free, total, frac, expected):
    assert vp.plan_batch(free, total, 0.9, 12, frac) == expected


def test_peak_never_exceeds_the_cap():
    for free in (2.0, 6.5, 9.0, 13.0, 15.5):
        p = vp.plan(free, 16.0, 0.9, 12, 0.75)
        used_after = (16.0 - free) + p.batch * 0.9
        assert p.batch == 1 or used_after <= 0.75 * 16.0 + 1e-9


def test_gpu_prefs_fraction_setting_and_env(monkeypatch):
    from infra import gpu_prefs

    assert gpu_prefs.vram_fraction() == 0.75
    assert gpu_prefs.set_vram_fraction("0.95") == 0.80 and gpu_prefs.vram_fraction() == 0.80
    monkeypatch.setenv(gpu_prefs.ENV_FRACTION, "72")
    assert gpu_prefs.vram_fraction() == 0.72
    with pytest.raises(ValueError):
        gpu_prefs.set_vram_fraction("lots")


def test_engine_max_batch_uses_the_policy(monkeypatch):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "cuda"):                    # the no-PyTorch CI job has only a stand-in torch module
        pytest.skip("real PyTorch needed")
    from core import tts_engine

    gb = 1024 ** 3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (int(11 * gb), int(16 * gb)))
    eng = tts_engine.Qwen3AdapterEngine.__new__(tts_engine.Qwen3AdapterEngine)
    eng._graph, eng.device = None, "cuda:0"
    assert eng.max_batch() == 7                       # old rule (free - 2 GB) // 0.9 gave 10 and filled the card to ~14 GB
    from infra import gpu_prefs

    gpu_prefs.set_vram_fraction(0.80)
    assert eng.max_batch() == 8
    eng._graph = object()                             # CUDA Graphs decode one sequence
    assert eng.max_batch() == 1
    eng._graph, eng.device = None, "cpu"
    assert eng.max_batch() == 1


def test_plan_describe_is_readable():
    assert "cap 75%" in vp.plan(11, 16, 0.9, 12).describe()
