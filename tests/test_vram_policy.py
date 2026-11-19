"""VRAM policy: plan from the memory free right now, leaving max(2 GB, 8 % of the card). The 70-80 % cap is optional."""

import pytest

from core import vram_policy as vp


def test_fraction_is_clamped_and_accepts_percent():
    assert vp.clamp_fraction(0.75) == 0.75
    assert vp.clamp_fraction(75) == 0.75
    assert vp.clamp_fraction(0.5) == vp.MIN_FRACTION and vp.clamp_fraction(0.95) == vp.MAX_FRACTION
    for bad in (None, "x", float("nan"), 0, -1):
        assert vp.clamp_fraction(bad) == vp.DEFAULT_FRACTION


def test_reserve_is_the_larger_of_two_gigabytes_and_eight_percent():
    assert vp.reserve_gb(16) == 2.0          # 8 % of 16 is 1.28
    assert vp.reserve_gb(32) == pytest.approx(2.56)
    assert vp.reserve_gb(0) == 2.0


def test_budget_is_free_minus_the_reserve():
    assert vp.budget_gb(11.0, 16.0) == pytest.approx(9.0)
    assert vp.budget_gb(20.0, 32.0) == pytest.approx(17.44)
    assert vp.budget_gb(1.0, 16.0) == 0.0
    # optional user cap: 75 % of 16 GB is 12 GB, 5 GB already in use, so 7 GB may still be taken
    assert vp.budget_gb(11.0, 16.0, fraction=0.75) == pytest.approx(7.0)
    assert vp.budget_gb(11.0, 16.0, fraction=None) == pytest.approx(9.0)


@pytest.mark.parametrize("free,total,frac,expected", [
    (11.0, 16.0, None, 10),     # 16 GB card, model loaded: (11 - 2) // 0.9
    (20.0, 24.0, None, 12),     # 24 GB card: capped by MAX_BATCH
    (3.0, 16.0, None, 1),       # another program holds the card: still one at a time
    (11.0, 16.0, 0.75, 7),      # the optional cap, when the user turns it on
    (11.0, 16.0, 0.80, 8),
    (11.0, 16.0, 0.70, 6),
])
def test_plan_batch(free, total, frac, expected):
    assert vp.plan_batch(free, total, 0.9, 12, frac) == expected


def test_batch_stays_inside_the_reserve():
    for free, total in ((11.0, 16.0), (20.0, 24.0), (6.0, 8.0), (28.0, 32.0), (1.5, 16.0)):
        p = vp.plan(free, total, 0.9, 12)
        assert p.batch >= 1
        if p.batch > 1:
            assert p.batch * 0.9 <= max(0.0, free - vp.reserve_gb(total)) + 1e-6


def test_gpu_prefs_fraction_is_off_unless_set(monkeypatch):
    from infra import gpu_prefs

    assert gpu_prefs.vram_fraction() is None
    assert gpu_prefs.set_vram_fraction("0.95") == 0.80 and gpu_prefs.vram_fraction() == 0.80
    assert gpu_prefs.set_vram_fraction("off") is None and gpu_prefs.vram_fraction() is None
    monkeypatch.setenv(gpu_prefs.ENV_FRACTION, "72")
    assert gpu_prefs.vram_fraction() == 0.72
    monkeypatch.setenv(gpu_prefs.ENV_FRACTION, "off")
    assert gpu_prefs.vram_fraction() is None
    with pytest.raises(ValueError):
        gpu_prefs.set_vram_fraction("lots")


def test_engine_max_batch_uses_the_policy(monkeypatch):
    torch = pytest.importorskip("torch")
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)) or not callable(getattr(cuda, "mem_get_info", None)):
        pytest.skip("stand-in torch has no cuda")
    from core import tts_engine

    gb = 1024 ** 3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (int(11 * gb), int(16 * gb)))
    eng = tts_engine.Qwen3AdapterEngine.__new__(tts_engine.Qwen3AdapterEngine)
    eng._graph, eng.device = None, "cuda:0"
    assert eng.max_batch() == 10                      # (11 GB free - 2 GB reserve) // 0.9
    from infra import gpu_prefs

    gpu_prefs.set_vram_fraction(0.80)
    assert eng.max_batch() == 8
    eng._graph = object()                             # CUDA Graphs decode one sequence
    assert eng.max_batch() == 1
    eng._graph, eng.device = None, "cpu"
    assert eng.max_batch() == 1


def test_plan_describe_names_the_reserve_and_an_optional_cap():
    text = vp.plan(11, 16, 0.9, 12).describe()
    assert "reserve 2.0 GB" in text and "cap" not in text
    assert "cap 75%" in vp.plan(11, 16, 0.9, 12, 0.75).describe()
