import numpy as np

from core import quality as q

SR = 24000


def _speech(db=-20.0, secs=3.0, seed=0):
    rng = np.random.RandomState(seed)
    n = int(SR * secs)
    t = np.arange(n) / SR
    env = (np.sin(2 * np.pi * 2.5 * t) > -0.2).astype(np.float32)          # слоги + паузы
    x = np.sin(2 * np.pi * 180 * t) * env
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 10 ** (db / 20)
    return (x + rng.randn(n) * 10 ** (-70 / 20)).astype(np.float32)


def test_good_segment_passes():
    r = q.assess_segments([_speech()], SR)
    assert r.keep == [True] and r.n_dropped == 0


def test_clipping_quiet_and_noisy_are_dropped():
    clipped = _speech(-12)
    clipped[:200] = 1.0
    quiet = _speech(-55)
    noisy = np.random.RandomState(1).randn(SR * 3).astype(np.float32) * 0.05    # шум без пауз: SNR ~ 0
    r = q.assess_segments([_speech(), clipped, quiet, noisy, _speech(seed=2), _speech(seed=3)], SR)
    assert r.keep == [True, False, False, False, True, True]
    assert r.reasons[1] == "clipping" and r.reasons[2] == "too_quiet" and r.reasons[3] == "low_snr"
    assert r.summary() == {"clipping": 1, "too_quiet": 1, "low_snr": 1}


def test_snr_criterion_switched_off_when_it_would_drop_most():
    flat = [np.random.RandomState(i).randn(SR * 3).astype(np.float32) * 0.05 for i in range(4)] + [_speech()]
    r = q.assess_segments(flat, SR)
    assert r.snr_disabled and all(r.keep)


def test_clip_threshold_scales_with_length():
    x = _speech()
    x[:10] = 1.0                      # ровно 10 отсчётов - ещё допустимо
    assert q.reject_reason(q.audio_stats(x, SR)) is None
    x[:50] = 1.0
    assert q.reject_reason(q.audio_stats(x, SR)) == "clipping"
