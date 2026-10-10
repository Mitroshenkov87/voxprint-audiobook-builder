"""Automatic cooling: full speed for 2.5 h, then a 2-3 min pause when the GPU stays at or above 83 C."""

import pytest

from core import gpu_thermal
from core.events import CancelToken
from core.i18n import tr


def _monitor(now, **kw):
    return gpu_thermal.Monitor(started=0.0, now=now, read_temp=lambda: None, **kw)


def test_pause_length_is_between_two_and_three_minutes():
    assert 120 <= gpu_thermal.COOL_S <= 180
    clock = {"t": 10_000.0}

    def now():
        return clock["t"]

    def sleep(seconds):
        clock["t"] += seconds

    mon = _monitor(now, sleep=sleep)
    mon.cool(None)
    assert clock["t"] == pytest.approx(10_000.0 + gpu_thermal.COOL_S)


def test_the_first_two_and_a_half_hours_stay_at_full_speed():
    moment = {"t": gpu_thermal.FULL_SPEED_S - 1}
    mon = _monitor(lambda: moment["t"])
    for i in range(5):
        mon.observe(90, at=moment["t"] - 40 + i)
    assert mon.should_cool() is False
    moment["t"] = gpu_thermal.FULL_SPEED_S
    for i in range(5):
        mon.observe(90, at=moment["t"] - 40 + i)
    assert mon.should_cool() is True
    assert mon.median_c() == 90


def test_a_cooler_median_and_a_single_spike_do_not_pause():
    moment = gpu_thermal.FULL_SPEED_S + 30
    cool = _monitor(lambda: moment)
    for i in range(5):
        cool.observe(70, at=moment - 50 + i * 10)
    assert cool.should_cool() is False

    spike = _monitor(lambda: moment)
    spike.observe(95, at=moment - 5)
    assert spike.should_cool() is False

    mixed = _monitor(lambda: moment)
    mixed.observe(95, at=moment - 10 * 60)          # outside the five-minute window
    for i in range(3):
        mixed.observe(60, at=moment - 30 + i * 10)
    assert mixed.median_c() == 60
    assert mixed.should_cool() is False


def test_query_temperature_parses_nvidia_smi_and_skips_a_missing_tool(monkeypatch):
    monkeypatch.setattr(gpu_thermal.shutil, "which", lambda name: None)
    assert gpu_thermal.query_temperature(0, run=lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran"))) is None

    monkeypatch.setattr(gpu_thermal.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def run(cmd, **_kw):
        assert cmd[1:4] == ["-i", "1", "--query-gpu=temperature.gpu"]

        class Result:
            returncode = 0
            stdout = "84\n"

        return Result()

    assert gpu_thermal.query_temperature(1, run=run) == 84.0

    def unreadable(cmd, **_kw):
        class Result:
            returncode = 0
            stdout = "[N/A]\n"

        return Result()

    assert gpu_thermal.query_temperature(0, run=unreadable) is None


def test_start_does_not_spawn_a_sampler_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(gpu_thermal.shutil, "which", lambda name: None)
    mon = gpu_thermal.Monitor(started=0.0)
    mon.start()
    try:
        assert mon._thread is None
    finally:
        mon.stop()


def test_cool_honours_cancel():
    from core.errors import CancelledByUser

    mon = gpu_thermal.Monitor(started=0.0, now=lambda: 0.0, sleep=lambda _s: None, read_temp=lambda: None)
    token = CancelToken()
    token.cancel()
    with pytest.raises(CancelledByUser):
        mon.cool(token)


def test_cooling_message_is_cooling_gpu():
    from core import i18n

    i18n.set_language("en")
    assert tr("narr.cooling") == "Cooling GPU"


def test_narration_pauses_between_batches_and_shows_cooling(tmp_path, monkeypatch):
    from core import i18n
    from core.narration import NarrationOptions
    from tests.test_narration import run
    from tests.test_narration_batch import BatchEngine, long_book

    i18n.set_language("en")

    class Hot:
        def __init__(self, started, **_kw):
            self.pauses = 0
            Hot.instance = self

        def start(self):
            return self

        def stop(self):
            pass

        def retarget(self, _device):
            pass

        def should_cool(self):
            return True

        def cool(self, _cancel, _pause=None):
            self.pauses += 1

    monkeypatch.setattr(gpu_thermal, "Monitor", Hot)
    events = []
    eng = BatchEngine(limit=2)
    run(tmp_path, engine=eng, book=long_book(), events=events, options=NarrationOptions(max_chars=40))
    assert Hot.instance.pauses >= 1
    assert any(event.message == "Cooling GPU" for event in events)
    assert eng.batches and any(len(batch) > 1 for batch in eng.batches)
