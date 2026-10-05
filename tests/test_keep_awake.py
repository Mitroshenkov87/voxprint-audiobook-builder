"""keep_awake sets and clears the Windows execution state; elsewhere it is a no-op."""
from infra import keep_awake as ka


def test_sets_and_clears(monkeypatch):
    calls = []
    monkeypatch.setattr(ka, "_set", lambda f: calls.append(f) or True)
    with ka.keep_awake() as on:
        assert on
    assert calls == [ka.ES_CONTINUOUS | ka.ES_SYSTEM_REQUIRED, ka.ES_CONTINUOUS]


def test_noop_off_windows(monkeypatch):
    monkeypatch.setattr(ka.sys, "platform", "linux")
    with ka.keep_awake() as on:
        assert on is False
