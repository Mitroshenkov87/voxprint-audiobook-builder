"""The voice index cache survives parallel writers and a briefly locked target."""
import os
import threading

from infra import voice_repository as vr


def test_parallel_writes_leave_one_valid_file(tmp_path, monkeypatch):
    target = tmp_path / "voice_index_cache.json"
    monkeypatch.setattr(vr, "cache_path", lambda: target)
    ts = [threading.Thread(target=vr._save_cache, args=(b'{"n": %d}' % i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert target.read_bytes().startswith(b'{"n": ')
    assert not list(tmp_path.glob("*.tmp"))


def test_rename_is_retried_when_locked(tmp_path, monkeypatch):
    target = tmp_path / "voice_index_cache.json"
    monkeypatch.setattr(vr, "cache_path", lambda: target)
    real, calls = os.replace, {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(32, "in use")
        real(a, b)

    monkeypatch.setattr(vr.os, "replace", flaky)
    vr._save_cache(b"{}")
    assert target.read_bytes() == b"{}" and calls["n"] == 3
