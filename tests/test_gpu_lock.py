"""Shared GPU lock: JSON in the temp directory, re-entrant, stale when the pid is dead or eta is 2 h past."""

import json
import os
import threading
from datetime import datetime, timedelta, timezone

import pytest

from core import gpu_lock
from core.errors import CancelledByUser, NarrationError
from core.i18n import tr


def _foreign(path, *, pid=424242, eta_delta=timedelta(hours=1), owner="movie-dubber", job="Film"):
    now = datetime.now(timezone.utc)
    payload = {
        "owner": owner, "pid": pid, "job": job,
        "started": gpu_lock._iso(now - timedelta(hours=1)),
        "eta": gpu_lock._iso(now + eta_delta),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_lock_file_is_atomic_json_and_removed_afterwards(tmp_path):
    path = tmp_path / "voxprint-gpu.lock"
    with gpu_lock.hold("My Book", path=path) as held:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert list(data) == ["owner", "pid", "job", "started", "eta"]
        assert data["owner"] == "audiobook-builder" and data["pid"] == os.getpid() and data["job"] == "My Book"
        for key in ("started", "eta"):
            parsed = datetime.fromisoformat(data[key])
            assert parsed.tzinfo is not None and data[key].endswith("+00:00")
        assert not list(tmp_path.glob("*.tmp"))
        held.note(30)
        refreshed = json.loads(path.read_text(encoding="utf-8"))
        assert refreshed["started"] == data["started"]
        assert datetime.fromisoformat(refreshed["eta"]) >= datetime.now(timezone.utc) + timedelta(hours=1)
    assert not path.exists()
    assert gpu_lock._depth == 0


def test_reentrant_hold_keeps_the_outer_lock_until_the_outer_block_ends(tmp_path):
    path = tmp_path / "voxprint-gpu.lock"
    with gpu_lock.hold("Outer", path=path) as outer:
        with gpu_lock.hold("Inner", path=path) as inner:
            assert inner is outer
            assert json.loads(path.read_text(encoding="utf-8"))["job"] == "Outer"
        assert path.is_file()
    assert not path.exists()


def test_a_dead_pid_is_stale_and_does_not_block(tmp_path):
    path = tmp_path / "voxprint-gpu.lock"
    _foreign(path, eta_delta=timedelta(hours=4))

    def boom(_seconds):
        raise AssertionError("a dead holder must not make us wait")

    with gpu_lock.hold("Book", path=path, sleep=boom, pid_alive_fn=lambda _pid: False):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["pid"] == os.getpid() and data["owner"] == "audiobook-builder"
    assert not path.exists()


def test_eta_more_than_two_hours_past_is_stale_even_if_the_pid_looks_alive(tmp_path):
    path = tmp_path / "voxprint-gpu.lock"
    _foreign(path, eta_delta=timedelta(hours=-3))

    def boom(_seconds):
        raise AssertionError("an expired eta must not make us wait")

    with gpu_lock.hold("Book", path=path, sleep=boom, pid_alive_fn=lambda pid: True):
        assert json.loads(path.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_a_live_holder_is_waited_on_and_reported(tmp_path, caplog):
    path = tmp_path / "voxprint-gpu.lock"
    _foreign(path)
    state = {"alive": True}
    seen = []

    def alive(pid):
        return True if pid == os.getpid() else state["alive"]

    def on_busy(owner, job):
        seen.append((owner, job))

    def sleep(_seconds):
        state["alive"] = False

    with caplog.at_level("INFO", logger="voxprint.gpu_lock"):
        with gpu_lock.hold("Book", on_busy=on_busy, path=path, poll_s=0.05, sleep=sleep, pid_alive_fn=alive):
            assert json.loads(path.read_text(encoding="utf-8"))["pid"] == os.getpid()
    assert seen == [("movie-dubber", "Film")]
    assert "GPU busy: movie-dubber Film" in caplog.text


def test_cancel_while_waiting_does_not_delete_the_other_programs_lock(tmp_path):
    path = tmp_path / "voxprint-gpu.lock"
    _foreign(path)
    original = path.read_text(encoding="utf-8")

    class Cancel:
        def __init__(self):
            self.n = 0

        def check(self):
            self.n += 1
            if self.n > 1:
                raise CancelledByUser()

    with pytest.raises(CancelledByUser):
        with gpu_lock.hold("Book", path=path, cancel=Cancel(), poll_s=0.01, sleep=lambda _s: None,
                           pid_alive_fn=lambda pid: pid == os.getpid() or True):
            raise AssertionError("must not enter while the other program holds the lock")
    assert path.read_text(encoding="utf-8") == original
    assert gpu_lock._depth == 0


def test_a_failed_job_still_drops_the_lock(tmp_path):
    path = tmp_path / "voxprint-gpu.lock"
    try:
        with gpu_lock.hold("Book", path=path):
            assert path.is_file()
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not path.exists()
    assert gpu_lock._depth == 0


def test_default_path_is_the_temp_directory():
    import tempfile
    from pathlib import Path

    assert gpu_lock.lock_path() == Path(tempfile.gettempdir()) / "voxprint-gpu.lock"


def test_narrate_book_holds_the_lock_for_the_job_and_removes_it(tmp_path, monkeypatch):
    from tests.test_narration import FakeEngine, run

    path = tmp_path / "lock" / "voxprint-gpu.lock"
    monkeypatch.setattr(gpu_lock, "lock_path", lambda: path)
    seen = {}

    class Watching(FakeEngine):
        def synthesize(self, text):
            seen["held"] = path.is_file()
            seen["data"] = json.loads(path.read_text(encoding="utf-8"))
            return super().synthesize(text)

    run(tmp_path, engine=Watching())
    assert seen["held"] is True
    assert seen["data"]["owner"] == "audiobook-builder"
    assert seen["data"]["job"] == "The Test Book"
    assert seen["data"]["pid"] == os.getpid()
    assert not path.exists()


def test_narrate_book_drops_the_lock_when_synthesis_fails(tmp_path, monkeypatch):
    from tests.test_narration import FakeEngine, run

    path = tmp_path / "voxprint-gpu.lock"
    monkeypatch.setattr(gpu_lock, "lock_path", lambda: path)

    seen = []

    class Boom(FakeEngine):
        def synthesize(self, text):
            seen.append(path.is_file())
            raise RuntimeError("stop")

    with pytest.raises(NarrationError):
        run(tmp_path, engine=Boom())
    assert seen and all(seen)
    assert not path.exists()
    assert gpu_lock._depth == 0


def test_busy_message_uses_the_catalog(monkeypatch):
    from core import i18n

    i18n.set_language("en")
    assert tr("narr.gpu_busy", owner="movie-dubber", job="Film") == "GPU busy: movie-dubber Film"


def test_two_threads_in_one_process_share_the_lock(tmp_path):
    path = tmp_path / "voxprint-gpu.lock"
    entered = threading.Event()
    release = threading.Event()
    errors = []

    def outer():
        try:
            with gpu_lock.hold("Outer", path=path):
                entered.set()
                assert release.wait(2)
        except Exception as exc:  # noqa: BLE001 - the thread must report it
            errors.append(exc)

    thread = threading.Thread(target=outer)
    thread.start()
    assert entered.wait(2)
    with gpu_lock.hold("Inner", path=path):
        assert json.loads(path.read_text(encoding="utf-8"))["job"] == "Outer"
    release.set()
    thread.join(2)
    assert not thread.is_alive()
    assert errors == []
    assert not path.exists()
    assert gpu_lock._depth == 0
