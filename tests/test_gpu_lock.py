"""Shared GPU lock: JSON in the temp directory, re-entrant, stale when the pid is dead or eta is 2 h past."""

import json
import os
import threading
from datetime import datetime, timedelta, timezone

import pytest

from core import gpu_lock
from core.errors import CancelledByUser, NarrationError
from core.i18n import tr


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _foreign(path, *, pid=99999999, eta_delta=timedelta(hours=1), app="movie-dubber", stage="Film"):
    now = datetime.now(timezone.utc)
    payload = {
        "app": app, "pid": pid, "stage": stage,
        "started": _iso(now - timedelta(hours=1)),
        "eta": _iso(now + eta_delta),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_lock_file_is_atomic_json_and_removed_afterwards(tmp_path):
    path = tmp_path / "voxprint-gpu.lock"
    with gpu_lock.hold("My Book", path=path) as held:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) == {"app", "stage", "pid", "started", "eta"}
        assert data["app"] == "audiobook-builder" and data["pid"] == os.getpid() and data["stage"] == "My Book"
        for key in ("started", "eta"):
            assert data[key].endswith("Z")
            assert datetime.fromisoformat(data[key].replace("Z", "+00:00")).tzinfo is not None
        assert not list(tmp_path.glob("*.tmp"))
        held.note(30)
        refreshed = json.loads(path.read_text(encoding="utf-8"))
        assert refreshed["started"] == data["started"]
        assert datetime.fromisoformat(refreshed["eta"].replace("Z", "+00:00")) >= datetime.now(timezone.utc) + timedelta(hours=1)
    assert not path.exists()


def test_reentrant_hold_keeps_the_outer_lock_until_the_outer_block_ends(tmp_path):
    path = tmp_path / "voxprint-gpu.lock"
    with gpu_lock.hold("Outer", path=path):
        with gpu_lock.hold("Inner", path=path):
            assert json.loads(path.read_text(encoding="utf-8"))["stage"] == "Outer"
        assert path.is_file()
    assert not path.exists()


def test_a_dead_pid_is_stale_and_does_not_block(tmp_path):
    path = tmp_path / "voxprint-gpu.lock"
    _foreign(path, eta_delta=timedelta(hours=4))
    with gpu_lock.hold("Book", path=path, timeout=0.2):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["pid"] == os.getpid() and data["app"] == "audiobook-builder"
    assert not path.exists()


def test_eta_more_than_two_hours_past_is_stale_even_if_the_pid_is_alive(tmp_path):
    path = tmp_path / "voxprint-gpu.lock"
    _foreign(path, pid=os.getpid(), eta_delta=timedelta(hours=-3))
    with gpu_lock.hold("Book", path=path, timeout=0.2):
        assert json.loads(path.read_text(encoding="utf-8"))["pid"] == os.getpid()
        assert json.loads(path.read_text(encoding="utf-8"))["app"] == "audiobook-builder"


def test_a_live_holder_times_out_and_is_reported(tmp_path):
    path = tmp_path / "voxprint-gpu.lock"
    _foreign(path, pid=os.getpid())
    original = path.read_text(encoding="utf-8")
    seen = []
    with pytest.raises(gpu_lock.GpuLockTimeout):
        with gpu_lock.hold("Book", on_busy=lambda app, stage: seen.append((app, stage)),
                           path=path, poll_s=0.02, timeout=0.05):
            raise AssertionError("must not enter while the other program holds the lock")
    assert ("movie-dubber", "Film") in seen
    assert path.read_text(encoding="utf-8") == original


def test_an_older_owner_job_record_is_still_read(tmp_path):
    path = tmp_path / "voxprint-gpu.lock"
    now = datetime.now(timezone.utc)
    path.write_text(json.dumps({
        "owner": "movie-dubber", "job": "Film", "pid": os.getpid(),
        "started": _iso(now - timedelta(minutes=5)),
        "eta": _iso(now + timedelta(hours=1)),
    }), encoding="utf-8")
    seen = []
    with pytest.raises(gpu_lock.GpuLockTimeout):
        with gpu_lock.hold("Book", on_busy=lambda app, stage: seen.append((app, stage)),
                           path=path, poll_s=0.02, timeout=0.05):
            raise AssertionError("must not enter")
    assert seen[0] == ("movie-dubber", "Film")


def test_cancel_while_waiting_does_not_delete_the_other_programs_lock(tmp_path):
    path = tmp_path / "voxprint-gpu.lock"
    _foreign(path, pid=os.getpid())
    original = path.read_text(encoding="utf-8")

    class Cancel:
        def check(self):
            raise CancelledByUser()

    with pytest.raises(CancelledByUser):
        with gpu_lock.hold("Book", path=path, cancel=Cancel(), poll_s=0.01, timeout=2):
            raise AssertionError("must not enter while the other program holds the lock")
    assert path.read_text(encoding="utf-8") == original


def test_a_failed_job_still_drops_the_lock(tmp_path):
    path = tmp_path / "voxprint-gpu.lock"
    try:
        with gpu_lock.hold("Book", path=path):
            assert path.is_file()
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not path.exists()


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
    assert seen["data"]["app"] == "audiobook-builder"
    assert seen["data"]["stage"] == "The Test Book"
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
        assert json.loads(path.read_text(encoding="utf-8"))["stage"] == "Outer"
    release.set()
    thread.join(2)
    assert not thread.is_alive()
    assert errors == []
    assert not path.exists()
