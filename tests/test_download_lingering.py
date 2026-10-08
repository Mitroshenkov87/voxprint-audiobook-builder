"""A source abandoned by the stall watchdog must not collide with the next one (Windows WinError 32 / 5), a source that
stalled after delivering data is resumed before switching, and locked files are not network failures."""
import io
import logging
import threading
import time
from pathlib import Path

import pytest

from infra import download_watch as dw
from infra import model_downloader as md
from infra import parallel_download as pd
from tests.test_download_watch import _ms_ok, fast_watch  # noqa: F401  (fixture)
from tests.test_model_mirrors import FILES, REPO, SHA_A, manifest  # noqa: F401  (fixture)


def meter():
    return dw.Meter("Test", lambda: 0)


def test_the_next_download_in_a_folder_waits_for_a_lingering_helper(tmp_path, caplog):
    cancel = threading.Event()
    state = {"first_done": False, "second_saw": None}

    def stuck_in_a_read():                      # ignores the cancel for a while, like a blocked socket read
        while not cancel.is_set():
            time.sleep(0.02)
        time.sleep(0.6)
        state["first_done"] = True

    with pytest.raises(dw.Stalled) as ei:
        dw.run_watched(stuck_in_a_read, tmp_path, meter(), cancel, stall=0.2, poll=0.05, grace=0.05)
    assert not ei.value.progressed and not state["first_done"]

    def next_source():
        state["second_saw"] = state["first_done"]
        return "ok"

    with caplog.at_level(logging.INFO, logger="voxprint.models"):
        assert dw.run_watched(next_source, tmp_path, meter(), threading.Event(), poll=0.05) == "ok"
    assert state["second_saw"] is True
    assert "waiting for the previous download to release files" in caplog.text
    assert dw.wait_released(tmp_path)           # forgotten once it ended


def test_a_stall_after_data_is_marked_as_progressed(tmp_path):
    cancel = threading.Event()

    def some_then_nothing():
        (tmp_path / "a.incomplete").write_bytes(b"x" * 100)
        while not cancel.is_set():
            time.sleep(0.02)

    with pytest.raises(dw.Stalled) as ei:
        dw.run_watched(some_then_nothing, tmp_path, meter(), cancel, stall=0.3, poll=0.05, grace=1.0)
    assert ei.value.progressed


def _ms_stalls_once(order, calls):
    ok = _ms_ok(order)

    def ms_download(repo, dest, progress, expected):
        calls.append(1)
        if len(calls) == 1:                     # delivers part of a file, then the route goes silent
            order.append("ms")
            Path(dest).mkdir(parents=True, exist_ok=True)
            (Path(dest) / "model.safetensors.incomplete").write_bytes(b"x" * 10)
            for _ in range(300):
                time.sleep(0.02)
                progress(0.1)                   # the watchdog's abort is raised here
            return
        Path(dest, "model.safetensors.incomplete").unlink(missing_ok=True)
        ok(repo, dest, progress, expected)
    return ms_download


def test_a_source_that_stalls_after_progress_is_resumed_before_switching(manifest, fast_watch, monkeypatch):  # noqa: F811
    monkeypatch.setattr(md, "STALL_PAUSE", 0)
    order, calls = [], []

    def hf_must_not_run(**kw):
        raise AssertionError("Hugging Face tried although ModelScope was only resumed")

    got = md.ensure_model(REPO, snapshot_download=hf_must_not_run, revision=SHA_A, hf_probe=lambda r: False,
                          mirror_download=_ms_stalls_once(order, calls), mirror_manifest=manifest)
    assert order == ["ms", "ms"] and md.verify_local_model(got)


def test_a_source_that_never_gives_data_is_left_at_once(manifest, fast_watch, monkeypatch):  # noqa: F811
    monkeypatch.setattr(md, "STALL_PAUSE", 0)
    order = []

    def ms_silent(repo, dest, progress, expected):
        order.append("ms")
        for _ in range(300):
            time.sleep(0.02)
            progress(0.0)

    def hf(repo_id, local_dir, **kw):
        order.append("hf")
        for n, b in FILES.items():
            t = Path(local_dir).joinpath(*n.split("/"))
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(b)

    md.ensure_model(REPO, snapshot_download=hf, revision=SHA_A, hf_probe=lambda r: False,
                    mirror_download=ms_silent, mirror_manifest=manifest)
    assert order == ["ms", "hf"]


def _locked_error(code=32):
    exc = PermissionError(13, "file used by another process")
    exc.winerror = code
    return exc


def test_a_briefly_locked_file_is_retried(monkeypatch, tmp_path):
    monkeypatch.setattr(pd, "LOCK_PAUSE", 0)
    tries = []

    def flaky():
        tries.append(1)
        if len(tries) < 3:
            raise _locked_error(32 if len(tries) == 1 else 5)
        return "ok"

    assert pd.retry_locked(flaky) == "ok" and len(tries) == 3
    with pytest.raises(PermissionError):           # a real permission problem (no WinError 32 / 5) is not retried
        pd.retry_locked(lambda: (_ for _ in ()).throw(PermissionError("denied")))
    always = []
    with pytest.raises(PermissionError):
        pd.retry_locked(lambda: always.append(1) or (_ for _ in ()).throw(_locked_error()))
    assert len(always) == pd.LOCK_RETRIES + 1


def test_a_locked_file_is_not_a_network_failure():
    try:
        try:
            raise _locked_error()
        except OSError as inner:
            raise pd.ParallelError("wrapped") from inner
    except pd.ParallelError as exc:
        wrapped = exc
    assert not md._is_network_failure(_locked_error()) and not md._is_network_failure(wrapped)
    assert md._is_network_failure(dw.Stalled("x"))


class _Resp(io.BytesIO):
    status = 206


def test_an_abandoned_range_download_is_not_turned_into_a_fallback(tmp_path):
    def opener(req, timeout):
        return _Resp(b"y" * 1000)

    def on_bytes(n):
        raise dw.Stalled("download abandoned (no data)")

    with pytest.raises(dw.Stalled):
        pd.download_file("https://x/f", tmp_path / "f.bin", 1000, opener=opener, on_bytes=on_bytes, conn=4)


def test_startup_log_settings_are_short_and_missing_torch_is_said_once(monkeypatch, caplog):
    import builtins

    from infra import diagnostics, vram_optimizer

    out = diagnostics._short_values({"lang.txt": "en", "modules.json": {"m": ["x" * 50] * 40}})
    assert out["lang.txt"] == "en" and out["modules.json"].startswith("<")

    real_import = builtins.__import__

    def no_torch(name, *a, **k):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    monkeypatch.setattr(vram_optimizer, "_no_torch_logged", False)
    with caplog.at_level(logging.INFO, logger=vram_optimizer.log.name):
        assert not vram_optimizer.detect_gpu().available and not vram_optimizer.detect_gpu().available
    assert caplog.text.count("GPU detection skipped") == 1 and "GPU detection failed" not in caplog.text
