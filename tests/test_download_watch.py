"""Stall watchdog, session verdict "Hugging Face is slow", plain-HTTP switch, parallel route probe, progress line."""
import os
import threading
import time
from pathlib import Path

import pytest

from infra import download_watch as dw
from infra import model_downloader as md
from infra import modelscope_mirror as ms
from infra import netroute as nr
from tests.test_model_mirrors import FILES, REPO, SHA_A, _fetcher, _fail_ms, manifest  # noqa: F401  (fixture)


def meter():
    return dw.Meter("Test", lambda: 0)


# ------------------------------------------------------------------------------------------------ watchdog
def test_a_stalled_source_is_abandoned_and_its_helper_is_told_to_stop(tmp_path):
    cancel = threading.Event()
    seen = []

    def blocked():
        while not cancel.is_set():
            time.sleep(0.02)
        seen.append("stopped")

    t0 = time.monotonic()
    with pytest.raises(dw.Stalled):
        dw.run_watched(blocked, tmp_path, meter(), cancel, stall=0.3, poll=0.05, grace=1.0)
    assert time.monotonic() - t0 < 3 and cancel.is_set() and seen == ["stopped"]


def test_growing_data_is_not_a_stall_and_results_and_errors_pass_through(tmp_path):
    def slow_but_alive():
        for i in range(12):
            (tmp_path / f"f{i}.bin").write_bytes(b"x" * 100)
            time.sleep(0.05)
        return "done"

    # The gap between writes is 0.05 s. The stall window has to be much wider than that:
    # a busy Windows runner can pause the watcher long enough to trip a 0.3 s limit.
    assert dw.run_watched(slow_but_alive, tmp_path, meter(), threading.Event(), stall=3.0, poll=0.05) == "done"
    with pytest.raises(ValueError):
        dw.run_watched(lambda: (_ for _ in ()).throw(ValueError("boom")), tmp_path, meter(), threading.Event(), poll=0.05)


def test_incomplete_files_of_the_hub_count_as_progress(tmp_path):
    (tmp_path / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (tmp_path / ".cache" / "huggingface" / "download" / "x.incomplete").write_bytes(b"1234")
    (tmp_path / "a").write_bytes(b"12")
    assert dw.dir_bytes(tmp_path) == 6 and dw.dir_bytes(tmp_path / "missing") == 0


def test_progress_line_is_plain_until_data_arrives_then_shows_size_speed_and_source():
    t = [0.0]
    m = dw.Meter("ModelScope", lambda: 4 * 1024 ** 3, clock=lambda: t[0])
    tr = lambda key, **kw: key + str(sorted(kw.items()))          # noqa: E731 - shows the arguments
    assert m.text(tr, "M", 3).startswith("progress.downloading")
    m.sample(0)
    t[0] = 10.0
    m.sample(100 * 1024 ** 2)
    line = m.text(tr, "M", 2)
    assert line.startswith("progress.detail[") and "ModelScope" in line and "100 MB" in line and "4.00 GB" in line and "MB/s" in line
    m2 = dw.Meter("X", lambda: 0)
    m2.sample(5)
    m2.sample(2048)
    assert m2.text(tr, "M", 0).startswith("progress.detail_unknown")


# ------------------------------------------------------------------------------------------------ model download: switch + verdict
@pytest.fixture
def fast_watch(monkeypatch, tmp_path):
    monkeypatch.setattr(dw, "STALL_SECONDS", 0.4)
    monkeypatch.setattr(dw, "POLL", 0.05)
    monkeypatch.setattr(dw, "ABORT_GRACE", 1.0)
    monkeypatch.setenv("VOXPRINT_NO_EXTERNAL_MODELS", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "h"))
    monkeypatch.chdir(tmp_path)


def _ms_ok(order):
    def ms_download(repo, dest, progress, expected):
        order.append("ms")
        for n, b in FILES.items():
            t = Path(dest).joinpath(*n.split("/"))
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(b)
        progress(1.0)
    return ms_download


def test_stalled_hugging_face_switches_to_modelscope_and_later_models_start_there(manifest, fast_watch, monkeypatch):
    order = []

    def hf_stalls(repo_id, local_dir, tqdm_class=None, **kw):       # connected, but no byte ever arrives (the xet/CAS symptom)
        order.append("hf")
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        bar = tqdm_class(total=100, unit="B")
        for _ in range(200):
            time.sleep(0.02)
            bar.update(0)                                            # alive, zero bytes; the watchdog's abort is raised here

    got = md.ensure_model(REPO, snapshot_download=hf_stalls, revision=SHA_A, hf_probe=lambda r: True,
                          mirror_download=_ms_ok(order), mirror_manifest=manifest)
    assert order == ["hf", "ms"] and md.verify_local_model(got)
    assert ms.hf_verdict() is False                                   # remembered for the session

    # the next model: no probe at all (it would raise), ModelScope first, Hugging Face is not even tried
    monkeypatch.setattr(ms, "hf_is_fast", lambda *a, **k: (_ for _ in ()).throw(AssertionError("probe repeated")))
    import shutil
    shutil.rmtree(got)
    order.clear()
    got2 = md.ensure_model(REPO, snapshot_download=lambda **kw: order.append("hf"), revision=SHA_A,
                           mirror_download=_ms_ok(order), mirror_manifest=manifest)
    assert order == ["ms"] and md.verify_local_model(got2)


def test_a_successful_hugging_face_download_remembers_that_it_is_fast(manifest, fast_watch):
    def hf(repo_id, local_dir, **kw):
        for n, b in FILES.items():
            t = Path(local_dir).joinpath(*n.split("/"))
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(b)

    md.ensure_model(REPO, snapshot_download=hf, revision=SHA_A, hf_probe=lambda r: True, mirror_download=_fail_ms,
                    mirror_manifest=manifest)
    assert ms.hf_verdict() is True


def test_the_probe_is_capped(monkeypatch):
    monkeypatch.setattr(md, "PROBE_CAP", 0.2)
    t0 = time.monotonic()
    assert md._hf_fast("Org/X", lambda r: time.sleep(3) or True) is False
    assert time.monotonic() - t0 < 1.5
    assert ms.hf_verdict() is None                                    # an injected probe does not set the session verdict
    monkeypatch.setattr(ms, "hf_is_fast", lambda r: time.sleep(3) or True)
    assert md._hf_fast("Org/X", None) is False and ms.hf_verdict() is False


# ------------------------------------------------------------------------------------------------ routes, xet
def test_routes_are_probed_in_parallel_and_the_best_working_one_wins(monkeypatch):
    a, b, c = nr.Route("default", "", False), nr.Route("tun0", "10.0.0.2", False), nr.Route("wlan0", "10.0.0.3", False)

    def probe(host, port, route, timeout):
        if route is a:
            time.sleep(1.0)                       # the default route hangs
            return False
        return route is c                         # tun0 refuses, wlan0 connects

    monkeypatch.setattr(nr, "_probe", probe)
    t0 = time.monotonic()
    assert nr._first_reachable("h", [a, b, c], cap=0.3) is c
    assert time.monotonic() - t0 < 0.8
    monkeypatch.setattr(nr, "_probe", lambda host, port, route, timeout: route is a)
    assert nr._first_reachable("h", [a, b, c], cap=2.0) is a          # the default wins if it works
    monkeypatch.setattr(nr, "_probe", lambda *x: False)
    assert nr._first_reachable("h", [a, b, c], cap=2.0) is None


def test_xet_is_switched_off_unless_allowed(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.delenv("VOXPRINT_ALLOW_XET", raising=False)
    nr.disable_xet()
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    monkeypatch.setenv("VOXPRINT_ALLOW_XET", "1")
    nr.disable_xet()
    assert os.environ["HF_HUB_DISABLE_XET"] == "0"


# ------------------------------------------------------------------------------------------------ finishing a download
def _write_partial(repo, files=FILES):
    part = md.local_dir_for(repo).with_name(md.local_dir_for(repo).name + ".partial")
    for n, b in files.items():
        t = part.joinpath(*n.split("/"))
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes(b)
    return part


def _no_network(*a, **k):
    raise AssertionError("nothing may be downloaded: the .partial folder is complete")


def test_a_complete_partial_folder_is_finished_without_any_download(manifest, fast_watch):
    part = _write_partial(REPO)
    (part / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (part / ".cache" / "huggingface" / "download" / "x.lock").write_bytes(b"")
    got = md.ensure_model(REPO, snapshot_download=_no_network, revision=SHA_A, hf_probe=_no_network,
                          mirror_download=_no_network, hf_mirror_fetch=_no_network, mirror_manifest=manifest)
    assert md.verify_local_model(got) and not part.exists() and not (got / ".cache").exists()
    assert (got / ".revision").read_text() == SHA_A


def test_an_incomplete_partial_folder_is_not_taken_for_complete(manifest, fast_watch):
    part = _write_partial(REPO, {"config.json": FILES["config.json"], "model.safetensors": b"short"})
    assert not md.partial_is_complete(part, REPO, SHA_A, manifest)
    part = _write_partial(REPO)
    assert md.partial_is_complete(part, REPO, SHA_A, manifest)
    (part / "model.safetensors.x.incomplete").write_bytes(b"1")
    assert not md.partial_is_complete(part, REPO, SHA_A, manifest)


def _flaky_rename(monkeypatch, failures):
    real = Path.rename
    calls = []

    def rename(self, target):
        if self.name.endswith(".partial") and (failures is None or len(calls) < failures):
            calls.append(1)
            raise PermissionError(5, "Access is denied (WinError 5)")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    return calls


def test_rename_is_retried_when_windows_denies_access(manifest, fast_watch, monkeypatch):
    sleeps = []
    monkeypatch.setattr(md, "_sleep", sleeps.append)
    calls = _flaky_rename(monkeypatch, 3)
    _write_partial(REPO)
    got = md.ensure_model(REPO, snapshot_download=_no_network, revision=SHA_A, hf_probe=_no_network,
                          mirror_download=_no_network, mirror_manifest=manifest)
    assert md.verify_local_model(got) and len(calls) == 3 and len(sleeps) == 3 and sleeps == sorted(sleeps)


def test_when_rename_never_works_the_folder_is_copied(manifest, fast_watch, monkeypatch):
    monkeypatch.setattr(md, "_sleep", lambda s: None)
    calls = _flaky_rename(monkeypatch, None)
    part = _write_partial(REPO)
    got = md.ensure_model(REPO, snapshot_download=_no_network, revision=SHA_A, hf_probe=_no_network,
                          mirror_download=_no_network, mirror_manifest=manifest)
    assert len(calls) == md.FINALIZE_ATTEMPTS and md.verify_local_model(got) and not part.exists()
    assert (got / "model.safetensors").read_bytes() == FILES["model.safetensors"]


def test_when_everything_fails_the_error_says_nothing_must_be_downloaded_again(manifest, fast_watch, monkeypatch):
    from core.errors import ModelDownloadError

    monkeypatch.setattr(md, "_sleep", lambda s: None)
    _flaky_rename(monkeypatch, None)
    monkeypatch.setattr(md, "_move_finished", lambda *a, **k: (_ for _ in ()).throw(PermissionError("move denied")))
    monkeypatch.setattr(md.shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(PermissionError("copy denied")))
    part = _write_partial(REPO)
    with pytest.raises(ModelDownloadError) as ei:
        md.ensure_model(REPO, snapshot_download=_no_network, revision=SHA_A, hf_probe=_no_network,
                        mirror_download=_no_network, mirror_manifest=manifest)
    assert "WinError 5" in (ei.value.details or "") and "copy denied" in (ei.value.details or "")
    assert part.is_dir() and not md.local_dir_for(REPO).exists()       # the complete .partial stays: the next try only renames


def test_all_files_present_is_not_a_stall(tmp_path):
    cancel = threading.Event()

    def hangs_on_verification():                  # everything is on disk, the hub still talks to the server
        while not cancel.is_set():
            time.sleep(0.02)

    t0 = time.monotonic()
    dw.run_watched(hangs_on_verification, tmp_path, meter(), cancel, stall=0.3, poll=0.05, grace=1.0, idle_ok=lambda: True)
    assert time.monotonic() - t0 < 3 and cancel.is_set()


def test_a_disk_error_does_not_make_hugging_face_look_slow(manifest, fast_watch):
    def hf(**kw):
        raise PermissionError("disk")

    with pytest.raises(Exception):
        md.ensure_model(REPO, snapshot_download=hf, revision=SHA_A, hf_probe=lambda r: True, mirror_download=_fail_ms,
                        hf_mirror_fetch=_fetcher({}), mirror_manifest=manifest)
    assert ms.hf_verdict() is not False
    assert md._is_network_failure(dw.Stalled("x")) and md._is_network_failure(TimeoutError()) and not md._is_network_failure(PermissionError())


# ------------------------------------------------------------------------------------------------ one download per model
def test_opening_a_held_lock_file_counts_as_not_acquired(tmp_path, monkeypatch):
    """Windows raises PermissionError on open() of a lock another thread holds (run 42). That is 'taken', not a crash."""
    real_open = open
    opens = []

    def held(path, mode="r", *args, **kwargs):
        if mode == "a+b" and str(path).endswith(".lock"):
            opens.append(path)
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", held)
    assert md.ModelLock(tmp_path / ".m.lock").try_acquire() is False
    assert len(opens) == 5


def test_a_transient_lock_open_error_is_retried(tmp_path, monkeypatch):
    real_open = open
    opens = []

    def flaky(path, mode="r", *args, **kwargs):
        if mode == "a+b" and str(path).endswith(".lock"):
            opens.append(path)
            if len(opens) < 3:
                raise PermissionError(13, "in use")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", flaky)
    lock = md.ModelLock(tmp_path / ".m.lock")
    assert lock.try_acquire() is True
    lock.release(remove=True)
    assert len(opens) == 3


def test_the_model_lock_excludes_a_second_holder_and_is_released(tmp_path):
    a, b = md.ModelLock(tmp_path / ".m.lock"), md.ModelLock(tmp_path / ".m.lock")
    assert a.try_acquire() and not b.try_acquire()
    a.release()
    assert b.try_acquire()
    b.release()


def test_the_lock_dies_with_its_process(tmp_path):
    import subprocess
    import sys

    code = ("import sys,time; sys.path.insert(0, %r)\nfrom infra.model_downloader import ModelLock\n"
            "l = ModelLock(__import__('pathlib').Path(%r)); assert l.try_acquire(); print('held', flush=True); time.sleep(60)"
            % (str(Path(md.__file__).resolve().parent.parent), str(tmp_path / ".m.lock")))
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "held"
        mine = md.ModelLock(tmp_path / ".m.lock")
        assert not mine.try_acquire()                  # another process holds it
        proc.kill()
        proc.wait()
        deadline = time.monotonic() + 5
        while not mine.try_acquire():                  # a killed holder never leaves a stale lock
            assert time.monotonic() < deadline
            time.sleep(0.05)
        mine.release()
    finally:
        proc.kill()


def test_two_requests_for_one_model_download_it_once(manifest, fast_watch):
    calls, first_in = [], threading.Event()

    def snap(repo_id, local_dir, revision=None, **kw):
        calls.append(repo_id)
        first_in.set()
        time.sleep(0.4)                                # the second request arrives while this one is downloading
        for n, b in FILES.items():
            t = Path(local_dir) / Path(*n.split("/"))
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(b)
        return str(local_dir)

    out, waits = [], []

    def run(progress):
        out.append(md.ensure_model(REPO, progress, snapshot_download=snap, revision=SHA_A, hf_probe=lambda r: True,
                                   mirror_manifest=manifest))

    t1 = threading.Thread(target=run, args=(lambda *a: None,))
    t1.start()
    assert first_in.wait(5)
    run(lambda stage, f, m: waits.append(m))
    t1.join(10)
    assert len(calls) == 1 and len(out) == 2 and out[0] == out[1] and md.verify_local_model(out[0])
    assert waits                                       # the second request said it was waiting
