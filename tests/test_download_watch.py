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

    assert dw.run_watched(slow_but_alive, tmp_path, meter(), threading.Event(), stall=0.3, poll=0.05) == "done"
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
