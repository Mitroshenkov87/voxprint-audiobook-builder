"""Model download locks leave no empty ``.<model>.lock`` file behind (build 667 left one per model)."""
import threading
import time
from pathlib import Path

from infra import auto_repair
from infra import model_downloader as md
from tests.test_download_watch import fast_watch  # noqa: F401  (fixture)
from tests.test_model_mirrors import FILES, REPO, SHA_A, manifest  # noqa: F401  (fixture)


def _snap(calls=None, delay=0.0):
    def snap(repo_id, local_dir, revision=None, **kw):
        if calls is not None:
            calls.append(repo_id)
        time.sleep(delay)
        for n, b in FILES.items():
            t = Path(local_dir) / Path(*n.split("/"))
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(b)
        return str(local_dir)
    return snap


def _locks(folder: Path):
    return sorted(p.name for p in folder.glob(".*.lock"))


def test_a_finished_download_removes_its_lock_file(manifest, fast_watch):
    path = md.ensure_model(REPO, snapshot_download=_snap(), revision=SHA_A, hf_probe=lambda r: True,
                           mirror_manifest=manifest)
    assert md.verify_local_model(path)
    assert _locks(path.parent) == []


def test_two_requests_still_download_once_and_leave_no_lock(manifest, fast_watch):
    calls, out = [], []

    def run():
        out.append(md.ensure_model(REPO, snapshot_download=_snap(calls, 0.3), revision=SHA_A, hf_probe=lambda r: True,
                                   mirror_manifest=manifest))

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
        time.sleep(0.05)
    for t in threads:
        t.join(10)
    assert len(calls) == 1 and len(out) == 2 and out[0] == out[1]
    assert _locks(out[0].parent) == []


def test_a_failed_download_keeps_working_and_the_next_one_cleans_up(manifest, fast_watch, tmp_path):
    def broken(repo_id, local_dir, revision=None, **kw):
        raise RuntimeError("boom")

    try:
        md.ensure_model(REPO, snapshot_download=broken, revision=SHA_A, hf_probe=lambda r: True,
                        mirror_manifest=manifest, mirror_download=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")),
                        hf_mirror_fetch=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    except Exception:  # noqa: BLE001 - any failure is fine here
        pass
    path = md.ensure_model(REPO, snapshot_download=_snap(), revision=SHA_A, hf_probe=lambda r: True,
                           mirror_manifest=manifest)
    assert md.verify_local_model(path) and _locks(path.parent) == []


def test_a_lock_removed_by_its_holder_does_not_let_two_holders_in(tmp_path):
    p = tmp_path / ".m.lock"
    a, b, c = md.ModelLock(p), md.ModelLock(p), md.ModelLock(p)
    assert a.try_acquire() and not b.try_acquire()
    a.release(remove=True)
    assert not p.exists()
    assert b.try_acquire() and not c.try_acquire()    # a fresh file at the path: still exclusive
    b.release(remove=True)
    assert not p.exists()


def test_sweep_removes_only_free_empty_locks(tmp_path):
    (tmp_path / ".Qwen--Qwen3-ASR-1.7B.lock").write_bytes(b"")
    (tmp_path / ".Qwen--Qwen3-TTS-12Hz-1.7B-Base.lock").write_bytes(b"")
    (tmp_path / ".not-ours.lock").write_bytes(b"data")             # not empty: someone else's file
    held = md.ModelLock(tmp_path / ".Helsinki-NLP--opus-mt.lock")
    assert held.try_acquire()                                      # a download in progress
    try:
        assert md.sweep_stale_locks(tmp_path) == 2
        assert _locks(tmp_path) == [".Helsinki-NLP--opus-mt.lock", ".not-ours.lock"]
    finally:
        held.release()
    assert md.sweep_stale_locks(tmp_path / "missing") == 0


def test_check_and_repair_sweeps_stale_locks(monkeypatch, tmp_path):
    from infra import paths

    models = paths.models_dir()
    models.mkdir(parents=True, exist_ok=True)
    (models / ".Qwen--Qwen3-ASR-1.7B.lock").write_bytes(b"")
    monkeypatch.setattr(auto_repair, "check_environment", lambda *a, **k: auto_repair.Item("environment", "env", auto_repair.OK))
    monkeypatch.setattr(auto_repair, "check_modules", lambda *a, **k: [])
    monkeypatch.setattr(auto_repair, "check_ffmpeg", lambda *a, **k: [])
    auto_repair.run(repos=[], tools=[])
    assert _locks(models) == []
