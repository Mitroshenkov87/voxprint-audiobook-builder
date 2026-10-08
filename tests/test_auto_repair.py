"""Auto-repair (infra/auto_repair.py): hash check of every model file, re-download of missing/broken parts, summary."""
import hashlib
import json

import pytest

from core.errors import CancelledByUser
from core.events import CancelToken
from infra import auto_repair as ar
from infra import install_state
from infra import model_downloader as md
from infra import model_mirrors
from infra import paths

REPO = "Owner/Model"
FILES = {"config.json": b"{}", "model.safetensors": b"weights" * 100}
REV = "a" * 40


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    data = {"schema": 1, "models": {REPO: {"source_revision": REV, "mirror_repo": "m/m", "mirror_revision": "b" * 40,
                                           "files": {n: {"size": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                                                     for n, b in FILES.items()}}}}
    p = tmp_path / "mirrors.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    real = model_mirrors.load
    monkeypatch.setattr(model_mirrors, "load", lambda path=None: real(path or p))
    return p


def install(folder, damage=None):
    folder.mkdir(parents=True, exist_ok=True)
    for n, b in FILES.items():
        (folder / n).write_bytes(b if n != damage else b"x" * len(b))
    (folder / ".revision").write_text(REV, encoding="utf-8")
    return folder


class FakeEnsure:
    """Stands in for ensure_model: completes the .partial folder with the good files (like a resumed download)."""

    def __init__(self):
        self.calls = []

    def __call__(self, repo, progress, root=None, **kw):
        self.calls.append((repo, root, kw))
        target = md.local_dir_for(repo, root)
        partial = target.with_name(target.name + ".partial")
        src = partial if partial.is_dir() else target
        src.mkdir(parents=True, exist_ok=True)
        for n, b in FILES.items():
            if not (src / n).exists():
                (src / n).write_bytes(b)
        if src != target:
            src.replace(target)
        return target


def fixed_env():
    return dict(env_verify=lambda: install_state.HealthReport(), modules_api=None, ffmpeg_ensure=lambda p: "ffmpeg")


def test_healthy_model_is_only_checked(manifest):
    install(paths.models_dir() / "Owner--Model")
    ens = FakeEnsure()
    rep = ar.run(repos=[REPO], ensure=ens, **fixed_env())
    assert rep.ok and ens.calls == [] and [i.status for i in rep.items if i.kind == "model"] == [ar.OK]
    assert "0 failed" in rep.summary() or "0" in rep.summary()


def test_damaged_file_is_replaced_and_only_it_is_fetched(manifest):
    folder = install(paths.models_dir() / "Owner--Model", damage="model.safetensors")
    ens = FakeEnsure()
    seen = []
    rep = ar.run(lambda f, m: seen.append(f), repos=[REPO], ensure=ens, **fixed_env())
    item = [i for i in rep.items if i.kind == "model"][0]
    assert item.status == ar.REPAIRED and ens.calls[0][1] == folder.parent and ens.calls[0][2]["revision"] == REV
    assert (folder / "model.safetensors").read_bytes() == FILES["model.safetensors"] and (folder / "config.json").exists()
    assert seen == sorted(seen) and seen[-1] == 1.0                         # progress never goes back


def test_missing_model_is_downloaded_with_the_normal_path(manifest):
    ens = FakeEnsure()
    rep = ar.run(repos=[REPO], ensure=ens, **fixed_env())
    assert [i.status for i in rep.items if i.kind == "model"] == [ar.DOWNLOADED] and ens.calls[0][1] is None


def test_still_broken_after_download_is_reported(manifest):
    install(paths.models_dir() / "Owner--Model", damage="model.safetensors")

    def bad_ensure(repo, progress, root=None, **kw):
        t = md.local_dir_for(repo, root)
        t.with_name(t.name + ".partial").replace(t)
        (t / "model.safetensors").write_bytes(b"still wrong")
        return t
    rep = ar.run(repos=[REPO], ensure=bad_ensure, **fixed_env())
    assert not rep.ok and "Model" in rep.summary()


def test_other_revision_and_unknown_models_are_not_hash_checked(manifest, tmp_path):
    folder = install(paths.models_dir() / "Owner--Model", damage="model.safetensors")
    (folder / ".revision").write_text("c" * 40, encoding="utf-8")
    other = install(paths.models_dir() / "X--Y")
    ens = FakeEnsure()
    rep = ar.run(repos=[REPO, "X/Y"], ensure=ens, **fixed_env())
    assert [i.status for i in rep.items if i.kind == "model"] == [ar.OK, ar.OK] and ens.calls == [] and other.exists()


def test_environment_repair_and_frozen_report(manifest, monkeypatch):
    broken = install_state.HealthReport()
    broken.add(install_state.R_MISSING, "torch")
    calls = []
    item = ar.check_environment(lambda f, m: None, lambda: broken, lambda p: calls.append(1) or (0, "ok"))
    assert item.status == ar.REPAIRED and calls
    monkeypatch.setattr("sys.frozen", True, raising=False)
    item = ar.check_environment(lambda f, m: None, lambda: broken)
    assert item.status == ar.FAILED and "torch" in item.detail


def test_thin_modules_missing_are_installed():
    class Api:
        installed = []

        def is_thin(self):
            return True

        def load_manifest(self, offline_ok=True):
            return {}

        def modules(self, man):
            from infra.modules import Module
            return [Module("core", "Core", True, 1, 1, [], True), Module("torch", "PyTorch", True, 1, 1, [], False)]

        def install(self, ids, progress, cancelled):
            self.installed += ids
    api = Api()
    items = ar.check_modules(lambda f, m: None, CancelToken(), api)
    assert [(i.name, i.status) for i in items] == [("Core", ar.OK), ("PyTorch", ar.DOWNLOADED)] and api.installed == ["torch"]


def test_cancel_stops(manifest):
    install(paths.models_dir() / "Owner--Model")
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(CancelledByUser):
        ar.run(repos=[REPO], ensure=FakeEnsure(), cancel=tok, **fixed_env())


def test_settings_button_runs_and_shows_the_summary(app, lib):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from core import i18n
    from tests.test_studio import make_studio, wait_for
    i18n.set_language("en")
    s = make_studio(lib)
    d = s.settings_dialog()
    rep = ar.Report([ar.Item("model", "Base", ar.REPAIRED)])
    d.autorepair_job = lambda progress, cancel: (progress(0.5, "Checking Base"), rep)[1]
    assert d.btn_autorepair.text() == "Check && repair" and "checksum" in d.btn_autorepair.toolTip() and d.btn_repair is d.btn_autorepair
    assert d.toggle_autorepair()
    assert wait_for(lambda: not d.autorepair_running) and d.autorepair_worker.wait(3000)
    for _ in range(5):
        QApplication.processEvents()
    assert "1 repaired or downloaded" in d.lbl_autorepair_status.text() and not d.bar_autorepair.isVisibleTo(d)
    s.shutdown()


from tests.test_studio import app, lib  # noqa: E402,F401  (fixtures)


def test_cli_flag_runs_the_same_job_and_writes_a_log(monkeypatch, capsys):
    import main as vox_main
    calls = []
    monkeypatch.setattr(ar, "run", lambda progress: calls.append(1) or ar.Report([ar.Item("model", "Base", ar.FAILED, "x")]))
    assert vox_main.main(["voxprint", "--auto-repair"]) == 1 and calls
    assert "FAILED     model: Base - x" in (paths.logs_dir() / "auto_repair.txt").read_text(encoding="utf-8")
