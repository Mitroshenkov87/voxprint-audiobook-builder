"""Auto-repair (infra/auto_repair.py): the environment check, the thin-build modules, ffmpeg, and the model store.

The model store belongs to VOXPrint MDL: Check & repair asks it to verify and repair (``voxprint-mdl verify --repair``);
this app never re-downloads a model itself.
"""
import pytest

from core.errors import CancelledByUser
from core.events import CancelToken
from infra import auto_repair as ar
from infra import install_state


def fixed_env():
    return dict(env_verify=lambda: install_state.HealthReport(), modules_api=None, ffmpeg_ensure=lambda p: "ffmpeg")


def test_healthy_environment_only_checks():
    rep = ar.run(models_verify=lambda progress: True, **fixed_env())
    kinds = [i.kind for i in rep.items]
    assert rep.ok and kinds == ["environment", "ffmpeg", "model"]


def test_model_store_failure_is_reported(monkeypatch):
    monkeypatch.setenv("VOXPRINT_SKIP_SUITE", "0")
    rep = ar.run(models_verify=lambda progress: False, **fixed_env())
    item = [i for i in rep.items if i.kind == "model"][0]
    assert not rep.ok and item.status == ar.FAILED and "Manage models" in item.detail
    assert "Models (VOXPrint MDL)" in rep.summary() or "Models" in rep.summary()


def test_model_store_is_skipped_under_the_suite_skip():
    rep = ar.run(**fixed_env())
    assert [i.status for i in rep.items if i.kind == "model"] == [ar.SKIPPED]


def test_environment_repair_and_frozen_report(monkeypatch):
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


def test_cancel_stops():
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(CancelledByUser):
        ar.run(models_verify=lambda progress: True, cancel=tok, **fixed_env())


def test_settings_button_runs_and_shows_the_summary(app, lib):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from core import i18n
    from tests.test_studio import make_studio, wait_for
    i18n.set_language("en")
    s = make_studio(lib)
    d = s.settings_dialog()
    rep = ar.Report([ar.Item("model", "Models (VOXPrint MDL)", ar.OK)])
    d.autorepair_job = lambda progress, cancel: (progress(0.5, "Checking the model store"), rep)[1]
    assert d.btn_autorepair.text() == "Check && repair" and "VOXPrint MDL" in d.btn_autorepair.toolTip() and d.btn_repair is d.btn_autorepair
    assert d.toggle_autorepair()
    assert wait_for(lambda: not d.autorepair_running) and d.autorepair_worker.wait(3000)
    for _ in range(5):
        QApplication.processEvents()
    assert "0 failed" in d.lbl_autorepair_status.text() and not d.bar_autorepair.isVisibleTo(d)
    s.shutdown()


from tests.test_studio import app, lib  # noqa: E402,F401  (fixtures)


def test_cli_flag_runs_the_same_job_and_writes_a_log(monkeypatch, capsys):
    import main as vox_main
    calls = []
    monkeypatch.setattr(ar, "run", lambda progress: calls.append(1) or ar.Report([ar.Item("model", "Base", ar.FAILED, "x")]))
    assert vox_main.main(["voxprint", "--auto-repair"]) == 1 and calls
    assert "FAILED     model: Base - x" in (__import__("infra.paths", fromlist=["paths"]).logs_dir() / "auto_repair.txt").read_text(encoding="utf-8")
