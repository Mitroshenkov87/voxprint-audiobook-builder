"""Shared suite folder, refcounted manifest, and the install-during-setup command."""
import json
from pathlib import Path

import pytest

from infra import paths, suite_install, suite_shared


def test_register_and_unregister_keep_a_resource_another_app_still_uses(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_SHARED", str(tmp_path / "shared"))
    suite_shared.register("audiobook-builder", ["runtime", "models", "tools"])
    suite_shared.register("movie-dubber", ["runtime", "models"])
    (suite_shared.resource_dir("runtime") / "torch").mkdir()
    (suite_shared.resource_dir("tools") / "ffmpeg").mkdir()
    assert "shared-models" not in suite_shared.load_manifest()["components"]
    unused = suite_shared.unregister("audiobook-builder")
    assert unused == ["ffmpeg-8"]
    assert not (suite_shared.root() / "tools").exists()
    assert (suite_shared.resource_dir("runtime") / "torch").is_dir()
    assert suite_shared.apps_using("runtime") == ["movie-dubber"]
    assert suite_shared.unregister("movie-dubber") == ["torch-2.11.0-cu130"]


def test_install_suite_runs_the_runtime_and_ffmpeg_and_records_the_app(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_SHARED", str(tmp_path / "shared"))
    monkeypatch.delenv("VOXPRINT_SKIP_SUITE", raising=False)
    seen = []

    def step(name):
        def run(progress):
            seen.append(name)
            progress(1.0, name)
            (suite_shared.resource_dir(name if name != "ffmpeg" else "tools") / "marker").write_text(name, encoding="utf-8")
        return run

    result = suite_install.install_suite(
        lambda fraction, message: seen.append(f"{fraction:.2f}"),
        modules_fn=step("runtime"), ffmpeg_fn=step("tools"),
        driver_fn=lambda: "", free_fn=lambda _path: 80 << 30, need_bytes=1 << 30, force=True,
    )
    assert result["ok"] and result["exit_code"] == 0
    assert [name for name in seen if name in ("runtime", "tools")] == ["runtime", "tools"]
    manifest = json.loads((suite_shared.root() / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == 1
    refs = {name: manifest["components"][spec.id]["refs"] for name, spec in (
        ("runtime", suite_shared._SPECS["runtime"]),
        ("tools", suite_shared._SPECS["tools"]),
    )}
    assert refs["runtime"] == ["audiobook-builder"]
    assert refs["tools"] == ["audiobook-builder"]
    assert "models" not in suite_install.APP_RESOURCES


def test_a_too_old_driver_stops_the_suite_install_with_the_download_link(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_SHARED", str(tmp_path / "shared"))
    monkeypatch.delenv("VOXPRINT_SKIP_SUITE", raising=False)
    called = []
    result = suite_install.install_suite(
        lambda _f, _m: called.append(1),
        modules_fn=lambda _p: called.append("modules"),
        driver_fn=lambda: "The NVIDIA driver is too old (535.98). https://www.nvidia.com/Download/index.aspx",
        force=True,
    )
    assert result["exit_code"] == 7 and "nvidia.com" in result["message"]
    assert "modules" not in called


def test_skip_env_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_SHARED", str(tmp_path / "shared"))
    monkeypatch.setenv("VOXPRINT_SKIP_SUITE", "1")
    result = suite_install.install_suite(modules_fn=lambda _p: (_ for _ in ()).throw(AssertionError("download")))
    assert result["exit_code"] == 0 and result["message"] == "skipped"


def test_default_models_folder_is_the_suite_root_and_shared_models_still_win_when_they_hold_files(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    assert paths.default_models_dir() == paths.app_home() / "models"
    previous = paths.shared_root() / "models" / "Org--Model"
    previous.mkdir(parents=True)
    (previous / "config.json").write_text("{}", encoding="utf-8")
    assert paths.models_dir() == paths.shared_root() / "models"
    live = paths.default_models_dir() / "Org--Other"
    live.mkdir(parents=True)
    (live / "config.json").write_text("{}", encoding="utf-8")
    assert paths.models_dir() == paths.default_models_dir()


def test_cli_does_not_write_a_status_file(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_SHARED", str(tmp_path / "shared"))
    monkeypatch.delenv("VOXPRINT_SKIP_SUITE", raising=False)
    status = tmp_path / "status.txt"

    def fake_install(progress, **kwargs):
        progress(0.4, "runtime")
        progress(1.0, "done")
        return {"ok": True, "exit_code": 0, "message": "installed", "missing": []}

    monkeypatch.setattr(suite_install, "install_suite", fake_install)
    code = suite_install.cli(["--install-suite", "--status", str(status), "--force"])
    assert code == 0 and not status.exists()
    assert not hasattr(suite_install, "write_status")


def test_installer_installs_the_suite_and_ships_the_console_exe():
    root = Path(__file__).resolve().parents[1]
    iss = (root / "installer" / "Voxprint.iss").read_text(encoding="utf-8-sig")
    assert "voxprint-audiobook-cli.exe" in iss and "--install-suite" in iss and "SkipSuite" in iss
    assert "voxprint-mdl.cmd" in iss and "download --app audiobook-builder" in iss
    suite = iss[iss.index("procedure RunSuiteInstall"):iss.index("procedure AddAppToUserPath")]
    assert "--gui" not in suite and "--json" in suite
    assert "ewWaitUntilTerminated" in iss and "wrote no status" not in iss
    assert "Abort;" in iss[iss.index("procedure RunSuiteInstall"):iss.index("procedure AddAppToUserPath")]
    assert "--unregister-suite" in iss and "ChangesEnvironment=yes" in iss
    bat = (root / "build_thin.bat").read_text(encoding="utf-8")
    assert "--console" in bat and "--name voxprint-audiobook-cli" in bat
    assert bat.index("\npyinstaller ") < bat.index("embed_build_stamp.py --exe") < bat.index("--name voxprint-audiobook-cli")
    wf = (root / ".github" / "workflows" / "build-installer.yml").read_text(encoding="utf-8")
    assert "pytest-timeout" in wf and "--timeout=180" in wf and "--timeout-method=thread" in wf
    assert "VOXPRINT_TEST_TRACEBACK_S" in wf and "timeout-minutes: 25" in wf
    assert wf.count('"/SkipSuite=1"') >= 3
    assert "pytest-timeout" not in (root / "requirements.txt").read_text(encoding="utf-8")


def test_selftest_json_repairs_and_keeps_progress_off_stdout(monkeypatch, capsys):
    pytest.importorskip("PySide6")
    import main as app_main

    monkeypatch.setattr(suite_install, "should_repair", lambda: True)
    monkeypatch.setattr(suite_install, "skipped", lambda *args, **kwargs: False)
    monkeypatch.setattr(suite_install, "inspect_shared", lambda: {
        "ok": False, "missing": ["tools"], "root": str(suite_shared.root()), "skipped": False,
    })
    seen = []

    def fake_install(progress, **kwargs):
        progress(0.5, "Repairing ffmpeg")
        seen.append("install")
        return {"ok": True, "exit_code": 0, "message": "installed", "missing": [],
                "root": str(suite_shared.root())}

    monkeypatch.setattr(suite_install, "install_suite", fake_install)
    assert app_main.main(["Voxprint.exe", "--selftest", "--json"]) == 0
    captured = capsys.readouterr()
    assert seen == ["install"]
    assert "Repairing ffmpeg" in captured.err and "Repairing ffmpeg" not in captured.out
    result = json.loads([line for line in captured.out.splitlines() if line.startswith("{")][-1])
    assert result["type"] == "result" and result["command"] == "selftest"
    assert result["suite"]["repaired"] is True and result["exit_code"] == 0
