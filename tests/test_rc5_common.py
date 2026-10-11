"""rc.5: voxprint-common pin, schema-1 runtime migration, suite VRAM tiers, mirror download."""
import json
from pathlib import Path

import pytest

from packaging.requirements import Requirement

from core.vram_policy import suite_tier
from infra import model_mirrors, modules, paths, suite_shared
from tools import make_runtime_lock as mrl

ROOT = Path(__file__).resolve().parents[1]
PIN = "voxprint-common @ git+https://github.com/Mitroshenkov87/voxprint-common.git@7bb4f4078839cc7348e2bbe621a220c673ed7eb8"


def test_voxprint_common_is_pinned_by_commit_in_requirements_and_the_lock():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert PIN in req
    lock = json.loads((ROOT / "infra" / "runtime_lock.json").read_text(encoding="utf-8"))
    assert lock["git"] == [PIN]
    assert lock["python"] == "3.14"
    assert mrl.GIT_PINS == [PIN]
    assert "7bb4f4078839cc7348e2bbe621a220c673ed7eb8" in mrl.GIT_SHA
    parsed = Requirement(PIN)
    assert parsed.name == "voxprint-common"


def test_test_requirements_emit_the_git_pin_and_shell_requirements_do_not(capsys):
    assert mrl.main(["--test-requirements"]) == 0
    out = capsys.readouterr().out
    assert PIN in out and "pyside6==" in out
    assert mrl.main(["--shell-requirements"]) == 0
    shell = capsys.readouterr().out
    assert "voxprint-common" not in shell and "pyside6==" in shell


def test_an_apps_manifest_migrates_into_schema_1_components(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    monkeypatch.setenv("VOXPRINT_SHARED", str(shared))
    (shared).mkdir()
    (shared / "manifest.json").write_text(json.dumps({
        "schema": 1,
        "apps": {
            "audiobook-builder": {"resources": ["runtime", "models", "tools"]},
            "movie-dubber": {"resources": ["models", "runtime"]},
        },
    }), encoding="utf-8")
    data = suite_shared.load_manifest()
    torch = data["components"]["torch-2.11.0-cu130"]
    assert set(torch["refs"]) == {"audiobook-builder", "movie-dubber"}
    assert set(data["components"]["shared-models"]["refs"]) == {"audiobook-builder", "movie-dubber"}
    assert data["components"]["ffmpeg-8"]["refs"] == ["audiobook-builder"]
    assert "apps" in json.loads((shared / "manifest.json").read_text(encoding="utf-8"))


def test_a_legacy_runtime_splits_cpu_onnxruntime_away_from_torch(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("VOXPRINT_HOME", str(home))
    monkeypatch.setenv("VOXPRINT_SHARED", str(tmp_path / "shared"))
    legacy = paths.app_home() / "runtime"
    (legacy / "torch").mkdir(parents=True)
    (legacy / "onnxruntime").mkdir()
    (legacy / "onnxruntime-1.20.0.dist-info").mkdir()
    (legacy / "scipy").mkdir()
    (legacy / ".users.json").write_text("{}", encoding="utf-8")
    suite_shared.migrate_installed_runtime()
    torch = suite_shared.torch_dir()
    wheels = suite_shared.wheels_dir()
    assert (torch / "torch").is_dir()
    assert (wheels / "onnxruntime").is_dir()
    assert (wheels / "onnxruntime-1.20.0.dist-info").is_dir()
    assert (wheels / "scipy").is_dir()
    assert (legacy / ".users.json").is_file()
    assert suite_shared.apps_using("runtime") == ["audiobook-builder"]
    assert not (torch / "scipy").exists()


def test_a_runtime_ref_is_deleted_only_when_nobody_else_holds_it(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_SHARED", str(tmp_path / "shared"))
    suite_shared.register("audiobook-builder", ["runtime", "models"])
    suite_shared.register("movie-dubber", ["runtime"])
    marker = suite_shared.torch_dir() / "torch"
    marker.mkdir(parents=True)
    assert suite_shared.release_runtime("audiobook-builder") == []
    assert marker.is_dir() and suite_shared.apps_using("runtime") == ["movie-dubber"]
    assert suite_shared.release_runtime("movie-dubber") == ["runtime"]
    assert not suite_shared.torch_dir().exists()
    assert (suite_shared.root() / "models").is_dir()


def test_another_apps_torch_tree_is_not_where_we_write(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VOXPRINT_SHARED", str(tmp_path / "shared"))
    suite_shared.register("movie-dubber", ["runtime"])
    (suite_shared.torch_dir() / "torch").mkdir(parents=True)
    assert suite_shared.foreign_torch_path() == suite_shared.torch_dir()
    assert modules.runtime_dir() == suite_shared.wheels_dir()
    assert (suite_shared.torch_dir() / "torch").is_dir()
    assert "movie-dubber" in suite_shared.apps_using("runtime")


def test_suite_tier_boundaries_are_15_5_and_23_5_gib():
    mib = 1024.0
    assert suite_tier(15.5 * mib - 1) == "8gb"
    assert suite_tier(15.5 * mib) == "16gb"
    assert suite_tier(23.5 * mib - 1) == "16gb"
    assert suite_tier(23.5 * mib) == "24gb"


def test_the_mirror_default_fetch_calls_the_suite_downloader(tmp_path, monkeypatch):
    calls = []

    def fake(dest, urls, sha256="", size=None):
        calls.append((Path(dest), list(urls), sha256, size))
        Path(dest).write_bytes(b"ok")

    monkeypatch.setattr(model_mirrors, "download_file", fake)
    model_mirrors._default_fetch("Org/repo", "weights.bin", "abc", tmp_path, size=4, sha256="ab" * 32)
    dest, urls, digest, size = calls[0]
    assert dest == tmp_path / "weights.bin" and digest == "ab" * 32 and size == 4
    assert urls[0].startswith("https://huggingface.co/") and "Org/repo" in urls[0]

    with pytest.raises(model_mirrors.MirrorError):
        model_mirrors._default_fetch("Org/repo", "weights.bin", "abc", tmp_path)
