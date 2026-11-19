"""rc.5: voxprint-common pin, schema-1 runtime migration, suite VRAM tiers, mirror download."""
import json
from pathlib import Path

from packaging.requirements import Requirement

from core.vram_policy import suite_tier
from infra import modules, paths, suite_shared
from tools import make_runtime_lock as mrl

ROOT = Path(__file__).resolve().parents[1]


COMMIT = "2ec81bb17c33c00d748aa9e21301d2bad80df7db"
UI_PIN = (
    "voxprint-common[ui] @ git+https://github.com/Mitroshenkov87/voxprint-common"
    f"@{COMMIT}"
)
# a pip pin on a git URL carries no --hash (pip rejects hashes on VCS pins);
# the wheel built from the commit is hash-checked by the pr-check pin step
SUITE_PIN = f"voxprint-common @ git+https://github.com/Mitroshenkov87/voxprint-common@{COMMIT}"


def test_voxprint_common_keeps_the_released_wheel_and_pins_the_ui_shell():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    lock = json.loads((ROOT / "infra" / "runtime_lock.json").read_text(encoding="utf-8"))
    suite = lock["suite"]
    assert suite == mrl.suite_record()
    assert suite["version"] == "0.8.0"
    assert suite["commit"] == COMMIT
    assert suite["sha256"] == "9020ddb9639831b5e12a38ecdf47eb2c01a882e0138c5439563e41bf8b1a6952"
    assert suite["size"] == 689311
    pin = mrl.suite_requirement(suite)
    assert pin not in req                       # no bare pin: requirements.txt carries the [ui] extra
    assert pin == SUITE_PIN
    assert "--hash" not in pin
    assert UI_PIN in req
    assert lock["git"] == [UI_PIN]
    assert all(row["dist"] != "voxprint-common" for row in lock["wheels"])
    assert lock["python"] == "3.14"
    parsed = Requirement(pin)
    assert parsed.name == "voxprint-common"
    assert parsed.url == suite["url"]
    assert Requirement(UI_PIN).name == "voxprint-common"
    assert mrl.git_dist_names(lock) == ["voxprint-common"]


def test_test_requirements_emit_the_git_pin_and_shell_requirements_do_not(capsys):
    wheel = mrl.suite_requirement()
    assert mrl.main(["--test-requirements"]) == 0
    out = capsys.readouterr().out
    assert UI_PIN in out and wheel not in out and "pyside6==" in out
    assert mrl.main(["--shell-requirements"]) == 0
    shell = capsys.readouterr().out
    assert "voxprint-common" not in shell and "pyside6==" in shell


def test_an_empty_git_list_keeps_the_suite_pin():
    lock = json.loads((ROOT / "infra" / "runtime_lock.json").read_text(encoding="utf-8"))
    lock["git"] = []
    lines = mrl.requirement_lines(lock, tests=True)
    assert mrl.suite_requirement() in lines
    assert UI_PIN not in lines
    rebuilt = mrl.keep_git({"suite": lock["suite"], "wheels": []}, {"git": [UI_PIN]})
    assert list(rebuilt)[:2] == ["suite", "git"]
    assert rebuilt["git"] == [UI_PIN]


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
    assert "shared-models" not in data["components"]
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
    assert "shared-models" not in suite_shared.load_manifest()["components"]
    assert suite_shared.release_runtime("audiobook-builder") == []
    assert marker.is_dir() and suite_shared.apps_using("runtime") == ["movie-dubber"]
    assert suite_shared.release_runtime("movie-dubber") == ["runtime"]
    assert not suite_shared.torch_dir().exists()


def test_registering_models_does_not_record_the_models_root(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("VOXPRINT_HOME", str(home))
    monkeypatch.setenv("VOXPRINT_SHARED", str(tmp_path / "shared"))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    suite_shared.register("audiobook-builder", ["models"])
    data = suite_shared.load_manifest()
    assert data["components"] == {}
    assert not suite_shared.manifest_path().is_file()


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
