"""Uninstall and a failed setup must not delete another app's models."""
from pathlib import Path

import pytest
from voxprint_common.runtime_manifest import add_ref

from infra import paths, suite_install, suite_shared


def _marker() -> Path:
    folder = paths.models_dir() / "movie-dubber"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "weights.bin"
    target.write_bytes(b"other-app")
    return target


def _plant_models_root(models: Path) -> None:
    """The rc.5 mistake: one component whose path is the whole models root."""
    add_ref(
        "shared-models", "audiobook-builder", kind="model", version="1",
        path=str(models.resolve()), sha256="ab" * 32, manifest=suite_shared.manifest_path(),
    )


def test_another_apps_models_survive_uninstall_rollback_and_a_failed_setup(monkeypatch):
    monkeypatch.delenv("VOXPRINT_SKIP_SUITE", raising=False)
    marker = _marker()
    keep = suite_shared.root() / "runtime" / "movie-dubber.txt"
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_text("keep", encoding="utf-8")
    _plant_models_root(paths.models_dir())
    suite_shared.register(suite_shared.APP_ID, ["runtime", "tools"])
    (suite_shared.torch_dir() / "torch").mkdir(parents=True)
    (suite_shared.resource_dir("tools") / "ffmpeg").mkdir(parents=True)

    assert suite_install.cli(["--unregister-suite"]) == 0
    assert marker.is_file() and marker.read_bytes() == b"other-app"
    assert keep.is_file()
    assert not suite_shared.torch_dir().exists()
    assert not (suite_shared.root() / "tools").exists()
    assert "shared-models" not in suite_shared.load_manifest()["components"]

    _plant_models_root(paths.models_dir())
    suite_shared.rollback()
    assert marker.is_file()
    assert "shared-models" not in suite_shared.load_manifest()["components"]

    _plant_models_root(paths.models_dir())

    def boom(progress):
        progress(0.2, "runtime")
        raise RuntimeError("setup failed")

    with pytest.raises(RuntimeError, match="setup failed"):
        suite_install.install_suite(
            lambda *_args: None, modules_fn=boom, ffmpeg_fn=lambda progress: progress(1.0, "ffmpeg"),
            driver_fn=lambda: "", force=True,
        )
    assert marker.is_file()
    assert "shared-models" not in suite_shared.load_manifest()["components"]
    assert keep.is_file()


def test_unregister_runtime_user_deletes_only_our_runtime_directory():
    marker = _marker()
    keep = suite_shared.root() / "runtime" / "movie-dubber.txt"
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_text("keep", encoding="utf-8")
    suite_shared.register(suite_shared.APP_ID, ["runtime"])
    (suite_shared.torch_dir() / "torch").mkdir(parents=True)
    assert suite_shared.release_runtime() == ["runtime"]
    assert marker.is_file() and keep.is_file()
    assert not suite_shared.torch_dir().exists()
    assert (suite_shared.root() / "runtime").is_dir()


def test_a_runtime_component_pointing_at_the_runtime_root_is_not_deleted():
    keep = suite_shared.root() / "runtime" / "movie-dubber.txt"
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_text("keep", encoding="utf-8")
    add_ref(
        "runtime-root", "audiobook-builder", kind="runtime", version="1", path="runtime",
        sha256="cd" * 32, manifest=suite_shared.manifest_path(),
    )
    suite_shared.unregister("audiobook-builder")
    assert keep.is_file()
    assert (suite_shared.root() / "runtime").is_dir()


def test_the_guard_raises_on_shared_roots_and_their_parents():
    marker = _marker()
    home = paths.app_home()
    shared = suite_shared.root()
    roots = [
        paths.models_dir(),
        paths.default_models_dir(),
        shared / "runtime",
        home / "voices",
        shared / "voices",
        home / "cache",
        home / "setup-cache",
        home,
        home.parent,
    ]
    for folder in roots:
        with pytest.raises(suite_shared.SharedRootError):
            suite_shared.refuse_shared_root(folder)
        with pytest.raises(suite_shared.SharedRootError):
            suite_shared._delete_tree(folder)
    assert marker.is_file()
    owned = suite_shared.torch_dir()
    owned.mkdir(parents=True)
    (owned / "torch").mkdir()
    suite_shared.refuse_shared_root(owned)
    suite_shared._delete_tree(owned)
    assert not owned.exists()
    assert (shared / "runtime").is_dir()


def test_setup_rolls_back_our_ref_and_never_deletes_the_models_root():
    root = Path(__file__).resolve().parents[1]
    iss = (root / "installer" / "Voxprint.iss").read_text(encoding="utf-8-sig")
    body = iss[iss.index("procedure RunSuiteInstall"):iss.index("procedure AddAppToUserPath")]
    assert body.count("UnregisterSuite();") >= 3
    assert "DelTree(SharedModelsDir" not in iss
    linux = (root / "installer" / "linux" / "install-voxprint-linux.sh").read_text(encoding="utf-8")
    failed = linux[linux.index("voxprint-mdl"):linux.index("Creating the launcher")]
    assert "--unregister-suite" in failed


def test_pr_check_workflow_stays_inside_its_run_block():
    text = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pr-check.yml").read_text(encoding="utf-8")
    assert not any(line.startswith(("{", "[", '"@')) for line in text.splitlines())
    assert "ConvertTo-Json" in text and "shared-models" in text
