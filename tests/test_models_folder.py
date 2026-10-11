"""User-chosen models folder (installer page "Models folder", infra/paths.py + infra/model_downloader.py)."""
import pytest

from infra import model_downloader as md
from infra import paths

REPO = "Owner/Model"


def make_model(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"w")
    return folder


def test_default_without_a_choice():
    assert paths.configured_models_dir() is None
    assert paths.models_dir() == paths.default_models_dir() == paths.app_home() / "models"


def test_chosen_folder_is_read_created_and_forgotten(tmp_path):
    chosen = tmp_path / "D" / "models"
    paths.set_models_dir(chosen)
    assert paths.configured_models_dir() == chosen and paths.models_dir() == chosen and chosen.is_dir()
    paths.set_models_dir(paths.default_models_dir())                   # the default = no choice stored
    assert paths.configured_models_dir() is None


def test_relative_path_and_env_override(tmp_path, monkeypatch):
    (paths.state_dir() / paths.MODELS_DIR_FILE).write_text("relative\\models\n", encoding="utf-8")
    assert paths.configured_models_dir() is None and paths.models_dir() == paths.default_models_dir()
    monkeypatch.setenv(paths.MODELS_DIR_ENV, str(tmp_path / "env"))
    assert paths.models_dir() == tmp_path / "env"


def test_unusable_folder_falls_back_to_the_default(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")                         # a file where the folder should be
    paths.set_models_dir(blocker / "models")
    assert paths.models_dir() == paths.default_models_dir()


def test_models_in_the_default_folder_are_reused_new_ones_go_to_the_chosen_one(tmp_path):
    old = make_model(paths.default_models_dir() / "Owner--Model")
    chosen = tmp_path / "D"
    paths.set_models_dir(chosen)
    assert md.local_dir_for(REPO) == old and md.verify_local_model(md.local_dir_for(REPO))
    assert md.local_dir_for("Owner/Other") == chosen / "Owner--Other"     # download target
    assert md.ensure_model(REPO) == old                                     # no download
    mine = make_model(chosen / "Owner--Model")
    assert md.local_dir_for(REPO) == mine                                   # the chosen folder wins when complete


def test_models_already_in_the_chosen_folder_in_another_layout_are_picked_up(tmp_path, monkeypatch):
    chosen = tmp_path / "D"
    snap = make_model(chosen / "models--Owner--Model" / "snapshots" / ("a" * 40))
    (chosen / "models--Owner--Model" / "refs").mkdir()
    (chosen / "models--Owner--Model" / "refs" / "main").write_text("a" * 40, encoding="utf-8")
    paths.set_models_dir(chosen)
    from core import model_locator
    monkeypatch.setattr(model_locator, "check_model_dir", lambda p, r: "")
    monkeypatch.setattr(md, "pinned_revision", lambda r: None)
    found = md.external_model(REPO)
    assert found is not None and found.path == snap
    assert md.model_state(REPO) == md.STATE_READY
    paths.set_models_dir(None)
    assert md.external_model(REPO) is None                    # without a choice the folder is not searched (tests: no guessing)


@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"])
def test_installer_file_format(tmp_path, bom):
    chosen = tmp_path / "Models"
    (paths.state_dir() / paths.MODELS_DIR_FILE).write_bytes(bom + str(chosen).encode("utf-8") + b"\r\n")
    assert paths.models_dir() == chosen
