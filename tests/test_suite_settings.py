"""Shared suite settings (state/suite.json, agreed with the Movie Dubber), runtime users file, installer integration."""
import json
import re
from pathlib import Path

import pytest

from infra import models_users as mu
from infra import paths
from infra import suite_settings as ss

ROOT = Path(__file__).resolve().parents[1]


def iss() -> str:
    return (ROOT / "installer" / "Voxprint.iss").read_bytes().decode("utf-8-sig")


def test_defaults_without_a_file(monkeypatch):
    monkeypatch.setattr("core.i18n.system_language", lambda: "de")
    assert not ss.path().exists()
    assert ss.values() == {"ui_language": "de", "theme": "glass-dark", "models_dir": None, "gpu": "auto"}
    monkeypatch.setattr("core.i18n.system_language", lambda: None)
    assert ss.ui_language() == "en"


def test_path_and_format():
    assert ss.path() == paths.app_home() / "state" / "suite.json"
    ss.set_value("gpu", "cuda:1")
    data = json.loads(ss.path().read_text(encoding="utf-8"))
    assert data == {"schema": 1, "gpu": "cuda:1"}
    assert not list(ss.path().parent.glob("*.tmp"))          # atomic write leaves nothing behind


def test_unknown_keys_are_kept_and_missing_keys_are_defaults():
    ss.path().parent.mkdir(parents=True, exist_ok=True)
    ss.path().write_text(json.dumps({"schema": 1, "dubber_only": {"x": 1}, "theme": "neon-pink"}), encoding="utf-8")
    assert ss.theme() == "glass-dark"                         # another program's theme: our default
    ss.set_value("ui_language", "ru")
    data = json.loads(ss.path().read_text(encoding="utf-8"))
    assert data["dubber_only"] == {"x": 1} and data["theme"] == "neon-pink" and data["ui_language"] == "ru"
    assert ss.gpu() == "auto" and ss.models_dir() is None


def test_damaged_file_counts_as_defaults_and_is_rewritten():
    ss.path().parent.mkdir(parents=True, exist_ok=True)
    ss.path().write_text("{broken", encoding="utf-8")
    assert ss.gpu() == "auto"
    ss.set_value("gpu", "cpu")
    assert json.loads(ss.path().read_text(encoding="utf-8")) == {"schema": 1, "gpu": "cpu"}


@pytest.mark.parametrize("value,expected", [("auto", "auto"), ("CPU", "cpu"), ("cuda", "cuda:0"), ("cuda:2", "cuda:2")])
def test_gpu_values(value, expected):
    assert ss.set_value("gpu", value) == expected and ss.gpu() == expected


@pytest.mark.parametrize("key,value", [("gpu", "vulkan"), ("gpu", "cuda:x"), ("ui_language", "zz"), ("models_dir", "rel/dir"),
                                       ("theme", ""), ("volume", 3)])
def test_invalid_values_are_refused(key, value):
    with pytest.raises(ValueError):
        ss.set_value(key, value)


def test_language_choice_is_shared(monkeypatch):
    from core import i18n

    monkeypatch.delenv("VOXPRINT_LANG")
    i18n.set_language("uk", persist=True)
    assert json.loads(ss.path().read_text(encoding="utf-8"))["ui_language"] == "uk"
    i18n.reset()
    ss.set_value("ui_language", "lv")                         # changed by the Movie Dubber
    assert i18n.get_language() == "lv"


def test_models_dir_order_env_suite_own_file(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.MODELS_DIR_ENV, raising=False)
    own = tmp_path / "own"
    (paths.state_dir() / paths.MODELS_DIR_FILE).write_text(str(own), encoding="utf-8")
    assert paths.configured_models_dir() == own
    ss.set_value("models_dir", None)                          # null = no shared choice: our own file still counts
    assert paths.configured_models_dir() == own
    shared = tmp_path / "shared"
    ss.set_value("models_dir", str(shared))
    assert paths.configured_models_dir() == shared and paths.models_dir() == shared
    env = tmp_path / "env"
    monkeypatch.setenv(paths.MODELS_DIR_ENV, str(env))
    assert paths.configured_models_dir() == env


def test_set_models_dir_writes_both(tmp_path):
    chosen = tmp_path / "D" / "models"
    paths.set_models_dir(chosen)
    assert ss.models_dir() == chosen and paths.app_models_dir_file() == chosen
    paths.set_models_dir(None)
    assert ss.has("models_dir") and ss.models_dir() is None and paths.app_models_dir_file() is None


def test_a_backup_never_becomes_the_shared_folder(tmp_path):
    usb = tmp_path / "E"
    (usb / "Voxprint-backup" / "models").mkdir(parents=True)
    paths.set_models_dir(usb)
    assert not ss.has("models_dir")


def test_migrate_and_installer_sync(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.MODELS_DIR_ENV, raising=False)
    old = tmp_path / "old"
    (paths.state_dir() / paths.MODELS_DIR_FILE).write_text(str(old), encoding="utf-8")
    (paths.state_dir() / "language").write_text("de", encoding="utf-8")
    ss.migrate_quietly()
    assert ss.models_dir() == old and ss.read_raw()["ui_language"] == "de"
    ss.set_value("ui_language", "ru")
    new = tmp_path / "new"
    (paths.state_dir() / paths.MODELS_DIR_FILE).write_text(str(new), encoding="utf-8")
    ss.migrate_quietly()                                      # a value already shared is not overwritten at start ...
    assert ss.models_dir() == old
    assert ss.sync_cli() == 0                                 # ... but the installer's explicit choice is
    assert ss.models_dir() == new and ss.read_raw()["ui_language"] == "ru"


def test_main_flags(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    import main as app_main

    monkeypatch.delenv(paths.MODELS_DIR_ENV, raising=False)
    (paths.state_dir() / paths.MODELS_DIR_FILE).write_text(str(tmp_path / "m"), encoding="utf-8")
    assert app_main.main(["Voxprint.exe", "--sync-suite-settings"]) == 0
    assert ss.models_dir() == tmp_path / "m"
    rt = mu.runtime_root()
    assert app_main.main(["Voxprint.exe", "--register-runtime-user"]) == 0
    assert mu.read_users(rt) == {"audiobook-builder": True}
    (rt / ".users.json").write_text(json.dumps({"audiobook-builder": True, "movie-dubber": True}), encoding="utf-8")
    out = tmp_path / "o.txt"
    assert app_main.main(["Voxprint.exe", "--unregister-runtime-user", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").splitlines() == ["1", str(rt)]
    assert mu.read_users(rt) == {"movie-dubber": True}


def test_runtime_register_only_when_the_folder_exists():
    mu.register_runtime_quietly()
    assert not mu.runtime_root().exists()
    mu.runtime_root().mkdir(parents=True)
    mu.register_runtime_quietly()
    assert mu.read_users(mu.runtime_root()) == {"audiobook-builder": True}


def test_installer_one_start_menu_folder_and_old_one_removed():
    t = iss()
    assert re.search(r"^DefaultGroupName=Voxprint\r?$", t, re.M) and re.search(r"^UsePreviousGroup=no\r?$", t, re.M)
    assert '{commonprograms}\\{#AppDisplayName}\\{#AppDisplayName}.lnk' in t and 'dirifempty; Name: "{commonprograms}\\{#AppDisplayName}"' in t


def test_installer_detects_the_sibling_and_reads_the_shared_folder():
    t = iss()
    assert "function SuiteJsonString" in t and "SuiteJsonString('models_dir')" in t and "{localappdata}\\Voxprint\\state\\suite.json'" in t
    body = t.split("function InitialModelsDir")[1].split("\nend;")[0]
    assert body.index("{param:ModelsFolder|}") < body.index("SuiteJsonString('models_dir')") < body.index("models_dir.txt")
    assert "function MovieDubberInstalled" in t and "'\"movie-dubber\"'" in t and "MovieDubberInstalled(ModelsEdit.Text)" in t
    for lang in ("english", "russian", "german"):
        assert f"{lang}.ModelsSiblingFound=" in t
    assert "ExecAsOriginalUser" in t and "'--sync-suite-settings'" in t and "'--unregister-runtime-user'" in t


def test_installer_uninstall_never_touches_the_sibling():
    t = iss()
    assert not re.search(r'filesandordirs; Name: "\{app\}"', t)
    sib = t.split("function IsSiblingFolder")[1].split("\nend;")[0]
    assert "'dubber'" in sib and "'unins*.exe'" in sib
    body = t.split("procedure DeleteProgramFolder")[1].split("\nend;")[0]
    assert "if not IsSiblingFolder(Path) then" in body and "RemoveDir(Dir)" in body
    # the models are still only offered for deletion when no other program uses them
    assert "(OtherModelUsers = 0)" in t


def test_pascal_strings_have_no_python_escapes():
    t = iss()
    code = t.split("function SuiteJsonString")[1].split("\nend;")[0]
    assert "if C = '\\' then" in code and "'\\\\'" not in code.replace("(Copy(Result, 1, 2) = '\\\\')", "")
