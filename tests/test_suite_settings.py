"""Shared suite settings (state/suite.json, agreed with the Movie Dubber), runtime users file, installer integration."""
import json
import re
from pathlib import Path

import pytest

from infra import modules, paths
from infra import suite_settings as ss


def runtime_root() -> Path:
    return paths.app_home() / "runtime"


def read_users(root: Path) -> dict:
    try:
        return json.loads((root / ".users.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

ROOT = Path(__file__).resolve().parents[1]


def iss() -> str:
    return (ROOT / "installer" / "Voxprint.iss").read_bytes().decode("utf-8-sig")


def test_defaults_without_a_file(monkeypatch):
    monkeypatch.setattr("core.i18n.system_language", lambda: "de")
    assert not ss.path().exists()
    assert ss.values() == {"ui_language": "de", "theme": "glass-dark", "gpu": "auto"}
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
    ss.set_value("ui_language", "en")
    data = json.loads(ss.path().read_text(encoding="utf-8"))
    assert data["dubber_only"] == {"x": 1} and data["theme"] == "neon-pink" and data["ui_language"] == "en"
    assert ss.gpu() == "auto" and "models_dir" not in ss.values()


def test_damaged_file_counts_as_defaults_and_is_rewritten():
    ss.path().parent.mkdir(parents=True, exist_ok=True)
    ss.path().write_text("{broken", encoding="utf-8")
    assert ss.gpu() == "auto"
    ss.set_value("gpu", "cuda:0")
    assert json.loads(ss.path().read_text(encoding="utf-8")) == {"schema": 1, "gpu": "cuda:0"}


@pytest.mark.parametrize("value,expected", [("auto", "auto"), ("cuda", "cuda:0"), ("cuda:2", "cuda:2")])
def test_gpu_values(value, expected):
    assert ss.set_value("gpu", value) == expected and ss.gpu() == expected


@pytest.mark.parametrize("key,value", [("gpu", "cpu"), ("gpu", "CPU"), ("gpu", "vulkan"), ("gpu", "cuda:x"),
                                       ("ui_language", "ru"), ("ui_language", "zz"),
                                       ("theme", ""), ("volume", 3)])
def test_invalid_values_are_refused(key, value):
    with pytest.raises(ValueError):
        ss.set_value(key, value)


def test_language_choice_is_shared(monkeypatch):
    from core import i18n

    monkeypatch.delenv("VOXPRINT_LANG")
    i18n.set_language("en", persist=True)
    assert json.loads(ss.path().read_text(encoding="utf-8"))["ui_language"] == "en"
    i18n.reset()
    ss.set_value("ui_language", "en")                         # English-only: the only shared UI language
    assert i18n.get_language() == "en"


def test_migrate_and_installer_sync(tmp_path, monkeypatch):
    (paths.state_dir() / "language").write_text("en", encoding="utf-8")
    ss.migrate_quietly()
    assert ss.read_raw()["ui_language"] == "en"
    ss.set_value("ui_language", "en")
    assert ss.sync_cli() == 0
    assert ss.read_raw()["ui_language"] == "en"


def test_main_flags(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    import main as app_main

    assert app_main.main(["Voxprint.exe", "--sync-suite-settings"]) == 0
    rt = runtime_root()
    assert app_main.main(["Voxprint.exe", "--register-runtime-user"]) == 0
    assert read_users(rt) == {"audiobook-builder": True}
    (rt / ".users.json").write_text(json.dumps({"audiobook-builder": True, "movie-dubber": True}), encoding="utf-8")
    out = tmp_path / "o.txt"
    assert app_main.main(["Voxprint.exe", "--unregister-runtime-user", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").splitlines() == ["1", str(rt)]
    assert read_users(rt) == {"movie-dubber": True}


def test_runtime_register_creates_the_users_file():
    assert not runtime_root().exists()
    modules.register_runtime_user()
    path = runtime_root() / ".users.json"
    assert path.is_file() and read_users(runtime_root()) == {"audiobook-builder": True}
    before = path.stat().st_mtime_ns
    modules.register_runtime_user()                      # already registered: no rewrite
    assert path.stat().st_mtime_ns == before


def test_gui_start_registers_audiobook_builder_in_the_runtime_users_file():
    pytest.importorskip("PySide6")
    import main as app_main

    assert not (runtime_root() / ".users.json").exists()
    assert app_main.main(["Voxprint.exe", "--selftest"]) == 0
    assert read_users(runtime_root()) == {"audiobook-builder": True}


def test_installer_one_start_menu_folder_and_old_one_removed():
    t = iss()
    assert re.search(r"^DefaultGroupName=\{#StartMenu\}\r?$", t, re.M) and re.search(r"^UsePreviousGroup=no\r?$", t, re.M)
    assert '#define StartMenu "VOXPrint AI Suite"' in t
    assert '#define SuiteDir "VOXPrint AI Suite"' in t
    assert '#define AppFolder "Audiobook Builder"' in t
    assert '{commonprograms}\\{#AppDisplayName}\\{#AppDisplayName}.lnk' in t and 'dirifempty; Name: "{commonprograms}\\{#AppDisplayName}"' in t


def test_installer_detects_the_sibling_and_reads_the_shared_folder():
    t = iss()
    assert "function SuiteJsonString" in t and "SuiteJsonString('models_dir')" in t and "{localappdata}\\Voxprint\\state\\suite.json'" in t
    body = t.split("function InitialModelsDir")[1].split("\nend;")[0]
    assert body.index("{param:ModelsFolder|}") < body.index("SuiteJsonString('models_dir')") < body.index("models_dir.txt")
    assert "function MovieDubberInstalled" in t and "'\"movie-dubber\"'" in t and "MovieDubberInstalled(ModelsEdit.Text)" in t
    for lang in ("english",):
        assert f"{lang}.ModelsSiblingFound=" in t
    assert "ExecAsOriginalUser" in t and "'--sync-suite-settings'" in t and "'--unregister-runtime-user'" in t


def test_installer_uninstall_never_touches_the_sibling():
    t = iss()
    assert not re.search(r'filesandordirs; Name: "\{app\}"', t)
    sib = t.split("function IsSiblingFolder")[1].split("\nend;")[0]
    assert "'dubber'" in sib and "'unins*.exe'" in sib
    body = t.split("procedure DeleteProgramFolder")[1].split("\nend;")[0]
    assert "if not IsSiblingFolder(Path) then" in body and "RemoveDir(Dir)" in body
    # Shared models stay with MDL. Uninstall does not offer to delete that folder.
    assert "OtherModelUsers" not in t and "DelTree(SharedModelsDir" not in t


def test_pascal_strings_have_no_python_escapes():
    t = iss()
    code = t.split("function SuiteJsonString")[1].split("\nend;")[0]
    assert "if C = '\\' then" in code and "'\\\\'" not in code.replace("(Copy(Result, 1, 2) = '\\\\')", "")
