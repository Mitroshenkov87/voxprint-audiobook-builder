"""installer/Voxprint.iss: the optional "existing models" wizard page, trilingual wizard texts, encoding.

Inno Setup cannot be compiled on Linux, so these tests lint the script: encoding, message tables, the Pascal block structure and
the contract with the app (the file the page writes is the file ``infra.existing_models`` reads)."""
import re
from pathlib import Path

from infra import existing_models, paths

ISS = Path(__file__).resolve().parents[1] / "installer" / "Voxprint.iss"
LANGS = ("english", "russian", "german")


def text() -> str:
    return ISS.read_bytes().decode("utf-8-sig")


def section(name: str) -> str:
    t = text()
    m = re.search(rf"^\[{name}\]\r?\n(.*?)(?=^\[|\Z)", t, re.S | re.M)
    assert m, name
    return m.group(1)


def test_bom_and_crlf_are_kept():
    raw = ISS.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and raw.count(b"\r\n") == raw.count(b"\n")


def test_three_wizard_languages():
    langs = re.findall(r'^Name: "(\w+)"; MessagesFile: "([^"]+)"', section("Languages"), re.M)
    assert [n for n, _ in langs] == list(LANGS)
    assert dict(langs)["german"].endswith("German.isl") and dict(langs)["russian"].endswith("Russian.isl")


def test_every_custom_message_exists_in_all_languages_and_every_use_is_defined():
    defined = {l: set() for l in LANGS}
    for line in section("CustomMessages").splitlines():
        m = re.match(r"(\w+)\.(\w+)=(.+)", line)
        if m:
            assert m.group(3).strip(), line
            defined[m.group(1)].add(m.group(2))
    assert defined["english"] == defined["russian"] == defined["german"] and len(defined["english"]) >= 9
    # {cm:Name} in script sections, CustomMessage('Name') in [Code]
    builtin = set()
    used = set(re.findall(r"\{cm:(\w+)", text())) | set(re.findall(r"CustomMessage\('(\w+)'\)", text()))
    assert used - builtin <= defined["english"], used - builtin - defined["english"]
    assert {"ModelsPageCaption", "ModelsPageDescription", "ModelsPageSubCaption", "ModelsPagePrompt", "ModelsPageBadFolder", "ModelsPageProtected"} <= used
    # the %1/%2 placeholders agree between the languages
    for name in defined["english"]:
        ph = {l: sorted(set(re.findall(r"%\d", next(x for x in section("CustomMessages").splitlines() if x.startswith(f"{l}.{name}=")))))
              for l in LANGS}
        assert ph["english"] == ph["russian"] == ph["german"], name


def test_no_russian_text_is_left_outside_the_message_table():
    t = text()
    rest = t.replace(section("CustomMessages"), "").replace(section("Messages"), "")
    code_lines = [l for l in rest.splitlines() if re.search("[А-Яа-яЁё]", l) and not l.lstrip().startswith((";", "{"))]
    assert code_lines == []


def test_pascal_blocks_are_balanced():
    code = re.sub(r"\{[^}]*\}", "", section("Code"))           # { comments }
    code = re.sub(r"'[^']*'", "''", code)
    words = re.findall(r"\b(begin|end|case|try)\b", code, re.I)
    opens = sum(1 for w in words if w.lower() in ("begin", "case", "try"))
    assert opens == sum(1 for w in words if w.lower() == "end")
    assert code.lower().count("procedure ") + code.lower().count("function ") >= 5


def test_models_folder_page_defaults_to_local_validates_and_only_remembers_the_path():
    code = section("Code")
    assert "CreateCustomPage(wpSelectDir" in code and "CreateInputDirPage(wpSelectDir" not in code.split("InitializeWizard")[1].split("end;")[0]   # right after the install folder
    assert "{localappdata}\\Voxprint\\models" in code                       # default = the app's default_models_dir
    assert "ModelsEdit.Text := InitialModelsDir()" in code and "{param:ModelsFolder|}" in code   # silent: /ModelsFolder=
    assert "models_dir.txt" in code and paths.MODELS_DIR_FILE == "models_dir.txt"
    assert "InProgramFiles(Dir)" in code and "{commonpf64}" in code             # the app could not write there
    assert "FolderUsable(Dir)" in code and ".voxprint-write-test" in code       # creatable and writable
    assert "DeleteFile(StateDir + '\\models_dir.txt')" in code                  # the default: the app follows its own default
    assert "{param:ModelsDir|}" in code and existing_models.CONFIG_NAME in code # the older import folder still works silently
    assert "ssPostInstall" in code and "SaveStringsToUTF8File" in code and "{localappdata}\\Voxprint\\state" in code
    # the installer must not copy gigabytes: no model files / folders in [Files], no copy helpers in [Code]
    files = section("Files")
    assert "models" not in files.lower() and ".safetensors" not in files.lower()
    assert not re.search(r"FileCopy|CopyFile|xcopy|robocopy", code, re.I)


def test_what_the_page_writes_is_what_the_app_reads(tmp_path):
    folder = tmp_path / "old"
    folder.mkdir()
    # the installer writes a UTF-8 file (SaveStringsToUTF8File: BOM, CRLF) with one line
    existing_models.config_file().write_bytes(b"\xef\xbb\xbf" + str(folder).encode("utf-8") + b"\r\n")
    assert existing_models.configured() == folder
    chosen = tmp_path / "D" / "Voxprint models"
    (paths.state_dir() / paths.MODELS_DIR_FILE).write_bytes(b"\xef\xbb\xbf" + str(chosen).encode("utf-8") + b"\r\n")
    assert paths.models_dir() == chosen and chosen.is_dir()


def test_no_desktop_shortcut_only_start_menu_and_upgrades_remove_the_old_one():
    t = text()
    assert "desktopicon" not in t and "{autodesktop}" not in t and "CreateDesktopIcon" not in t
    assert not re.search(r"^\[Tasks\]", t, re.M)
    icons = section("Icons")
    assert "{group}\\{#AppDisplayName}" in icons and "desktop" not in icons.lower()
    deletes = section("InstallDelete")
    assert "{commondesktop}\\{#AppDisplayName}.lnk" in deletes and "{userdesktop}\\{#AppDisplayName}.lnk" in deletes


def test_setup_refuses_windows_older_than_24h2():
    """Build < 26100 is a hard stop, in every wizard language, with no offer to continue."""
    t = text()
    assert "MinVersion=10.0.26100" in section("Setup")
    assert "WinBuildWarning" not in t and "Continue the installation" not in t
    body = section("Code").split("function InitializeSetup")[1].split("\nend;")[0]
    assert "V.Build < 26100" in body and "Result := False" in body and "mbError" in body
    assert "MB_YESNO" not in body and "IDYES" not in body and "WinBuildTooOld" in body
    for lang in LANGS:
        custom = next(x for x in section("CustomMessages").splitlines() if x.startswith(f"{lang}.WinBuildTooOld="))
        low = next(x for x in section("Messages").splitlines() if x.startswith(f"{lang}.WinVersionTooLowError="))
        for line in (custom, low):
            assert "26100" in line and "Windows 11 24H2" in line
            assert "Windows 10" not in line and "Continue" not in line and "Продолжить" not in line


def test_a_backup_folder_is_a_restore_source_not_the_models_folder():
    """Hybrid rule: an empty / normal folder -> models_dir.txt (download target); a Voxprint backup -> existing_models_dir.txt
    (restored into the default folder by the app) and models_dir.txt is removed.  Same detection as paths.backup_root_of."""
    code = section("Code")
    looks = code.split("function LooksLikeBackup")[1].split("\nend;")[0]
    assert f"'{paths.BACKUP_MANIFEST}'" in looks and f"'{paths.BACKUP_DIRNAME}'" in looks
    for sub in paths.BACKUP_CONTENT:                                                  # a backup without its manifest
        assert f"'{sub}')" in looks
    body = code.split("function BackupRootOf")[1].split("\nend;")[0]
    assert "LooksLikeBackup(D)" in body and f"LooksLikeBackup(AddBackslash(D) + '{paths.BACKUP_DIRNAME}')" in body
    assert "LooksLikeBackup(ExtractFileDir(D))" in body
    for sub in paths.BACKUP_SUBDIRS:
        assert f"(N = '{sub}')" in body
    nxt = code.split("function NextButtonClick")[1].split("\nend;")[0]
    assert "BackupRootOf(Dir)" in nxt and "CustomMessage('ModelsPageBackupFound')" in nxt
    assert nxt.index("BackupRootOf(Dir)") < nxt.index("InProgramFiles(Dir)")          # a backup is only read: no write test
    post = code.split("procedure CurStepChanged")[1].split("\nend;")[0]
    backup_branch = post.split("if Backup <> '' then")[1].split("else if")[0]
    assert "DeleteFile(StateDir + '\\models_dir.txt')" in backup_branch
    assert f"'\\{existing_models.CONFIG_NAME}'" in backup_branch and "Lines[0] := Backup" in backup_branch
