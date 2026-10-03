"""installer/Voxprint.iss: the optional "existing models" wizard page, trilingual wizard texts, encoding.

Inno Setup cannot be compiled on Linux, so these tests lint the script: encoding, message tables, the Pascal block structure and
the contract with the app (the file the page writes is the file ``infra.existing_models`` reads)."""
import re
from pathlib import Path

from infra import existing_models

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
    # {cm:Name} in script sections, CustomMessage('Name') in [Code]; CreateDesktopIcon / AdditionalIcons ship with Inno
    builtin = {"CreateDesktopIcon", "AdditionalIcons"}
    used = set(re.findall(r"\{cm:(\w+)", text())) | set(re.findall(r"CustomMessage\('(\w+)'\)", text()))
    assert used - builtin <= defined["english"], used - builtin - defined["english"]
    assert {"ModelsPageCaption", "ModelsPageDescription", "ModelsPageSubCaption", "ModelsPagePrompt", "ModelsPageBadFolder"} <= used
    # the %1/%2 placeholders agree between the languages
    for name in defined["english"]:
        ph = {l: sorted(set(re.findall(r"%\d", next(x for x in section("CustomMessages").splitlines() if x.startswith(f"{l}.{name}=")))))
              for l in LANGS}
        assert ph["english"] == ph["russian"] == ph["german"], name


def test_no_russian_text_is_left_outside_the_message_table():
    t = text()
    rest = t.replace(section("CustomMessages"), "")
    code_lines = [l for l in rest.splitlines() if re.search("[А-Яа-яЁё]", l) and not l.lstrip().startswith((";", "{"))]
    assert code_lines == []


def test_pascal_blocks_are_balanced():
    code = re.sub(r"\{[^}]*\}", "", section("Code"))           # { comments }
    code = re.sub(r"'[^']*'", "''", code)
    words = re.findall(r"\b(begin|end|case|try)\b", code, re.I)
    opens = sum(1 for w in words if w.lower() in ("begin", "case", "try"))
    assert opens == sum(1 for w in words if w.lower() == "end")
    assert code.lower().count("procedure ") + code.lower().count("function ") >= 5


def test_models_page_is_optional_skippable_and_only_remembers_the_path():
    code = section("Code")
    assert "CreateInputDirPage(wpSelectDir" in code                       # right after the install folder
    assert "{param:ModelsDir|}" in code                                    # silent installs: /ModelsDir="D:\models"
    assert "Trim(ModelsPage.Values[0])" in code and "Dir <> ''" in code     # empty = skipped
    assert "not DirExists(Dir)" in code                                    # a typo is caught before installing
    assert "ssPostInstall" in code and "SaveStringsToUTF8File" in code
    assert f"\\state" in code and existing_models.CONFIG_NAME in code
    assert "{localappdata}\\Voxprint\\state" in code
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
