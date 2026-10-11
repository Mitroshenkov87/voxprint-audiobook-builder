"""Suite install folder, silent migration, discovery and the download fixes."""
import json
from pathlib import Path

import pytest

from infra import download_watch, install_layout, model_downloader, parallel_download, suite_shared

ROOT = Path(__file__).resolve().parents[1]


def test_windows_default_layout_names():
    assert install_layout.windows_install_dir() == r"C:\Program Files\VOXPrint AI Suite\Audiobook Builder"
    assert install_layout.windows_cli_path().endswith(r"Audiobook Builder\voxprint-audiobook-cli.exe")
    assert install_layout.GUI_EXE == "voxprint-audiobook-builder.exe"
    assert install_layout.CLI_EXE == "voxprint-audiobook-cli.exe"
    assert install_layout.START_MENU == "VOXPrint AI Suite"
    assert install_layout.SHORTCUT == "VOXPrint AI Audiobook Builder"
    assert install_layout.LINUX_INSTALL == "/opt/voxprint-ai-suite/audiobook-builder"
    assert install_layout.LINUX_DESKTOP.endswith("voxprint-audiobook-builder.desktop")
    assert install_layout.legacy_program_dir() == r"C:\Program Files\Voxprint"
    assert install_layout.kebab_install_dir() == r"C:\Program Files\voxprint-ai-suite\audiobook-builder"


def test_migration_removes_the_old_program_and_keeps_user_data():
    old = r"C:\Program Files\Voxprint\Voxprint.exe"
    data = r"C:\Users\me\AppData\Local\Voxprint\voices\narrator.txt"
    shared = r"C:\Users\me\AppData\Local\Voxprint\shared\manifest.json"
    new = install_layout.windows_install_dir()
    cased = r"C:\Program Files\Voxprint AI Suite\Audiobook Builder"
    kebab = install_layout.kebab_install_dir()
    assert install_layout.migration_removes_program_dir(old)
    assert install_layout.migration_removes_program_dir(kebab)
    assert install_layout.migration_removes_program_dir(kebab + r"\voxprint-audiobook-builder.exe")
    assert not install_layout.migration_removes_program_dir(cased)
    assert not install_layout.migration_removes_program_dir(new)
    assert not install_layout.migration_removes_program_dir(new + r"\voxprint-audiobook-builder.exe")
    assert not install_layout.migration_removes_program_dir(data)
    assert not install_layout.migration_removes_program_dir(shared)
    assert install_layout.user_data_dir(r"C:\Users\me\AppData\Local").endswith(r"\Voxprint")


def test_old_suite_spelling_is_renamed_and_the_kebab_folder_is_removed():
    assert install_layout.case_only_rename("Voxprint AI Suite", install_layout.SUITE_DIR)
    assert install_layout.case_only_rename("Voxprint AI Suite", install_layout.START_MENU)
    assert not install_layout.case_only_rename("voxprint-ai-suite", install_layout.SUITE_DIR)
    assert not install_layout.case_only_rename(install_layout.LEGACY_SUITE_GUI, install_layout.GUI_EXE)
    iss = (ROOT / "installer" / "Voxprint.iss").read_text(encoding="utf-8-sig")
    assert '#define LegacySuiteDir "Voxprint AI Suite"' in iss
    assert '#define SuiteDir "VOXPrint AI Suite"' in iss
    assert '#define AppFolder "Audiobook Builder"' in iss
    assert '#define LegacyKebabSuite "voxprint-ai-suite"' in iss
    assert '#define LegacyKebabApp "audiobook-builder"' in iss
    assert '#define StartMenu "VOXPrint AI Suite"' in iss
    assert "procedure FixStartMenuCasing" in iss
    assert "procedure FixInstallFolderCasing" in iss
    assert "procedure RemoveOldExeNames" in iss
    assert "RmDir(" not in iss
    post = iss.split("RunSuiteInstall();", 1)[1].split("RegisterSuiteDiscovery();", 1)[0]
    assert post.index("RemoveLegacyInstall();") < post.index("FixInstallFolderCasing();")
    assert post.index("FixInstallFolderCasing();") < post.index("FixStartMenuCasing();")
    assert "RemoveOldExeNames();" in post
    body = iss.split("procedure RemoveLegacyInstall", 1)[1].split("procedure ", 1)[0]
    assert "LegacyKebabAppDir" in body
    assert "DelTree(Legacy, True, True, True)" in body
    assert "{localappdata}\\Voxprint" not in body


def test_installer_script_migrates_without_the_old_uninstaller():
    iss = (ROOT / "installer" / "Voxprint.iss").read_text(encoding="utf-8-sig")
    assert r"DefaultDirName={autopf}\{#SuiteDir}\{#AppFolder}" in iss
    assert "UsePreviousAppDir=no" in iss
    assert "UninstallDisplayName={#AppDisplayName}" in iss
    assert '#define AppExe "voxprint-audiobook-builder.exe"' in iss
    assert '#define CliExe "voxprint-audiobook-cli.exe"' in iss
    assert r"{group}\{#AppDisplayName}" in iss
    assert "procedure RemoveLegacyInstall" in iss
    assert "procedure DropLegacyUninstallKey" in iss
    assert r"Software\voxprint\Apps\audiobook-builder" in iss
    body = iss.split("procedure RemoveLegacyInstall", 1)[1].split("procedure ", 1)[0]
    assert "unins" not in body.lower()
    assert "unregister-models" not in body.lower()
    assert "{localappdata}\\Voxprint" not in body


def test_discovery_record_keeps_components_and_falls_back_to_the_default_path(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_SHARED", str(tmp_path / "shared"))
    suite_shared.register("movie-dubber", ["runtime"])
    install = tmp_path / "Audiobook Builder"
    cli = install / "voxprint-audiobook-cli.exe"
    install.mkdir()
    cli.write_bytes(b"cli")
    install_layout.register_install(install, cli)
    raw = json.loads(suite_shared.manifest_path().read_text(encoding="utf-8"))
    assert raw["components"]
    assert "apps" not in raw
    assert raw["installs"]["audiobook-builder"]["cli_path"] == str(cli)
    assert install_layout.locate_cli() == cli
    install_layout.unregister_install()
    after = json.loads(suite_shared.manifest_path().read_text(encoding="utf-8"))
    assert "installs" not in after
    assert after["components"]
    monkeypatch.setattr(install_layout, "default_cli_path", lambda: cli)
    assert install_layout.locate_cli() == cli


def test_ci_smoke_installs_the_old_layout_then_the_new_one():
    wf = (ROOT / ".github" / "workflows" / "build-installer.yml").read_text(encoding="utf-8")
    assert r"C:\Program Files\Voxprint" in wf
    assert r"C:\Program Files\Voxprint AI Suite\Audiobook Builder" in wf
    assert r"C:\Program Files\voxprint-ai-suite\audiobook-builder" in wf
    assert r"C:\Program Files\VOXPrint AI Suite\Audiobook Builder" in wf
    assert "kebab-marker.txt" in wf
    assert "suite-marker.txt" in wf
    assert "voxprint-audiobook-builder.exe" in wf
    assert "VOXPrint AI Audiobook Builder.lnk" in wf
    assert "suite-marker.txt" in wf
    assert "StringComparison]::Ordinal" in wf
    assert "voxprint-audiobook-cli.exe" in wf
    assert "--version" in wf
    assert "migration-marker.txt" in wf
    assert '/DIR=`"$dest`"' in wf
    assert "voxprint-audiobook-builder-patch-" in wf


def test_current_docs_name_the_kebab_install_and_keep_the_display_name():
    manual = (ROOT / "docs" / "manual" / "src" / "manual_en.md").read_text(encoding="utf-8")
    cli = (ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
    assert r"C:\Program Files\VOXPrint AI Suite\Audiobook Builder\voxprint-audiobook-cli.exe" in manual
    assert "voxprint-audiobook-builder.exe" in manual
    assert "VOXPrint AI Audiobook Builder" in manual
    assert r"C:\Program Files\Voxprint\Voxprint.exe" not in manual
    assert "voxprint-audiobook-builder.exe --modules-status" in cli
    assert "voxprint-ai-suite" in cli and "audiobook-builder" in cli


def test_linux_launcher_name_and_compatibility_link():
    script = (ROOT / "installer" / "linux" / "install-voxprint-linux.sh").read_text(encoding="utf-8")
    assert 'PREFIX="/opt/voxprint-ai-suite/audiobook-builder"' in script
    assert 'LAUNCHER="$BIN_DIR/voxprint-audiobook-cli"' in script
    assert 'ln -sfn "voxprint-audiobook-cli"' in script
    assert "voxprint-audiobook-builder.desktop" in script
    assert "remove_old_user_package" in script
    assert "~/.local/share/voxprint/app" in script


def test_choose_primary_prefers_a_fast_hugging_face_over_a_slow_modelscope():
    assert parallel_download.choose_primary(500 * 1024, 84 * 1024) == "hf"
    assert parallel_download.choose_primary(90 * 1024, 84 * 1024) == "hf"
    assert parallel_download.choose_primary(100 * 1024, 2 * 1024 * 1024) == "ms"
    assert parallel_download.choose_primary(0, 0) is None
    assert parallel_download.choose_primary(0, 84 * 1024) == "ms"
    assert parallel_download.order_for(500 * 1024, 84 * 1024)[0] == "hf"
    assert parallel_download.too_slow(84 * 1024, 4.0, parallel_download.SLOW_BPS, parallel_download.SLOW_AFTER, True)
    assert not parallel_download.too_slow(84 * 1024, 1.0, parallel_download.SLOW_BPS, parallel_download.SLOW_AFTER, True)


def test_a_slow_stall_is_not_retried_on_the_same_source():
    slow = download_watch.Stalled("slow", progressed=True, slow=True)
    stalled = download_watch.Stalled("stalled", progressed=True)
    assert model_downloader._retry_same_source(slow, 0) is False
    assert model_downloader._retry_same_source(stalled, 0) is True
    assert model_downloader._retry_same_source(stalled, model_downloader.STALL_RETRIES) is False


def test_download_retries_a_transient_error_and_keeps_the_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel_download, "_retry_sleep", lambda _s: None)
    dest = tmp_path / "weights.bin"
    calls = {"n": 0}

    def fetch(destination, sources, *, sha256, size=None, **kwargs):
        calls["n"] += 1
        partial = Path(str(destination) + ".partial")
        partial.write_bytes(b"part")
        if calls["n"] < 3:
            raise TimeoutError("reset")
        destination.write_bytes(b"ok")
        return destination

    got = parallel_download.download_with_retry(dest, ["https://example.invalid/a"], sha256="a" * 64, download=fetch)
    assert got.read_bytes() == b"ok" and calls["n"] == 3
    assert (tmp_path / "weights.bin.partial").read_bytes() == b"part"

    def full(destination, sources, *, sha256, size=None, **kwargs):
        raise parallel_download.DiskSpaceError(destination, 10, 1)

    with pytest.raises(parallel_download.DiskSpaceError):
        parallel_download.download_with_retry(dest, ["https://example.invalid/a"], sha256="b" * 64, download=full, tries=5)


def test_json_stdout_is_utf8_and_keeps_cyrillic(monkeypatch):
    import io

    import cli

    buf = io.BytesIO()
    wrapper = io.TextIOWrapper(buf, encoding="cp1252", newline="\n")
    monkeypatch.setattr(cli.sys, "stdout", wrapper)
    monkeypatch.setattr(cli.sys, "stderr", io.StringIO())
    cli._json_stdout = None
    cli.begin_json_stdout()
    cli._emit_result(
        ok=True, command="status", exit_code=0, outputs=[], warnings=[], duration_s=0.0,
        error=None, hint=None, extra={"message": "Готово"},
    )
    wrapper.flush()
    raw = buf.getvalue()
    text = raw.decode("utf-8")
    assert "Готово" in text
    assert "\\u0413" not in text
    cli.end_json_stdout()


def test_faster_hugging_face_is_tried_before_modelscope(tmp_path, monkeypatch):
    monkeypatch.delenv("VOXPRINT_NO_MIRROR", raising=False)
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(parallel_download, "measure", lambda _repo: (500 * 1024, 84 * 1024))
    order = []

    def snap(repo_id, local_dir, **kwargs):
        order.append("hf")
        folder = Path(local_dir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text("{}", encoding="utf-8")
        (folder / "model.bin").write_bytes(b"weights")

    def ms_download(repo, dest, progress, expected):
        order.append("ms")
        progress(1.0)

    from infra import modelscope_mirror

    modelscope_mirror.reset_verdict()
    model_downloader.ensure_model(
        "Org/Speed", snapshot_download=snap, revision="a" * 40, mirror_download=ms_download,
    )
    assert order[0] == "hf"
