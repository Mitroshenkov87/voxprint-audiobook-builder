"""Suite install folder, silent migration, discovery and the download fixes."""
import json
from pathlib import Path

import pytest

from infra import download_retry, download_watch, install_layout, mirror_speed, model_downloader, suite_install, suite_shared

ROOT = Path(__file__).resolve().parents[1]


def test_windows_default_layout_names():
    assert install_layout.windows_install_dir() == r"C:\Program Files\VOXPrint AI Suite\Audiobook Builder"
    assert install_layout.windows_cli_path().endswith(r"Audiobook Builder\voxprint-audiobook-cli.exe")
    assert install_layout.GUI_EXE == "VOXPrint AI Audiobook Builder.exe"
    assert install_layout.CLI_EXE == "voxprint-audiobook-cli.exe"
    assert install_layout.START_MENU == "VOXPrint AI Suite"
    assert install_layout.SHORTCUT == "Audiobook Builder"
    assert install_layout.legacy_program_dir() == r"C:\Program Files\Voxprint"


def test_migration_removes_the_old_program_and_keeps_user_data():
    old = r"C:\Program Files\Voxprint\Voxprint.exe"
    data = r"C:\Users\me\AppData\Local\Voxprint\voices\narrator.txt"
    shared = r"C:\Users\me\AppData\Local\Voxprint\shared\manifest.json"
    suite = install_layout.legacy_suite_dir()
    assert suite == r"C:\Program Files\Voxprint AI Suite\Audiobook Builder"
    assert install_layout.migration_removes_program_dir(old)
    assert not install_layout.migration_removes_program_dir(data)
    assert not install_layout.migration_removes_program_dir(shared)
    assert not install_layout.migration_removes_program_dir(suite)
    assert not install_layout.migration_removes_program_dir(suite + r"\Voxprint AI Audiobook Builder.exe")
    assert install_layout.user_data_dir(r"C:\Users\me\AppData\Local").endswith(r"\Voxprint")


def test_old_suite_spelling_is_renamed_in_place():
    assert install_layout.case_only_rename("Voxprint AI Suite", install_layout.SUITE_DIR)
    assert install_layout.case_only_rename(install_layout.LEGACY_SUITE_GUI, install_layout.GUI_EXE)
    assert not install_layout.case_only_rename("Voxprint", install_layout.SUITE_DIR)
    assert not install_layout.case_only_rename(install_layout.SUITE_DIR, install_layout.SUITE_DIR)
    iss = (ROOT / "installer" / "Voxprint.iss").read_text(encoding="utf-8-sig")
    assert '#define LegacySuiteDir "Voxprint AI Suite"' in iss
    assert '#define LegacyDisplayName "Voxprint AI Audiobook Builder"' in iss
    assert "procedure FixBrandCasing" in iss
    assert "procedure RenameChildCase" in iss
    post = iss.split("RunSuiteInstall();", 1)[1].split("RegisterSuiteDiscovery();", 1)[0]
    assert "RemoveLegacyInstall();" in post and "FixBrandCasing();" in post
    assert post.index("RemoveLegacyInstall();") < post.index("FixBrandCasing();")
    body = iss.split("procedure RemoveLegacyInstall", 1)[1].split("procedure ", 1)[0]
    assert "LegacySuite" not in body
    assert "DelTree(Legacy, True, True, True)" in body


def test_installer_script_migrates_without_the_old_uninstaller():
    iss = (ROOT / "installer" / "Voxprint.iss").read_text(encoding="utf-8-sig")
    assert r"DefaultDirName={autopf}\{#SuiteDir}\{#AppFolder}" in iss
    assert "UsePreviousAppDir=no" in iss
    assert '#define AppExe "VOXPrint AI Audiobook Builder.exe"' in iss
    assert '#define CliExe "voxprint-audiobook-cli.exe"' in iss
    assert r"{group}\{#AppFolder}" in iss
    assert "procedure RemoveLegacyInstall" in iss
    assert "procedure DropLegacyUninstallKey" in iss
    assert r"Software\Voxprint\Apps\audiobook-builder" in iss
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
    assert "Voxprint AI Audiobook Builder.exe" in wf
    assert "suite-marker.txt" in wf
    assert "StringComparison]::Ordinal" in wf
    assert "voxprint-audiobook-cli.exe" in wf
    assert "--version" in wf
    assert "migration-marker.txt" in wf
    assert '/DIR=`"$dest`"' in wf
    assert "Voxprint-AI-Audiobook-Builder-patch-" in wf


def test_linux_launcher_name_and_compatibility_link():
    script = (ROOT / "installer" / "linux" / "install-voxprint-linux.sh").read_text(encoding="utf-8")
    assert 'LAUNCHER="$BIN_DIR/voxprint-audiobook-cli"' in script
    assert 'ln -sfn "voxprint-audiobook-cli"' in script
    assert "voxprint-audiobook.desktop" in script
    assert "voxprint-audiobook-cli" in script


def test_choose_primary_prefers_a_fast_hugging_face_over_a_slow_modelscope():
    assert mirror_speed.choose_primary(500 * 1024, 84 * 1024) == "hf"
    assert mirror_speed.choose_primary(90 * 1024, 84 * 1024) == "hf"
    assert mirror_speed.choose_primary(100 * 1024, 2 * 1024 * 1024) == "ms"
    assert mirror_speed.choose_primary(0, 0) is None
    assert mirror_speed.choose_primary(0, 84 * 1024) == "ms"
    assert mirror_speed.order_for(500 * 1024, 84 * 1024)[0] == "hf"
    assert mirror_speed.too_slow(84 * 1024, 4.0, mirror_speed.SLOW_BPS, mirror_speed.SLOW_AFTER, True)
    assert not mirror_speed.too_slow(84 * 1024, 1.0, mirror_speed.SLOW_BPS, mirror_speed.SLOW_AFTER, True)


def test_a_slow_stall_is_not_retried_on_the_same_source():
    slow = download_watch.Stalled("slow", progressed=True, slow=True)
    stalled = download_watch.Stalled("stalled", progressed=True)
    assert model_downloader._retry_same_source(slow, 0) is False
    assert model_downloader._retry_same_source(stalled, 0) is True
    assert model_downloader._retry_same_source(stalled, model_downloader.STALL_RETRIES) is False


def test_download_retries_a_transient_error_and_keeps_the_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(download_retry, "_sleep", lambda _s: None)
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

    got = download_retry.download_with_retry(dest, ["https://example.invalid/a"], sha256="a" * 64, download=fetch)
    assert got.read_bytes() == b"ok" and calls["n"] == 3
    assert (tmp_path / "weights.bin.partial").read_bytes() == b"part"

    def full(destination, sources, *, sha256, size=None, **kwargs):
        raise download_retry.DiskSpaceError(destination, 10, 1)

    with pytest.raises(download_retry.DiskSpaceError):
        download_retry.download_with_retry(dest, ["https://example.invalid/a"], sha256="b" * 64, download=full, tries=5)


def test_suite_status_is_utf8_with_a_bom_on_windows(tmp_path, monkeypatch):
    path = tmp_path / "suite-status.txt"
    monkeypatch.setattr(suite_install.sys, "platform", "win32")
    suite_install.write_status(path, "done", 1000, "Готово", 2)
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert raw.decode("utf-8-sig").splitlines()[2] == "Готово"
    monkeypatch.setattr(suite_install.sys, "platform", "linux")
    suite_install.write_status(path, "done", 1000, "ok", 3)
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert path.read_text(encoding="utf-8").splitlines()[0] == "done"


def test_faster_hugging_face_is_tried_before_modelscope(tmp_path, monkeypatch):
    monkeypatch.delenv("VOXPRINT_NO_MIRROR", raising=False)
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(mirror_speed, "measure", lambda _repo: (500 * 1024, 84 * 1024))
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
