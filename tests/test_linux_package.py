"""The experimental Linux package: XDG data folder, the package builder (tools/make_linux_package.py), the installer script
and the menu entry.  Nothing is installed; the script is only syntax-checked and run with --check/--help."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from infra import paths
from tests.test_online_installer import Server
from tools import make_linux_package as mk
from tools import online_fetch as of

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "installer" / "linux" / "install-voxprint-linux.sh"


def test_linux_data_dir_is_xdg_and_lower_case(monkeypatch, tmp_path):
    monkeypatch.delenv("VOXPRINT_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert paths.app_home() == tmp_path / "xdg" / "voxprint"
    monkeypatch.delenv("XDG_DATA_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "h"))
    assert paths.app_home() == tmp_path / "h" / ".local" / "share" / "voxprint"
    monkeypatch.setenv("XDG_DATA_HOME", "relative/path")       # XDG says: relative values are invalid
    assert paths.app_home() == tmp_path / "h" / ".local" / "share" / "voxprint"


def test_voxprint_home_overrides_everywhere(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "x"))
    monkeypatch.setattr(sys, "platform", "linux")
    assert paths.app_home() == tmp_path / "x"


def test_earlier_capitalised_linux_folder_is_still_found(monkeypatch, tmp_path):
    """Before the lower-case name an XDG install used ``~/.local/share/Voxprint``: its models/settings are adopted."""
    monkeypatch.delenv("VOXPRINT_HOME", raising=False)
    monkeypatch.delenv("VOXPRINT_PREVIOUS_HOMES", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    h = tmp_path / "h"
    (h / ".local" / "share" / "Voxprint" / "models").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: h))
    monkeypatch.setenv("XDG_DATA_HOME", str(h / ".local" / "share"))
    cur = paths.app_home()
    assert cur.name == "voxprint"
    old = h / ".local" / "share" / "Voxprint"
    if not cur.exists() or not old.samefile(cur):             # (a case-insensitive file system would make them one folder)
        assert old.name in [p.name for p in paths.previous_homes()]


@pytest.fixture
def built(tmp_path):
    out = tmp_path / "srv"
    srv = Server(out)
    man = mk.build(ROOT, out, "v0.1.0-beta", "o/r", base_url=srv.url)
    yield srv, man, out
    srv.close()


def test_package_contents_and_manifest(built):
    srv, man, out = built
    m = json.loads(man.read_text(encoding="utf-8"))
    assert man.name == "manifest-linux.json" and m["schema"] == 1 and m["experimental"] is True and m["platform"] == "linux"
    (c,) = m["components"]
    assert c["file"] == "Voxprint-linux-app.zip" and c["sha256"] == mk.sha256_of(out / c["file"]) and c["size"] == (out / c["file"]).stat().st_size
    with zipfile.ZipFile(out / c["file"]) as z:
        names = set(z.namelist())
    assert {"main.py", "credits.json", "requirements.txt", "core/translate.py", "ui/studio.py", "locales/de.json",
            "installer/linux/voxprint.desktop", "installer/linux/install-voxprint-linux.sh", "assets/voxprint.png"} <= names
    assert not any(n.startswith(("tests/", ".github/", "docs/manual")) or "__pycache__" in n or n.endswith(".pyc") for n in names)
    assert "installer/Voxprint.iss" not in names and "build.bat" not in names
    sums = (out / "SHA256SUMS-linux.txt").read_text(encoding="utf-8")
    for n in ("Voxprint-linux-app.zip", "manifest-linux.json", "voxprint-fetch.py", "install-voxprint-linux.sh", "Voxprint-linux-experimental.tar.gz"):
        assert f"{mk.sha256_of(out / n)}  {n}" in sums


def test_tarball_has_installer_and_app_tree(built):
    _, _, out = built
    with tarfile.open(out / "Voxprint-linux-experimental.tar.gz") as t:
        names = t.getnames()
        mode = t.getmember("voxprint-linux/install-voxprint-linux.sh").mode
    assert "voxprint-linux/app/main.py" in names and "voxprint-linux/voxprint-fetch.py" in names and mode & 0o111


def test_release_and_tag_are_filled_into_the_script(built):
    _, _, out = built
    text = (out / "install-voxprint-linux.sh").read_text(encoding="utf-8")
    assert "@REPO@" not in text and "@TAG@" not in text and 'TAG="${VOXPRINT_TAG:-v0.1.0-beta}"' in text
    assert "@REPO@" in SCRIPT.read_text(encoding="utf-8")        # the template stays a template


def test_the_windows_downloader_installs_the_linux_component(built, tmp_path):
    srv, man, out = built
    dest = tmp_path / "app"
    rc = of.main(["--manifest", str(man), "--dest", str(dest), "--cache", str(tmp_path / "cache")])
    assert rc == 0 and (dest / "main.py").is_file() and (dest / "core" / "translate.py").is_file()
    assert any("Voxprint-linux-app.zip" in p for p, _ in srv.requests)


@pytest.mark.skipif(shutil.which("bash") is None or sys.platform == "win32", reason="bash needed")
def _os_release(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


@pytest.mark.skipif(shutil.which("bash") is None or sys.platform == "win32", reason="bash needed")
def test_old_distro_warns_and_a_current_one_does_not(tmp_path):
    """Release year from os-release: before 2025 warns and does not change the exit code."""
    old = tmp_path / "old"
    new = tmp_path / "new"
    plain = tmp_path / "plain"
    missing = tmp_path / "missing"
    _os_release(old, 'NAME="Ubuntu"\nVERSION_ID="24.04"\nVERSION="24.04.3 LTS (Noble Numbat)"\nPRETTY_NAME="Ubuntu 24.04.3 LTS"\n')
    _os_release(new, 'NAME="Ubuntu"\nVERSION_ID="25.04"\nPRETTY_NAME="Ubuntu 25.04"\n')
    _os_release(plain, 'NAME="Debian GNU/Linux"\nVERSION_ID="13"\nVERSION="13 (trixie)"\nPRETTY_NAME="Debian GNU/Linux 13 (trixie)"\n')
    dated = tmp_path / "dated"
    _os_release(dated, 'NAME="openSUSE Tumbleweed"\nVERSION_ID="20241001"\nPRETTY_NAME="openSUSE Tumbleweed"\n')

    def check(path: Path):
        env = dict(os.environ)
        env["VOXPRINT_OS_RELEASE"] = str(path)
        env["VOXPRINT_PYTHON"] = sys.executable
        return subprocess.run(["bash", str(SCRIPT), "--check"], capture_output=True, text=True, env=env)

    warned = check(old)
    quiet = check(new)
    undated = check(plain)
    tumble = check(dated)
    absent = check(missing)
    assert "older than 2025" in warned.stderr and "Installation continues" in warned.stderr
    assert "Ubuntu 24.04.3 LTS" in warned.stderr
    assert "older than 2025" not in quiet.stderr
    assert "older than 2025" not in undated.stderr          # no year in the file: best effort, no warning
    assert "older than 2025" in tumble.stderr               # calendar year in VERSION_ID
    assert "older than 2025" not in absent.stderr
    assert warned.returncode == quiet.returncode == undated.returncode == tumble.returncode == absent.returncode


def test_installer_script_syntax_and_help():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0
    r = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--uninstall" in r.stdout and "--install-deps" in r.stdout
    r = subprocess.run(["bash", str(SCRIPT), "--bogus"], capture_output=True, text=True)
    assert r.returncode != 0


def test_desktop_entry_and_icon():
    text = (ROOT / "installer" / "linux" / "voxprint.desktop").read_text(encoding="utf-8")
    kv = dict(line.split("=", 1) for line in text.splitlines() if "=" in line and not line.startswith("#"))
    assert kv["Type"] == "Application" and kv["Icon"] == "voxprint" and kv["Exec"].startswith("@LAUNCHER@")
    assert kv["Categories"].endswith(";") and kv["Terminal"] == "false"
    png = (ROOT / "installer" / "linux" / "voxprint-256.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_text_selftest_passes_without_models():
    from workers import selftest_text

    assert selftest_text.run() == 0
