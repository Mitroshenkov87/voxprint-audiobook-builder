"""install.ps1 downloads only the official release, and a dry run can be pointed at 127.0.0.1."""
import hashlib
import shutil
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = "Mitroshenkov87/voxprint-audiobook-builder"


def test_script_and_readme_stay_on_the_official_release():
    script = (ROOT / "install.ps1").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "build-installer.yml").read_text(encoding="utf-8")
    assert OFFICIAL in script
    assert "releases/download/" in script
    assert "-SourceUrl is only allowed with -DryRun" in script
    assert "/VERYSILENT" in script
    assert "RunAs" in script
    assert "Option 2: one command in PowerShell (as Administrator)" in readme
    assert "irm https://raw.githubusercontent.com/Mitroshenkov87/voxprint-audiobook-builder/main/install.ps1 | iex" in readme
    assert "](install.ps1)" in readme
    assert "\n  install-script:\n" in workflow
    assert "http://127.0.0.1:8765/" in workflow


def _pwsh():
    return shutil.which("pwsh") or shutil.which("powershell")


def test_dry_run_checks_a_local_release_and_refuses_anything_else(tmp_path):
    pwsh = _pwsh()
    if not pwsh:
        pytest.skip("PowerShell is not installed")
    release = tmp_path / "release"
    release.mkdir()
    payload = b"voxprint-fake-installer"
    (release / "Voxprint-Setup-online.exe").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    sha = release / "Voxprint-Setup-online.exe.sha256"
    sha.write_text(f"{digest}  Voxprint-Setup-online.exe\n", encoding="ascii")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(release), **kwargs)

        def log_message(self, fmt, *args):
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    script = str(ROOT / "install.ps1")
    local = f"http://127.0.0.1:{port}/"

    def run(*extra):
        return subprocess.run(
            [pwsh, "-NoProfile", "-File", script, *extra],
            capture_output=True, text=True, check=False,
        )

    try:
        ok = run("-DryRun", "-SourceUrl", local)
        assert ok.returncode == 0, ok.stdout + ok.stderr
        assert "Dry run" in ok.stdout and "SHA-256 matches" in ok.stdout
        sha.write_text(("0" * 64) + "  Voxprint-Setup-online.exe\n", encoding="ascii")
        bad = run("-DryRun", "-SourceUrl", local)
        assert bad.returncode != 0 and "does not match" in (bad.stdout + bad.stderr)
        evil = run("-DryRun", "-SourceUrl", "https://example.com/Voxprint-Setup-online.exe")
        assert evil.returncode != 0 and "127.0.0.1" in (evil.stdout + evil.stderr)
        real = run("-SourceUrl", local)
        assert real.returncode != 0 and "only allowed with -DryRun" in (real.stdout + real.stderr)
    finally:
        httpd.shutdown()
        httpd.server_close()
