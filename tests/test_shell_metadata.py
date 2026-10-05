"""Every package bundled into the thin shell must keep its dist-info (real PC failure: transformers -> "No package metadata was found
for packaging"), and the dead multi-GB full build must stay switched off."""
import json
from pathlib import Path

from tools import make_runtime_lock as mrl

ROOT = Path(__file__).resolve().parents[1]


def test_copy_metadata_args_cover_every_shell_package_of_the_lock():
    lock = json.loads((ROOT / "infra" / "runtime_lock.json").read_text(encoding="utf-8"))
    args = mrl.pyinstaller_metadata_args(lock)
    copied = {args[i + 1] for i, a in enumerate(args) if a == "--copy-metadata"}
    assert copied == set(lock["shell"]) and {"packaging", "psutil", "soundfile", "cffi", "pycparser", "pyside6", "shiboken6"} <= copied
    assert set(lock["shell"]) == mrl.SHELL_PROVIDED          # the lock is in sync with the list of what the shell provides


def test_cli_prints_one_line_for_the_build_script(capsys):
    assert mrl.main(["--pyinstaller-metadata-args"]) == 0
    out = capsys.readouterr().out.strip()
    assert "\n" not in out and "--copy-metadata packaging" in out


def test_build_script_uses_it_and_ships_the_lock_for_the_frozen_check():
    bat = (ROOT / "build_thin.bat").read_text(encoding="utf-8")
    assert "--pyinstaller-metadata-args" in bat and "%META%" in bat and "runtime_lock.json;infra" in bat
    main = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "runtime_lock.json" in main and "_md.version(dist)" in main


def test_the_heavy_full_build_cannot_run_any_more():
    wf = (ROOT / ".github" / "workflows" / "build-installer.yml").read_text(encoding="utf-8")
    assert "\n  build:\n" not in wf and "build_full" not in wf and "tags:" not in wf
    assert "build.bat onedir" not in wf
    ps1 = (ROOT / "installer" / "build_online.ps1").read_text(encoding="utf-8")
    assert "LegacyPayload" in ps1 and "Refusing to build multi-GB payload parts" in ps1
