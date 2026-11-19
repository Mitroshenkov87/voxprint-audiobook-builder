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
    assert copied == set(lock["shell"]) | set(mrl.git_dist_names(lock))
    assert {"packaging", "psutil", "soundfile", "cffi", "pycparser", "pyside6", "shiboken6", "voxprint-common"} <= copied
    assert set(lock["shell"]) == mrl.SHELL_PROVIDED          # the lock is in sync with the list of what the shell provides


def test_cli_prints_one_line_for_the_build_script(capsys):
    assert mrl.main(["--pyinstaller-metadata-args"]) == 0
    out = capsys.readouterr().out.strip()
    assert "\n" not in out and "--copy-metadata packaging" in out


def test_full_build_targets_python_314_and_bundles_audioop():
    bat = (ROOT / "build.bat").read_text(encoding="utf-8")
    assert "py -3.14" in bat and "--hidden-import audioop" in bat and "cu130" in bat
    assert "py -3.11" not in bat and "cu128" not in bat


def test_build_script_uses_it_and_ships_the_lock_for_the_frozen_check():
    bat = (ROOT / "build_thin.bat").read_text(encoding="utf-8")
    assert "--pyinstaller-metadata-args" in bat and "%META%" in bat and "runtime_lock.json;infra" in bat
    write_at = bat.index("embed_build_stamp.py --write-module")
    pack_at = bat.index("\npyinstaller ")
    restore_at = bat.index("embed_build_stamp.py --restore-module")
    exe_at = bat.index("embed_build_stamp.py --exe")
    assert write_at < pack_at < restore_at < exe_at
    main = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "runtime_lock.json" in main and "_md.version(dist)" in main


def test_workflows_stamp_the_explicit_build_and_ignore_the_run_number():
    inst = (ROOT / ".github" / "workflows" / "build-installer.yml").read_text(encoding="utf-8")
    linux = (ROOT / ".github" / "workflows" / "build-linux.yml").read_text(encoding="utf-8")
    assert "GITHUB_RUN_NUMBER +" not in inst and "GITHUB_RUN_NUMBER +" not in linux
    assert "tools\\build_number.py --stamp" in inst
    assert "tools/build_number.py --stamp" in linux
    assert linux.index("build_number.py --stamp") < linux.index("make_linux_package.py")


def test_the_heavy_full_build_cannot_run_any_more():
    wf = (ROOT / ".github" / "workflows" / "build-installer.yml").read_text(encoding="utf-8")
    assert "\n  build:\n" not in wf and "build_full" not in wf
    assert "build.bat onedir" not in wf
    # a v* tag push may only run the thin build (CI artifacts only); the heavy full-payload job stays gone
    assert 'push:\n    tags: ["v*"]' in wf
    assert "REL_TAG: ${{ startsWith(github.ref, 'refs/tags/') && github.ref_name || '' }}" in wf


def test_no_workflow_creates_or_uploads_to_a_release():
    """Repo rule: builds stay CI artifacts; no workflow touches a GitHub release."""
    for wf_path in (ROOT / ".github" / "workflows").glob("*.yml"):
        wf = wf_path.read_text(encoding="utf-8")
        assert "gh release" not in wf, wf_path.name
        assert "action-gh-release" not in wf and "release_tag" not in wf, wf_path.name
        assert "contents: write" not in wf, wf_path.name


def test_thin_job_publishes_an_exe_patch():
    """build-thin uploads the GUI exe plus patch.json as a CI artifact."""
    wf = (ROOT / ".github" / "workflows" / "build-installer.yml").read_text(encoding="utf-8")
    assert "name: Voxprint-patch\n" in wf
    assert "dist/thin/voxprint-audiobook-builder/voxprint-audiobook-builder.exe\n" in wf
    assert "dist/thin/voxprint-audiobook-builder/voxprint-audiobook-builder.exe.sha256\n" in wf
    assert "dist/thin/voxprint-audiobook-builder/patch.json\n" in wf
    patch = wf.split("name: Voxprint-patch", 1)[1][:500]
    assert "retention-days: 14" in patch
    assert "runtime_lock_sha256" in wf and "VOXPRINT_BUILD" in wf
    ps1 = (ROOT / "installer" / "build_online.ps1").read_text(encoding="utf-8")
    assert "LegacyPayload" in ps1 and "Refusing to build multi-GB payload parts" in ps1
