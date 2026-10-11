"""scripts/vm_test.sh with stand-in nvidia-smi and voxprint. No GPU."""
from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

from core.vxbook import load_vxbook

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "vm_test.sh"
PS1 = ROOT / "scripts" / "vm_test.ps1"

FAKE_VOX = r"""
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
log = Path(os.environ["FAKE_LOG"])
with log.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(args, ensure_ascii=False) + "\n")

def out_path():
    return Path(args[args.index("--out") + 1])

if "--version" in args:
    payload = {"ok": True, "command": "version", "version": "1.0.0-rc.3", "build": 1002, "codename": "Tamim"}
    print(json.dumps(payload) if "--json" in args else 'Voxprint 1.0.0-rc.3 build 1002 "Tamim"')
    sys.exit(0)
if "--selftest" in args:
    print("selftest ok")
    sys.exit(0)
if "--dry-run" in args:
    cmd = "prepare" if "prepare" in args else "narrate"
    print(json.dumps({
        "type": "result", "ok": True, "command": cmd, "exit_code": 0,
        "outputs": [], "warnings": [], "dry_run": True, "durations": {"audio_s": None},
    }))
    sys.exit(0)
if args and args[0] == "prepare":
    out_path().write_text("Ещё одно слово.\n", encoding="utf-8")
    sys.exit(0)
if args and args[0] == "speakers":
    out_path().write_text("1. NARRATOR\n2. MALE: Иван\n3. FEMALE: Анна\n", encoding="utf-8")
    print(json.dumps({"type": "result", "ok": True, "command": "speakers"}))
    sys.exit(0)
if args and args[0] == "narrate":
    if os.environ.get("FAKE_NARRATE_RC") == "1":
        print("narration failed", file=sys.stderr)
        sys.exit(1)
    dest = out_path()
    dest.mkdir(parents=True, exist_ok=True)
    names = ["01 - Утро.mp3", "02 - Вечер.mp3"] if any(str(a).endswith(".vxbook") for a in args) else ["01 - Chapter.mp3"]
    written = []
    for name in names:
        (dest / name).write_bytes(b"ID3")
        written.append(str(dest / name))
    print(json.dumps({
        "type": "result", "ok": True, "command": "narrate", "exit_code": 0,
        "outputs": written, "out_dir": str(dest), "durations": {"audio_s": 1.25},
    }))
    sys.exit(0)
print("unexpected", args, file=sys.stderr)
sys.exit(2)
"""


def _exe(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _layout(tmp_path: Path) -> dict:
    home = tmp_path / "home"
    home.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    (bindir / "fake_vox.py").write_text(FAKE_VOX, encoding="utf-8", newline="\n")
    # A bash wrapper so Git Bash does not depend on a python3 store alias.
    _exe(bindir / "voxprint", """#!/bin/bash
if python3 -c "import sys" >/dev/null 2>&1; then PY=python3; else PY=python; fi
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
exec "$PY" "$(cd "$(dirname "$0")" && pwd)/fake_vox.py" "$@"
""")
    _exe(bindir / "nvidia-smi", "#!/bin/sh\necho 2048\nexit 0\n")
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    env["FAKE_LOG"] = str(log)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.pop("VOXPRINT_BIN", None)
    env.pop("VOXPRINT_VM_RESULTS", None)
    env.pop("VOXPRINT_VM_DONE", None)
    env.pop("FAKE_NARRATE_RC", None)
    return {"home": home, "log": log, "env": env, "cwd": tmp_path}


def _run(layout: dict, *args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(layout["env"])
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=layout["cwd"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _objects(proc: subprocess.CompletedProcess) -> list:
    assert proc.stdout, proc.stderr
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def test_scripts_are_noninteractive_and_share_the_contract():
    sh = SCRIPT.read_text(encoding="utf-8")
    ps = PS1.read_text(encoding="utf-8")
    assert not re.search(r"(?m)^\s*read\b", sh)
    assert "Read-Host" not in ps
    for name in ("version", "selftest", "dry-run", "smoke-narrate", "vram", "dialogue", "yo", "vxbook", "mp3-labels"):
        assert name in sh and name in ps
    assert "nvidia-smi" in ps and ".done" in ps and "USERPROFILE" in ps
    assert "vm_vxbook.py" in sh and "vm_vxbook.py" in ps
    assert "/workspace/vm-results/audiobook-builder/" in sh


def test_smoke_json_then_resume_skips(tmp_path):
    layout = _layout(tmp_path)
    proc = _run(layout, "--smoke", "--json", "--tag", "v1.0.0-rc.3",
                "--results", str(tmp_path / "out"))
    assert proc.returncode == 0, proc.stderr
    rows = _objects(proc)
    assert rows[-1]["ok"] is True and rows[-1]["exit"] == 0 and rows[-1]["mode"] == "smoke"
    assert rows[-1]["peak_vram_mib"] == 2048
    ids = [row["id"] for row in rows if row.get("type") == "step"]
    assert ids == ["version", "selftest", "dry-run", "smoke-narrate", "vram"]
    assert all(row["status"] == "pass" for row in rows if row.get("type") == "step")
    done = layout["home"] / ".done"
    for step in ids:
        assert (done / step).is_file()
    calls = layout["log"].read_text(encoding="utf-8")
    again = _run(layout, "--smoke", "--json", "--tag", "v1.0.0-rc.3", "--results", str(tmp_path / "out"))
    assert again.returncode == 0, again.stderr
    skipped = [row for row in _objects(again) if row.get("type") == "step"]
    assert [row["status"] for row in skipped] == ["skipped"] * 5
    assert layout["log"].read_text(encoding="utf-8") == calls
    saved = json.loads((tmp_path / "out" / "result.json").read_text(encoding="utf-8"))
    assert saved["tag"] == "v1.0.0-rc.3" and saved["peak_vram_mib"] == 2048


def test_full_writes_a_soundscape_book_yo_and_labelled_mp3s(tmp_path):
    layout = _layout(tmp_path)
    results = tmp_path / "full"
    proc = _run(layout, "--full", "--json", "--results", str(results))
    assert proc.returncode == 0, proc.stderr
    final = _objects(proc)[-1]
    assert final["ok"] is True and final["mode"] == "full" and final["tag"] == "v1.0.0-rc.3"
    book = load_vxbook(results / "work" / "story.vxbook")
    assert book.extensions == ("sound/1",) and book.sound_document["extension"] == "sound/1"
    assert "\u0451" in (results / "work" / "yo-only.txt").read_text(encoding="utf-8")
    marks = (results / "work" / "marks.txt").read_text(encoding="utf-8")
    assert "MALE" in marks and "FEMALE" in marks
    names = sorted(p.name for p in (results / "work" / "vxbook-out").glob("*.mp3"))
    assert names == ["01 - Утро.mp3", "02 - Вечер.mp3"]
    assert (layout["home"] / ".done" / "mp3-labels").is_file()


def test_narration_failure_is_exit_1_and_leaves_no_marker(tmp_path):
    layout = _layout(tmp_path)
    proc = _run(layout, "--smoke", "--json", "--tag", "vtest", "--results", str(tmp_path / "out"),
                env_extra={"FAKE_NARRATE_RC": "1"})
    assert proc.returncode == 1
    final = _objects(proc)[-1]
    assert final["ok"] is False and final["exit"] == 1
    assert not (layout["home"] / ".done" / "smoke-narrate").is_file()
    assert (layout["home"] / ".done" / "dry-run").is_file()


def test_missing_gpu_is_exit_3_and_missing_program_is_exit_2(tmp_path):
    layout = _layout(tmp_path)
    (tmp_path / "bin" / "nvidia-smi").unlink()
    proc = _run(layout, "--smoke", "--json", "--tag", "vtest", "--results", str(tmp_path / "out"))
    assert proc.returncode == 3
    assert _objects(proc)[-1]["exit"] == 3
    assert not (layout["home"] / ".done").exists() or not any((layout["home"] / ".done").iterdir())
    bad = _run(layout, "--bin", str(tmp_path / "missing-bin"), "--smoke")
    assert bad.returncode == 2
    bare = _run(layout)
    assert bare.returncode == 2
