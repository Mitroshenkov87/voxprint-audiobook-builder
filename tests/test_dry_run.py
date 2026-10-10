"""Each suite command's --dry-run runs in a subprocess with no GPU and no models."""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from core.dry_run import schema_error

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sound-sample.vxbook"
BOOK = (
    "The man and the woman walked to the house in the morning and he said that she was the one.\n\n"
    "They were not there when we arrived but the story was all we had from the start and it was good. "
    "She had a book and he had a pen and they would read it for the people who were there with them.\n"
)
REFUSAL = "RTX 40-series"


def _env(home: Path) -> dict:
    env = os.environ.copy()
    env.pop("VOXPRINT_ALLOW_NO_GPU", None)
    env.update({
        "VOXPRINT_HOME": str(home),
        "VOXPRINT_LANG": "en",
        "VOXPRINT_NO_ENV_PROBE": "1",
        "VOXPRINT_NO_EXTERNAL_MODELS": "1",
        "VOXPRINT_NO_MIRROR": "1",
        "VOXPRINT_VOICES_INDEX": "offline://dry-run-test",
        "QT_QPA_PLATFORM": "offscreen",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONUTF8": "1",
    })
    return env


def _run(home: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "main.py"), *args],
        cwd=ROOT, env=_env(home), capture_output=True, text=True, timeout=180,
    )


def _payload(proc: subprocess.CompletedProcess) -> dict:
    assert proc.returncode != 7, proc.stderr
    assert REFUSAL not in proc.stderr, proc.stderr
    found = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "result":
            found = obj
    assert found is not None, proc.stdout + proc.stderr
    assert schema_error(found) == "", found
    assert found["dry_run"] is True and found["outputs"] == []
    assert found["exit_code"] == proc.returncode
    return found


@pytest.fixture
def book(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text(BOOK, encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    return path, out


def test_each_dry_run_command_prints_json_without_a_gpu(tmp_path, book):
    path, out = book
    home = tmp_path / "home"
    home.mkdir()
    cases = [
        (["--dry-run", "--json", "narrate", str(path), "--voice", "narrator", "--out", str(out)],
         ("input", "output", "voice", "formats"), {"voice": "narrator"}),
        (["--dry-run", "--json", "narrate", str(FIXTURE), "--voice", "narrator", "--out", str(out)],
         ("input", "output", "voice", "formats"), None),
        (["narrate", str(path), "--voice", "narrator", "--out", str(out), "--dry-run", "--json"],
         ("input", "output", "voice", "formats"), None),
        (["--dry-run", "--json", "speakers", str(path), "--out", str(out / "marks.txt")],
         ("input", "paragraphs", "planned_output"), None),
        (["--dry-run", "--json", "prepare", str(path), "--out", str(out / "prepared.txt")],
         ("input", "planned_output", "language", "steps"), {"language": "en"}),
        (["--dry-run", "--json", "translate", str(path), "--to", "ru", "--out", str(out / "book.ru.txt")],
         ("input", "source", "target", "planned_output"), {"source": "en", "target": "ru"}),
        (["--dry-run", "--json", "check"],
         ("items", "checked", "fixed", "failed"), {"command": "check", "items": []}),
        (["--dry-run", "--json", "repair"],
         ("items", "checked", "fixed", "failed"), {"command": "check", "items": []}),
        (["--dry-run", "--json", "status"],
         ("version", "build", "codename", "platform", "gpu", "voices", "formats", "modules", "runtime"), None),
        (["--dry-run", "capabilities"],
         ("version", "build", "codename", "platform", "gpu", "voices", "formats", "modules", "runtime"),
         {"command": "capabilities"}),
        (["--dry-run", "--json", "voices", "catalog"],
         ("voices",), {"voices": []}),
        (["--dry-run", "--json", "settings", "get", "narration.ordinals"],
         ("key", "value"), {"key": "narration.ordinals", "value": True}),
    ]
    for args, extra, expect in cases:
        proc = _run(home, args)
        payload = _payload(proc)
        assert proc.returncode == 0, (args, proc.stderr, payload)
        assert schema_error(payload, extra) == "", payload
        if expect:
            for key, value in expect.items():
                assert payload[key] == value, (args, payload)
        assert not (out / "marks.txt").exists()
        assert not (out / "prepared.txt").exists()
        assert not (out / "book.ru.txt").exists()
    status = _run(home, ["--dry-run", "--json", "status"])
    body = _payload(status)
    assert body["gpu"]["requirement"]["source"] == "dry-run"
    assert body["gpu"]["cuda_available"] is False


def test_dry_run_exit_codes_for_bad_input_and_settings_do_not_persist(tmp_path, book):
    path, out = book
    home = tmp_path / "home"
    home.mkdir()
    missing = tmp_path / "missing.txt"
    bad = tmp_path / "bad.vxbook"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("readme.txt", "not a book")
    nowhere = tmp_path / "no-such-dir" / "audiobooks"
    failures = [
        (["--dry-run", "--json", "narrate", str(missing), "--voice", "narrator", "--out", str(out)], 3),
        (["--dry-run", "--json", "narrate", str(bad), "--voice", "narrator", "--out", str(out)], 3),
        (["--dry-run", "--json", "narrate", str(path), "--voice", "narrator", "--out", str(nowhere)], 3),
        (["--dry-run", "--json", "narrate", str(path), "--voice", "narrator", "--out", str(out),
          "--format", "nope"], 2),
        (["--dry-run", "--json", "prepare", str(path), "--out", str(out / "p.txt"), "--steps", "bogus"], 2),
        (["--dry-run", "--json", "translate", str(path), "--to", "en", "--out", str(out / "same.txt")], 2),
        (["--dry-run", "--json", "settings", "get", "not-a-setting"], 2),
        (["--dry-run", "--json", "settings", "set", "narration.speed", "9"], 2),
    ]
    for args, code in failures:
        proc = _run(home, args)
        payload = _payload(proc)
        assert proc.returncode == code, (args, proc.stderr, payload)
        assert payload["ok"] is False and payload["error"]

    ordinals = home / "state" / "narration_ordinals.json"
    sound = home / "state" / "soundscape.json"
    assert not ordinals.exists()
    proc = _run(home, ["--dry-run", "--json", "settings", "set", "narration.ordinals", "off"])
    payload = _payload(proc)
    assert proc.returncode == 0 and payload["value"] is False
    assert not ordinals.exists()
    proc = _run(home, ["--dry-run", "--json", "settings", "set", "narration.soundscape", "on"])
    payload = _payload(proc)
    assert proc.returncode == 0 and payload["value"] is True
    assert not sound.exists()
    assert not list(home.rglob("ace-step*"))
    again = _payload(_run(home, ["--dry-run", "--json", "settings", "get", "narration.ordinals"]))
    assert again["value"] is True
