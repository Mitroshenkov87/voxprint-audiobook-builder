"""RTX 40-series gate: compute capability 8.9 or newer, at startup and in the installers.

``VOXPRINT_ALLOW_NO_GPU=1`` is internal (CI and this suite). It is not a user option.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ISS = ROOT / "installer" / "Voxprint.iss"
LINUX = ROOT / "installer" / "linux" / "install-voxprint-linux.sh"


class _Cuda:
    def __init__(self, devices):
        self._devices = list(devices)

    def is_available(self):
        return bool(self._devices)

    def device_count(self):
        return len(self._devices)

    def get_device_capability(self, index):
        return self._devices[index][1]

    def get_device_name(self, index):
        return self._devices[index][0]


class _Torch:
    def __init__(self, devices):
        self.cuda = _Cuda(devices)


def _no_torch():
    return _Torch([])


def _smi(*rows: str):
    text = "\n".join(rows)

    def run(args):
        assert "--query-gpu=name,compute_cap,driver_version" in args
        return 0, text

    return run


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.delenv("VOXPRINT_ALLOW_NO_GPU", raising=False)
    monkeypatch.setenv("VOXPRINT_LANG", "en")
    from infra import gpu_requirement as gr
    from core import i18n

    i18n.reset()
    return gr


def test_no_gpu_is_refused(gate):
    report = gate.check_gpu(torch_module=_no_torch(), run=_smi(), which=lambda name: None)
    assert report.ok is False and report.detected == "none" and report.best is None
    assert report.source == "none"
    message = gate.requirement_message(gate.detected_label(report))
    assert message == "Voxprint needs an NVIDIA GeForce RTX 40-series or newer GPU. Detected: none."


def test_rtx_3090_compute_8_6_is_refused(gate):
    report = gate.check_gpu(
        torch_module=_no_torch(),
        run=_smi("NVIDIA GeForce RTX 3090, 8.6, 550.54"),
        which=lambda name: "/usr/bin/nvidia-smi",
    )
    assert report.ok is False and report.detected == "NVIDIA GeForce RTX 3090"
    assert report.best is not None and (report.best.major, report.best.minor) == (8, 6)
    assert "RTX 3090" in gate.requirement_message(gate.detected_label(report))


def test_rtx_4090_compute_8_9_and_rtx_5090_compute_12_are_accepted(gate):
    ada = gate.check_gpu(
        torch_module=_Torch([("NVIDIA GeForce RTX 4090", (8, 9))]),
        run=lambda args: (_ for _ in ()).throw(AssertionError("nvidia-smi must not run when torch.cuda works")),
        which=lambda name: "/usr/bin/nvidia-smi",
    )
    assert ada.ok and ada.source == "torch" and ada.detected == "NVIDIA GeForce RTX 4090"
    assert ada.to_dict()["compute_cap"] == "8.9" and ada.to_dict()["min_compute"] == "8.9"

    blackwell = gate.check_gpu(
        torch_module=_no_torch(),
        run=_smi("NVIDIA GeForce RTX 5090, 12.0, 570.1"),
        which=lambda name: "nvidia-smi",
    )
    assert blackwell.ok and blackwell.source == "nvidia-smi"
    assert (blackwell.best.major, blackwell.best.minor) == (12, 0)


def test_nvidia_smi_missing_is_refused(gate):
    def boom(args):
        raise AssertionError("nvidia-smi is not installed")

    report = gate.check_gpu(torch_module=_no_torch(), run=boom, which=lambda name: None)
    assert report.ok is False and report.detected == "none" and report.source == "none"


def test_best_gpu_wins_when_one_card_is_old(gate):
    report = gate.check_gpu(
        torch_module=_no_torch(),
        run=_smi("NVIDIA GeForce RTX 3090, 8.6, 550.1", "NVIDIA GeForce RTX 4090, 8.9, 560.2"),
        which=lambda name: "nvidia-smi",
    )
    assert report.ok and report.detected == "NVIDIA GeForce RTX 4090"


def test_cpu_torch_falls_back_to_nvidia_smi(gate):
    report = gate.check_gpu(
        torch_module=_no_torch(),
        run=_smi("NVIDIA GeForce RTX 4090, 8.9, 560.2"),
        which=lambda name: "nvidia-smi",
    )
    assert report.ok and report.source == "nvidia-smi"


def test_override_env_is_honoured_only_as_a_flag(gate, monkeypatch, capsys):
    monkeypatch.setenv("VOXPRINT_ALLOW_NO_GPU", "1")
    report = gate.check_gpu(torch_module=_no_torch(), which=lambda name: None)
    assert report.ok is False and report.overridden is True
    assert gate.enforce(gui=False, torch_module=_no_torch(), which=lambda name: None) == 0
    assert capsys.readouterr().err == ""

    monkeypatch.setenv("VOXPRINT_ALLOW_NO_GPU", "0")
    shown = []
    code = gate.enforce(gui=True, dialog=shown.append, torch_module=_no_torch(), which=lambda name: None)
    assert code == gate.EXIT_GPU_REQUIRED == 7
    assert shown == ["Voxprint needs an NVIDIA GeForce RTX 40-series or newer GPU. Detected: none."]
    err = capsys.readouterr().err
    code = gate.enforce(gui=False, torch_module=_no_torch(), which=lambda name: None)
    assert code == 7 and "Detected: none." in capsys.readouterr().err and err == ""


def test_lightweight_commands_skip_the_gate_and_the_gui_does_not(gate):
    assert gate.startup_requires_gpu(["voxprint"]) is True
    assert gate.startup_is_gui(["voxprint"]) is True
    assert gate.startup_requires_gpu(["voxprint", "narrate", "book.epub"]) is True
    assert gate.startup_is_gui(["voxprint", "narrate", "book.epub"]) is False
    for argv in (
        ["voxprint", "--version"],
        ["voxprint", "-V"],
        ["voxprint", "--help"],
        ["voxprint", "help"],
        ["voxprint", "narrate", "--help"],
        ["voxprint", "status"],
        ["voxprint", "--json", "status"],
        ["voxprint", "capabilities"],
        ["voxprint", "diag"],
        ["voxprint", "diagnostics"],
        ["voxprint", "--register-models-user"],
        ["voxprint", "--sync-suite-settings"],
        ["voxprint", "narrate", "book.epub", "--dry-run"],
        ["voxprint", "--dry-run", "narrate", "a.epub"],
    ):
        assert gate.startup_requires_gpu(argv) is False, argv
    monkey_env = os.environ.copy()
    monkey_env["VOXPRINT_ALLOW_NO_GPU"] = "1"
    assert gate.startup_requires_gpu(["voxprint", "narrate", "a.epub"], environ=monkey_env) is False


def test_startup_refuses_on_stderr_for_cli_and_dialog_for_gui(monkeypatch):
    monkeypatch.delenv("VOXPRINT_ALLOW_NO_GPU", raising=False)
    monkeypatch.setenv("VOXPRINT_LANG", "en")
    from infra import gpu_requirement as gr
    from core import i18n

    i18n.reset()
    shown = []
    monkeypatch.setattr(gr, "show_dialog", shown.append)
    import main as app_main

    assert app_main.main(["voxprint", "narrate", "book.epub"]) == 7
    import sys
    from io import StringIO

    buf = StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    assert app_main.main(["voxprint", "train", "clip.wav"]) == 7
    assert "RTX 40-series" in buf.getvalue() and "Detected:" in buf.getvalue()
    assert app_main.main(["voxprint"]) == 7
    assert shown and "RTX 40-series" in shown[-1]
    assert app_main.main(["voxprint", "--version"]) == 0
    assert app_main.main(["voxprint", "status"]) == 0


def test_status_reports_the_gpu_check(monkeypatch, capsys):
    monkeypatch.delenv("VOXPRINT_ALLOW_NO_GPU", raising=False)
    monkeypatch.setenv("VOXPRINT_LANG", "en")
    from infra import gpu_requirement as gr

    monkeypatch.setattr(
        gr, "check_gpu",
        lambda **_k: gr.GpuReport(False, (), "none", None, "none", False),
    )
    import main as app_main

    assert app_main.main(["voxprint", "status"]) == 0
    data = json.loads(capsys.readouterr().out)
    req = data["gpu"]["requirement"]
    assert req["ok"] is False and req["detected"] == "none" and req["min_compute"] == "8.9"
    assert req["override"] is False


def test_message_is_translated(monkeypatch):
    monkeypatch.setenv("VOXPRINT_LANG", "ru")
    from core import i18n
    from infra.gpu_requirement import requirement_message

    i18n.reset()
    text = requirement_message("NVIDIA GeForce RTX 3090")
    assert "RTX 40" in text and "RTX 3090" in text and "видеокарт" in text


def test_windows_installer_checks_the_gpu_and_hides_the_ci_switch():
    text = ISS.read_text(encoding="utf-8")
    assert "SKIPGPUCHECK" in text
    assert "nvidia-smi" in text and "compute_cap" in text
    assert "Win32_VideoController" in text
    body = text.split("function InitializeSetup")[1].split("\nfunction ")[0]
    assert "ParamSwitch('SKIPGPUCHECK')" in body or 'ParamSwitch("SKIPGPUCHECK")' in body
    assert "GpuRequired" in body and "Result := False" in body
    for lang, none in (("english", "none"), ("russian", "нет"), ("german", "keine")):
        line = next(x for x in text.splitlines() if x.startswith(f"{lang}.GpuRequired="))
        assert "40" in line and "%1" in line
        assert next(x for x in text.splitlines() if x.startswith(f"{lang}.GpuNone=")).endswith(none)


@pytest.mark.skipif(shutil.which("bash") is None or sys.platform == "win32", reason="bash needed (Windows bash is the WSL stub)")
def test_linux_installer_gpu_check_and_no_cpu_option(tmp_path):
    script = str(LINUX)
    help_run = subprocess.run(["bash", script, "--help"], capture_output=True, text=True)
    assert help_run.returncode == 0 and "--cpu" not in help_run.stdout
    unknown = subprocess.run(["bash", script, "--cpu"], capture_output=True, text=True)
    assert unknown.returncode != 0

    def smi(text: str) -> dict:
        bindir = tmp_path / text.split(",")[0][-4:]
        bindir.mkdir(parents=True, exist_ok=True)
        exe = bindir / "nvidia-smi"
        exe.write_text(f"#!/bin/sh\nprintf '%s\\n' '{text}'\n", encoding="utf-8")
        exe.chmod(0o755)
        env = os.environ.copy()
        env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
        return env

    old = subprocess.run(["bash", script, "--gpu-check-only"], capture_output=True, text=True,
                         env=smi("NVIDIA GeForce RTX 3090, 8.6, 550.54"))
    assert old.returncode != 0 and "RTX 3090" in old.stderr and "RTX 40" in old.stderr
    ada = subprocess.run(["bash", script, "--gpu-check-only"], capture_output=True, text=True,
                         env=smi("NVIDIA GeForce RTX 4090, 8.9, 560.70"))
    assert ada.returncode == 0
    new = subprocess.run(["bash", script, "--gpu-check-only"], capture_output=True, text=True,
                         env=smi("NVIDIA GeForce RTX 5090, 12.0, 570.10"))
    assert new.returncode == 0
    empty = tmp_path / "nosmi"
    empty.mkdir()
    env = os.environ.copy()
    env["PATH"] = str(empty) + os.pathsep + env.get("PATH", "")
    missing = subprocess.run(["bash", script, "--gpu-check-only"], capture_output=True, text=True, env=env)
    assert missing.returncode != 0 and "Detected: none." in missing.stderr
    skipped = subprocess.run(["bash", script, "--gpu-check-only", "--skip-gpu-check"], capture_output=True, text=True, env=env)
    assert skipped.returncode == 0
    text = LINUX.read_text(encoding="utf-8")
    assert "--torch-backend=cpu" not in text and "falling back to the CPU" not in text
