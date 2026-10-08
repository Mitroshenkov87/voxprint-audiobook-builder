"""Logging set-up, exception hooks, settings snapshot and the diagnostic report (infra/diagnostics.py, Settings, CLI diag)."""
from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import threading
import zipfile

import pytest

from infra import diagnostics as dg
from tests.test_studio import app, lib, make_studio  # noqa: F401


@pytest.fixture
def isolated_logging():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    hooks = (sys.excepthook, threading.excepthook)
    yield
    for h in root.handlers[:]:
        if h not in handlers:
            root.removeHandler(h)
            h.close()
    root.setLevel(level)
    sys.excepthook, threading.excepthook = hooks
    logging.captureWarnings(False)


def test_rotating_log_capped_at_30_mb_and_thread_crashes_logged(tmp_path, isolated_logging):
    path = dg.setup_logging(tmp_path)
    h = [x for x in logging.getLogger().handlers if isinstance(x, logging.handlers.RotatingFileHandler)][-1]
    assert path == tmp_path / "voxprint.log" and h.maxBytes * (h.backupCount + 1) == 30_000_000

    def boom():
        raise ValueError("worker exploded")
    t = threading.Thread(target=boom, name="worker-x")
    t.start()
    t.join()
    for x in logging.getLogger().handlers:
        x.flush()
    text = path.read_text(encoding="utf-8")
    assert "uncaught exception in thread worker-x" in text and "ValueError: worker exploded" in text
    assert "Traceback" in text


def test_settings_snapshot_drops_secrets_and_cuts_paths(tmp_path):
    (tmp_path / "preload.json").write_text(json.dumps({"enabled": True, "folder": "C:\\Users\\anna\\Books\\x.epub",
                                                       "hf_token": "abc", "url": "https://example.org/a/b"}), "utf-8")
    (tmp_path / "models_dir.txt").write_text("/home/anna/models\n", "utf-8")
    (tmp_path / "api_token.txt").write_text("secret", "utf-8")
    snap = dg.settings_snapshot(tmp_path)
    assert snap["preload.json"] == {"enabled": True, "folder": "x.epub", "url": "https://example.org/a/b"}
    assert snap["models_dir.txt"] == "models" and "api_token.txt" not in snap


def test_report_zips_logs_info_and_settings(tmp_path):
    logs, state = tmp_path / "logs", tmp_path / "state"
    logs.mkdir(), state.mkdir()
    (logs / "voxprint.log").write_text("hello\n", "utf-8")
    (logs / "voxprint.log.1").write_text("older\n", "utf-8")
    (state / "language").write_text("ru", "utf-8")
    out = dg.write_report(tmp_path / "report", logs, state, info=lambda: {"app_version": "x"})
    assert out.name == "report.zip"
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == ["logs/voxprint.log", "logs/voxprint.log.1", "settings.json", "system_info.json"]
        assert json.loads(z.read("settings.json")) == {"language": "ru"}


def test_system_info_without_torch_import():
    info = dg.system_info(import_torch=False)
    assert info["python"] and info["cpu_logical"] and "cuda_available" in info["gpu"]


def test_cli_diag_and_settings_button(tmp_path, monkeypatch, app, lib):
    import cli

    made = []
    monkeypatch.setattr(dg, "write_report", lambda p, **kw: made.append(p) or p)
    assert cli.is_user_cli(["voxprint", "diag"])
    assert cli.main(["diag", "--out", str(tmp_path / "d.zip")]) == 0 and made == [tmp_path / "d.zip"]
    s = make_studio(lib)
    d = s.settings_dialog()
    notes = []
    d.notify = lambda title, text: notes.append(text)
    d.pick_save = lambda title, name: str(tmp_path / name)
    assert d.save_diagnostics().name.startswith("voxprint-diagnostics-") and notes
    d.pick_save = lambda title, name: ""
    assert d.save_diagnostics() is None


def test_report_lists_models_vulkan_and_a_cuda_summary(tmp_path):
    models = tmp_path / "models"
    (models / "Qwen--Qwen3-TTS-12Hz-1.7B-Base").mkdir(parents=True)
    (models / "Qwen--Qwen3-TTS-12Hz-1.7B-Base" / "model.safetensors").write_bytes(b"x" * 2048)
    (models / "Qwen--Qwen3-ASR-1.7B.partial").mkdir()
    ms = dg.installed_models(models)
    assert [m["name"] for m in ms] == ["Qwen--Qwen3-ASR-1.7B.partial", "Qwen--Qwen3-TTS-12Hz-1.7B-Base"] and ms[0]["partial"]
    info = {"app_version": "0.1.1", "os": "Windows-11", "python": "3.11.9", "models": ms,
            "gpu": {"cuda_available": True, "torch": "2.11.0+cu128", "cuda_build": "12.8", "gpu": "RTX 4090", "vram_total_gb": 16.0},
            "vulkan": {"loader": True, "devices": ["Vulkan0: NVIDIA GeForce RTX 4090"]}}
    text = "\n".join(dg.summary_lines(info))
    assert "CUDA available: yes" in text and "RTX 4090" in text and "Vulkan0: NVIDIA" in text and "loader present" in text
    assert "Models (2)" in text and "(partial)" in text
    assert dg.vulkan_info().get("llama_server") == "not installed"


def test_known_noise_is_not_logged():
    assert any("QFont::setPointSize: Point size <= 0 (-1), must be greater than 0".startswith(m) for m in dg.QT_NOISE)
    assert any("QThreadStorage: entry 1 destroyed before end of thread 0x1".startswith(m) for m in dg.QT_NOISE)
    import re
    assert any(re.match(p, "Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work") for p in dg.WARNING_NOISE)
