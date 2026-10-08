"""Optional noise clean-up (core/denoise.py, infra/denoise_tool.py) with a fake deep-filter program and a fake DNSMOS:
suggested only for noisy input, never ticked by itself, never downloaded silently, the original file is never touched."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core import denoise
from infra import denoise_tool, model_release
from tests.test_ui import app  # noqa: F401

SR = 48000


def _wav(path: Path, seconds=3.0, sr=SR):
    x = (0.1 * np.sin(np.linspace(0, 2000, int(seconds * sr)))).astype(np.float32)
    sf.write(str(path), x, sr)
    return path, x


def fake_tool(drop=1440, gain=0.5, code=0):
    """Behaves like deep-filter: reads the input WAV, writes <out_dir>/<name> a bit shorter and quieter."""
    calls = []

    def run(cmd):
        calls.append(cmd)
        if code:
            return code
        out_dir, inp = Path(cmd[cmd.index("-o") + 1]), Path(cmd[-1])
        x, sr = sf.read(str(inp), dtype="float32")
        assert sr == SR
        out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_dir / inp.name), x[:-drop] * gain, sr, subtype="PCM_16")
        return 0
    run.calls = calls
    return run


class FakeMos:
    def __init__(self, bak):
        self.bak, self.calls = bak, 0

    def score(self, audio, sr):
        self.calls += 1
        return {"ovrl": 3.0, "sig": 3.5, "bak": self.bak}


def test_suggested_only_below_the_background_threshold(tmp_path):
    f, _ = _wav(tmp_path / "a.wav", 35.0, 16000)
    m = FakeMos(2.4)
    assert denoise.background_score([f], m) == 2.4 and m.calls == 3          # 35 s -> three 10 s excerpts
    assert denoise.should_suggest(2.4) and not denoise.should_suggest(3.6) and not denoise.should_suggest(None)
    assert denoise.background_score([tmp_path / "missing.wav"], m) is None


def test_many_clips_are_sampled_not_all(tmp_path):
    files = [_wav(tmp_path / f"c{i}.wav", 2.0, 16000)[0] for i in range(20)]
    m = FakeMos(3.8)
    assert denoise.background_score(files, m) == 3.8 and m.calls == denoise.EXCERPTS


def test_denoise_file_keeps_length_and_leaves_the_original(tmp_path):
    src, x = _wav(tmp_path / "rec.wav")
    before = src.read_bytes()
    run = fake_tool()
    out = denoise.denoise_file(src, tmp_path / "job" / "clean.wav", Path("deep-filter"), run=run)
    y, sr = sf.read(str(out), dtype="float32")
    assert sr == SR and len(y) == len(x) and src.read_bytes() == before
    cmd = run.calls[0]
    assert cmd[0] == "deep-filter" and "-D" in cmd and cmd[cmd.index("-a") + 1] == str(denoise.ATTEN_LIMIT_DB)
    with pytest.raises(RuntimeError):
        denoise.denoise_file(src, tmp_path / "x.wav", Path("deep-filter"), run=fake_tool(code=2))


def test_tool_is_pinned_per_platform_and_checked_on_download(tmp_path):
    assert denoise_tool.platform_key("win32", "AMD64") == "win-x64"
    assert denoise_tool.platform_key("linux", "x86_64") == "linux-x64"
    assert denoise_tool.platform_key("darwin", "arm64") is None and denoise_tool.platform_key("linux", "aarch64") is None
    assert denoise_tool.ASSETS["win-x64"]["size"] == 26912256
    assert all(len(a["sha256"]) == 64 for a in denoise_tool.ASSETS.values())
    assert denoise_tool.ready(tmp_path, "win-x64") is None
    payload = b"x" * 100
    key = "test-x64"
    denoise_tool.ASSETS[key] = {"name": "deep-filter-test", "size": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest()}
    try:
        urls = []

        class Resp(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def opener(req, timeout):
            urls.append(req.full_url)
            return Resp(payload)

        p = denoise_tool.ensure(models_dir=tmp_path, opener=opener, key=key)
        assert p.read_bytes() == payload and denoise_tool.ready(tmp_path, key) == p
        assert urls == [f"https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/deep-filter-test"]
        denoise_tool.ASSETS[key]["sha256"] = "0" * 64
        p.unlink()
        with pytest.raises(model_release.ReleaseError):
            denoise_tool.ensure(models_dir=tmp_path, opener=opener, key=key)
        assert denoise_tool.ready(tmp_path, key) is None
    finally:
        del denoise_tool.ASSETS[key]


def test_runner_cleans_only_on_request_and_never_downloads(tmp_path, monkeypatch):
    from core.events import CancelToken
    from workers import pipeline_runner as pr

    src, x = _wav(tmp_path / "rec.wav")
    off = pr.TaskRequest(kind=pr.KIND_LORA, audio=src)
    assert pr._denoise_inputs(off, tmp_path, lambda *a: None, CancelToken()) == (off, "")
    on = pr.TaskRequest(kind=pr.KIND_LORA, audio=src, denoise=True)
    monkeypatch.setattr(denoise_tool, "ready", lambda *a, **k: None)
    monkeypatch.setattr(denoise_tool, "ensure", lambda *a, **k: pytest.fail("no silent download"))
    req, warn = pr._denoise_inputs(on, tmp_path, lambda *a: None, CancelToken())
    assert req.audio == src and warn                                         # missing program: original + warning
    req, warn = pr._denoise_inputs(on, tmp_path, lambda *a: None, CancelToken(), tool=Path("df"), run=fake_tool())
    assert req.audio.parent == tmp_path / "denoised" and req.audio.is_file() and warn == ""
    req, warn = pr._denoise_inputs(on, tmp_path, lambda *a: None, CancelToken(), tool=Path("df"), run=fake_tool(code=1))
    assert req.audio == src and "1" in warn                                  # failure: train on the original


def test_runner_expands_clip_folders(tmp_path):
    from core.events import CancelToken
    from workers import pipeline_runner as pr

    d = tmp_path / "clips"
    d.mkdir()
    for i in range(3):
        _wav(d / f"{i}.wav", 1.0)
    req = pr.TaskRequest(kind=pr.KIND_LORA, no_transcript=True, audio_files=[d], denoise=True)
    out, warn = pr._denoise_inputs(req, tmp_path / "job", lambda *a: None, CancelToken(), tool=Path("df"), run=fake_tool(drop=10))
    assert len(out.audio_files) == 3 and all(p.parent == tmp_path / "job" / "denoised" for p in out.audio_files)


def test_train_window_offers_the_cleanup_only_for_noisy_input(app, tmp_path, monkeypatch):
    from tests.test_ui import make_window

    monkeypatch.setattr(denoise_tool, "ready", lambda *a, **k: None)
    w = make_window(lambda *a: None)
    assert w.denoise_row.isHidden() and w._task_extras()["denoise"] is False
    w._apply_noise_score(3.9)                                                # clean: nothing shown
    assert w.denoise_row.isHidden()
    w._apply_noise_score(2.2)
    assert not w.denoise_row.isHidden() and "2.2" in w.lbl_denoise.text()
    assert not w.chk_denoise.isEnabled() and w.btn_denoise_dl.isVisibleTo(w)  # not downloaded: only the button
    assert not w.chk_denoise.isChecked() and w._task_extras()["denoise"] is False
    monkeypatch.setattr(denoise_tool, "ready", lambda *a, **k: Path("df"))   # "downloaded"
    w._on_denoise_downloaded("")
    assert w.chk_denoise.isEnabled() and w.chk_denoise.isChecked() and not w.btn_denoise_dl.isVisibleTo(w)
    assert w._task_extras()["denoise"] is True
    w._apply_noise_score(None)                                               # another recording without a score
    assert w.denoise_row.isHidden() and not w.chk_denoise.isChecked() and w._task_extras()["denoise"] is False
    w.close()


def test_train_window_scores_new_recordings_in_the_background(app, tmp_path, monkeypatch):
    from tests.test_studio import wait_for
    from tests.test_ui import make_window

    monkeypatch.setattr(denoise_tool, "platform_key", lambda *a: "win-x64")
    w = make_window(lambda *a: None)
    seen = []
    w.noise_scorer = lambda files: seen.append(files) or 2.0
    src, _ = _wav(tmp_path / "r.wav", 1.0)
    w.set_audio(src)
    wait_for(lambda: not w.denoise_row.isHidden())
    assert seen == [[src]] and w.noise_bak == 2.0
    w.close()
