"""OpenVoice V2 pins and the voice-conversion contract. Fake bytes only: nothing is downloaded or loaded."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core import voice_convert
from core.vc_openvoice import OpenVoiceConverter
from infra import model_mirrors, model_release, portable, vc_model
from tools.make_model_release import plan
from workers.pipeline_runner import required_model_repos

REPO = "myshell-ai/OpenVoiceV2"
REV = "f36e7edfe1684461a8343844af60babc2efbb727"


def test_converter_contract_keeps_duration_and_the_default_model_is_not_loaded(tmp_path):
    src = np.linspace(-0.4, 0.4, 1600, dtype=np.float32)
    ref = np.linspace(0.2, -0.2, 800, dtype=np.float32)
    sf.write(str(tmp_path / "s.wav"), src, 16000)
    sf.write(str(tmp_path / "r.wav"), ref, 16000)
    fake = voice_convert.FakeVoiceConverter()
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"opus")
        return type("R", (), {"returncode": 0})()

    out = voice_convert.convert_file(tmp_path / "s.wav", tmp_path / "r.wav", tmp_path / "out", fake, ffmpeg="ffmpeg", run=run)
    assert out.name.endswith(".opus") and out.read_bytes() == b"opus" and not (tmp_path / "out" / "s re-voiced.flac").exists()
    assert fake.calls == [(1600, 16000, 800, 16000)] and "libopus" in calls[0]
    assert voice_convert.fit_length(np.array([0.0, 1.0], np.float32), 4).shape == (4,)
    assert OpenVoiceConverter().key == "openvoice-v2" and voice_convert.make_converter().key == "openvoice-v2"
    assert voice_convert.make_converter(lambda: voice_convert.FakeVoiceConverter()).key == "fake"


def test_openvoice_is_pinned_optional_and_not_in_the_first_run_download():
    e = model_mirrors.load()[REPO]
    assert e.license == "MIT" and e.source_revision == REV and not e.has_mirror
    ckpt, cfg = e.files["converter/checkpoint.pth"], e.files["converter/config.json"]
    assert ckpt["size"] == 131320490 and ckpt["sha256"].startswith("9652c27e")
    assert cfg["size"] == 838 and vc_model.download_bytes() == 131321328 and vc_model.download_mb() == 131
    assert REPO not in model_release.load()
    assert REPO not in required_model_repos()
    assert REPO not in portable.models_for("all", 0)
    assert REPO not in plan(model_mirrors.load())


def test_ensure_verifies_the_hash_with_a_fake_download(tmp_path):
    payload, cfg = b"not-a-real-checkpoint", b'{"_version_":"v2"}'
    files = {"converter/config.json": cfg, "converter/checkpoint.pth": payload}
    rev = "a" * 40
    manifest = tmp_path / "mirrors.json"
    body = {"schema": 1, "models": {REPO: {
        "license": "MIT", "source_repo": REPO, "source_revision": rev,
        "files": {n: {"size": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in files.items()},
    }}}
    manifest.write_text(json.dumps(body), encoding="utf-8")
    urls = []

    class Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout):
        urls.append(req.full_url)
        name = "converter/config.json" if req.full_url.endswith("config.json") else "converter/checkpoint.pth"
        return Resp(files[name])

    folder = vc_model.ensure(models_dir=tmp_path / "models", opener=opener, manifest=manifest)
    assert (folder / "converter" / "checkpoint.pth").read_bytes() == payload
    assert vc_model.ready(tmp_path / "models", manifest)
    assert urls == [f"https://huggingface.co/{REPO}/resolve/{rev}/{n}" for n in vc_model.FILES]
    urls.clear()
    assert vc_model.ensure(models_dir=tmp_path / "models", opener=opener, manifest=manifest) == folder and urls == []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["models"][REPO]["files"]["converter/checkpoint.pth"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(data), encoding="utf-8")
    (folder / "converter" / "checkpoint.pth").unlink()
    with pytest.raises(model_release.ReleaseError):
        vc_model.ensure(models_dir=tmp_path / "models", opener=opener, manifest=manifest)
    assert not vc_model.ready(tmp_path / "models", manifest)
