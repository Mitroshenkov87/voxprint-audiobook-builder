"""OpenVoice V2 pins and the voice-conversion contract. Fake bytes only: nothing is downloaded or loaded."""
from pathlib import Path

import numpy as np
import soundfile as sf

from core import voice_convert
from core.vc_openvoice import OpenVoiceConverter
from infra import mdl_models, vc_model
from workers.pipeline_runner import required_model_repos

REPO = "myshell-ai/OpenVoiceV2"


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
    from voxprint_common.mdl.catalog import load_manifest

    from infra.mdl_gate import manifest_path

    row = next(m for m in load_manifest(manifest_path()).models if m.id == "openvoice-v2")
    assert row.on_demand
    files = {f.path: f for f in row.files}
    ckpt = files["myshell-ai--OpenVoiceV2/converter/checkpoint.pth"]
    cfg = files["myshell-ai--OpenVoiceV2/converter/config.json"]
    assert ckpt.size == 131320490 and ckpt.sha256.startswith("9652c27e")
    assert cfg.size == 838 and vc_model.download_bytes() == 131321328 and vc_model.download_mb() == 131
    assert REPO not in required_model_repos()


def test_ensure_with_an_explicit_root_does_not_fetch(tmp_path):
    """Tests pass a models root with plain files: ensure returns the folder, ready follows the files."""
    folder = vc_model.ensure(models_dir=tmp_path / "models")
    assert folder == tmp_path / "models" / "myshell-ai--OpenVoiceV2"
    assert not vc_model.ready(tmp_path / "models")
    conv = folder / "converter"
    conv.mkdir(parents=True)
    (conv / "config.json").write_bytes(b"{}")
    (conv / "checkpoint.pth").write_bytes(b"payload")
    assert vc_model.ready(tmp_path / "models")
    assert vc_model.ensure(models_dir=tmp_path / "models") == folder


def test_ensure_through_mdl_returns_the_view_folder(monkeypatch):
    from tests.mdl_harness import build_store

    build_store("openvoice-v2")
    folder = vc_model.ensure()
    assert folder == mdl_models.view_dir("openvoice-v2") / "myshell-ai--OpenVoiceV2"
    assert vc_model.ready()

