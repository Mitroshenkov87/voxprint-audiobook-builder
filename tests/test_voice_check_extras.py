"""Voice-check extras: CER, speaker similarity (the model's own encoder), DNSMOS MOS (scorer, pinned download, Components
"download all"), and their use in the quick preview and the post-training check."""
import hashlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

from core import voice_check as vc
from tests.test_preview_check import G, FakeEngine, _dataset, _train, tone


def test_cer_and_cosine():
    assert vc.cer("Привет, мир!", "привет мир") == 0.0
    assert abs(vc.cer("abcd", "abed") - 0.25) < 1e-9 and vc.cer("ab", "") == 1.0
    assert vc.cosine([1, 0], [1, 0]) == pytest.approx(1.0) and vc.cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert vc.cosine(None, [1]) is None and vc.cosine([0, 0], [1, 0]) is None and vc.cosine([1, 0], [1, 0, 0]) is None


def test_similarity_and_mos_only_warn():
    x, ref = tone(120, 8.0), tone(120, 6.0)
    ok = vc.check_sample(x, 24000, "a b c", ref, 24000, asr_text="a b c", sim=0.9, mos=3.4)
    assert ok.verdict == vc.GOOD and ok.as_dict()["sim"] == 0.9 and ok.as_dict()["mos"] == 3.4 and ok.as_dict()["cer"] == 0.0
    low = vc.check_sample(x, 24000, "a b c", ref, 24000, asr_text="a b c", sim=0.3, mos=1.5)
    assert low.verdict == vc.WARN and {"sim", "mos"} <= set(low.issues)
    none = vc.check_sample(x, 24000, "a b c", ref, 24000)
    assert none.as_dict()["sim"] is None and none.as_dict()["mos"] is None and none.as_dict()["cer"] is None


class FakeSession:
    """Stands in for the onnxruntime session: records the window shapes, returns fixed raw scores."""
    def __init__(self, raw=(3.0, 3.0, 3.0)):
        self.raw, self.shapes = raw, []

    def get_inputs(self):
        return [SimpleNamespace(name="input_1")]

    def run(self, _out, feeds):
        self.shapes.append(feeds["input_1"].shape)
        return [np.array([self.raw], dtype=np.float32)]


def test_dnsmos_windows_and_polynomial_mapping():
    from core import mos
    s = FakeSession()
    m = mos.DnsMos(session=s)
    r = m.score(tone(120, 3.0), 24000)                 # short: repeated to fill one 9.01 s window at 16 kHz
    assert s.shapes and set(s.shapes) == {(1, 144160)}
    assert r["ovrl"] == pytest.approx(round(float(np.poly1d([-0.06766283, 1.11546468, 0.04602535])(3.0)), 3))
    assert r["sig"] == pytest.approx(round(float(np.poly1d([-0.08397278, 1.22083953, 0.0052439])(3.0)), 3))
    s.shapes.clear()
    m.score(tone(120, 12.0, sr=16000), 16000)           # 12 s -> windows at 0, 1, 2 s (1 s hop)
    assert len(s.shapes) == 3
    s.shapes.clear()
    m.score(np.zeros(16000 * 200, dtype=np.float32) + 0.01, 16000)
    assert len(s.shapes) == mos.MAX_WINDOWS            # bounded cost for long audio
    assert m.score(np.zeros(1000, dtype=np.float32), 16000) is None


def test_default_mos_never_downloads(monkeypatch, tmp_path):
    from core import mos
    from infra import quality_models
    monkeypatch.setattr(quality_models, "ensure_dnsmos", lambda *a, **k: pytest.fail("must not download"))
    assert mos.default_mos() is None                   # not downloaded (isolated test home)
    monkeypatch.setattr(quality_models, "dnsmos_ready", lambda *a: True)
    monkeypatch.setenv("VOXPRINT_NO_MOS", "1")
    assert mos.default_mos() is None                   # switched off


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_dnsmos_download_is_pinned_and_falls_back(monkeypatch, tmp_path):
    from infra import model_release, quality_models as qm
    payload = b"onnx" * 1000
    monkeypatch.setattr(qm, "DNSMOS_META", {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    seen = []

    def opener(req, timeout):
        seen.append(req.full_url)
        if "github" in req.full_url:
            raise OSError("blocked")
        return _Resp(payload)
    assert qm.missing_bytes(tmp_path) == len(payload) and not qm.dnsmos_ready(tmp_path)
    msgs = []
    p = qm.ensure_dnsmos(lambda f, m="": msgs.append((f, m)), models_dir=tmp_path, opener=opener)
    assert p == tmp_path / "dnsmos" / "sig_bak_ovr.onnx" and p.read_bytes() == payload
    assert "github" in seen[0] and "huggingface.co" in seen[1] and qm.dnsmos_ready(tmp_path) and qm.missing_bytes(tmp_path) == 0
    assert msgs[-1][0] == 1.0
    qm.ensure_dnsmos(models_dir=tmp_path, opener=lambda *a: pytest.fail("already there"))
    # a wrong file from every source: nothing is kept, the error names the failures
    p.unlink()
    with pytest.raises(model_release.ReleaseError):
        qm.ensure_dnsmos(models_dir=tmp_path, opener=lambda req, t: _Resp(b"x" * len(payload)))
    assert not p.exists()
    # the pinned values are the real file's
    assert qm.__dict__["DNSMOS_URLS"][0].startswith("https://raw.githubusercontent.com/microsoft/DNS-Challenge/591184a9")


def test_components_download_all_includes_dnsmos(monkeypatch):
    pytest.importorskip("PySide6")
    from infra import quality_models, text_models
    from ui import modules_dialog
    calls = []
    monkeypatch.setattr(text_models, "missing_component_extras", lambda: [SimpleNamespace(size_mb=3)])
    monkeypatch.setattr(text_models, "ensure_component_extras", lambda p: (calls.append("sage"), p("s", 1.0, "sage"), ["sage-ru"])[2])
    monkeypatch.setattr(quality_models, "missing_bytes", lambda *a: 1 << 20)
    monkeypatch.setattr(quality_models, "ensure_dnsmos", lambda p: (calls.append("dnsmos"), p(1.0, "dnsmos")))
    assert modules_dialog.default_extras_bytes() == (3 << 20) + (1 << 20)
    prog = []
    assert modules_dialog.default_extras(lambda f, m="": prog.append(round(f, 2))) == ["sage-ru", "dnsmos"]
    assert calls == ["sage", "dnsmos"] and prog == [0.75, 1.0]


def test_first_run_prefetch_gets_dnsmos_best_effort(monkeypatch):
    from infra import quality_models
    from workers import pipeline_runner as prun
    monkeypatch.setattr(quality_models, "ensure_dnsmos", lambda p: (_ for _ in ()).throw(OSError("offline")))
    prun._prefetch_dnsmos(lambda *a: None)             # logged, never raised


# --------------------------------------------------------------------------- similarity / MOS in the checks
class EmbEngine(FakeEngine):
    """A fake with a "speaker encoder": the embedding is the pitch, so a pitch-shifted sample is less similar."""
    def speaker_embedding(self, audio, sr):
        f = vc.f0_median(audio, sr)
        return np.array([np.cos(f / 100.0), np.sin(f / 100.0)])


class FakeMos:
    def __init__(self, v):
        self.v, self.calls = v, 0

    def score(self, audio, sr):
        self.calls += 1
        return {"ovrl": self.v, "sig": self.v, "bak": self.v}


def test_preview_reports_similarity_and_mos(tmp_path):
    from core.asr import FakeASR
    from infra.vram_optimizer import plan_training
    from workers import preview_runner as pr
    m = FakeMos(3.3)
    items = pr.run_previews(_dataset(tmp_path), tmp_path / "out", plan_training(G, 70), compare=False, language="english",
                            gpu=G, train_fn=_train([]), engine_factory=lambda a, l: EmbEngine(), mos=m,
                            asr=FakeASR([pr.SAMPLE_TEXT["english"]]))
    chk = items[0].check
    assert chk["sim"] > 0.99 and chk["mos"] == 3.3 and chk["verdict"] == "good" and m.calls == 1
    items = pr.run_previews(_dataset(tmp_path / "x") if (tmp_path / "x").mkdir() is None else None, tmp_path / "o2",
                            plan_training(G, 70), compare=False, language="english", gpu=G, train_fn=_train([]),
                            engine_factory=lambda a, l: EmbEngine(f0=260.0), mos=FakeMos(1.2), asr=None)
    assert {"sim", "mos"} <= set(items[0].check["issues"]) and items[0].check["sim"] < vc.SIM_WARN


def test_post_training_check_has_similarity_and_mos(tmp_path, monkeypatch):
    import core.lora_trainer as lt
    from core import audio_utils as au
    from core.asr import FakeASR
    from core.voice_library import VoiceLibrary
    from tests.test_preview_check import TrueRateAligner, _inputs, _NoUpd
    from workers import preview_runner as pr
    from workers.pipeline_runner import KIND_LORA, TaskRequest, run_task
    _inputs(tmp_path)

    def fake_train(d, o, *a, **k):
        o.mkdir(parents=True, exist_ok=True)
        for f in ("adapter_model.safetensors", "adapter_config.json"):
            (o / f).write_bytes(b"x")
        au.write_wav(o / "ref_sample.wav", tone(120, 6.0), 24000)
        (o / "training_meta.json").write_text('{"epochs": 1, "ref_sample_text": "t"}', encoding="utf-8")
        return o
    monkeypatch.setattr(lt, "train_lora_from_dataset", fake_train)
    res = run_task(TaskRequest(kind=KIND_LORA, audio=tmp_path / "in.wav", text=tmp_path / "in.txt", out_root=tmp_path / "o",
                               quality_check=True), updater=_NoUpd(), aligner_factory=TrueRateAligner,
                   voice_library=VoiceLibrary(tmp_path / "lib"), asr_factory=lambda: FakeASR([pr.SAMPLE_TEXT["russian"]]),
                   synth_deps=dict(engine_factory=lambda a, l: EmbEngine(), mos=FakeMos(3.1)))
    assert res.quality["sim"] > 0.99 and res.quality["mos"] == 3.1 and res.quality["cer"] == 0.0


def test_engine_speaker_embedding_uses_the_models_own_encoder():
    torch = pytest.importorskip("torch")
    from core.tts_engine import Qwen3AdapterEngine
    from tests.test_lora_trainer import tiny_model
    eng = Qwen3AdapterEngine.__new__(Qwen3AdapterEngine)
    eng._q = SimpleNamespace(model=tiny_model().eval())
    a = eng.speaker_embedding(tone(120, 2.0, sr=16000), 16000)       # resampled to 24 kHz inside
    b = eng.speaker_embedding(tone(120, 2.0), 24000)
    assert a.shape == (32,) and vc.cosine(a, b) > 0.95
