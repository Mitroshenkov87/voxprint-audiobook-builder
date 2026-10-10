"""Optional CUDA Graphs decode path (faster-qwen3-tts) and ``voxprint bench`` - with fakes, no GPU."""
import json
import sys
import types

import numpy as np
import pytest

from core import bench
from core import fast_decode as fd


class Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_tolerant_rope_init_gives_a_config_without_rope_theta_a_fallback():
    seen = []

    def default(config, device=None, seq_len=None, **kw):
        seen.append(config.rope_theta)
        return config.head_dim

    f = fd.tolerant(default)
    assert fd.tolerant(f) is f                                     # idempotent
    assert f(Cfg(rope_theta=5.0, head_dim=64)) == 64               # unchanged when present
    assert f(Cfg(head_dim=32)) == 32                               # MimiConfig-like: no rope_theta
    assert f(Cfg(head_dim=8, rope_parameters={"rope_theta": 777.0}), "cpu") == 8
    assert seen == [5.0, fd.DEFAULT_ROPE_THETA, 777.0]


def test_install_rope_compat_patches_both_places(monkeypatch):
    calls = []
    table = {"default": lambda config, *a, **k: calls.append(config.rope_theta)}
    mru = types.SimpleNamespace(ROPE_INIT_FUNCTIONS=table)
    compat = types.ModuleType("qwen_tts._transformers_compat")
    compat._default_rope_parameters = lambda config, *a, **k: calls.append(("compat", config.rope_theta))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(modeling_rope_utils=mru))
    monkeypatch.setitem(sys.modules, "transformers.modeling_rope_utils", mru)
    monkeypatch.setitem(sys.modules, "qwen_tts", types.ModuleType("qwen_tts"))
    monkeypatch.setitem(sys.modules, "qwen_tts._transformers_compat", compat)
    assert fd.install_rope_compat() == ["transformers.ROPE_INIT_FUNCTIONS", "qwen_tts._transformers_compat"]
    fd.install_rope_compat()
    table["default"](Cfg())
    compat._default_rope_parameters(Cfg())
    assert calls == [fd.DEFAULT_ROPE_THETA, ("compat", fd.DEFAULT_ROPE_THETA)]


def test_install_rope_compat_is_harmless_with_the_real_transformers():
    pytest.importorskip("transformers")
    from transformers import modeling_rope_utils as mru

    before = mru.ROPE_INIT_FUNCTIONS["default"]
    try:
        assert "transformers.ROPE_INIT_FUNCTIONS" in fd.install_rope_compat()
        f = mru.ROPE_INIT_FUNCTIONS["default"]
        assert f.__wrapped__ is before
        cfg = types.SimpleNamespace(rope_theta=10000.0, hidden_size=64, num_attention_heads=4, partial_rotary_factor=1.0,
                                    head_dim=16)
        inv, scale = f(cfg, None)
        assert tuple(inv.shape) == (8,) and scale == 1.0
        del cfg.rope_theta                                          # like MimiConfig under transformers >= 5.18
        inv2, _ = f(cfg, None)
        assert np.allclose(inv.numpy(), inv2.numpy())
    finally:
        mru.ROPE_INIT_FUNCTIONS["default"] = before


def test_check_reports_why_it_cannot_run(monkeypatch):
    monkeypatch.setattr(fd, "installed_version", lambda: None)
    assert fd.check() == (False, "faster-qwen3-tts is not installed")
    monkeypatch.setattr(fd, "installed_version", lambda: "0.5.4")   # needs qwen-tts-hf + transformers 5: not for our pins
    ok, why = fd.check()
    assert not ok and "not supported" in why


def test_new_tokens_fit_the_static_cache():
    assert fd.cap_new_tokens(500) == 500
    assert fd.cap_new_tokens(5000) == fd.MAX_SEQ_LEN - fd.PROMPT_ROOM
    assert fd.cap_new_tokens(0) == 1


def test_gpu_prefs_fast_decode(monkeypatch):
    from infra import gpu_prefs

    assert gpu_prefs.fast_decode() == "off"
    assert gpu_prefs.set_fast_decode("cuda_graphs") == "graphs" and gpu_prefs.fast_decode() == "graphs"
    monkeypatch.setenv(gpu_prefs.ENV_FAST_DECODE, "off")
    assert gpu_prefs.fast_decode() == "off"
    monkeypatch.setenv(gpu_prefs.ENV_FAST_DECODE, "warp")
    assert gpu_prefs.fast_decode() == "off"
    with pytest.raises(ValueError):
        gpu_prefs.set_fast_decode("turbo")


def fake_faster_package(monkeypatch, log):
    pkg = types.ModuleType("faster_qwen3_tts")

    class FasterQwen3TTS:
        def __init__(self, base_model, predictor_graph, talker_graph, device, dtype, max_seq_len):
            log.append(("init", device, max_seq_len))
            self.model = base_model

        def warmup(self, prefill_len=100):
            log.append(("warmup", prefill_len))

        def generate_voice_clone(self, text, language, ref_text="", voice_clone_prompt=None, max_new_tokens=2048):
            log.append(("gen", text, language, ref_text, len(voice_clone_prompt), max_new_tokens))
            return [np.ones(240, dtype=np.float32)], 24000

    pkg.FasterQwen3TTS = FasterQwen3TTS
    pg = types.ModuleType("faster_qwen3_tts.predictor_graph")
    pg.PredictorGraph = lambda *a, **k: log.append(("predictor", k["top_k"], k["temperature"])) or "pg"
    tg = types.ModuleType("faster_qwen3_tts.talker_graph")
    tg.TalkerGraph = lambda *a, **k: log.append(("talker", k["max_seq_len"])) or "tg"
    for name, mod in (("faster_qwen3_tts", pkg), ("faster_qwen3_tts.predictor_graph", pg), ("faster_qwen3_tts.talker_graph", tg)):
        monkeypatch.setitem(sys.modules, name, mod)


def fake_q():
    predictor = types.SimpleNamespace(model=types.SimpleNamespace(config="pred-config"))
    talker = types.SimpleNamespace(code_predictor=predictor, model="talker-backbone")
    model = types.SimpleNamespace(talker=talker, config=types.SimpleNamespace(talker_config=types.SimpleNamespace(hidden_size=8)))
    return types.SimpleNamespace(model=model)


def test_graph_decoder_wraps_the_loaded_model(monkeypatch):
    pytest.importorskip("torch")
    log = []
    fake_faster_package(monkeypatch, log)
    dec = fd.GraphDecoder(fake_q(), "cuda:1")
    prompt = [types.SimpleNamespace(ref_text="Reference.")]
    audio, sr = dec.synthesize("Hello.", "Russian", prompt, 9999)
    assert sr == 24000 and audio.dtype == np.float32 and audio.size == 240
    assert ("predictor", 50, 0.9) in log and ("talker", fd.MAX_SEQ_LEN) in log and ("init", "cuda:1", fd.MAX_SEQ_LEN) in log
    assert ("warmup", 100) in log
    assert ("gen", "Hello.", "Russian", "Reference.", 1, fd.MAX_SEQ_LEN - fd.PROMPT_ROOM) in log


# ------------------------------------------------------------------------------------------------- bench
class BenchEngine:
    def __init__(self, mode, batch=4, fail=False):
        self.decode_mode = mode
        self.batch, self.fail, self.calls, self.closed = batch, fail, [], False
        self.sample_rate = 24000
        self.device = "cpu"

    def synthesize(self, text):
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        self.calls.append([text])
        return np.zeros(2400, dtype=np.float32)

    def synthesize_batch(self, texts):
        self.calls.append(list(texts))
        return [np.zeros(2400, dtype=np.float32) for _ in texts]

    def max_batch(self):
        return self.batch

    def close(self):
        self.closed = True


def test_bench_runs_both_modes_and_measures(tmp_path):
    engines = {}

    def make(mode):
        engines[mode] = BenchEngine(mode)
        return engines[mode]

    res = bench.run(bench.MODES, make, out_dir=tmp_path)
    b, g = res
    assert b.ok and g.ok and b.phrases == g.phrases == len(bench.TEXT_RU)
    assert b.batch == 4 and g.batch == 1
    assert max(len(c) for c in engines["batched"].calls[1:]) == 4            # warm-up first, then batches of 4
    assert all(len(c) == 1 for c in engines["graphs"].calls)
    assert b.audio_s == pytest.approx(len(bench.TEXT_RU) * 0.1) and b.x_realtime > 0 and b.rtf > 0
    assert all(e.closed for e in engines.values())
    assert (tmp_path / "bench-batched.wav").is_file() and (tmp_path / "bench-graphs.wav").is_file()
    s = bench.summary(res)
    assert "graphs_vs_batched" in s and len(s["modes"]) == 2


def test_bench_skips_graphs_when_the_engine_fell_back():
    res = bench.run(["graphs"], lambda mode: BenchEngine("batched"))
    assert not res[0].ok and "not available" in res[0].reason
    res = bench.run(["batched"], lambda mode: BenchEngine("batched", fail=True))
    assert not res[0].ok and "out of memory" in res[0].reason


def test_cli_bench(capsys, monkeypatch):
    import cli

    made = []

    def make(mode):
        made.append(mode)
        return BenchEngine("batched" if mode == "batched" else "batched")       # graphs not available here

    args = cli.build_parser().parse_args(["bench", "--json"])
    rc = cli.cmd_bench(args, make_engine=make)
    out = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    result = out[-1]
    assert rc == 0 and made == ["batched", "graphs"] and result["ok"]
    modes = {m["mode"]: m for m in result["bench"]["modes"]}
    assert modes["batched"]["ok"] and not modes["graphs"]["ok"]
    assert any("graphs" in w for w in result["warnings"])


def test_cli_bench_bad_mode_and_dispatch():
    import cli

    assert cli.main(["bench", "--modes", "warp"]) == cli.EXIT_BAD_ARGS
    assert cli.is_user_cli(["voxprint", "bench"])


def test_cli_settings_gpu_keys(capsys):
    import cli

    assert cli.main(["settings", "set", "gpu", "cuda:1"]) == 0
    assert cli.main(["settings", "set", "gpu.vram_fraction", "0.7"]) == 0
    assert cli.main(["settings", "set", "gpu.fast_decode", "graphs"]) == 0
    assert cli.main(["settings", "set", "gpu", "tpu"]) == cli.EXIT_BAD_ARGS
    assert cli.main(["settings", "set", "theme", "glass-dark"]) == 0
    capsys.readouterr()
    assert cli.main(["settings", "list", "--json"]) == 0
    values = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["settings"]
    assert values["gpu"] == "cuda:1" and values["gpu.vram_fraction"] == 0.7 and values["gpu.fast_decode"] == "graphs"
    assert values["theme"] == "glass-dark"


def test_engine_device_follows_the_shared_gpu_setting(monkeypatch):
    torch = pytest.importorskip("torch")
    from core import tts_engine
    from infra import suite_settings

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert tts_engine.resolve_device("auto") == "cuda:0"
    suite_settings.set_value("gpu", "cuda:1")
    assert tts_engine.resolve_device("auto") == "cuda:1" and tts_engine.resolve_device("cpu") == "cpu"
    suite_settings.set_value("gpu", "cuda:5")
    assert tts_engine.resolve_device("auto") == "cuda:0"                       # not there: first GPU
    with pytest.raises(ValueError):
        suite_settings.set_value("gpu", "cpu")
    assert tts_engine.resolve_device("auto") == "cuda:0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert tts_engine.resolve_device("cuda:1") == "cpu"


def test_engine_uses_graphs_only_when_asked_and_possible(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from core import tts_engine
    from tests.test_preload import fake_tts_modules, make_voice

    fake_tts_modules(monkeypatch, [])
    base = tmp_path / "base"
    base.mkdir()
    voice = make_voice(tmp_path)

    class FakeGraph:
        def __init__(self, q, device, dtype):
            self.device = device

        def synthesize(self, text, language, prompt, n):
            return np.full(10, 0.5, dtype=np.float32), 24000

    monkeypatch.setattr(fd, "GraphDecoder", FakeGraph)
    monkeypatch.setattr(fd, "check", lambda: (True, "faster-qwen3-tts 0.3.2"))
    eng = tts_engine.Qwen3AdapterEngine(voice, base, device="cpu", fast_decode="graphs")
    assert eng.decode_mode == "single" and eng._graph is None                 # never on the CPU
    eng.close()
    monkeypatch.setattr(tts_engine, "resolve_device", lambda device="auto": "cuda:0")   # pretend a GPU from here on
    monkeypatch.setattr(tts_engine, "plan_batch_now", lambda device: pytest.fail("graphs never plan a batch"))
    eng = tts_engine.Qwen3AdapterEngine(voice, base)                          # default setting: off
    assert eng.decode_mode == "batched" and eng._graph is None
    eng.close()
    eng = tts_engine.Qwen3AdapterEngine(voice, base, fast_decode="graphs")
    assert eng.decode_mode == "graphs" and eng.max_batch() == 1
    assert eng.synthesize("x").tolist() == [0.5] * 10 and len(eng.synthesize_batch(["a", "b"])) == 2
    eng.close()
    monkeypatch.setattr(fd, "check", lambda: (False, "faster-qwen3-tts is not installed"))
    eng = tts_engine.Qwen3AdapterEngine(voice, base, fast_decode="graphs")
    assert eng._graph is None
    eng.close()

    def broken(*a, **k):
        raise RuntimeError("operation not permitted when stream is capturing")

    monkeypatch.setattr(fd, "check", lambda: (True, "ok"))
    monkeypatch.setattr(fd, "GraphDecoder", broken)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
    eng = tts_engine.Qwen3AdapterEngine(voice, base, fast_decode="graphs")
    assert eng._graph is None and eng.decode_mode == "batched"                # capture failed: the normal path
    eng.close()


def test_install_wheel_checks_the_hash_and_unpacks_only_the_package(tmp_path):
    import hashlib
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("faster_qwen3_tts/__init__.py", "__version__ = '0.3.2'\n")
        zf.writestr("faster_qwen3_tts-0.3.2.dist-info/METADATA", "Name: faster-qwen3-tts\nVersion: 0.3.2\n")
    good = buf.getvalue()
    pinned = dict(fd.WHEEL)
    try:
        fd.WHEEL.update(size=len(good), sha256=hashlib.sha256(good).hexdigest())
        with pytest.raises(ValueError, match="SHA-256"):
            fd.install_wheel(tmp_path / "bad", fetch=lambda url: good + b"x")
        assert not (tmp_path / "bad").exists()
        where = fd.install_wheel(tmp_path / "pk", fetch=lambda url: good)
        assert (where / "__init__.py").is_file() and (tmp_path / "pk" / "faster_qwen3_tts-0.3.2.dist-info").is_dir()
    finally:
        fd.WHEEL.clear()
        fd.WHEEL.update(pinned)
    assert fd.WHEEL["url"].endswith(fd.WHEEL["file"]) and len(fd.WHEEL["sha256"]) == 64


def test_the_pinned_wheel_is_in_the_supported_range():
    from packaging.specifiers import SpecifierSet

    assert fd.WHEEL["version"] in SpecifierSet(fd.SUPPORTED)
