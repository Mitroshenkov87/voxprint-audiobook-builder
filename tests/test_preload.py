"""Optional "Preload models into memory at startup": RAM availability, the background preloader and the reuse path
(engines take the preloaded objects over instead of loading again).  No real models: fake loaders / fake model classes."""
import json
import struct
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from core import model_cache
from infra import preload as pl
from infra import sysinfo

GB = pl.GB


@pytest.fixture(autouse=True)
def empty_cache():
    model_cache.clear()
    yield
    model_cache.clear()


def write_safetensors(path: Path, dtype: str = "BF16", payload: int = 1000) -> None:
    """A minimal safetensors file: header with one tensor of ``dtype`` + ``payload`` bytes of data."""
    header = json.dumps({"w": {"dtype": dtype, "shape": [payload], "data_offsets": [0, payload]}}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * payload)


def T(kind, gb, path="m"):
    return pl.Target(kind, Path(path), int(gb * GB))


# ----------------------------------------------------------------------------- availability calculation
def test_availability_ok_when_the_pc_has_models_plus_headroom():
    av = pl.availability([T("tts", 4), T("asr", 2)], total=64 * GB, available=50 * GB)
    assert av.status == pl.OK and av.offered
    assert av.models == 6 * GB + 2 * pl.OVERHEAD_PER_MODEL
    assert av.need == av.models + pl.HEADROOM
    assert pl.gb(av.total) == 64 and pl.gb(av.need, up=True) == 11


def test_availability_too_small_names_need_and_total():
    av = pl.availability([T("tts", 8), T("asr", 4)], total=8 * GB, available=6 * GB)
    assert av.status == pl.TOO_SMALL and not av.offered
    assert pl.gb(av.need, up=True) == 17 and pl.gb(av.total) == 8


def test_availability_waits_while_free_ram_is_short_and_counts_held_models_as_free():
    targets = [T("tts", 4)]
    low = pl.availability(targets, total=16 * GB, available=3 * GB)
    assert low.status == pl.LOW_NOW and low.offered
    # the same models already sitting in RAM do not need to fit a second time
    assert pl.availability(targets, total=16 * GB, available=3 * GB, held=4 * GB).status == pl.OK


def test_availability_without_models_or_without_ram_figures():
    assert pl.availability([], 64 * GB, 60 * GB).status == pl.NOTHING
    av = pl.availability([T("asr", 2)], 0, 0)
    assert av.status == pl.UNKNOWN and av.offered


def test_low_water_is_eight_percent_but_at_least_one_gb():
    assert pl.low_water(64 * GB) == int(64 * GB * 0.08)
    assert pl.low_water(8 * GB) == GB


def test_ram_comes_from_the_real_weight_files(tmp_path):
    m = tmp_path / "Qwen--TTS"
    write_safetensors(m / "model.safetensors", "BF16", 4000)
    write_safetensors(m / "speech_tokenizer" / "model.safetensors", "BF16", 1000)   # sub-folder counts too
    (m / "config.json").write_text("{" + " " * 5000 + "}")                          # not weights
    size = pl.weights_bytes(m)
    assert size == sum(p.stat().st_size for p in m.rglob("*.safetensors"))
    assert pl.ram_for(m, cuda=True) == size            # loaded in bfloat16 for the GPU
    assert pl.ram_for(m, cuda=False) == 2 * size       # float32 on a CPU-only PC
    f32 = tmp_path / "f32"
    write_safetensors(f32 / "model.safetensors", "F32", 4000)
    assert pl.ram_for(f32, cuda=False) == pl.weights_bytes(f32)


def test_find_targets_lists_only_installed_models(tmp_path):
    from infra import model_downloader as md

    tts, asr = tmp_path / "tts", tmp_path / "asr"
    write_safetensors(tts / "model.safetensors", payload=3000)
    write_safetensors(asr / "model.safetensors", payload=2000)
    ready = {"Qwen/Base": tts, md.ASR_REPO: asr}
    got = pl.find_targets("Qwen/Base", True, ready=ready.get, sage_dir=lambda: None)
    assert [(t.kind, t.path) for t in got] == [("tts", tts), ("asr", asr)]
    assert got[0].ram == pl.weights_bytes(tts)
    assert [t.kind for t in pl.find_targets("", True, ready=ready.get, sage_dir=lambda: tts)] == ["asr", "sage"]
    assert pl.find_targets("Qwen/Other", True, ready=lambda r: None, sage_dir=lambda: None) == []


def test_setting_is_off_by_default_and_persisted():
    assert pl.enabled() is False
    pl.set_enabled(True)
    assert pl.enabled() is True
    pl.set_enabled(False)
    assert pl.enabled() is False


# ----------------------------------------------------------------------------- the background preloader
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_preloader(targets, memory=lambda: (64 * GB, 50 * GB), loaders=None, enabled=True, voice=None):
    state = {"on": enabled}
    calls = []

    def loader(kind):
        def load(path, cuda):
            calls.append((kind, path, cuda))
            return {"kind": kind, "path": path}
        return load

    clock = Clock()
    pre = pl.Preloader(lambda: voice, loaders=loaders or {k: loader(k) for k in ("tts", "asr", "sage")}, memory=memory,
                       cuda=lambda: True, targets=lambda repo, cuda: targets, is_enabled=lambda: state["on"],
                       clock=clock, start_delay=5.0)
    return pre, calls, clock, state


def run_tick(pre, clock, busy=False):
    clock.t += pl.CHECK_EVERY_S
    pre.tick(busy)
    if pre._thread is not None:
        pre._thread.join(5)


def test_preloader_warms_installed_models_after_the_start_delay(tmp_path):
    targets = [T("tts", 1, tmp_path / "tts"), T("asr", 1, tmp_path / "asr")]
    pre, calls, clock, _ = make_preloader(targets)
    pre.tick()                                         # right after the start: nothing yet
    assert pre._thread is None and calls == []
    run_tick(pre, clock)
    assert [c[0] for c in calls] == ["tts", "asr"]
    assert set(model_cache.keys()) == {model_cache.key("tts", tmp_path / "tts", True), model_cache.key("asr", tmp_path / "asr", True)}
    assert pre.held_bytes() == 2 * GB
    run_tick(pre, clock)                               # already warm: nothing is loaded twice
    assert len(calls) == 2


def test_preloader_does_nothing_when_off_busy_or_short_of_ram(tmp_path):
    targets = [T("tts", 4, tmp_path / "tts")]
    pre, calls, clock, _ = make_preloader(targets, enabled=False)
    run_tick(pre, clock)
    pre, calls2, clock2, _ = make_preloader(targets)
    run_tick(pre, clock2, busy=True)
    pre, calls3, clock3, _ = make_preloader(targets, memory=lambda: (8 * GB, 6 * GB))     # needs ~8.25 GB + headroom
    run_tick(pre, clock3)
    assert calls == calls2 == calls3 == [] and model_cache.keys() == []


def test_unticking_frees_the_preloaded_models(tmp_path):
    pre, calls, clock, state = make_preloader([T("asr", 1, tmp_path / "asr")])
    run_tick(pre, clock)
    assert model_cache.keys()
    pre.set_enabled(False)
    assert model_cache.keys() == [] and pre.held_bytes() == 0 and pl.enabled() is False


def test_low_memory_frees_the_models_and_waits_before_warming_again(tmp_path):
    mem = {"free": 50 * GB}
    pre, calls, clock, _ = make_preloader([T("asr", 1, tmp_path / "asr")], memory=lambda: (64 * GB, mem["free"]))
    run_tick(pre, clock)
    assert len(model_cache.keys()) == 1
    mem["free"] = 2 * GB                               # below 8 % of 64 GB
    run_tick(pre, clock)
    assert model_cache.keys() == []
    mem["free"] = 50 * GB
    run_tick(pre, clock)                               # cool-down: not at once
    assert model_cache.keys() == [] and len(calls) == 1
    clock.t += pl.COOLDOWN_S
    run_tick(pre, clock)
    assert len(model_cache.keys()) == 1 and len(calls) == 2


def test_switching_voice_to_another_base_model_drops_the_old_one(tmp_path):
    targets = [T("tts", 1, tmp_path / "base_a")]
    pre, calls, clock, _ = make_preloader(targets)
    pre.targets = lambda repo, cuda: targets
    run_tick(pre, clock)
    targets[:] = [T("tts", 1, tmp_path / "base_b")]
    run_tick(pre, clock)
    assert model_cache.keys() == [model_cache.key("tts", tmp_path / "base_b", True)]


def test_free_while_loading_discards_the_loaded_model(tmp_path):
    gate, started = threading.Event(), threading.Event()

    def slow(path, cuda):
        started.set()
        gate.wait(5)
        return "model"

    pre, _, clock, _ = make_preloader([T("tts", 1, tmp_path / "tts")], loaders={"tts": slow})
    clock.t += pl.CHECK_EVERY_S
    pre.tick()
    assert started.wait(5)
    pre.free("switched off")
    gate.set()
    pre._thread.join(5)
    assert model_cache.keys() == []


def test_preload_thread_runs_at_low_priority():
    out = []
    t = threading.Thread(target=lambda: out.append(sysinfo.lower_thread_priority()))
    t.start()
    t.join()
    assert out and isinstance(out[0], bool)


# ----------------------------------------------------------------------------- the registry
def test_take_moves_the_object_out():
    model_cache.put(("asr", "x", "cpu"), "obj")
    assert model_cache.take(("asr", "x", "cpu")) == "obj"
    assert model_cache.take(("asr", "x", "cpu")) is None


def test_take_waits_for_a_load_in_progress_instead_of_loading_again():
    key = ("tts", "base", "cuda")
    inside, release = threading.Event(), threading.Event()

    def preload():
        with model_cache.loading(key):
            inside.set()
            release.wait(5)
            model_cache.put(key, "warm")

    t = threading.Thread(target=preload)
    t.start()
    assert inside.wait(5)
    got = []
    taker = threading.Thread(target=lambda: got.append(model_cache.take(key)))
    taker.start()
    time.sleep(0.1)
    assert got == []                                   # still waiting
    release.set()
    taker.join(5)
    t.join(5)
    assert got == ["warm"]


def test_to_device_updates_the_cached_devices():
    torch = pytest.importorskip("torch")
    tok = types.SimpleNamespace(model=torch.nn.Linear(2, 2), device=None)
    inner = torch.nn.Linear(2, 2)
    inner.speech_tokenizer = tok
    wrapper = types.SimpleNamespace(model=inner, device=None)
    assert model_cache.to_device(wrapper, "cpu") is wrapper
    assert wrapper.device == torch.device("cpu") and tok.device == torch.device("cpu")


# ----------------------------------------------------------------------------- reuse in the engines
class FakeTalker:
    def merge_and_unload(self):
        return "merged-talker"


def fake_tts_modules(monkeypatch, loads):
    class Qwen3TTSModel:
        @classmethod
        def from_pretrained(cls, path, **kw):
            loads.append((path, kw))
            return FakeWrapper()

    class PeftModel:
        @staticmethod
        def from_pretrained(talker, path):
            return FakeTalker()

    monkeypatch.setitem(sys.modules, "qwen_tts", types.SimpleNamespace(Qwen3TTSModel=Qwen3TTSModel))
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=PeftModel))


class FakeWrapper:
    def __init__(self):
        self.model = types.SimpleNamespace(talker="talker", eval=lambda: None)

    def create_voice_clone_prompt(self, ref_audio, ref_text):
        return {"ref": ref_text}


def make_voice(tmp_path):
    from core.voice_library import VoiceRecord

    folder = tmp_path / "voice"
    folder.mkdir()
    (folder / "ref_sample.wav").write_bytes(b"RIFF")
    (folder / "training_meta.json").write_text(json.dumps({"ref_sample_text": "Text."}), encoding="utf-8")
    return VoiceRecord("v1", folder, {"name": "[model_voice]", "base_model": "Qwen/Base", "language": "ru"})


def test_tts_engine_takes_the_preloaded_base_model_instead_of_loading(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from core import tts_engine

    loads = []
    fake_tts_modules(monkeypatch, loads)
    base = tmp_path / "base"
    base.mkdir()
    warm = FakeWrapper()
    model_cache.put(model_cache.key("tts", base, False), (warm, "sdpa"))
    eng = tts_engine.Qwen3AdapterEngine(make_voice(tmp_path), base, device="cpu")
    try:
        assert loads == [] and eng._q is warm and eng.attn == "sdpa"
        assert warm.model.talker == "merged-talker"        # the adapter went onto the preloaded weights
        assert model_cache.keys() == []                    # handed over, not shared
    finally:
        eng.close()


def test_tts_engine_loads_normally_without_a_preloaded_model(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from core import tts_engine

    loads = []
    fake_tts_modules(monkeypatch, loads)
    base = tmp_path / "base"
    base.mkdir()
    model_cache.put(model_cache.key("tts", tmp_path / "other", False), (FakeWrapper(), "sdpa"))
    eng = tts_engine.Qwen3AdapterEngine(make_voice(tmp_path), base, device="cpu")
    try:
        assert len(loads) == 1 and loads[0][0] == str(base)
        assert len(model_cache.keys()) == 1                # someone else's model is left alone
    finally:
        eng.close()


def test_asr_takes_the_preloaded_model(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from core.asr import Qwen3ASR

    class Never:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            raise AssertionError("loaded again")

    monkeypatch.setitem(sys.modules, "qwen_asr", types.SimpleNamespace(Qwen3ASRModel=Never))
    warm = types.SimpleNamespace(model=None, device=None)
    model_cache.put(model_cache.key("asr", tmp_path, False), warm)
    asr = Qwen3ASR(str(tmp_path), device="cpu")
    asr.load()
    assert asr._m is warm and model_cache.keys() == []


def test_sage_takes_the_preloaded_tokenizer_and_model(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from core.text_cleanup import SageEngine

    class Model:
        def to(self, dev):
            self.dev = dev
            return self

        def eval(self):
            return self

    import transformers

    def never(*a, **kw):
        raise AssertionError("loaded again")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", never)
    monkeypatch.setattr(transformers.AutoModelForSeq2SeqLM, "from_pretrained", never)
    model = Model()
    model_cache.put(model_cache.key("sage", tmp_path, False), ("tok", model))
    eng = SageEngine(tmp_path, device="cpu")
    eng._load()
    assert eng._tok == "tok" and eng._model is model and model.dev == "cpu"


# ----------------------------------------------------------------------------- Settings dialog
def test_settings_shows_availability_and_toggles_the_preloader(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from core.i18n import tr
    from tests.test_studio import make_studio
    from core.voice_library import VoiceLibrary
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    s = make_studio(VoiceLibrary(tmp_path / "voices"))
    try:
        d = s.settings_dialog()
        d.memory = lambda: (64 * GB, 40 * GB)
        d.preload_targets = lambda repo, cuda: [T("tts", 4), T("asr", 2)]
        d.refresh_preload()
        assert d.lbl_preload.text() == tr("preload.available", total=64)
        assert d.chk_preload.isEnabled() and not d.chk_preload.isChecked()
        assert d.chk_preload.text() == tr("preload.option")
        d.memory = lambda: (8 * GB, 6 * GB)
        d.refresh_preload()
        assert d.lbl_preload.text() == tr("preload.too_small", need=11, total=8)
        assert not d.chk_preload.isEnabled()
        d.memory = lambda: (64 * GB, 40 * GB)
        d.refresh_preload()
        model_cache.put(("asr", "x", "cuda"), "warm")
        s.preloader._held[("asr", "x", "cuda")] = GB
        d.chk_preload.setChecked(True)
        assert pl.enabled() is True
        d.chk_preload.setChecked(False)
        assert pl.enabled() is False and model_cache.keys() == []
        d.preload_targets = lambda repo, cuda: []
        d.refresh_preload()
        assert d.lbl_preload.text() == tr("preload.no_models")
    finally:
        s.shutdown()
