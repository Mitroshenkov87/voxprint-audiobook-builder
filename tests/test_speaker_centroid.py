"""Averaged speaker embedding over clean clips: selection, averaging, storage, and its use in training, narration (voice-clone
prompt), the universal model, the cache key, the library and the voice check."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core import audio_utils as au
from core import speaker_centroid as sc


def _clip(path, secs, amp=0.2, noise=0.0, f=150.0, seed=0):
    sr = 24000
    t = np.arange(int(secs * sr)) / sr
    x = amp * np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))   # speech-like loudness changes
    x = x + noise * np.random.default_rng(seed).standard_normal(len(x))
    au.write_wav(path, x.astype(np.float32), sr)
    return str(path)


def test_average_restores_the_typical_length_and_beats_single_clips():
    rng = np.random.default_rng(1)
    true = rng.standard_normal(64)
    true /= np.linalg.norm(true)
    embs = [(true + 0.9 * rng.standard_normal(64) / 8) * rng.uniform(8, 12) for _ in range(40)]
    c = sc.average(embs)
    cos = lambda a, b: float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
    assert cos(c, true) > max(cos(e, true) for e in embs)               # steadier than any single clip
    assert abs(np.linalg.norm(c) - np.median([np.linalg.norm(e) for e in embs])) < 1e-3
    assert np.linalg.norm(np.mean(embs, axis=0)) < np.linalg.norm(c)   # the naive mean would shrink it
    with pytest.raises(ValueError):
        sc.average([np.zeros(4)])


def test_select_clips_keeps_clean_ones_best_first(tmp_path):
    paths = [_clip(tmp_path / "short.wav", 1.5), _clip(tmp_path / "long.wav", 20.0),
             _clip(tmp_path / "clipped.wav", 5.0, amp=1.5), _clip(tmp_path / "quiet.wav", 5.0, amp=0.001),
             _clip(tmp_path / "noisy.wav", 5.0, noise=0.05), _clip(tmp_path / "clean.wav", 5.0)]
    got = [p for p, _ in sc.select_clips(paths + [str(tmp_path / "missing.wav")])]
    assert [Path(p).name for p in got] == ["clean.wav", "noisy.wav"]
    assert len(sc.select_clips(paths, max_clips=1)) == 1


def test_compute_save_load_and_switch(tmp_path, monkeypatch):
    paths = [_clip(tmp_path / f"c{i}.wav", 4.0, f=140 + i, seed=i) for i in range(6)]
    res = sc.compute(lambda x, sr: np.array([1.0, float(len(x)) / sr]), paths)
    assert res is not None and len(res[1]) == 6 and res[0].shape == (2,)
    assert sc.compute(lambda x, sr: np.ones(2), paths[:3]) is None      # fewer than MIN_CLIPS -> old path
    sc.save(tmp_path / "v", res[0], res[1])
    assert np.allclose(sc.load(tmp_path / "v"), res[0]) and sc.load(tmp_path / "nothing") is None
    monkeypatch.setenv("VOXPRINT_SPEAKER_CENTROID", "0")
    assert not sc.enabled() and sc.load(tmp_path / "v") is None and sc.compute(lambda x, sr: np.ones(2), paths) is None


def test_prompt_x_vector_is_replaced_but_the_icl_part_stays():
    torch = pytest.importorskip("torch")
    from core.tts_engine import use_centroid
    item = SimpleNamespace(ref_spk_embedding=torch.zeros(4, dtype=torch.bfloat16), ref_code="codes", ref_text="t")
    assert use_centroid([item], np.array([1, 2, 3, 4], dtype=np.float32))
    assert item.ref_spk_embedding.dtype == torch.bfloat16 and item.ref_spk_embedding.tolist() == [1, 2, 3, 4]
    assert item.ref_code == "codes" and item.ref_text == "t"
    other = SimpleNamespace(ref_spk_embedding=torch.zeros(3))
    assert not use_centroid([other], np.ones(4, dtype=np.float32)) and other.ref_spk_embedding.tolist() == [0, 0, 0]
    assert not use_centroid([other], None)


def test_cache_key_library_and_check_reference(tmp_path, monkeypatch):
    from core import tts_engine
    from core.voice_library import ALLOWED_FILES, VoiceLibrary, VoiceRecord
    from tests.test_voice_library import make_adapter
    from workers.preview_runner import reference_embedding
    d = make_adapter(tmp_path / "a")
    rec = VoiceRecord("v", d, {"language": "en", "base_model": "Qwen/B"})
    old = tts_engine.engine_tag(rec)
    sc.save(d, np.array([3.0, 4.0]))
    with_c = tts_engine.engine_tag(rec)
    assert with_c != old
    monkeypatch.setenv("VOXPRINT_SPEAKER_CENTROID", "0")
    assert tts_engine.engine_tag(rec) == old
    monkeypatch.delenv("VOXPRINT_SPEAKER_CENTROID")
    assert sc.FILENAME in ALLOWED_FILES
    lib = VoiceLibrary(tmp_path / "lib")
    assert (lib.add_from_adapter(d).path / sc.FILENAME).is_file()       # kept when the voice is added / imported
    eng = SimpleNamespace(speaker_embedding=lambda a, sr: pytest.fail("the centroid is used, not the clip"))
    assert np.allclose(reference_embedding(eng, d, np.zeros(10), 24000), [3.0, 4.0])
    assert reference_embedding(SimpleNamespace(), d, np.zeros(10), 24000) is None   # fakes without an encoder


# --------------------------------------------------------------------------- tiny real model
torch = pytest.importorskip("torch")
pytest.importorskip("peft")


def test_training_uses_and_stores_the_centroid(tmp_path):
    from core.events import CancelToken
    from core.lora_trainer import load_training_rows, speaker_conditioning, speaker_embedding_from_ref, train_on_model
    from tests.test_lora_trainer import _dataset, _encode, _plan, _tok, tiny_model
    d = _dataset(tmp_path)
    data = load_training_rows(d)
    m = tiny_model()
    spk, cen = speaker_conditioning(m, data, torch.device("cpu"), torch.float32)
    assert cen is not None and len(cen[1]) == 6 and torch.allclose(spk[0], torch.from_numpy(cen[0]))
    spk_ref, none = speaker_conditioning(m, data, torch.device("cpu"), torch.float32, use_centroid=False)
    assert none is None and torch.allclose(spk_ref, speaker_embedding_from_ref(m, data["ref_audio"], torch.device("cpu"), torch.float32))
    out = tmp_path / "out"
    train_on_model(tiny_model(), _tok, _encode, data, out, _plan(), "Qwen/B", lambda *a: None, CancelToken(), [])
    assert (out / sc.FILENAME).is_file()
    assert json.loads((out / "checkpoints" / "losses.json").read_text())["speaker_embedding"] == {"source": "centroid", "clips": 6}
    train_on_model(tiny_model(), _tok, _encode, data, out, _plan(speaker_centroid=False), "Qwen/B", lambda *a: None,
                   CancelToken(), [])
    assert not (out / sc.FILENAME).exists()                            # switched off: the old single-clip path
    assert json.loads((out / "checkpoints" / "losses.json").read_text())["speaker_embedding"]["source"] == "ref"


def test_universal_model_speaker_row_is_the_centroid(tmp_path):
    from core.model_export import merge_adapter_into_model
    from tests.test_adapter_strength import _train_adapter
    from tests.test_lora_trainer import tiny_model
    ad = _train_adapter(tmp_path)
    assert sc.load(ad) is not None                                     # training wrote one
    c = np.linspace(-1, 1, 32).astype(np.float32)                      # (all tiny clips are equal: use a distinct vector)
    sc.save(ad, c)
    state, spk = merge_adapter_into_model(tiny_model(), ad, spk_id=100)
    assert torch.allclose(state["talker.model.codec_embedding.weight"][100], torch.from_numpy(c))
    (ad / sc.FILENAME).unlink()                                        # an older voice: the reference clip
    from core.lora_trainer import speaker_embedding_from_ref
    m = tiny_model()
    state2, spk2 = merge_adapter_into_model(m, ad, spk_id=100)
    want = speaker_embedding_from_ref(m, str(ad / "ref_sample.wav"), torch.device("cpu"), torch.float32)
    assert torch.allclose(spk2, want)
