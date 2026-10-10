"""Soundscape: the sample fixture, re-anchoring, a synthetic mix, and the enable gate.

No ACE-Step weights and no network. Cue audio is a tone the test supplies.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import cli as user_cli
from core import audiobook_export as ex
from core import narration as nr
from core import soundscape
from core import voice_info
from core.audiobook_export import ChapterAudio
from core.book_parsers import Book, Chapter, load_book
from core.voice_library import VoiceLibrary
from infra import auto_repair, model_mirrors, modules, paths, portable, soundscape_model
from tests.test_narration import FakeEngine

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sound-sample.vxbook"
REPO = soundscape_model.REPO


def _cue(plan, cid):
    return next(c for c in plan.cues if c.id == cid)


def _copy(book, **changes):
    doc = json.loads(json.dumps(book.sound_document))
    cast = json.loads(json.dumps(book.sound_cast_document))
    return book.carry(sound_document=doc, sound_cast_document=cast, **changes)


def _tone(prompt, seconds, sr, kind):
    n = max(1, int(seconds * sr))
    return np.full(n, 0.25, np.float32)


def _add_voice(lib: VoiceLibrary, folder: Path, name: str, voice_id: str, voice_type: str) -> None:
    src = folder / f"_src_{voice_id}"
    src.mkdir(parents=True)
    (src / "adapter_model.safetensors").write_bytes(b"weights")
    (src / "adapter_config.json").write_text("{}", encoding="utf-8")
    (src / "ref_sample.wav").write_bytes(b"RIFFxxxx")
    (src / "training_meta.json").write_text(json.dumps({"ref_sample_text": "hello"}), encoding="utf-8")
    voice_info.write_voice_json(src, voice_info.build_voice_info(
        name, "ru", 60, 3, "Qwen/Base", license="CC-BY-4.0", voice_type=voice_type))
    rec = lib.add_from_adapter(src, name=name)
    if rec.id != voice_id and not (lib.root / voice_id).exists():
        rec.path.rename(lib.root / voice_id)
        info = voice_info.read_voice_json(lib.root / voice_id) or {}
        info["id"] = voice_id
        voice_info.write_voice_json(lib.root / voice_id, info)


def _cast(tmp_path: Path, *ids) -> VoiceLibrary:
    lib = VoiceLibrary(tmp_path / "lib")
    types = {"levi": "male", "miriam": "female", "natan": "male"}
    for vid in ids:
        _add_voice(lib, tmp_path, vid.title(), vid, types[vid])
    return VoiceLibrary(tmp_path / "lib")


def test_sample_plan_uses_defaults_cast_and_skips_a_plain_book():
    book = load_book(FIXTURE)
    plan = soundscape.plan_for(book)
    assert plan is not None and soundscape.requested(book)
    bed, sfx, accent, transition = (_cue(plan, cid) for cid in ("c1", "c2", "c3", "c4"))
    assert bed.kind == "bed" and bed.layer == "music"
    assert (bed.start_paragraph, bed.end_paragraph) == (1, 4)
    assert bed.gain_db == -22 and bed.duck_db == 8 and bed.fade_in_ms == 3000
    assert sfx.kind == "sfx" and sfx.gain_db == -20 and sfx.position == "after"
    assert accent.kind == "accent" and accent.gain_db == -18 and accent.position == "before"
    assert transition.kind == "transition" and transition.gain_db == -16
    assert transition.asset_library == "voxprint-sound" and transition.asset_id == "harp-gliss-major-03"
    assert any("harp-gliss-major-03" in w for w in plan.warnings)
    assert 30 <= soundscape.BED_SECONDS <= 60
    assert soundscape.plan_for(Book("T", chapters=[Chapter("C", "Hello there.\n\nSecond paragraph here.")])) is None


def test_fingerprint_reanchors_or_drops_and_cast_can_turn_sound_off():
    book = load_book(FIXTURE)
    moved = _copy(book)
    row = next(c for c in moved.sound_document["cues"] if c["id"] == "c3")
    row["start_paragraph"] = 99
    plan = soundscape.plan_for(moved)
    assert _cue(plan, "c3").start_paragraph == 2
    assert any("re-anchored" in w and "c3" in w for w in plan.warnings)

    lost = _copy(book)
    row = next(c for c in lost.sound_document["cues"] if c["id"] == "c3")
    row["start_paragraph"] = 99
    row["para_fp"] = "deadbeefdead"
    plan = soundscape.plan_for(lost)
    assert "c3" not in {c.id for c in plan.cues}
    assert any("dropped" in w and "c3" in w for w in plan.warnings)

    off = _copy(book)
    off.sound_cast_document["enabled"] = False
    assert soundscape.plan_for(off) is None

    clash = book.carry(sound_inline=((1, "c3"),))
    plan = soundscape.plan_for(clash)
    assert _cue(plan, "c3").start_paragraph == 2
    assert any("sound.json" in w and "c3" in w for w in plan.warnings)


def test_loop_crossfade_covers_the_scene():
    clip = np.ones(1000, np.float32)
    out = soundscape.loop_crossfade(clip, 4500, 200)
    assert out.shape == (4500,)
    assert out[0] == 1.0 and np.max(out) <= 1.0
    assert np.min(out[200:800]) > 0.0


def test_duck_keeps_speech_loudness_and_beds_sit_under_it(tmp_path):
    sr = 8000
    speech = (0.4 * np.sin(np.linspace(0, 40 * np.pi, sr, dtype=np.float32))).astype(np.float32)
    narr = np.concatenate([speech, np.zeros(sr, np.float32)])
    wav = tmp_path / "ch.wav"
    sf.write(str(wav), narr, sr)
    chapter = ChapterAudio(0, "Chapter", wav, len(narr) / sr)
    cue = soundscape.Cue(
        id="bed", kind="bed", prompt="room", start_paragraph=1, end_paragraph=1, chapter_index=0,
        gain_db=-22, duck_db=8, layer="music", fade_in_ms=0, fade_out_ms=0, crossfade_ms=0, loop=True,
    )
    plan = soundscape.SoundPlan([cue])
    spans = [(1, "Hello there friend.", 0, sr, len(narr))]
    soundscape.mix_chapter_file(chapter, spans, plan, 0, soundscape.SoundRequest(generate=_tone), tmp_path / "cache",
                                revision="test")
    mixed, got = sf.read(str(wav), dtype="float32")
    assert got == sr
    mask = soundscape._speech_mask(narr, sr) > 0.5
    speech_rms = soundscape._rms(narr[mask])
    assert soundscape._rms(mixed[mask]) == pytest.approx(speech_rms, rel=0.01)
    gap = mixed[sr + 500:sr + 7000]
    said = mixed[500:7000]
    gap_db = 20 * np.log10(soundscape._rms(gap) / soundscape._rms(said))
    assert gap_db == pytest.approx(-22, abs=0.4)
    level = speech_rms * (10 ** (-22 / 20))
    scale = float(gap[0] / level)
    bed_under = said - scale * speech[500:7000]
    duck_db = 20 * np.log10(soundscape._rms(bed_under) / soundscape._rms(gap))
    assert duck_db == pytest.approx(-8, abs=0.4)


def test_sfx_failure_is_skipped_and_a_plain_book_never_mixes(tmp_path, caplog):
    calls = []

    def generate(prompt, seconds, sr, kind):
        calls.append((kind, seconds))
        if kind == "sfx":
            raise RuntimeError("cannot render")
        return _tone(prompt, seconds, sr, kind)

    book = load_book(FIXTURE)
    caplog.set_level("WARNING", logger="voxprint.soundscape")
    res = nr.narrate_book(
        book, FakeEngine, "t", tmp_path / "out", language="russian", ffmpeg=None,
        options=nr.NarrationOptions(
            formats={ex.FORMAT_WAV_CHAPTERS}, speak_titles=False, yo=False, ordinals=False,
            sound=soundscape.SoundRequest(generate=generate),
        ),
    )
    assert res.chapters == 2 and res.files
    assert ("bed", soundscape.BED_SECONDS) in calls
    assert ("sfx", 4.0) in calls and ("accent", 3.0) in calls and ("transition", 3.0) in calls
    assert any("could not be rendered" in r.message for r in caplog.records)
    wavs = list((tmp_path / "out").rglob("*.wav"))
    assert len(wavs) == 2 and all(p.stat().st_size > 44 for p in wavs)

    plain_calls = []
    plain = Book("Plain", chapters=[Chapter("One", "Hello there friend.\n\nSecond paragraph is long enough.")])
    nr.narrate_book(
        plain, FakeEngine, "t", tmp_path / "plain", language="english", ffmpeg=None,
        options=nr.NarrationOptions(
            formats={ex.FORMAT_WAV_CHAPTERS}, speak_titles=False, yo=False, ordinals=False,
            sound=soundscape.SoundRequest(generate=lambda *a: plain_calls.append(a) or _tone(*a)),
        ),
    )
    assert plain_calls == []


def test_enable_downloads_a_pinned_file_and_narrate_does_not(tmp_path, monkeypatch):
    payload = b"not-the-real-weights"
    rev = "b" * 40
    manifest = tmp_path / "mirrors.json"
    body = {"schema": 1, "models": {REPO: {
        "license": "MIT", "source_repo": REPO, "source_revision": rev,
        "files": {"config.json": {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}},
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
        return Resp(payload)

    assert soundscape_model.enabled() is False
    folder = soundscape_model.ensure(models_dir=tmp_path / "models", opener=opener, manifest=manifest)
    assert (folder / "config.json").read_bytes() == payload
    assert urls == [f"https://huggingface.co/{REPO}/resolve/{rev}/config.json"]
    assert soundscape_model.ready(tmp_path / "models", manifest)
    urls.clear()
    assert soundscape_model.ensure(models_dir=tmp_path / "models", opener=opener, manifest=manifest) == folder
    assert urls == []
    monkeypatch.setattr(soundscape_model, "ensure", lambda *a, **k: folder)
    assert soundscape_model.enable(manifest=manifest) == folder
    state = json.loads((paths.state_dir() / "soundscape.json").read_text(encoding="utf-8"))
    assert state["enabled"] is True

    pinned = model_mirrors.load()[REPO]
    assert pinned.license == "MIT" and pinned.source_revision == soundscape_model.SOURCE_REVISION
    assert pinned.has_mirror is False
    assert sum(int(meta["size"]) for meta in pinned.files.values()) == 10_092_102_593
    assert REPO not in portable.models_for("all", 0)
    assert REPO not in auto_repair.model_repos()
    assert modules.not_a_runtime_module("soundscape") is True
    assert "narration.soundscape" in user_cli.SETTINGS_HELP


def test_cli_flag_skips_the_mix_and_a_missing_cast_voice_exits_3(tmp_path, monkeypatch):
    real_enabled = soundscape_model.enabled
    lib = _cast(tmp_path, "levi", "miriam", "natan")
    seen = {}

    def fake_run(job, progress, cancel, pause):
        seen["sound"] = job.options.sound
        seen["yo"] = job.options.yo
        return type("R", (), {"out_dir": job.out_dir, "files": []})()

    def fail_download(*_a, **_k):
        raise AssertionError("narrate must not download the soundscape model")

    monkeypatch.setattr(soundscape_model, "enabled", lambda: True)
    monkeypatch.setattr(soundscape_model, "ensure", fail_download)
    monkeypatch.setattr(soundscape_model, "enable", fail_download)
    args = ["narrate", str(FIXTURE), "--voice", "levi", "--out", str(tmp_path / "out"), "--format", "wav"]
    assert user_cli.main(args + ["--no-soundscape"], run_narration_fn=fake_run, library=lib) == 0
    assert seen["sound"] is None and seen["yo"] is False
    assert user_cli.main(args, run_narration_fn=fake_run, library=lib) == 0
    assert isinstance(seen["sound"], soundscape.SoundRequest)

    def plan_fn():
        raise AssertionError("a .vxbook must not ask Gemma")

    assert user_cli.main(args + ["--speakers"], run_narration_fn=fake_run, library=lib, plan_fn=plan_fn) == 0

    only = _cast(tmp_path / "only", "levi")
    code = user_cli.main(
        ["narrate", str(FIXTURE), "--voice", "levi", "--out", str(tmp_path / "missing"), "--format", "wav"],
        library=only,
    )
    assert code == user_cli.EXIT_INPUT

    saved = []

    def fake_enable(*_a, **_k):
        soundscape_model._save(True)
        saved.append("on")
        return tmp_path

    monkeypatch.setattr(soundscape_model, "enable", fake_enable)
    monkeypatch.setattr(soundscape_model, "enabled", real_enabled)
    assert user_cli.main(["settings", "set", "narration.soundscape", "off"]) == 0
    assert soundscape_model.enabled() is False
    assert user_cli.main(["settings", "set", "narration.soundscape", "on"]) == 0
    assert saved == ["on"] and soundscape_model.enabled() is True
