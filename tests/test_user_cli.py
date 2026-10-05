"""User-facing CLI (cli.py): argparse parsing and that narrate/train call the runners (fakes, no GPU)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli as user_cli
from core import audiobook_export as ex
from core import voice_info
from core.errors import DatasetMakerError
from core.narration import NarrationOptions
from core.voice_library import VoiceLibrary
from workers.narration_runner import NarrationJob
from workers.pipeline_runner import KIND_LORA, TaskRequest


def _make_voice(lib_root: Path, name: str = "Model Voice", voice_id: str = "model-voice") -> VoiceLibrary:
    """Build a minimal usable voice library for resolve / list / export tests."""
    src = lib_root / "_src"
    src.mkdir(parents=True)
    (src / "adapter_model.safetensors").write_bytes(b"weights")
    (src / "adapter_config.json").write_text("{}", encoding="utf-8")
    (src / "ref_sample.wav").write_bytes(b"RIFFxxxx")
    (src / "training_meta.json").write_text(json.dumps({"ref_sample_text": "hello"}), encoding="utf-8")
    voice_info.write_voice_json(src, voice_info.build_voice_info(
        name, "english", 60, 3, "Qwen/Base", license="CC-BY-4.0", voice_type="female"))
    lib = VoiceLibrary(lib_root / "lib")
    rec = lib.add_from_adapter(src, name=name)
    # force a stable id when the slug differs
    if rec.id != voice_id and not (lib.root / voice_id).exists():
        rec.path.rename(lib.root / voice_id)
        info = voice_info.read_voice_json(lib.root / voice_id) or {}
        info["id"] = voice_id
        voice_info.write_voice_json(lib.root / voice_id, info)
    return VoiceLibrary(lib_root / "lib")


# --------------------------------------------------------------------------- argparse


def test_parse_narrate_defaults_and_flags():
    ap = user_cli.build_parser()
    args = ap.parse_args(["narrate", "book.txt", "--voice", "[model_voice]", "--out", "/tmp/out"])
    assert args.command == "narrate" and args.book == "book.txt"
    assert args.voice == "[model_voice]" and args.pauses is False and args.ai_disclosure is False
    assert args.formats == [] and args.work_dir is None


def test_parse_narrate_options():
    ap = user_cli.build_parser()
    args = ap.parse_args([
        "narrate", "b.epub", "--voice", "anna", "--out", "out",
        "--format", "mp3,m4b", "--format", "opus", "--pauses", "--ai-disclosure",
        "--work-dir", "/work",
    ])
    assert args.pauses is True and args.ai_disclosure is True
    assert args.formats == ["mp3,m4b", "opus"] and str(args.work_dir) == "/work"
    args2 = ap.parse_args(["narrate", "b.txt", "--voice", "v", "--out", "o", "--no-pauses"])
    assert args2.pauses is False


def test_parse_train_and_voices():
    ap = user_cli.build_parser()
    t = ap.parse_args(["train", "a.wav", "--text", "t.txt", "--name", "Anna", "--type", "female"])
    assert t.command == "train" and t.name == "Anna" and t.voice_type == "female"
    assert str(t.text).endswith("t.txt")
    t2 = ap.parse_args(["train", "clips/"])
    assert t2.text is None
    vl = ap.parse_args(["voices", "list"])
    assert vl._handler == "voices_list"
    ve = ap.parse_args(["voices", "export", "[model_voice]", "--out", "v.zip"])
    assert ve._handler == "voices_export" and ve.voice == "[model_voice]"


def test_parse_formats_aliases_and_unknown():
    assert user_cli.parse_formats([]) == set(ex.DEFAULT_FORMATS)
    assert user_cli.parse_formats(["mp3", "opus_single"]) == {ex.FORMAT_MP3_CHAPTERS, ex.FORMAT_OPUS_SINGLE}
    assert user_cli.parse_formats(["flac,wav"]) == {ex.FORMAT_FLAC_CHAPTERS, ex.FORMAT_WAV_CHAPTERS}
    with pytest.raises(SystemExit):
        user_cli.parse_formats(["bogus"])


def test_is_user_cli():
    assert user_cli.is_user_cli(["main.py", "narrate", "b.txt", "--voice", "v", "--out", "o"])
    assert user_cli.is_user_cli(["Voxprint.exe", "voices", "list"])
    assert not user_cli.is_user_cli(["main.py", "--selftest"])
    assert not user_cli.is_user_cli(["main.py"])


# --------------------------------------------------------------------------- resolve / list / export


def test_resolve_voice_by_id_and_name(tmp_path):
    lib = _make_voice(tmp_path)
    by_id = user_cli.resolve_voice(lib, "model-voice")
    assert by_id.id == "model-voice"
    by_name = user_cli.resolve_voice(lib, "Model Voice")
    assert by_name.id == by_id.id
    with pytest.raises(SystemExit):
        user_cli.resolve_voice(lib, "missing")


def test_voices_list_and_export(tmp_path, capsys):
    lib = _make_voice(tmp_path)
    args = SimpleNamespace()
    assert user_cli.cmd_voices_list(args, library=lib) == 0
    out = capsys.readouterr().out
    assert "model-voice" in out and "Model Voice" in out
    dest = tmp_path / "pack.zip"
    assert user_cli.cmd_voices_export(SimpleNamespace(voice="model-voice", out=dest), library=lib) == 0
    assert dest.is_file() and dest.stat().st_size > 0


# --------------------------------------------------------------------------- narrate / train call runners with fakes


def test_narrate_calls_runner_with_options(tmp_path, monkeypatch):
    lib = _make_voice(tmp_path)
    book = tmp_path / "story.txt"
    book.write_text("Chapter One\n\nHello world. Another sentence.\n", encoding="utf-8")
    out = tmp_path / "audiobooks"
    called = {}

    def fake_run(job, progress, cancel, pause):
        called["job"] = job
        progress  # keep signature used
        assert isinstance(job, NarrationJob)
        assert job.voice.id == "model-voice"
        assert job.out_dir == out
        assert isinstance(job.options, NarrationOptions)
        assert job.options.ai_disclosure is True
        assert job.options.pauses is None  # --no-pauses default
        assert ex.FORMAT_MP3_CHAPTERS in job.options.formats
        return SimpleNamespace(out_dir=out / "story", files=[out / "story" / "story.mp3"])

    code = user_cli.main(
        ["narrate", str(book), "--voice", "model-voice", "--out", str(out),
         "--format", "mp3", "--ai-disclosure", "--no-pauses"],
        run_narration_fn=fake_run, library=lib,
    )
    assert code == 0 and "job" in called


def test_narrate_with_pauses_flag(tmp_path):
    lib = _make_voice(tmp_path)
    book = tmp_path / "b.txt"
    book.write_text("Hi there.\n", encoding="utf-8")
    seen = {}

    def fake_run(job, progress, cancel, pause):
        seen["pauses"] = job.options.pauses
        return SimpleNamespace(out_dir=job.out_dir, files=[])

    assert user_cli.main(
        ["narrate", str(book), "--voice", "Model Voice", "--out", str(tmp_path / "o"), "--pauses"],
        run_narration_fn=fake_run, library=lib,
    ) == 0
    assert seen["pauses"] is not None


def test_train_calls_runner_with_text(tmp_path):
    audio = tmp_path / "rec.wav"
    text = tmp_path / "script.txt"
    audio.write_bytes(b"RIFFdata")
    text.write_text("hello", encoding="utf-8")
    seen = {}

    def fake_task(req, progress, cancel=None, **_kw):
        seen["req"] = req
        assert isinstance(req, TaskRequest)
        assert req.kind == KIND_LORA
        assert req.audio == audio and req.text == text
        assert req.no_transcript is False
        assert req.voice_display_name == "Anna" and req.voice_type == "female"
        return SimpleNamespace(voice_id="anna_female", adapter_path=tmp_path / "adapter",
                               root_dir=tmp_path / "root", warnings=[])

    code = user_cli.main(
        ["train", str(audio), "--text", str(text), "--name", "Anna", "--type", "female",
         "--out", str(tmp_path / "out")],
        run_task_fn=fake_task,
    )
    assert code == 0 and seen["req"].out_root == tmp_path / "out"


def test_train_no_transcript_without_text(tmp_path):
    audio = tmp_path / "rec.wav"
    audio.write_bytes(b"RIFFdata")
    seen = {}

    def fake_task(req, progress, cancel=None, **_kw):
        seen["req"] = req
        assert req.no_transcript is True and req.audio_files == [audio]
        return SimpleNamespace(voice_id="rec", adapter_path=None, root_dir=tmp_path, warnings=["w"])

    assert user_cli.main(["train", str(audio)], run_task_fn=fake_task) == 0
    assert seen["req"].text is None


def test_narrate_missing_book_returns_2(tmp_path, capsys):
    lib = _make_voice(tmp_path)
    code = user_cli.main(
        ["narrate", str(tmp_path / "missing.txt"), "--voice", "model-voice", "--out", str(tmp_path / "o")],
        library=lib, run_narration_fn=lambda *a, **k: None,
    )
    assert code == 2 and "ERROR" in capsys.readouterr().err


def test_narrate_runner_error_is_friendly(tmp_path, capsys):
    lib = _make_voice(tmp_path)
    book = tmp_path / "b.txt"
    book.write_text("Hi.\n", encoding="utf-8")

    def boom(*_a, **_k):
        raise DatasetMakerError("no gpu", details="oom")

    code = user_cli.main(
        ["narrate", str(book), "--voice", "model-voice", "--out", str(tmp_path / "o")],
        library=lib, run_narration_fn=boom,
    )
    err = capsys.readouterr().err
    assert code == 2 and "no gpu" in err and "oom" in err


def test_main_dispatches_user_cli(monkeypatch):
    """``main.main(['…', 'voices', 'list'])`` must not open the GUI."""
    import main as app_main

    called = {}

    def fake_cli(argv):
        called["argv"] = list(argv)
        return 0

    monkeypatch.setattr("cli.main", fake_cli)
    # re-import path used inside main.main: it does `from cli import ...` each call
    import cli as cli_mod
    monkeypatch.setattr(cli_mod, "main", fake_cli)
    monkeypatch.setattr(cli_mod, "is_user_cli", lambda argv: True)
    # Bypass overlay / logging side effects that need paths by injecting after those run —
    # easiest: call the branch logic via main with voices list; activate_overlay is light.
    code = app_main.main(["main.py", "voices", "list"])
    assert code == 0 and called["argv"] == ["voices", "list"]
