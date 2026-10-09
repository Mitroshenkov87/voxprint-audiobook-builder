"""User-facing CLI (cli.py): argparse parsing and that narrate/train call the runners (fakes, no GPU)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli as user_cli
from core import audiobook_export as ex
from core import narration as nr
from core import voice_info
from core.errors import CancelledByUser, DatasetMakerError, ModelDownloadError, OutOfMemoryError_
from core.events import Stage
from core.llm_text import LLMPlan
from core.narration import NarrationOptions, NarrationProgress
from core.voice_library import VoiceLibrary
from infra import auto_repair
from tests.test_llm_text import FakeModel
from tests.test_narration import FakeEngine, FakeFfmpeg
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
    assert args.formats == ["mp3,m4b", "opus"] and args.work_dir == Path("/work")
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
    with pytest.raises(BaseException) as ei:
        user_cli.parse_formats(["bogus"])
    assert getattr(ei.value, "code", None) == user_cli.EXIT_BAD_ARGS


def test_parse_backup_and_restore():
    ap = user_cli.build_parser()
    b = ap.parse_args(["backup", "--out", "D:/bk", "--no-models", "--json"])
    assert b._handler == "backup" and b.no_models and not b.no_voices and b.json_output and b.out == Path("D:/bk")
    r = ap.parse_args(["restore", "--from", "D:/bk", "--link", "--json"])
    assert r._handler == "restore" and r.link and r.json_output and r.src == Path("D:/bk")
    plain = ap.parse_args(["backup", "--out", "out", "--no-voices"])
    assert plain.no_voices and not plain.no_models and not getattr(plain, "json_output", False)


def test_is_user_cli():
    assert user_cli.is_user_cli(["main.py", "narrate", "b.txt", "--voice", "v", "--out", "o"])
    assert user_cli.is_user_cli(["voxprint", "backup", "--out", "d"])
    assert user_cli.is_user_cli(["voxprint", "restore", "--from", "d"])
    assert user_cli.is_user_cli(["Voxprint.exe", "voices", "list"])
    assert user_cli.is_user_cli(["Voxprint.exe", "--version"])
    assert user_cli.is_user_cli(["Voxprint.exe", "--json", "status"])
    assert user_cli.is_user_cli(["Voxprint.exe", "capabilities"])
    assert user_cli.is_user_cli(["Voxprint.exe", "models", "download", "denoise"])
    assert user_cli.is_user_cli(["Voxprint.exe", "revoice", "clip.wav"])
    assert user_cli.is_user_cli(["Voxprint.exe", "--json"])
    assert not user_cli.is_user_cli(["main.py", "--selftest"])
    assert not user_cli.is_user_cli(["main.py", "--install-modules"])
    assert not user_cli.is_user_cli(["main.py", "--modules-status"])
    assert not user_cli.is_user_cli(["main.py"])


# --------------------------------------------------------------------------- resolve / list / export


def test_resolve_voice_by_id_and_name(tmp_path):
    lib = _make_voice(tmp_path)
    by_id = user_cli.resolve_voice(lib, "model-voice")
    assert by_id.id == "model-voice"
    by_name = user_cli.resolve_voice(lib, "Model Voice")
    assert by_name.id == by_id.id
    with pytest.raises(BaseException) as ei:
        user_cli.resolve_voice(lib, "missing")
    assert getattr(ei.value, "code", None) == user_cli.EXIT_INPUT


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
    assert code == 0 and seen["req"].out_root == tmp_path / "out" / "Anna_Voxprint"      # one work folder per voice


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


def test_narrate_missing_book_returns_input_code(tmp_path, capsys):
    lib = _make_voice(tmp_path)
    code = user_cli.main(
        ["narrate", str(tmp_path / "missing.txt"), "--voice", "model-voice", "--out", str(tmp_path / "o")],
        library=lib, run_narration_fn=lambda *a, **k: None,
    )
    err = capsys.readouterr().err
    assert code == user_cli.EXIT_INPUT and "ERROR" in err and "Fix:" in err


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
    assert code == user_cli.EXIT_INTERNAL and "no gpu" in err and "oom" in err


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


def _json_lines(text: str):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_exit_codes_are_stable():
    assert user_cli.EXIT_OK == 0
    assert user_cli.EXIT_INTERNAL == 1
    assert user_cli.EXIT_BAD_ARGS == 2
    assert user_cli.EXIT_INPUT == 3
    assert user_cli.EXIT_MISSING == 4
    assert user_cli.EXIT_GPU == 5
    assert user_cli.EXIT_CANCELLED == 6


def test_version_prints_build_json_codename(capsys):
    meta = json.loads(Path("BUILD.json").read_text(encoding="utf-8"))
    assert user_cli.main(["--version"]) == 0
    out = capsys.readouterr().out
    info = user_cli.version_payload()
    assert info["codename"] == meta["codename"]
    assert meta["codename"] in out
    assert str(info["build"]) in out
    assert info["version"].split("-")[0] in out
    assert "build" in out


def test_version_codename_prefers_build_json(monkeypatch):
    import core.appinfo as appinfo

    monkeypatch.setattr(appinfo, "APP_CODENAME", "NotFromFile")
    meta = json.loads(Path("BUILD.json").read_text(encoding="utf-8"))
    assert user_cli.version_payload()["codename"] == meta["codename"]


def test_version_json(capsys):
    assert user_cli.main(["--json", "--version"]) == 0
    data = json.loads(capsys.readouterr().out)
    info = user_cli.version_payload()
    assert data["ok"] is True and data["command"] == "version"
    assert data["codename"] == info["codename"]
    assert data["build"] == info["build"] and isinstance(data["build"], int)
    assert data["version"] == info["version"]


def test_help_is_layered_and_has_examples(capsys):
    assert user_cli.main(["--help"]) == 0
    top = capsys.readouterr().out
    assert "Examples:" in top and "--json" in top and "--version" in top
    assert "narrate" in top and "status" in top
    assert "--voice" not in top
    assert user_cli.main(["narrate", "--help"]) == 0
    narrate = capsys.readouterr().out
    assert "Examples:" in narrate and "--voice" in narrate and "--json" in narrate
    assert "voxprint narrate" in narrate
    assert user_cli.main(["models", "download", "--help"]) == 0
    download = capsys.readouterr().out
    assert "Examples:" in download and "denoise" in download


def test_bad_args_exit_and_json(capsys):
    assert user_cli.main(["narrate"]) == user_cli.EXIT_BAD_ARGS
    assert user_cli.main(["--json", "narrate"]) == user_cli.EXIT_BAD_ARGS
    data = _json_lines(capsys.readouterr().out)[-1]
    assert data["ok"] is False and data["exit_code"] == user_cli.EXIT_BAD_ARGS
    assert data["hint"]


def test_yes_is_accepted_without_a_prompt(tmp_path, capsys):
    lib = _make_voice(tmp_path)
    assert user_cli.main(["--yes", "voices", "list"], library=lib) == 0
    assert "model-voice" in capsys.readouterr().out


def test_voices_list_json(tmp_path, capsys):
    lib = _make_voice(tmp_path)
    assert user_cli.main(["voices", "list", "--json"], library=lib) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is True and data["command"] == "voices list"
    voice = data["voices"][0]
    assert voice["id"] == "model-voice" and voice["license"]
    assert "consent_scope" in voice and "duration_s" in data


def test_narrate_json_progress_and_result(tmp_path, capsys):
    lib = _make_voice(tmp_path)
    book = tmp_path / "story.txt"
    book.write_text("Hello world.\n", encoding="utf-8")
    out = tmp_path / "audiobooks"
    target = out / "story" / "story.mp3"

    def fake_run(job, progress, cancel, pause):
        progress(NarrationProgress(1, 2, 12.0, "chapter", "synth"))
        return SimpleNamespace(out_dir=out / "story", files=[target])

    code = user_cli.main(
        ["--json", "narrate", str(book), "--voice", "model-voice", "--out", str(out), "--format", "mp3"],
        run_narration_fn=fake_run, library=lib,
    )
    assert code == 0
    lines = _json_lines(capsys.readouterr().out)
    assert lines[0]["type"] == "progress" and lines[0]["stage"] == "synth"
    assert 0 <= lines[0]["percent"] <= 100
    result = lines[-1]
    assert result["type"] == "result" and result["ok"] is True
    assert str(target) in result["outputs"]
    assert "duration_s" in result and result["warnings"] == []


def test_train_json_progress(tmp_path, capsys):
    audio = tmp_path / "rec.wav"
    audio.write_bytes(b"RIFFdata")

    def fake_task(req, progress, cancel=None, **_kw):
        progress(Stage.TRAIN, 0.4, "epoch 1")
        return SimpleNamespace(voice_id="anna", adapter_path=tmp_path / "adapter",
                               root_dir=tmp_path / "root", warnings=["check the clips"])

    code = user_cli.main(["train", str(audio), "--json"], run_task_fn=fake_task)
    assert code == 0
    lines = _json_lines(capsys.readouterr().out)
    assert lines[0]["type"] == "progress" and lines[0]["stage"] == "train"
    assert 0 <= lines[0]["percent"] <= 100
    result = lines[-1]
    assert result["voice_id"] == "anna" and result["warnings"] == ["check the clips"]
    assert "duration_s" in result


def test_classified_exit_codes(tmp_path, capsys):
    lib = _make_voice(tmp_path)
    book = tmp_path / "b.txt"
    book.write_text("Hi.\n", encoding="utf-8")
    out = str(tmp_path / "o")
    base = ["narrate", str(book), "--voice", "model-voice", "--out", out]

    def run(exc):
        return user_cli.main(base, library=lib, run_narration_fn=lambda *_a, **_k: (_ for _ in ()).throw(exc))

    assert run(OutOfMemoryError_("Not enough video memory (VRAM).", details="cuda")) == user_cli.EXIT_GPU
    err = capsys.readouterr().err
    assert "status" in err
    assert run(CancelledByUser()) == user_cli.EXIT_CANCELLED
    assert run(ModelDownloadError("model missing", details="tts")) == user_cli.EXIT_MISSING
    assert "models download" in capsys.readouterr().err
    assert run(RuntimeError("boom")) == user_cli.EXIT_INTERNAL
    assert "diag" in capsys.readouterr().err


def test_missing_voice_names_the_fix(tmp_path, capsys):
    lib = _make_voice(tmp_path)
    book = tmp_path / "b.txt"
    book.write_text("Hi.\n", encoding="utf-8")
    code = user_cli.main(
        ["narrate", str(book), "--voice", "no-such-voice", "--out", str(tmp_path / "o"), "--json"],
        library=lib, run_narration_fn=lambda *_a, **_k: None,
    )
    assert code == user_cli.EXIT_INPUT
    captured = capsys.readouterr()
    assert "voices list" in captured.err
    data = _json_lines(captured.out)[-1]
    assert data["exit_code"] == user_cli.EXIT_INPUT and data["hint"]


def test_status_and_capabilities_json(tmp_path, capsys):
    lib = _make_voice(tmp_path)
    assert user_cli.main(["status", "--json"], library=lib) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["command"] == "status" and data["ok"] is True
    assert data["codename"] == user_cli.version_payload()["codename"]
    assert isinstance(data["gpu"]["cuda_available"], bool)
    assert any(v["id"] == "model-voice" for v in data["voices"])
    ids = {m["id"] for m in data["modules"]}
    assert {"aligner", "denoise", "llm", "dnsmos", "tts-1.7b"} <= ids
    assert "opus_single" in data["formats"]
    assert data["format_aliases"]["mp3"]
    assert user_cli.main(["capabilities"], library=lib) == 0
    again = json.loads(capsys.readouterr().out)
    assert again["command"] == "capabilities" and again["modules"]


def test_models_list_json(capsys):
    assert user_cli.main(["models", "list", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    ids = {m["id"] for m in data["modules"]}
    assert "denoise" in ids and "sage-ru" in ids
    assert all("installed" in m and "optional" in m for m in data["modules"])


def test_models_download_unknown_names_choices(capsys):
    code = user_cli.main(["models", "download", "not-a-module"])
    assert code == user_cli.EXIT_BAD_ARGS
    err = capsys.readouterr().err
    assert "denoise" in err and "Example:" in err


def test_models_download_json_and_idempotent(tmp_path, capsys):
    seen = {}

    def fake_download(module_id, progress):
        seen["id"] = module_id
        progress(0.4, "fetch")
        return str(tmp_path / "deep-filter")

    code = user_cli.main(
        ["models", "download", "deepfilternet", "--json"],
        download_fn=fake_download,
    )
    assert code == 0 and seen["id"] == "denoise"
    lines = _json_lines(capsys.readouterr().out)
    assert lines[0]["type"] == "progress" and "percent" in lines[0] and lines[0]["stage"] == "download"
    assert lines[-1]["downloaded"] is True and lines[-1]["ok"] is True

    def boom(*_a, **_k):
        raise AssertionError("must not download twice")

    code = user_cli.main(
        ["--json", "models", "download", "denoise", "--yes"],
        download_fn=boom, installed_fn=lambda _module: True,
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["downloaded"] is False and data["installed"] is True


def test_fetch_module_dispatches_without_network(monkeypatch, tmp_path):
    import infra.denoise_tool as denoise_tool
    import infra.text_models as text_models

    def ensure(progress, **_kwargs):
        progress(1.0, "ok")
        return tmp_path / "deep-filter"

    monkeypatch.setattr(denoise_tool, "ensure", ensure)
    info = user_cli.fetch_module("denoise", lambda _f, _m: None)
    assert info["module"] == "denoise" and info["downloaded"] is True

    def already(*_a, **_k):
        return tmp_path / "deep-filter"

    monkeypatch.setattr(denoise_tool, "ready", already)

    def fail_ensure(*_a, **_k):
        raise AssertionError("already installed")

    monkeypatch.setattr(denoise_tool, "ensure", fail_ensure)
    again = user_cli.fetch_module("denoise", lambda _f, _m: None)
    assert again["downloaded"] is False and again["installed"] is True

    def fetch_text(model, progress, **_kw):
        assert model.key == "sage-ru"
        progress(None, 1.0, "done")
        return tmp_path / "sage"

    monkeypatch.setattr(text_models, "state", lambda _model: text_models.STATE_NEEDS_DOWNLOAD)
    monkeypatch.setattr(text_models, "ensure", fetch_text)
    got = user_cli.fetch_module("sage-ru", lambda _f, _m: None)
    assert got["downloaded"] is True and got["module"] == "sage-ru"


def test_revoice_writes_a_book(tmp_path, capsys):
    audio = tmp_path / "01 - Intro.wav"
    audio.write_bytes(b"not-a-real-wav")
    out = tmp_path / "revoice"

    def fake(files, language, progress, cancel):
        assert files[0] == audio and language is None
        progress(1.0, "Intro")
        return [("Intro", "Hello from the recording.")]

    code = user_cli.main(
        ["revoice", str(audio), "--out", str(out), "--json"],
        transcribe_fn=fake,
    )
    assert code == 0
    lines = _json_lines(capsys.readouterr().out)
    assert lines[0]["stage"] == "transcribe"
    text_path = Path(lines[-1]["outputs"][0])
    assert text_path.is_file()
    body = text_path.read_text(encoding="utf-8")
    assert body.startswith("# Intro") and "Hello from the recording." in body


def test_diag_json_and_human(tmp_path, monkeypatch, capsys):
    import infra.diagnostics as dg

    made = []
    monkeypatch.setattr(dg, "system_info", lambda **_kw: {"app_version": "x", "gpu": {}})
    monkeypatch.setattr(dg, "summary_lines", lambda _info: ["Voxprint x"])
    monkeypatch.setattr(dg, "write_report", lambda p, **_kw: made.append(Path(p)) or Path(p))
    dest = tmp_path / "d.zip"
    assert user_cli.main(["diag", "--out", str(dest)]) == 0
    assert made == [dest]
    assert "Diagnostic report" in capsys.readouterr().out
    assert user_cli.main(["diag", "--out", str(dest), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is True and data["command"] == "diag"
    assert str(dest) in data["outputs"]
    assert data["summary"] == ["Voxprint x"]


def test_revoice_missing_audio(tmp_path, capsys):
    code = user_cli.main(
        ["revoice", str(tmp_path / "missing.wav"), "--out", str(tmp_path / "out")],
        transcribe_fn=lambda *_a, **_k: [],
    )
    assert code == user_cli.EXIT_INPUT
    assert "Fix:" in capsys.readouterr().err


# --------------------------------------------------------------------------- speakers, multi-voice narrate, check


DIALOGUE = (
    "He walked along the quiet road for a while.\n\n"
    '"Hello there," she said softly.\n\n'
    '"Good day to you," he answered at once.\n'
)


def _add_voice(lib: VoiceLibrary, name: str, voice_id: str, voice_type: str = "male") -> None:
    src = lib.root.parent / f"_src_{voice_id}"
    src.mkdir(parents=True, exist_ok=True)
    (src / "adapter_model.safetensors").write_bytes(b"weights")
    (src / "adapter_config.json").write_text("{}", encoding="utf-8")
    (src / "ref_sample.wav").write_bytes(b"RIFFxxxx")
    (src / "training_meta.json").write_text(json.dumps({"ref_sample_text": "hello"}), encoding="utf-8")
    voice_info.write_voice_json(src, voice_info.build_voice_info(
        name, "english", 60, 3, "Qwen/Base", license="CC-BY-4.0", voice_type=voice_type))
    rec = lib.add_from_adapter(src, name=name)
    if rec.id != voice_id and not (lib.root / voice_id).exists():
        rec.path.rename(lib.root / voice_id)
        info = voice_info.read_voice_json(lib.root / voice_id) or {}
        info["id"] = voice_id
        voice_info.write_voice_json(lib.root / voice_id, info)


def _cast_library(tmp_path: Path) -> VoiceLibrary:
    lib = _make_voice(tmp_path, name="Narrator", voice_id="narrator")
    _add_voice(lib, "Tom", "tom", "male")
    _add_voice(lib, "Ann", "ann", "female")
    return VoiceLibrary(lib.root)


def _marks_answer(prompt: str) -> str:
    text = prompt.rsplit("TEXT:\n", 1)[-1]
    n = len([p for p in text.split("\n\n") if p.strip()])
    roles = ["NARRATOR", "FEMALE: Ann", "MALE: Tom"]
    return "\n".join(roles[:n])


def _plan():
    return LLMPlan(lambda: FakeModel(_marks_answer), "fake")


def _narrate_with_fakes(engines: dict):
    """``run_narration`` stand-in: the real ``narrate_book`` with silent engines. No GPU."""

    def run(job, progress, cancel, pause):
        narr = FakeEngine()
        narr.tag = "narr"
        engines["narr"] = narr
        extra = {}
        for vid in job.extra_voices:
            eng = FakeEngine()
            eng.tag = vid
            engines[vid] = eng
            extra[vid] = (lambda e=eng: e, vid)
        return nr.narrate_book(
            job.book, lambda: narr, "narr", job.out_dir,
            language="english", narrator=job.voice.name, options=job.options,
            progress=progress, cancel=cancel, pause=pause,
            ffmpeg="ffmpeg", run=FakeFfmpeg(), extra_engines=extra or None,
        )

    return run


def test_parse_speakers_and_check():
    ap = user_cli.build_parser()
    args = ap.parse_args([
        "narrate", "book.txt", "--voice", "narrator", "--out", "out",
        "--speakers", "--male-voice", "tom", "--female-voice", "ann",
    ])
    assert args.speakers and args.male_voice == "tom" and args.female_voice == "ann"
    assert args.speaker_marks is None
    marks = ap.parse_args([
        "narrate", "book.txt", "--voice", "narrator", "--out", "out", "--speaker-marks", "marks.txt",
    ])
    assert marks.speaker_marks == Path("marks.txt") and marks.speakers is False
    plain = ap.parse_args(["narrate", "book.txt", "--voice", "v", "--out", "o"])
    assert plain.speakers is False and plain.male_voice == "" and plain.female_voice == ""
    sp = ap.parse_args(["speakers", "book.txt", "--out", "marks.txt", "--json"])
    assert sp._handler == "speakers" and sp.out == Path("marks.txt") and sp.json_output
    repair = ap.parse_args(["repair", "--json"])
    assert repair.command == "repair" and repair._handler == "check" and repair.json_output


def test_is_user_cli_speakers_and_check():
    assert user_cli.is_user_cli(["Voxprint.exe", "speakers", "book.txt", "--out", "marks.txt"])
    assert user_cli.is_user_cli(["Voxprint.exe", "check"])
    assert user_cli.is_user_cli(["Voxprint.exe", "--json", "repair"])
    assert not user_cli.is_user_cli(["Voxprint.exe", "--auto-repair"])


def test_speakers_command_writes_marks_the_narrate_flag_reads(tmp_path, capsys):
    from core import speakers as spk

    book = tmp_path / "book.txt"
    book.write_text(DIALOGUE, encoding="utf-8")
    dest = tmp_path / "marks.txt"
    code = user_cli.main(
        ["speakers", str(book), "--out", str(dest), "--json"],
        plan_fn=_plan,
    )
    assert code == 0
    data = _json_lines(capsys.readouterr().out)[-1]
    assert data["command"] == "speakers" and data["ok"] and data["paragraphs"] == 3
    assert str(dest) in data["outputs"]
    lines = spk.load_marks(dest.read_text(encoding="utf-8"))
    assert [ln.role for ln in lines] == ["narrator", "female", "male"]
    dest.write_text("1. NARRATOR\n2. MALE: Tom\n3. FEMALE: Ann\n", encoding="utf-8")
    assert [ln.role for ln in spk.load_marks(dest.read_text(encoding="utf-8"))] == ["narrator", "male", "female"]


def test_speakers_missing_model_is_exit_4(tmp_path, capsys):
    book = tmp_path / "book.txt"
    book.write_text("Hello there, this is a paragraph.\n", encoding="utf-8")
    code = user_cli.main(
        ["speakers", str(book), "--out", str(tmp_path / "marks.txt"), "--json"],
        plan_fn=lambda: None,
    )
    assert code == user_cli.EXIT_MISSING
    err = capsys.readouterr()
    data = _json_lines(err.out)[-1]
    assert data["ok"] is False and data["exit_code"] == 4
    assert "models download llm" in (data["hint"] or "")
    assert "Fix:" in err.err


def test_narrate_speaker_flags_reject_bad_combinations(tmp_path, capsys):
    lib = _cast_library(tmp_path)
    book = tmp_path / "book.txt"
    book.write_text(DIALOGUE, encoding="utf-8")
    marks = tmp_path / "marks.txt"
    marks.write_text("1. NARRATOR\n2. FEMALE: Ann\n3. MALE: Tom\n", encoding="utf-8")
    out = tmp_path / "audiobooks"
    base = ["narrate", str(book), "--voice", "narrator", "--out", str(out), "--json"]

    both = user_cli.main(
        base + ["--speakers", "--speaker-marks", str(marks), "--male-voice", "tom"],
        library=lib, plan_fn=_plan,
    )
    assert both == user_cli.EXIT_BAD_ARGS
    bare = user_cli.main(base + ["--male-voice", "tom", "--female-voice", "ann"], library=lib)
    assert bare == user_cli.EXIT_BAD_ARGS
    same = user_cli.main(base + ["--speakers", "--male-voice", "narrator"], library=lib, plan_fn=_plan)
    assert same == user_cli.EXIT_BAD_ARGS
    missing = user_cli.main(
        base + ["--speaker-marks", str(tmp_path / "no-such.txt"), "--female-voice", "ann"],
        library=lib,
    )
    assert missing == user_cli.EXIT_INPUT
    bad = tmp_path / "bad.txt"
    bad.write_text("1. NARRATOR\nhello\n", encoding="utf-8")
    unreadable = user_cli.main(base + ["--speaker-marks", str(bad), "--female-voice", "ann"], library=lib)
    assert unreadable == user_cli.EXIT_INPUT
    err = capsys.readouterr()
    assert "Fix:" in err.err
    assert any(row.get("exit_code") == 2 for row in _json_lines(err.out))


def test_narrate_from_marks_uses_the_core_path(tmp_path, capsys):
    lib = _cast_library(tmp_path)
    book = tmp_path / "book.txt"
    book.write_text(DIALOGUE, encoding="utf-8")
    marks = tmp_path / "marks.txt"
    marks.write_text("1. NARRATOR\n2. FEMALE: Ann\n3. MALE: Tom\n", encoding="utf-8")
    engines = {}
    code = user_cli.main(
        ["narrate", str(book), "--voice", "narrator", "--male-voice", "tom", "--female-voice", "ann",
         "--speaker-marks", str(marks), "--out", str(tmp_path / "audiobooks"), "--format", "wav", "--json"],
        run_narration_fn=_narrate_with_fakes(engines), library=lib,
    )
    assert code == 0
    data = _json_lines(capsys.readouterr().out)[-1]
    assert data["ok"] and data["warnings"] == []
    speakers = [p for p in data["outputs"] if p.endswith("speakers.txt")]
    assert len(speakers) == 1 and Path(speakers[0]).is_file()
    text = Path(speakers[0]).read_text(encoding="utf-8")
    assert "NARRATOR" in text and "FEMALE" in text and "MALE" in text
    assert any("walked" in c for c in engines["narr"].calls)
    assert any("Hello" in c for c in engines["ann"].calls)
    assert any("Good day" in c for c in engines["tom"].calls)


def test_narrate_speakers_flag_asks_the_text_model(tmp_path, capsys):
    lib = _cast_library(tmp_path)
    book = tmp_path / "book.txt"
    book.write_text(DIALOGUE, encoding="utf-8")
    engines = {}
    code = user_cli.main(
        ["narrate", str(book), "--voice", "narrator", "--male-voice", "tom", "--female-voice", "ann",
         "--speakers", "--out", str(tmp_path / "audiobooks"), "--format", "wav", "--json"],
        run_narration_fn=_narrate_with_fakes(engines), library=lib, plan_fn=_plan,
    )
    assert code == 0, capsys.readouterr().err
    data = _json_lines(capsys.readouterr().out)[-1]
    assert data["warnings"] == []
    assert any("Hello" in c for c in engines["ann"].calls)
    assert any("Good day" in c for c in engines["tom"].calls)
    note = next(p for p in data["outputs"] if p.endswith("speakers.txt"))
    assert Path(note).is_file()


def test_narrate_mismatch_warns_and_the_narrator_reads_all(tmp_path, capsys):
    lib = _cast_library(tmp_path)
    book = tmp_path / "book.txt"
    book.write_text(DIALOGUE, encoding="utf-8")
    marks = tmp_path / "short.txt"
    marks.write_text("1. FEMALE: Ann\n", encoding="utf-8")
    engines = {}
    code = user_cli.main(
        ["narrate", str(book), "--voice", "narrator", "--female-voice", "ann",
         "--speaker-marks", str(marks), "--out", str(tmp_path / "audiobooks"), "--format", "wav", "--json"],
        run_narration_fn=_narrate_with_fakes(engines), library=lib,
    )
    assert code == 0
    data = _json_lines(capsys.readouterr().out)[-1]
    assert data["warnings"] == [user_cli.SPEAKER_MISMATCH]
    note = Path(next(p for p in data["outputs"] if p.endswith("speakers.txt")))
    assert "mismatch" in note.read_text(encoding="utf-8")
    assert engines["ann"].calls == []
    heard = " ".join(engines["narr"].calls)
    assert "Hello" in heard and "Good day" in heard and "walked" in heard


def test_check_and_repair_json(tmp_path, capsys):
    def ok_report(progress):
        progress(0.4, "checking models")
        progress(1.0, "done")
        return auto_repair.Report([
            auto_repair.Item("model", "gemma", auto_repair.OK),
            auto_repair.Item("lock", ".gemma.lock", auto_repair.REPAIRED, "removed"),
        ])

    code = user_cli.main(["check", "--json"], repair_fn=ok_report)
    assert code == 0
    lines = _json_lines(capsys.readouterr().out)
    assert lines[0]["type"] == "progress" and lines[0]["stage"] == "check"
    data = lines[-1]
    assert data["command"] == "check" and data["ok"] and data["checked"] == 2 and data["fixed"] == 1
    assert data["failed"] == 0 and data["items"][1]["status"] == "repaired"

    def bad_report(progress):
        progress(1.0, "done")
        return auto_repair.Report([
            auto_repair.Item("model", "tts", auto_repair.FAILED, "still bad"),
        ])

    code = user_cli.main(["repair", "--json"], repair_fn=bad_report)
    assert code == user_cli.EXIT_INTERNAL
    data = _json_lines(capsys.readouterr().out)[-1]
    assert data["command"] == "check" and data["ok"] is False and data["exit_code"] == 1
    assert data["failed"] == 1 and any("tts" in w for w in data["warnings"])


def test_json_flushes_and_a_missing_console_keeps_the_exit_code(tmp_path, monkeypatch, capsys):
    class FlushStream:
        def __init__(self):
            self.buf = []
            self.flushed = 0

        def write(self, text):
            self.buf.append(text)
            return len(text)

        def flush(self):
            self.flushed += 1

    stream = FlushStream()
    monkeypatch.setattr(sys, "stdout", stream)
    lib = _make_voice(tmp_path)
    code = user_cli.main(["voices", "list", "--json"], library=lib)
    assert code == 0 and stream.flushed >= 1
    data = json.loads("".join(stream.buf).strip().splitlines()[-1])
    assert data["ok"] and data["command"] == "voices list"

    class DeadStream:
        def write(self, _text):
            raise OSError("no console")

        def flush(self):
            raise OSError("no console")

    monkeypatch.setattr(sys, "stdout", DeadStream())
    monkeypatch.setattr(sys, "stderr", DeadStream())
    assert user_cli.main(["voices", "list", "--json"], library=lib) == 0
    assert user_cli.main(["check", "--json"], repair_fn=lambda progress: auto_repair.Report([])) == 0


def test_errno_22_invalid_handle_exits_without_a_traceback_window(tmp_path, monkeypatch):
    """PowerShell ``& Voxprint.exe status --json > file`` leaves a non-None stdout whose write raises errno 22."""
    from infra.stdio_guard import install_cli_excepthook

    class InvalidHandle:
        def write(self, _text):
            raise OSError(22, "Invalid argument")

        def flush(self):
            raise OSError(22, "Invalid argument")

        def fileno(self):
            return 1

    shown = []
    monkeypatch.setenv("VOXPRINT_NO_ENV_PROBE", "1")
    monkeypatch.setattr(sys, "__excepthook__", lambda *args: shown.append(args))
    monkeypatch.setattr(sys, "stdout", InvalidHandle())
    monkeypatch.setattr(sys, "stderr", InvalidHandle())
    lib = _make_voice(tmp_path)
    assert user_cli.main(["voices", "list", "--json"], library=lib) == 0
    assert user_cli.main(["status", "--json"], library=lib) == 0
    sys.stdout.flush()
    sys.stderr.flush()
    assert shown == []
    previous = sys.excepthook
    install_cli_excepthook()
    try:
        sys.excepthook(RuntimeError, RuntimeError("boom"), None)
        assert shown == []
    finally:
        sys.excepthook = previous


def test_speakers_json_warns_and_saves_the_raw_reply(tmp_path, capsys):
    from core import speakers as spk
    from core.book_parsers import parse_txt
    from core.llm_text import LLMPlan
    from tests.test_llm_text import FakeModel
    from tests.test_speakers import _FIXTURE, _expected_roles, _gemma_reply

    book = parse_txt(_FIXTURE.read_text(encoding="utf-8"), "spor")
    paras = [text for _ci, text in spk.paragraphs(book)]
    dest = tmp_path / "marks.txt"

    def plan():
        return LLMPlan(lambda: FakeModel(lambda prompt: _gemma_reply(prompt, paras)), "fake")

    code = user_cli.main(["speakers", str(_FIXTURE), "--out", str(dest), "--json"], plan_fn=plan)
    assert code == 0
    data = _json_lines(capsys.readouterr().out)[-1]
    assert data["ok"] and data["warnings"] == [] and data["paragraphs"] == 21
    raw = dest.with_name("marks.speakers-raw.txt")
    assert str(dest) in data["outputs"] and str(raw) in data["outputs"]
    assert raw.is_file() and "<|channel>thought" in raw.read_text(encoding="utf-8")
    assert spk.load_marks(dest.read_text(encoding="utf-8")) == _expected_roles(21)

    def garbage(_prompt):
        return "I cannot tell who is speaking."

    bad = tmp_path / "bad.txt"
    code = user_cli.main(
        ["speakers", str(_FIXTURE), "--out", str(bad), "--json"],
        plan_fn=lambda: LLMPlan(lambda: FakeModel(garbage), "fake"),
    )
    assert code == 0
    data = _json_lines(capsys.readouterr().out)[-1]
    assert data["warnings"] == [user_cli.SPEAKER_UNPARSED]
    saved = bad.with_name("bad.speakers-raw.txt")
    assert saved.is_file() and "cannot tell" in saved.read_text(encoding="utf-8")
    assert user_cli.speaker_warning_lines("unparsed;no_speakers") == [
        user_cli.SPEAKER_UNPARSED, user_cli.SPEAKER_NO_SPEAKERS]


def test_narrate_second_male_voice_and_character_pins(tmp_path, capsys):
    lib = _make_voice(tmp_path, name="Narrator", voice_id="narrator")
    for name, vid, kind in (("Tom", "tom", "male"), ("Bob", "bob", "male"), ("Ann", "ann", "female")):
        _add_voice(lib, name, vid, kind)
    lib = VoiceLibrary(lib.root)
    book = tmp_path / "book.txt"
    book.write_text(
        'They sat down.\n\n"I start," David said.\n\n"I follow," Michael said.\n\n"And I," Hannah said.\n',
        encoding="utf-8")
    marks = tmp_path / "marks.txt"
    marks.write_text("1. NARRATOR\n2. MALE: David\n3. MALE: Michael\n4. FEMALE: Hannah\n", encoding="utf-8")
    base = ["narrate", str(book), "--voice", "narrator", "--speaker-marks", str(marks),
            "--out", str(tmp_path / "audiobooks"), "--format", "wav", "--json"]
    engines = {}
    code = user_cli.main(base + ["--male-voice", "tom", "--male2-voice", "bob", "--female-voice", "ann"],
                         run_narration_fn=_narrate_with_fakes(engines), library=lib)
    assert code == 0, capsys.readouterr().err
    capsys.readouterr()
    assert any("I start" in c for c in engines["tom"].calls)
    assert any("I follow" in c for c in engines["bob"].calls)
    assert any("And I" in c for c in engines["ann"].calls)

    engines = {}
    code = user_cli.main(base + ["--male-voice", "tom", "--character", "Michael=Bob", "--character", "David=tom"],
                         run_narration_fn=_narrate_with_fakes(engines), library=lib)
    assert code == 0
    capsys.readouterr()
    assert any("I follow" in c for c in engines["bob"].calls)
    assert any("I start" in c for c in engines["tom"].calls)
    assert any("And I" in c for c in engines["narr"].calls)

    assert user_cli.main(base + ["--male2-voice", "bob"], library=lib) == 2
    assert user_cli.main(base + ["--male-voice", "tom", "--character", "Michael"], library=lib) == 2
    plain = ["narrate", str(book), "--voice", "narrator", "--out", str(tmp_path / "o")]
    assert user_cli.main(plain + ["--character", "David=tom"], library=lib) == 2
    capsys.readouterr()


def test_resolve_voice_returns_the_library_id_when_the_folder_lookup_ignores_case():
    """Windows: get("Bob") opens the folder "bob" and names the record "Bob". The listed record wins."""
    listed = SimpleNamespace(id="bob", name="Bob")

    class Lib:
        def get(self, needle):
            return SimpleNamespace(id=needle, name="Bob") if needle.lower() == "bob" else None

        def list_voices(self):
            return [listed]

    assert user_cli.resolve_voice(Lib(), "Bob") is listed
    assert user_cli.resolve_voice(Lib(), "bob") is listed


def test_voices_catalog_and_download(tmp_path, capsys):
    """``voices catalog`` lists the index, ``voices download`` imports one entry once (fake index and download)."""
    from infra import voice_repository as repo

    lib = _make_voice(tmp_path)
    entries = repo.parse_index({"schema": repo.INDEX_SCHEMA, "voices": [
        {"id": "eitan", "name": "Eitan", "language": "ru", "license": "CC0-1.0", "gender": "male",
         "url": "https://example.org/eitan.zip", "sha256": "0" * 64, "size_bytes": 53_000_000},
        {"id": "noa", "name": "Noa", "language": "ru", "license": "CC0-1.0", "gender": "female",
         "url": "https://example.org/noa.zip", "sha256": "1" * 64, "size_bytes": 53_000_000}]})
    fetch = lambda: repo.IndexResult(voices=entries)
    parse = user_cli.build_parser().parse_args

    assert user_cli.cmd_voices_catalog(parse(["voices", "catalog", "--json"]), library=lib, fetch_fn=fetch) == 0
    rows = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["voices"]
    assert [(r["id"], r["installed"]) for r in rows] == [("eitan", False), ("noa", False)]

    calls = []

    def fake_download(entry, library, progress=None):
        calls.append(entry.id)
        rec = library.list_voices()[0]
        rec.info["repo_id"] = entry.id
        return rec

    code = user_cli.cmd_voices_download(parse(["voices", "download", "Eitan", "--json"]), library=lib, fetch_fn=fetch,
                                        download_fn=fake_download)
    assert code == 0 and calls == ["eitan"]
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["downloaded"] is True
    assert user_cli.cmd_voices_download(parse(["voices", "download", "nobody"]), library=lib, fetch_fn=fetch) == 3
    empty = lambda: repo.IndexResult(error="unreachable")
    assert user_cli.cmd_voices_catalog(parse(["voices", "catalog"]), library=lib, fetch_fn=empty) == 4


def test_narrate_speakers_alone_uses_the_default_cast(tmp_path, capsys):
    """Build 703: ``--speakers`` without voice flags picks the shipped cast (here: Natan and Miriam), else the first voice of
    each gender. Tom and Ann come first in the library but are not the shipped cast."""
    lib = _cast_library(tmp_path)
    _add_voice(lib, "Natan", "natan", "male")
    _add_voice(lib, "Miriam", "miriam", "female")
    lib = VoiceLibrary(lib.root)
    book = tmp_path / "book.txt"
    book.write_text(DIALOGUE, encoding="utf-8")
    engines = {}
    code = user_cli.main(
        ["narrate", str(book), "--voice", "narrator", "--speakers", "--out", str(tmp_path / "audiobooks"),
         "--format", "wav", "--json"],
        run_narration_fn=_narrate_with_fakes(engines), library=lib, plan_fn=_plan,
    )
    assert code == 0, capsys.readouterr().err
    assert any("Hello" in c for c in engines["miriam"].calls)
    assert any("Good day" in c for c in engines["natan"].calls)
    assert "ann" not in engines                       # Tom may stand in for the missing Shimon as the second male voice


def test_narrate_speakers_alone_falls_back_to_first_voices(tmp_path, capsys):
    lib = _cast_library(tmp_path)
    book = tmp_path / "book.txt"
    book.write_text(DIALOGUE, encoding="utf-8")
    engines = {}
    code = user_cli.main(
        ["narrate", str(book), "--voice", "narrator", "--speakers", "--out", str(tmp_path / "audiobooks"),
         "--format", "wav", "--json"],
        run_narration_fn=_narrate_with_fakes(engines), library=lib, plan_fn=_plan,
    )
    assert code == 0, capsys.readouterr().err
    assert any("Hello" in c for c in engines["ann"].calls)
    assert any("Good day" in c for c in engines["tom"].calls)


def test_revoice_language_accepts_iso_codes(tmp_path, capsys):
    """Build 702 bug: ``--language ru`` reached Qwen3-ASR as "ru". Codes and names in any case become the ASR's name."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    seen = []

    def fake(files, language, progress, cancel):
        seen.append(language)
        return [("A", "text")]

    for value, want in (("ru", "Russian"), ("EN", "English"), ("de-DE", "German"), ("Russian", "Russian"), ("auto", None)):
        assert user_cli.main(["revoice", str(audio), "--out", str(tmp_path / "o"), "--language", value, "--json"],
                             transcribe_fn=fake) == 0
        assert seen[-1] == want
    capsys.readouterr()
    assert user_cli.main(["revoice", str(audio), "--out", str(tmp_path / "o"), "--language", "xx", "--json"],
                         transcribe_fn=fake) == user_cli.EXIT_BAD_ARGS
    assert len(seen) == 5
