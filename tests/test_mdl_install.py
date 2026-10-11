"""VOXPrint MDL: the app manifest, the ready gate, voice migration, and release-tag checks."""
import io
import json
from pathlib import Path

import pytest

import cli
from infra import mdl_gate, paths
from tools import check_manifest_tags as tags

ROOT = Path(__file__).resolve().parents[1]


def test_app_manifest_lists_required_models_and_leaves_the_rest_on_demand():
    document = json.loads((ROOT / "voxprint-app-models.json").read_text(encoding="utf-8"))
    assert document["schema"] == 1 and document["app"] == "audiobook-builder"
    by_id: dict[str, list[dict]] = {}
    for model in document["models"]:
        by_id.setdefault(model["id"], []).append(model)
    tiers = {row["vram_tier"] for row in by_id["qwen3-tts"]}
    assert tiers == {"8gb", "16gb"}
    assert all(not row["on_demand"] for row in by_id["qwen3-tts"])
    assert by_id["qwen3-tts-tokenizer"][0]["on_demand"] is False
    assert by_id["voice-levi"][0]["on_demand"] is False
    assert by_id["voice-miriam"][0]["on_demand"] is False
    for name in ("opus-mt-tc-big-en-zle", "opus-mt-en-ru"):
        assert by_id[name][0]["on_demand"] is False
    for name in ("gemma-4-12b", "ace-step", "voice-natan", "voice-rivka"):
        assert by_id[name][0]["on_demand"] is True
    blob = json.dumps(document)
    assert "yo" not in {row["id"] for row in document["models"]}
    assert "core/data" not in blob


def test_narrate_exits_4_when_models_are_not_ready(tmp_path, monkeypatch):
    monkeypatch.delenv("VOXPRINT_SKIP_SUITE", raising=False)
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(mdl_gate, "ready", lambda: False)
    book = tmp_path / "book.txt"
    book.write_text("Hello\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    code = cli.main(["narrate", str(book), "--voice", "levi", "--out", str(out), "--json"])
    assert code == cli.EXIT_MISSING


def test_status_runs_when_models_are_not_ready(tmp_path, monkeypatch):
    monkeypatch.delenv("VOXPRINT_SKIP_SUITE", raising=False)
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(mdl_gate, "ready", lambda: False)
    assert cli.main(["status", "--json"]) == 0


def test_json_result_writes_cyrillic_as_utf8(monkeypatch):
    buf = io.BytesIO()
    wrapper = io.TextIOWrapper(buf, encoding="cp1252", newline="\n")
    monkeypatch.setattr(cli.sys, "stdout", wrapper)
    monkeypatch.setattr(cli.sys, "stderr", io.StringIO())
    cli._json_stdout = None
    cli.begin_json_stdout()
    try:
        cli._emit_result(
            ok=False, command="narrate", exit_code=4, outputs=[], warnings=[], duration_s=0.1,
            error="Готово", hint=None,
        )
        wrapper.flush()
        text = buf.getvalue().decode("utf-8")
    finally:
        cli.end_json_stdout()
    assert "Готово" in text and "\\u" not in text


def test_existing_voices_move_into_the_shared_folder(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("VOXPRINT_HOME", str(home))
    old = tmp_path / "old"
    (old / "state").mkdir(parents=True)
    monkeypatch.setattr(paths, "previous_homes", lambda: [old])
    voice = old / "voices" / "levi"
    voice.mkdir(parents=True)
    (voice / "voice.json").write_text('{"id": "levi"}\n', encoding="utf-8")
    shared = home / "shared" / "voices" / "miriam"
    shared.mkdir(parents=True)
    (shared / "voice.json").write_text('{"id": "miriam"}\n', encoding="utf-8")
    target = paths.voices_dir()
    assert (target / "levi" / "voice.json").is_file()
    assert (target / "miriam" / "voice.json").is_file()
    assert not (shared / "voice.json").exists()


def test_a_missing_model_is_not_downloaded_here(tmp_path, monkeypatch):
    from core.errors import ModelDownloadError
    from infra import model_downloader as md

    monkeypatch.setattr(md.paths, "models_dir", lambda: tmp_path)
    monkeypatch.setattr(md, "pinned_revision", lambda _repo: None)
    with pytest.raises(ModelDownloadError) as ei:
        md.ensure_model("Org/Missing", root=tmp_path)
    assert "voxprint-mdl" in ei.value.user_message
    assert [p.name for p in tmp_path.iterdir() if not p.name.startswith(".")] == []


def test_manage_models_uses_the_download_gui_flag():
    argv = mdl_gate.manage_argv()
    assert argv[argv.index("download") + 1] == "--gui"
    assert "--app" in argv and "audiobook-builder" in argv


def test_a_manifest_url_for_a_missing_tag_fails(tmp_path):
    good = tmp_path / "local.json"
    good.write_text(json.dumps({"url": "http://127.0.0.1:8097/manifest-thin-beta.json"}), encoding="utf-8")
    bad = tmp_path / "remote.json"
    bad.write_text(json.dumps({
        "url": "https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/v0.0.0-dev/manifest-thin-beta.json",
    }), encoding="utf-8")
    seen = []

    def exists(repo, tag):
        seen.append((repo, tag))
        return tag != "v0.0.0-dev"

    assert tags.check([good, bad], exists=exists) == [
        f"{bad}: GitHub tag v0.0.0-dev does not exist (Mitroshenkov87/voxprint-audiobook-builder)",
    ]
    assert seen == [("Mitroshenkov87/voxprint-audiobook-builder", "v0.0.0-dev")]
    assert tags.check([good], exists=exists) == []
