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


def _file(name: str, digest: str) -> dict:
    return {
        "path": name,
        "size": 4,
        "sha256": digest,
        "mirrors": [{"source": "url", "url": "https://example.invalid/file.bin"}],
    }


def _model(model_id: str, files: list[dict], *, tier: str = "any", on_demand: bool = False) -> dict:
    return {
        "id": model_id,
        "version": "1",
        "vram_tier": tier,
        "on_demand": on_demand,
        "files": files,
    }


def _write_manifest(folder: Path, models: list[dict]) -> Path:
    path = folder / "voxprint-app-models.json"
    path.write_text(json.dumps({"schema": 1, "app": "audiobook-builder", "models": models}), encoding="utf-8")
    return path


def _plant(root: Path, digest: str, payload: bytes = b"data") -> None:
    blob = root / "blobs" / digest
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(payload)


def _arm(tmp_path, monkeypatch, models: list[dict]):
    monkeypatch.delenv("VOXPRINT_SKIP_SUITE", raising=False)
    monkeypatch.setattr(mdl_gate, "ready", lambda: True)
    store = tmp_path / "models"
    monkeypatch.setattr(mdl_gate, "suite_models_dir", lambda: store)
    monkeypatch.setattr(paths, "models_dir", lambda: store)
    torch = tmp_path / "runtime" / "torch"
    torch.mkdir(parents=True)
    monkeypatch.setattr("infra.suite_shared.torch_dir", lambda: torch.parent)
    manifest = _write_manifest(tmp_path, models)
    monkeypatch.setattr(mdl_gate, "manifest_path", lambda: manifest)
    return store


def test_a_ready_status_with_a_missing_file_is_not_ready(tmp_path, monkeypatch):
    digest = "a" * 64
    _arm(tmp_path, monkeypatch, [_model("voice-levi", [_file("voices/levi.zip", digest)])])
    assert mdl_gate.models_ready() is False


def test_an_empty_required_file_is_not_ready(tmp_path, monkeypatch):
    digest = "b" * 64
    store = _arm(tmp_path, monkeypatch, [_model("voice-levi", [_file("voices/levi.zip", digest)])])
    _plant(store, digest, b"")
    assert mdl_gate.models_ready() is False


def test_one_vram_tier_plus_torch_is_ready_and_on_demand_can_be_missing(tmp_path, monkeypatch):
    small = "c" * 64
    large = "d" * 64
    voice = "e" * 64
    optional = "f" * 64
    store = _arm(tmp_path, monkeypatch, [
        _model("qwen3-tts", [_file("Qwen--small/model.safetensors", small)], tier="8gb"),
        _model("qwen3-tts", [_file("Qwen--large/model.safetensors", large)], tier="16gb"),
        _model("voice-levi", [_file("voices/levi.zip", voice)]),
        _model("gemma-4-12b", [_file("llm/model.safetensors", optional)], on_demand=True),
    ])
    _plant(store, small)
    _plant(store, voice)
    assert mdl_gate.models_ready() is True


def test_a_named_path_counts_when_the_blob_is_absent(tmp_path, monkeypatch):
    digest = "1" * 64
    store = _arm(tmp_path, monkeypatch, [_model("voice-levi", [_file("voices/levi.zip", digest)])])
    named = store / "voices" / "levi.zip"
    named.parent.mkdir(parents=True)
    named.write_bytes(b"zip")
    assert mdl_gate.models_ready() is True


def test_missing_torch_is_not_ready(tmp_path, monkeypatch):
    digest = "2" * 64
    store = _arm(tmp_path, monkeypatch, [_model("voice-levi", [_file("voices/levi.zip", digest)])])
    _plant(store, digest)
    monkeypatch.setattr("infra.suite_shared.torch_dir", lambda: tmp_path / "no-torch")
    assert mdl_gate.models_ready() is False


def test_narrate_exits_4_when_the_status_file_says_ready_but_a_file_is_missing(tmp_path, monkeypatch):
    digest = "3" * 64
    _arm(tmp_path, monkeypatch, [_model("voice-levi", [_file("voices/levi.zip", digest)])])
    book = tmp_path / "book.txt"
    book.write_text("Hello\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    code = cli.main(["narrate", str(book), "--voice", "levi", "--out", str(out), "--json"])
    assert code == cli.EXIT_MISSING


def test_skip_suite_does_not_require_the_files(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_SKIP_SUITE", "1")
    monkeypatch.setattr(mdl_gate, "ready", lambda: True)
    monkeypatch.setattr(mdl_gate, "manifest_path", lambda: tmp_path / "missing.json")
    assert mdl_gate.models_ready() is True


def test_installers_download_without_asking_ready():
    iss = (ROOT / "installer" / "Voxprint.iss").read_text(encoding="utf-8-sig")
    body = iss[iss.index("procedure RunSuiteInstall"):iss.index("procedure AddAppToUserPath")]
    assert "SkipSuiteInstall()" in body
    assert body.index("SkipSuiteInstall()") < body.index("download --app audiobook-builder")
    assert "--json" in body and "ready()" not in body and "mdl.ready" not in body
    linux = (ROOT / "installer" / "linux" / "install-voxprint-linux.sh").read_text(encoding="utf-8")
    block = linux[linux.index("Downloading the required models"):linux.index("Creating the launcher")]
    assert "download --app audiobook-builder" in block and "--json" in block
    assert "ready" not in block


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
