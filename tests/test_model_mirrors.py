"""Hugging Face backup mirror (infra/model_mirrors.py): SHA-256 verification and the order local -> original -> mirror."""
import hashlib
import json
from pathlib import Path

import pytest

from core.errors import ModelDownloadError
from infra import model_downloader as md
from infra import model_mirrors as mir

REPO = "Org/Tiny-1.7B"
SHA_A = "a" * 40
MIRROR = "Someone/voxprint-mirror-tiny"
FILES = {"config.json": b'{"model_type": "x"}', "model.safetensors": b"W" * 64, "speech_tokenizer/config.json": b"{}"}


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    monkeypatch.delenv("VOXPRINT_NO_MIRROR", raising=False)       # conftest disables mirrors; these tests enable them
    monkeypatch.delenv("VOXPRINT_NO_HF_MIRROR", raising=False)
    files = {n: {"size": len(b), "sha256": _h(b)} for n, b in FILES.items()}
    files["README.md"] = {"size": 5, "sha256": _h(b"card!"), "card": True}
    p = tmp_path / "mirrors.json"
    p.write_text(json.dumps({"schema": 1, "models": {REPO: {
        "source_repo": REPO, "source_revision": SHA_A, "license": "Apache-2.0", "mirror_repo": MIRROR,
        "mirror_revision": "m" * 40, "files": files}}}), encoding="utf-8")
    return p


def _fetcher(content=FILES, calls=None):
    def fetch(repo, name, rev, local_dir):
        if calls is not None:
            calls.append((repo, name, rev))
        t = Path(local_dir).joinpath(*name.split("/"))
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes(content[name])
    return fetch


def _broken_hf(**kw):
    raise OSError("network down")


def _fail_ms(*a, **k):
    raise OSError("modelscope down")


def test_bundled_manifest_is_valid_and_permissive():
    entries = mir.load()
    assert set(entries) >= {"Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ForcedAligner-0.6B",
                            "ai-forever/sage-fredt5-distilled-95m"}
    for repo, e in entries.items():
        assert e.license in ("Apache-2.0", "MIT", "CC0-1.0", "CC-BY-4.0"), repo
        assert len(e.source_revision) == 40
        if e.has_mirror:
            assert e.mirror_repo.startswith("Mitroshenkov87/voxprint-mirror-") and len(e.mirror_revision) == 40
        else:                                    # hashes only (no mirror yet)
            assert (repo in {"Qwen/Qwen3-ASR-1.7B", "myshell-ai/OpenVoiceV2", "ACE-Step/Ace-Step1.5"}
                    or repo.startswith("Helsinki-NLP/opus-mt-tc-")) and not e.mirror_repo
        assert any(n.endswith((".safetensors", ".bin", ".pth")) for n in e.files)
        assert any(n == "config.json" or n.endswith("/config.json") for n in e.files)
        assert all(len(m["sha256"]) == 64 and m["size"] > 0 for m in e.files.values())


def test_load_rejects_bad_manifests(tmp_path):
    bad = tmp_path / "bad.json"
    assert mir.load(tmp_path / "missing.json") == {}
    bad.write_text("{not json", encoding="utf-8")
    assert mir.load(bad) == {}
    bad.write_text(json.dumps({"schema": 1, "models": {REPO: {
        "source_revision": SHA_A, "mirror_repo": MIRROR, "mirror_revision": "m" * 40,
        "files": {"../evil.bin": {"size": 1, "sha256": "0" * 64}}}}}), encoding="utf-8")
    assert mir.load(bad) == {}


def test_download_verifies_and_skips_card(manifest, tmp_path):
    calls = []
    e = mir.load(manifest)[REPO]
    prog = []
    assert mir.download(e, tmp_path / "d", prog.append, _fetcher(calls=calls)) == SHA_A
    assert sorted(c[1] for c in calls) == sorted(FILES)              # README (card) is not downloaded
    assert all(c[0] == MIRROR and c[2] == "m" * 40 for c in calls)
    assert (tmp_path / "d" / "speech_tokenizer" / "config.json").read_bytes() == b"{}"
    assert prog[-1] == 1.0


def test_download_resumes_and_replaces_corrupt_file(manifest, tmp_path):
    e = mir.load(manifest)[REPO]
    d = tmp_path / "d"
    d.mkdir()
    (d / "config.json").write_bytes(FILES["config.json"])           # already good: not fetched again
    (d / "model.safetensors").write_bytes(b"truncated")             # bad: replaced
    calls = []
    mir.download(e, d, fetch=_fetcher(calls=calls))
    assert sorted(c[1] for c in calls) == ["model.safetensors", "speech_tokenizer/config.json"]
    assert (d / "model.safetensors").read_bytes() == FILES["model.safetensors"]


def test_hash_mismatch_is_rejected_and_file_deleted(manifest, tmp_path):
    e = mir.load(manifest)[REPO]
    evil = dict(FILES, **{"model.safetensors": b"X" * 64})           # same size, different content
    with pytest.raises(mir.MirrorError, match="mismatch"):
        mir.download(e, tmp_path / "d", fetch=_fetcher(evil))
    assert not (tmp_path / "d" / "model.safetensors").exists()


def test_fetch_error_becomes_mirror_error(manifest, tmp_path):
    def boom(*a):
        raise OSError("401")
    with pytest.raises(mir.MirrorError, match="401"):
        mir.download(mir.load(manifest)[REPO], tmp_path / "d", fetch=boom)


def test_ensure_model_falls_back_to_hf_mirror_right_after_the_original(manifest, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "h"))
    monkeypatch.setenv("VOXPRINT_NO_EXTERNAL_MODELS", "1")
    monkeypatch.chdir(tmp_path)
    order = []

    def ms(*a, **k):
        order.append("ms")
        raise OSError("modelscope down")

    def hf(**kw):
        order.append("hf")
        raise OSError("network down")

    calls = []
    got = md.ensure_model(REPO, snapshot_download=hf, revision=SHA_A, hf_probe=lambda r: True, mirror_download=ms,
                          hf_mirror_fetch=_fetcher(calls=calls), mirror_manifest=manifest)
    assert order == ["hf", "hf"] and calls                                 # original first, then our mirror - ModelScope is not needed
    assert md.verify_local_model(got)
    assert (got / ".revision").read_text() == SHA_A


def test_modelscope_is_the_last_resort_after_the_original_and_our_mirror(manifest, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "h"))
    monkeypatch.chdir(tmp_path)
    order = []

    def hf(**kw):
        order.append("hf")
        raise OSError("network down")

    def broken_fetch(repo, name, rev, dest):
        order.append("hfm")
        raise OSError("mirror down")

    def ms(repo, dest, progress, expected):
        order.append("ms")
        for n, b in FILES.items():
            t = Path(dest).joinpath(*n.split("/"))
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(b)

    got = md.ensure_model(REPO, snapshot_download=hf, revision=SHA_A, hf_probe=lambda r: True, mirror_download=ms,
                          hf_mirror_fetch=broken_fetch, mirror_manifest=manifest)
    assert order[:2] == ["hf", "hf"] and order[-2:] == ["hfm", "ms"] and md.verify_local_model(got)


def test_original_success_never_touches_the_mirror(manifest, tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_NO_EXTERNAL_MODELS", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "h"))
    monkeypatch.chdir(tmp_path)

    def hf(repo_id, local_dir, **kw):
        for n, b in FILES.items():
            t = Path(local_dir).joinpath(*n.split("/"))
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(b)

    def no_mirror(*a):
        raise AssertionError("mirror must not be used")

    got = md.ensure_model(REPO, snapshot_download=hf, revision=SHA_A, hf_probe=lambda r: True,
                          mirror_download=_fail_ms, hf_mirror_fetch=no_mirror, mirror_manifest=manifest)
    assert md.verify_local_model(got)


def test_mirror_failure_reports_all_errors_and_cache_is_first(manifest, tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_NO_EXTERNAL_MODELS", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "h"))
    monkeypatch.chdir(tmp_path)
    evil = dict(FILES, **{"model.safetensors": b"X" * 64})
    with pytest.raises(ModelDownloadError) as ei:
        md.ensure_model(REPO, snapshot_download=_broken_hf, revision=SHA_A, hf_probe=lambda r: True,
                        mirror_download=_fail_ms, hf_mirror_fetch=_fetcher(evil), mirror_manifest=manifest)
    d = ei.value.details or ""
    assert "network down" in d and "modelscope down" in d and "mismatch" in d
    assert not md.local_dir_for(REPO).exists()
    # a complete local copy wins over everything (no source is contacted at all)
    ok = md.local_dir_for(REPO)
    ok.mkdir(parents=True)
    (ok / "config.json").write_text("{}")
    (ok / "model.safetensors").write_bytes(b"x")
    assert md.ensure_model(REPO, snapshot_download=_broken_hf, hf_mirror_fetch=_fetcher(evil),
                           mirror_manifest=manifest) == ok


def test_mirror_not_used_for_other_revision_staging_or_when_disabled(manifest, tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_NO_EXTERNAL_MODELS", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "h"))
    monkeypatch.chdir(tmp_path)
    calls = []
    kw = dict(snapshot_download=_broken_hf, hf_probe=lambda r: True, mirror_download=_fail_ms,
              hf_mirror_fetch=_fetcher(calls=calls), mirror_manifest=manifest)
    with pytest.raises(ModelDownloadError):                          # another commit than the mirrored one
        md.ensure_model(REPO, revision="b" * 40, **kw)
    with pytest.raises(ModelDownloadError):                          # update through a staging folder
        md.ensure_model(REPO, root=tmp_path / "stage", revision=SHA_A, **kw)
    monkeypatch.setenv("VOXPRINT_NO_HF_MIRROR", "1")
    with pytest.raises(ModelDownloadError):
        md.ensure_model(REPO, revision=SHA_A, **kw)
    assert calls == []
