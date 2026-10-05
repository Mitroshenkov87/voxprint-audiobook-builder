"""A finished download is checked by size + SHA-256 against infra/model_mirrors.json BEFORE it is renamed into place
(infra/model_downloader.py: manifest_bad_files / _verify_or_repair)."""
import hashlib
import json
from pathlib import Path

import pytest

from core.errors import ModelDownloadError
from infra import model_downloader as md

REPO = "Org/Tiny-1.7B"
SHA_A = "a" * 40
FILES = {"config.json": b'{"model_type": "x"}', "model.safetensors": b"W" * 64, ".gitattributes": b"*.bin lfs"}


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def manifest(tmp_path):
    files = {n: {"size": len(b), "sha256": _h(b)} for n, b in FILES.items()}
    p = tmp_path / "mirrors.json"
    p.write_text(json.dumps({"schema": 1, "models": {REPO: {
        "source_repo": REPO, "source_revision": SHA_A, "license": "Apache-2.0", "mirror_repo": "Someone/mirror",
        "mirror_revision": "m" * 40, "files": files}}}), encoding="utf-8")
    return p


def _hf(content):
    def snapshot(repo_id, local_dir, revision=None, **kw):
        for name, data in content.items():
            t = Path(local_dir).joinpath(*name.split("/"))
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(data)
    return snapshot


def test_a_correct_download_passes_and_git_metadata_is_not_checked(manifest):
    content = dict(FILES, **{".gitattributes": b"rewritten by the hub"})      # metadata may differ: ignored
    got = md.ensure_model(REPO, snapshot_download=_hf(content), revision=SHA_A, mirror_manifest=manifest)
    assert got == md.local_dir_for(REPO) and md.verify_local_model(got)
    assert md.manifest_bad_files(got, REPO, SHA_A, manifest) == []


def test_a_corrupt_download_is_never_marked_done(manifest):
    evil = dict(FILES, **{"model.safetensors": b"X" * 64})                    # same size, other content
    with pytest.raises(ModelDownloadError) as ei:
        md.ensure_model(REPO, snapshot_download=_hf(evil), revision=SHA_A, mirror_manifest=manifest)
    assert "model.safetensors" in ei.value.details
    target = md.local_dir_for(REPO)
    partial = target.with_name(target.name + ".partial")
    assert not target.exists()                                                  # not renamed into place
    assert partial.is_dir() and not (partial / "model.safetensors").exists()   # only the bad file is gone: the rest resumes
    assert (partial / "config.json").is_file()


def test_a_truncated_file_is_repaired_from_the_backup_mirror(manifest, monkeypatch):
    monkeypatch.delenv("VOXPRINT_NO_MIRROR", raising=False)
    monkeypatch.delenv("VOXPRINT_NO_HF_MIRROR", raising=False)
    calls = []

    def fetch(repo, name, rev, local_dir):
        calls.append(name)
        t = Path(local_dir).joinpath(*name.split("/"))
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes(FILES[name])

    short = dict(FILES, **{"model.safetensors": b"W" * 10})
    got = md.ensure_model(REPO, snapshot_download=_hf(short), revision=SHA_A, mirror_manifest=manifest,
                          hf_mirror_fetch=fetch, hf_probe=lambda r: True, mirror_download=lambda *a, **k: None)
    assert (got / "model.safetensors").read_bytes() == FILES["model.safetensors"]
    assert calls == ["model.safetensors"]                                       # only the bad file came from the mirror


def test_unknown_revision_or_repository_means_structural_check_only(manifest, tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    assert md.manifest_bad_files(d, REPO, "b" * 40, manifest) is None
    assert md.manifest_bad_files(d, "Other/Repo", SHA_A, manifest) is None
    assert md.manifest_bad_files(d, REPO, None, manifest) is None
    assert md.manifest_bad_files(d, REPO, SHA_A, manifest) == ["config.json", "model.safetensors"]


def test_the_log_names_who_asked_for_a_download(manifest, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="voxprint.models"):
        md.ensure_model(REPO, snapshot_download=_hf(FILES), revision=SHA_A, mirror_manifest=manifest)
    line = next(r.getMessage() for r in caplog.records if "requested by" in r.getMessage())
    assert REPO in line and "test_the_log_names_who_asked_for_a_download" in line


def test_ready_model_path_never_downloads(tmp_path, monkeypatch):
    """Voice check / consent must see a missing model as None, not trigger ensure_model."""
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "h"))
    calls = []
    monkeypatch.setattr(md, "ensure_model", lambda *a, **k: calls.append(a) or tmp_path)
    assert md.ready_model_path(REPO) is None and calls == []
    d = md.local_dir_for(REPO)
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"W" * 8)
    assert md.ready_model_path(REPO) == d and calls == []


def test_voice_check_asr_uses_local_copy_only(tmp_path, monkeypatch):
    from workers import pipeline_runner as pr
    from workers.pipeline_runner import TaskRequest

    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "h"))
    calls = []
    monkeypatch.setattr(md, "ensure_model", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError("surprise download")))
    assert pr._asr_for_check(TaskRequest(kind="lora", audio=tmp_path / "a.wav", text=tmp_path / "t.txt"), None) is None
    assert calls == []
