"""Small models as GitHub release assets (infra/model_release.py): hash-checked, resumable, first in the source order."""
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from core.errors import ModelDownloadError
from infra import download_watch as dw
from infra import model_downloader as md
from infra import model_release as rel

REPO, SHA = "Org/Small-Model", "b" * 40
FILES = {"config.json": b'{"model_type": "x"}', "model.safetensors": b"W" * 300_000, "sub/tok.json": b"{}"}


def _h(b):
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def server():
    hits = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            name = self.path.rsplit("/", 1)[-1]
            data = {"small--" + n.replace("/", "--"): b for n, b in FILES.items()}.get(name)
            hits.append((name, self.headers.get("Range")))
            if data is None:
                self.send_error(404)
                return
            start = 0
            r = self.headers.get("Range")
            if r:
                start = int(r.split("=")[1].split("-")[0])
            self.send_response(206 if r else 200)
            self.send_header("Content-Length", str(len(data) - start))
            self.end_headers()
            self.wfile.write(data[start:])

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", hits
    srv.shutdown()


def _manifest(tmp_path, base, corrupt=None):
    files = {n: {"asset": "small--" + n.replace("/", "--"), "size": len(b), "sha256": _h(b)} for n, b in FILES.items()}
    if corrupt:
        files[corrupt]["sha256"] = "0" * 64
    p = tmp_path / "rel.json"
    p.write_text(json.dumps({"schema": 1, "repo": "o/r", "tag": "models-v1", "base_url": base,
                             "models": {REPO: {"source_revision": SHA, "files": files}}}), encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.delenv("VOXPRINT_NO_MIRROR", raising=False)
    monkeypatch.setenv("VOXPRINT_LANG", "en")


def test_bundled_manifest_is_consistent_with_the_mirror_manifest():
    from infra import model_mirrors as mir

    entries, mirrors = rel.load(), mir.load()
    for repo, e in entries.items():
        m = mirrors[repo]
        assert e.source_revision == m.source_revision
        total = 0
        for n, meta in e.files.items():
            assert m.files[n]["sha256"] == meta["sha256"] and int(m.files[n]["size"]) == int(meta["size"])
            assert int(meta["size"]) < 2 * 1024 ** 3 - 1
            total += int(meta["size"])
        assert total <= rel.SMALL_MODEL_MAX                    # only small models are hosted there


def test_download_verifies_resumes_and_reports_progress(server, tmp_path):
    base, hits = server
    e = rel.entry_for(REPO, SHA, _manifest(tmp_path, base))
    dest = tmp_path / "m"
    dest.mkdir()
    (dest / "model.safetensors.incomplete").write_bytes(b"W" * 100_000)         # an interrupted earlier download
    seen = []
    assert rel.download(e, dest, seen.append) == SHA
    assert (dest / "model.safetensors").read_bytes() == FILES["model.safetensors"]
    assert (dest / "sub" / "tok.json").read_bytes() == b"{}"
    assert ("small--model.safetensors", "bytes=100000-") in hits               # resumed with Range
    assert seen == sorted(seen) and seen[-1] == 1.0
    assert not list(dest.rglob("*.incomplete"))


def test_a_wrong_hash_deletes_the_file_and_raises(server, tmp_path):
    base, _ = server
    e = rel.entry_for(REPO, SHA, _manifest(tmp_path, base, corrupt="model.safetensors"))
    with pytest.raises(rel.ReleaseError):
        rel.download(e, tmp_path / "m")
    assert not (tmp_path / "m" / "model.safetensors").exists() and not (tmp_path / "m" / "model.safetensors.incomplete").exists()


def test_entry_only_for_the_pinned_revision_and_when_enabled(tmp_path, monkeypatch):
    p = _manifest(tmp_path, "http://x")
    assert rel.entry_for(REPO, SHA, p) is not None and rel.entry_for(REPO, None, p) is not None
    assert rel.entry_for(REPO, "c" * 40, p) is None and rel.entry_for("Other/Model", SHA, p) is None
    monkeypatch.setenv("VOXPRINT_NO_GITHUB_MODELS", "1")
    assert rel.entry_for(REPO, SHA, p) is None


def test_a_bad_manifest_is_ignored(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"schema": 1, "repo": "o/r", "tag": "t", "models": {"A/b": {"source_revision": "x", "files": {"../x": {}}}}}')
    assert rel.load(p) == {}


@pytest.fixture
def quick(monkeypatch, tmp_path):
    monkeypatch.setattr(dw, "STALL_SECONDS", 0.5)
    monkeypatch.setattr(dw, "POLL", 0.05)
    monkeypatch.setattr(dw, "ABORT_GRACE", 1.0)
    monkeypatch.setenv("VOXPRINT_NO_EXTERNAL_MODELS", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "h"))
    monkeypatch.chdir(tmp_path)


def _hf(order):
    def hf(repo_id, local_dir, **kw):
        order.append("hf")
        for n, b in FILES.items():
            t = Path(local_dir).joinpath(*n.split("/"))
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(b)
    return hf


def test_a_small_model_comes_from_github_first_and_hugging_face_is_not_touched(server, quick, tmp_path):
    base, _ = server
    order, lines = [], []
    got = md.ensure_model(REPO, lambda s, f, m: lines.append(m), snapshot_download=_hf(order), revision=SHA,
                          hf_probe=lambda r: True, release_manifest=_manifest(tmp_path, base))
    assert md.verify_local_model(got) and order == []
    assert (got / ".revision").read_text() == SHA
    assert any("GitHub" in m for m in lines)


def test_a_failing_github_falls_back_to_hugging_face(quick, tmp_path):
    order = []

    def broken_opener(req, timeout):
        order.append("gh")
        raise OSError("github down")

    got = md.ensure_model(REPO, snapshot_download=_hf(order), revision=SHA, hf_probe=lambda r: True,
                          release_manifest=_manifest(tmp_path, "http://x"), release_opener=broken_opener)
    assert order == ["gh", "hf"] and md.verify_local_model(got)


def test_a_stalled_github_is_abandoned_by_the_watchdog(quick, tmp_path):
    order = []

    class Slow:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            time.sleep(3)                       # connected, but no byte arrives
            return b""

    got = md.ensure_model(REPO, snapshot_download=_hf(order), revision=SHA, hf_probe=lambda r: True,
                          release_manifest=_manifest(tmp_path, "http://x"), release_opener=lambda req, t: Slow())
    assert order == ["hf"] and md.verify_local_model(got)


def test_large_models_and_pattern_mismatches_never_use_github(quick, tmp_path):
    order = []
    got = md.ensure_model("Org/Other-Big", snapshot_download=_hf(order), revision=None, hf_probe=lambda r: True,
                          release_manifest=_manifest(tmp_path, "http://x"),
                          get_remote_sha=lambda r: "d" * 40)
    assert order == ["hf"] and md.verify_local_model(got)
