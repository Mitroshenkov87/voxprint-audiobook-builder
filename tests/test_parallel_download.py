"""Multi-connection Range downloads (infra/parallel_download.py) and the post-download verifying status."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.request

import pytest

from infra import model_downloader as md
from infra import parallel_download as pd

PAYLOAD = b"ABCDEFGH" * (64 * 1024)  # 512 KiB - large enough to split with a low PARALLEL_MIN
REPO = "Org/Tiny-Parallel"
SHA = "c" * 40
FILES = {"config.json": b'{"model_type": "x"}', "model.safetensors": PAYLOAD}


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def range_server():
    """HTTP server that serves PAYLOAD and answers Range with 206; records every request."""
    hits = []
    body = PAYLOAD

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            hits.append((self.path, rng))
            if self.path.endswith("/missing"):
                self.send_error(404)
                return
            if rng:
                # bytes=start-end
                spec = rng.split("=", 1)[1]
                start_s, end_s = spec.split("-", 1)
                start = int(start_s)
                end = int(end_s) if end_s else len(body) - 1
                end = min(end, len(body) - 1)
                chunk = body[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/file.bin", hits, body
    srv.shutdown()


@pytest.fixture
def fast_parallel(monkeypatch):
    monkeypatch.setenv("VOXPRINT_DL_PARALLEL_MIN", "1024")
    monkeypatch.setenv("VOXPRINT_DL_CONNECTIONS", "4")
    monkeypatch.delenv("VOXPRINT_NO_PARALLEL_DL", raising=False)


def test_parallel_range_download_uses_several_connections_and_resumes(range_server, tmp_path, fast_parallel):
    url, hits, body = range_server
    target = tmp_path / "model.safetensors"
    # seed one finished part so resume skips it
    ranges = pd._plan_ranges(len(body), 4)
    parts = pd._parts_dir(target)
    parts.mkdir(parents=True)
    start, end = ranges[0]
    (parts / "0000").write_bytes(body[start:end + 1])

    seen = []
    pd.download_file(url, target, len(body), on_bytes=lambda n: seen.append(n),
                     opener=lambda req, t: urllib.request.urlopen(req, timeout=t), conn=4)
    assert target.read_bytes() == body
    assert not parts.exists()
    # at least two distinct Range requests (resume of part 0 may be skipped entirely)
    range_hits = [h for h in hits if h[1]]
    assert len(range_hits) >= 2
    assert sum(seen) >= len(body)  # resumed part counted + new bytes


def test_small_file_stays_single_stream(range_server, tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_DL_PARALLEL_MIN", str(10 * 1024 * 1024))  # above our payload
    monkeypatch.delenv("VOXPRINT_NO_PARALLEL_DL", raising=False)
    url, hits, body = range_server
    target = tmp_path / "small.bin"
    pd.download_file(url, target, len(body),
                     opener=lambda req, t: urllib.request.urlopen(req, timeout=t), conn=8)
    assert target.read_bytes() == body
    # one full GET (no Range) or a single resume Range - never a parts directory left behind
    assert not pd._parts_dir(target).exists()
    assert len(hits) == 1


def test_cancel_aborts_a_parallel_download(range_server, tmp_path, fast_parallel):
    url, hits, body = range_server
    cancel = threading.Event()
    # open slowly: cancel after first byte reports
    real_open = urllib.request.urlopen

    def slow_open(req, timeout):
        cancel.set()
        return real_open(req, timeout=timeout)

    with pytest.raises(Exception) as ei:
        pd.download_file(url, tmp_path / "x.bin", len(body), opener=slow_open, cancel=cancel, conn=4)
    assert "abandon" in str(ei.value).lower() or ei.type.__name__ == "Stalled"


def test_download_listed_and_plan_ranges():
    assert pd._plan_ranges(10, 4) == [(0, 1), (2, 4), (5, 6), (7, 9)]
    assert sum(e - s + 1 for s, e in pd._plan_ranges(1000, 8)) == 1000


@pytest.fixture
def manifest(tmp_path):
    files = {n: {"size": len(b), "sha256": _h(b)} for n, b in FILES.items()}
    p = tmp_path / "mirrors.json"
    p.write_text(json.dumps({"schema": 1, "models": {REPO: {
        "source_repo": REPO, "source_revision": SHA, "license": "Apache-2.0", "mirror_repo": "Someone/mirror",
        "mirror_revision": "m" * 40, "files": files}}}), encoding="utf-8")
    return p


def _hf(content):
    def snapshot(repo_id, local_dir, revision=None, **kw):
        for name, data in content.items():
            t = Path(local_dir).joinpath(*name.split("/"))
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(data)
    return snapshot


def test_after_download_status_says_verifying_checksum_not_downloading(manifest, monkeypatch, tmp_path):
    """The UI must leave the 'downloading' look once bytes are on disk and hashes are checked."""
    messages = []

    def progress(stage, f, msg):
        messages.append(msg)

    from core import i18n
    i18n.reset()
    i18n._catalogs.clear()
    got = md.ensure_model(REPO, progress, snapshot_download=_hf(FILES), revision=SHA, mirror_manifest=manifest)
    assert md.verify_local_model(got)
    verifying = [m for m in messages if "SHA-256" in m]
    assert verifying, f"expected a verifying status in {messages!r}"
    v_idx = messages.index(verifying[0])
    assert v_idx < len(messages) - 1, "verifying must come before the final ready line"
    # must not look like an in-flight transfer percentage
    assert "%" not in verifying[0] and "Downloading" not in verifying[0]


def test_ensure_model_uses_parallel_listed_path_when_sizes_known(range_server, manifest, monkeypatch, tmp_path, fast_parallel):
    """With no injected snapshot_download, listed sizes go through parallel_download (local fake CDN)."""
    url, hits, body = range_server
    monkeypatch.setenv("VOXPRINT_LANG", "en")
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "h"))
    monkeypatch.setenv("VOXPRINT_NO_EXTERNAL_MODELS", "1")
    monkeypatch.setenv("VOXPRINT_NO_MIRROR", "1")  # only HF path
    from core import i18n
    i18n.reset()
    i18n._catalogs.clear()

    # tiny config over the same server path pattern: map names -> bytes via a custom opener + url_for
    # We intercept download_listed by patching hf_file_url and opener through download_file.
    sizes = {n: len(b) for n, b in FILES.items()}
    # Serve both files from one body for the big one; config from a side map.
    store = {"/model.safetensors": FILES["model.safetensors"], "/config.json": FILES["config.json"]}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            data = store.get(self.path)
            if data is None:
                self.send_error(404)
                return
            rng = self.headers.get("Range")
            if rng:
                start_s, end_s = rng.split("=", 1)[1].split("-", 1)
                start, end = int(start_s), int(end_s) if end_s else len(data) - 1
                chunk = data[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"

    monkeypatch.setattr(pd, "hf_file_url", lambda repo, name, rev=None, endpoint=None: f"{base}/{name}")
    monkeypatch.setattr(pd, "_open", lambda req, t: urllib.request.urlopen(req, timeout=t))
    # force the production branch (no injected snapshot) and supply sizes
    messages = []
    got = md.ensure_model(
        REPO, lambda s, f, m: messages.append(m), snapshot_download=None, revision=SHA,
        mirror_manifest=manifest, get_remote_sizes=lambda *a: sizes, hf_probe=lambda r: True,
        mirror_download=lambda *a, **k: (_ for _ in ()).throw(AssertionError("ms should not run")),
    )
    srv.shutdown()
    assert md.verify_local_model(got)
    assert (got / "model.safetensors").read_bytes() == FILES["model.safetensors"]
    assert any("SHA-256" in m or "checksum" in m.lower() for m in messages)
