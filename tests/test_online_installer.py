"""The ONLINE installer: payload splitting + manifest (tools/make_online_payload.py) and the downloader inside the installer
(tools/online_fetch.py): reuse of installed parts, SHA-256 verification, resumable downloads, safe extraction."""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import re
import threading
import zipfile
from pathlib import Path

import pytest

from tools import make_online_payload as mk
from tools import online_fetch as of

ROOT = Path(__file__).resolve().parents[1]


class Server:
    """Local HTTP server with Range support; ``cut_first`` closes the first big response early (a dropped connection)."""

    def __init__(self, root: Path) -> None:
        self.root, self.requests, self.cut_first, self.cut_done = root, [], 0, set()
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: D401
                pass

            def do_GET(self):  # noqa: N802
                f = outer.root / self.path.lstrip("/")
                outer.requests.append((self.path, self.headers.get("Range")))
                if not f.is_file():
                    self.send_error(404)
                    return
                data = f.read_bytes()
                rng = self.headers.get("Range")
                start = 0
                if rng and (m := re.match(r"bytes=(\d+)-", rng)):
                    start = int(m.group(1))
                    if start >= len(data):
                        self.send_error(416)
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if outer.cut_first and self.path not in outer.cut_done and len(body) > 4096:
                    outer.cut_done.add(self.path)
                    self.wfile.write(body[: len(body) // 3])
                    self.wfile.flush()
                    self.connection.close()
                    return
                self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()


@pytest.fixture
def dist(tmp_path):
    d = tmp_path / "dist" / "Voxprint"
    (d / "_internal" / "torch" / "lib").mkdir(parents=True)
    (d / "Voxprint.exe").write_bytes(b"MZ" + os.urandom(30000))
    (d / "python311.dll").write_bytes(os.urandom(40000))
    for i in range(6):
        (d / "_internal" / "torch" / "lib" / f"lib{i}.dll").write_bytes(os.urandom(30000))
    (d / "_internal" / "base.txt").write_text("hello " * 1000, encoding="utf-8")
    return d


@pytest.fixture
def served(tmp_path, dist):
    out = tmp_path / "srv"
    srv = Server(out)
    man = mk.build(dist, out, "v0.1.0-beta", "o/r", base_url=srv.url, limit_mib=0)
    yield srv, man
    srv.close()


def test_channel_for_tag():
    assert mk.channel_for("v0.1.0-beta") == "beta" and mk.channel_for("v1.2.3-rc1") == "beta"
    assert mk.channel_for("v1.0.0") == "stable" and mk.channel_for("1.0.0") == "stable"


def test_payload_parts_and_manifest(tmp_path, dist):
    out = tmp_path / "o"
    mp = mk.build(dist, out, "v0.1.0-beta", "Mitroshenkov87/voxprint-audiobook-builder", limit_mib=0)
    assert mp.name == "manifest-beta.json"
    man = json.loads(mp.read_text(encoding="utf-8"))
    assert man["schema"] == 1 and man["channel"] == "beta" and man["tag"] == "v0.1.0-beta"
    comps = man["components"]
    assert len(comps) >= 2 and [c["id"] for c in comps] == [f"payload-{i:02d}" for i in range(1, len(comps) + 1)]
    names = []
    for c in comps:
        p = out / c["file"]
        assert c["size"] == p.stat().st_size and c["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
        assert c["url"] == f"https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/v0.1.0-beta/{c['file']}"
        with zipfile.ZipFile(p) as z:
            names += z.namelist()
    want = sorted(f.relative_to(dist).as_posix() for f in dist.rglob("*") if f.is_file())
    assert sorted(names) == want                             # every file exactly once
    assert "Voxprint.exe" in zipfile.ZipFile(out / comps[0]["file"]).namelist()      # the program itself is in part 01
    of.validate_manifest(man)


def test_install_reuse_and_update(tmp_path, dist, served):
    srv, mp = served
    man = json.loads(mp.read_text(encoding="utf-8"))
    dest, cache = tmp_path / "app", tmp_path / "cache"
    st = tmp_path / "status.txt"
    n = of.run(str(mp), dest, cache, of.Status(st))
    assert n == len(man["components"]) >= 2
    for f in dist.rglob("*"):
        if f.is_file():
            assert (dest / f.relative_to(dist)).read_bytes() == f.read_bytes()
    assert st.read_text(encoding="ascii").split("\n")[0] == "done"
    assert list(cache.glob("*.zip")) == [] and list(cache.glob("*.part")) == []
    before = len(srv.requests)
    assert of.run(str(mp), dest, cache, of.Status(None)) == 0          # everything is reused
    assert len(srv.requests) == before
    # one part changes (a new build): only that part is downloaded again
    (dist / "_internal" / "base.txt").write_text("changed " * 500, encoding="utf-8")
    mk.build(dist, srv.root, "v0.1.0-beta", "o/r", base_url=srv.url, limit_mib=0)
    new = json.loads(mp.read_text(encoding="utf-8"))
    changed = [a["id"] for a, b in zip(man["components"], new["components"]) if a["sha256"] != b["sha256"]]
    assert changed and len(changed) < len(new["components"])
    assert of.run(str(mp), dest, cache, of.Status(None)) == len(changed)
    assert (dest / "_internal" / "base.txt").read_text(encoding="utf-8").startswith("changed")
    # a deleted marker file makes the part count as missing
    (dest / "Voxprint.exe").unlink()
    assert of.run(str(mp), dest, cache, of.Status(None)) >= 1 and (dest / "Voxprint.exe").is_file()


def test_resume_after_a_dropped_connection(tmp_path, served):
    srv, mp = served
    srv.cut_first = 1
    dest, cache = tmp_path / "app", tmp_path / "cache"
    assert of.run(str(mp), dest, cache, of.Status(None), sleep=lambda s: None) >= 2
    ranged = [r for r in srv.requests if r[1]]
    assert ranged and all(r[1].startswith("bytes=") and not r[1].startswith("bytes=0-") for r in ranged)


def test_resume_from_a_leftover_partial_file(tmp_path, served):
    srv, mp = served
    man = json.loads(mp.read_text(encoding="utf-8"))
    c = man["components"][0]
    cache = tmp_path / "cache"
    cache.mkdir()
    data = (srv.root / c["file"]).read_bytes()
    (cache / f"{c['sha256']}.part").write_bytes(data[:1000])
    of.run(str(mp), tmp_path / "app", cache, of.Status(None))
    assert ("/" + c["file"], "bytes=1000-") in srv.requests


def test_corrupt_download_is_rejected(tmp_path, served):
    srv, mp = served
    man = json.loads(mp.read_text(encoding="utf-8"))
    f = srv.root / man["components"][0]["file"]
    raw = bytearray(f.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    f.write_bytes(bytes(raw))
    st = tmp_path / "s.txt"
    with pytest.raises(of.FetchError, match="SHA-256"):
        of.run(str(mp), tmp_path / "app", tmp_path / "cache", of.Status(st), sleep=lambda s: None)
    assert not list((tmp_path / "cache").glob("*.part"))                 # a corrupt file is never resumed on top of
    assert of.main(["--manifest", str(mp), "--dest", str(tmp_path / "app"), "--cache", str(tmp_path / "cache"),
                    "--status", str(st)]) == 1
    assert st.read_text(encoding="ascii").split("\n")[0] == "error"


def test_http_404_fails_fast_with_a_message(tmp_path, served):
    srv, mp = served
    man = json.loads(mp.read_text(encoding="utf-8"))
    (srv.root / man["components"][1]["file"]).unlink()
    with pytest.raises(of.FetchError, match="HTTP 404"):
        of.run(str(mp), tmp_path / "app", tmp_path / "cache", of.Status(None), sleep=lambda s: None)


def test_zip_slip_is_refused(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(of.FetchError, match="Unsafe"):
        of.extract(z, tmp_path / "dest", lambda a, b: None)
    assert not (tmp_path / "evil.txt").exists()


def test_manifest_validation(tmp_path):
    good = {"schema": 1, "components": [{"id": "a", "file": "a.zip", "url": "https://example.org/a.zip", "size": 5,
                                         "sha256": "0" * 64}]}
    assert of.validate_manifest(json.loads(json.dumps(good)))
    for bad in ({**good, "schema": 2}, {**good, "components": []},
                {**good, "components": [{**good["components"][0], "url": "http://example.org/a.zip"}]},
                {**good, "components": [{**good["components"][0], "url": "file:///etc/passwd"}]},
                {**good, "components": [{**good["components"][0], "sha256": "xyz"}]},
                {**good, "components": [{**good["components"][0], "id": "../x"}]},
                {**good, "components": good["components"] * 2}):
        with pytest.raises(of.FetchError):
            of.validate_manifest(json.loads(json.dumps(bad)))
    p = tmp_path / "m.json"
    p.write_text("not json", encoding="utf-8")
    with pytest.raises(of.FetchError, match="unreadable"):
        of.read_manifest(str(p))


def test_inno_script_has_the_online_variant():
    iss = (ROOT / "installer" / "Voxprint.iss").read_text(encoding="utf-8-sig")
    assert "#ifdef ONLINE" in iss and "voxprint-fetch.exe" in iss and "Voxprint-Setup-online" in iss
    assert "{param:Manifest|" in iss and "procedure DeleteProgramFolder" in iss     # uninstall: program files, never a sibling
    assert "PrivilegesRequired=admin" in iss                  # one UAC prompt for the whole install
    assert "PrepareToInstall" in iss and "ExtractTemporaryFile" in iss      # a failure stops the setup (non-zero exit)
    wf = (ROOT / ".github" / "workflows" / "build-installer.yml").read_text(encoding="utf-8")
    for needle in ("build_online.ps1", "Voxprint-Setup-online.exe", "online-smoke", "manifest-", "release upload"):
        assert needle in wf or needle in wf.replace("gh release upload", "release upload")
    ps1 = (ROOT / "installer" / "build_online.ps1").read_text(encoding="utf-8")
    assert "make_online_payload.py" in ps1 and "/DONLINE" in ps1 and "online_fetch.py" in ps1


def test_status_file_has_a_heartbeat(tmp_path):
    p = tmp_path / "st.txt"
    st = of.Status(p)
    st.write("running", 0.5, "half", force=True)
    lines = p.read_text(encoding="ascii").split("\n")
    assert lines[:3] == ["running", "500", "half"] and lines[3].isdigit()
    st.start_heartbeat()
    import time
    first = int(lines[3])
    time.sleep(1.8)
    st.stop()
    assert int(p.read_text(encoding="ascii").split("\n")[3]) > first
