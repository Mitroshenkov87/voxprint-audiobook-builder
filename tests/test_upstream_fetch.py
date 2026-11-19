"""online_fetch: components from upstream sites with fallback addresses, and wheels."""
import hashlib
import http.server
import threading
import zipfile
from pathlib import Path

import pytest

from tools import online_fetch as of


def make_wheel(path: Path, name="demo", version="1.0"):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{name}/__init__.py", "X = 1\n")
        z.writestr(f"{name}-{version}.dist-info/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        z.writestr(f"{name}-{version}.data/purelib/{name}_extra.py", "Y = 2\n")
        z.writestr(f"{name}-{version}.data/scripts/tool.exe", "x")
        z.writestr(f"{name}-{version}.data/headers/h.h", "x")
    return path


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "www"
    root.mkdir()

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(root), **k)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def comp(whl: Path, url, urls=()):
    data = whl.read_bytes()
    return {"id": "whl-demo", "file": whl.name, "url": url, "urls": list(urls), "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "kind": "wheel", "markers": ["demo-1.0.dist-info/METADATA"]}


class St:
    def write(self, *a, **k):
        pass


def manifest(c):
    return {"schema": 1, "components": [c]}


def test_a_wheel_is_installed_like_site_packages(tmp_path, server):
    root, base = server
    whl = make_wheel(root / "demo-1.0-py3-none-any.whl")
    dest = tmp_path / "rt"
    mp = tmp_path / "m.json"
    import json
    mp.write_text(json.dumps(manifest(comp(whl, f"{base}/demo-1.0-py3-none-any.whl"))))
    assert of.run(str(mp), dest, tmp_path / "cache", St()) == 1
    assert (dest / "demo" / "__init__.py").is_file() and (dest / "demo-1.0.dist-info" / "METADATA").is_file()
    assert (dest / "demo_extra.py").is_file()                       # purelib merged into the root
    assert not (dest / "tool.exe").exists() and not any(dest.rglob("h.h"))
    assert of.run(str(mp), dest, tmp_path / "cache", St()) == 0     # second run: already installed


def test_a_dead_upstream_falls_back_to_the_mirror(tmp_path, server):
    root, base = server
    whl = make_wheel(root / "mirror.whl")
    c = comp(whl, f"{base}/gone/demo-1.0-py3-none-any.whl", [f"{base}/mirror.whl"])
    got = of.download(c, tmp_path / "c", lambda a, b: None, sleep=lambda s: None)
    assert got.read_bytes() == whl.read_bytes()


def test_a_corrupt_upstream_file_is_never_accepted_and_the_mirror_is_used(tmp_path, server):
    root, base = server
    good = make_wheel(root / "good.whl")
    (root / "bad.whl").write_bytes(b"x" * good.stat().st_size)      # same size, wrong content
    c = comp(good, f"{base}/bad.whl", [f"{base}/good.whl"])
    got = of.download(c, tmp_path / "c", lambda a, b: None, sleep=lambda s: None)
    assert of.sha256_file(got) == c["sha256"]


def test_all_sources_failing_is_an_error(tmp_path, server):
    root, base = server
    whl = make_wheel(root / "w.whl")
    c = comp(whl, f"{base}/nope1.whl", [f"{base}/nope2.whl"])
    with pytest.raises(of.FetchError):
        of.download(c, tmp_path / "c", lambda a, b: None, sleep=lambda s: None)


def test_unsafe_wheel_paths_are_refused(tmp_path):
    bad = tmp_path / "evil.whl"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../evil.py", "x")
    with pytest.raises(of.FetchError):
        of.extract_wheel(bad, tmp_path / "dest", lambda a, b: None)


def test_the_manifest_checks_fallback_addresses_and_kind():
    base = {"id": "a", "url": "https://x.org/a", "size": 1, "sha256": "0" * 64}
    of.validate_manifest({"schema": 1, "components": [dict(base, urls=["https://y.org/a"], kind="wheel")]})
    with pytest.raises(of.FetchError):
        of.validate_manifest({"schema": 1, "components": [dict(base, urls=["http://evil.example/a"])]})
    with pytest.raises(of.FetchError):
        of.validate_manifest({"schema": 1, "components": [dict(base, kind="exe")]})


# ------------------------------------------------------------------------------------------------ portable setup folder (foundation)
def test_a_portable_folder_keeps_everything_and_serves_an_offline_reinstall(tmp_path, server):
    import json

    root, base = server
    whl = make_wheel(root / "demo-1.0-py3-none-any.whl")
    mp = tmp_path / "m.json"
    c = comp(whl, f"{base}/demo-1.0-py3-none-any.whl")
    mp.write_text(json.dumps(manifest(c)))
    port, dest = tmp_path / "Voxprint Portable", tmp_path / "rt"
    assert of.run(str(mp), dest, tmp_path / "cache", St(), portable=port) == 1
    kept = port / "components" / "whl-demo" / "demo-1.0-py3-none-any.whl"
    assert kept.is_file() and (port / "manifest.json").is_file()
    assert (port / "SHA256SUMS.txt").read_text().strip() == f"{c['sha256']} *components/whl-demo/demo-1.0-py3-none-any.whl"
    # already installed on this PC, still kept in the folder (the folder is meant to be complete)
    port2 = tmp_path / "Portable2"
    assert of.run(str(mp), dest, tmp_path / "cache", St(), portable=port2) == 0 and (port2 / "components" / "whl-demo").is_dir()
    # offline re-install into a fresh folder from the portable one: the server is not asked for the file any more
    (root / "demo-1.0-py3-none-any.whl").unlink()
    dest2 = tmp_path / "rt2"
    assert of.run(str(mp), dest2, tmp_path / "cache2", St(), portable=port) == 1
    assert (dest2 / "demo" / "__init__.py").is_file()
