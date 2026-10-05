"""Portable setup folder: keep everything, install offline, fetch only what changed, models with the program's source order."""
import hashlib
import http.server
import json
import threading
import time
import zipfile
from pathlib import Path

import pytest

from infra import download_watch as dw
from infra import portable as pt
from tools import online_fetch as of


class St:
    def __init__(self):
        self.lines = []

    def write(self, state, fraction, text, force=False):
        self.lines.append((state, fraction, text))


def _zip(path: Path, files: dict) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for n, b in files.items():
            z.writestr(n, b)
    return path


def _comp(cid, path: Path, base, role, kind=None, flavor=None, markers=()):
    data = path.read_bytes()
    c = {"id": cid, "file": path.name, "url": f"{base}/{path.name}", "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
         "role": role, "markers": list(markers), "unpacked_bytes": len(data) * 2}
    if kind:
        c["kind"] = kind
    if flavor:
        c["flavor"] = flavor
    return c


@pytest.fixture
def web(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    hits = []

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(root), **k)

        def log_message(self, *a):
            pass

        def do_GET(self):
            hits.append(self.path)
            super().do_GET()

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{srv.server_address[1]}", hits
    srv.shutdown()


def _release(root, base, version="1.0", shell_extra=b"", torch_blob=b"T" * 5000):
    """A fake release: shell (core), one library wheel, PyTorch in three flavors; returns the manifest (also written to www/)."""
    shell = _zip(root / f"shell-{version}.zip", {"Voxprint.exe": b"exe" + shell_extra, "_internal/x.dll": b"d"})
    wheel = _zip(root / "lib-1.0-py3-none-any.whl", {"lib/__init__.py": "X=1\n", "lib-1.0.dist-info/METADATA": "Name: lib\n"})
    comps = [_comp("shell-01", shell, base, "core", markers=["Voxprint.exe"]),
             _comp("whl-lib", wheel, base, "runtime", "wheel", markers=["lib-1.0.dist-info/METADATA"])]
    for fl in ("cu128", "cu126", "cpu"):
        t = _zip(root / f"torch-{fl}.whl", {"torch/__init__.py": f"F='{fl}'\n", "torch-2.dist-info/METADATA": "Name: torch\n", "blob": torch_blob + fl.encode()})
        comps.append(_comp(f"whl-torch-{fl}", t, base, "runtime", "wheel", fl, ["torch-2.dist-info/METADATA"]))
    man = {"schema": 1, "app_version": version, "components": comps, "runtime": {"flavors": ["cu128", "cu126", "cpu"]}}
    (root / "manifest.json").write_text(json.dumps(man))
    return man


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("VOXPRINT_TORCH_FLAVOR", "cpu")


def test_the_portable_folder_keeps_all_components_of_the_pc_flavor_and_installs_only_the_shell(tmp_path, web):
    root, base, _ = web
    _release(root, base)
    folder, dest = tmp_path / "Voxprint Portable", tmp_path / "app"
    assert of.run(f"{base}/manifest.json", dest, tmp_path / "c", St(), roles=["core"], portable=folder, keep_all=True) == 1
    assert (dest / "Voxprint.exe").is_file() and not (dest / "lib").exists()           # installed: the shell only
    kept = sorted(p.parent.name for p in (folder / "components").rglob("*.*"))
    assert kept == ["shell-01", "whl-lib", "whl-torch-cpu"]                              # everything for this PC, other flavors not
    sums = (folder / "SHA256SUMS.txt").read_text()
    for p in (folder / "components").rglob("*.*"):
        assert hashlib.sha256(p.read_bytes()).hexdigest() in sums
    assert pt.is_setup_folder(folder) and pt.verify(folder, "cpu", deep=True).complete


def test_even_what_the_pc_already_has_is_kept_in_the_folder(tmp_path, web):
    root, base, _ = web
    _release(root, base)
    dest = tmp_path / "app"
    of.run(f"{base}/manifest.json", dest, tmp_path / "c", St(), roles=["core"])           # shell already installed
    folder = tmp_path / "P"
    assert of.run(f"{base}/manifest.json", dest, tmp_path / "c", St(), roles=["core"], portable=folder, keep_all=True) == 0
    assert (folder / "components" / "shell-01" / "shell-1.0.zip").is_file()


def test_a_later_run_installs_from_the_folder_without_internet(tmp_path, web):
    root, base, hits = web
    _release(root, base)
    folder = tmp_path / "P"
    of.run(f"{base}/manifest.json", tmp_path / "a1", tmp_path / "c", St(), roles=["core"], portable=folder, keep_all=True)
    hits.clear()
    # explicit offline mode
    assert of.run("", tmp_path / "a2", tmp_path / "c2", St(), roles=["core"], portable=folder, keep_all=False, offline=True) == 1
    assert (tmp_path / "a2" / "Voxprint.exe").is_file() and hits == []
    # the manifest address is dead: the folder is used automatically
    assert of.run("http://127.0.0.1:9/manifest.json", tmp_path / "a3", tmp_path / "c3", St(), roles=["core"], portable=folder, keep_all=True) == 1
    assert (tmp_path / "a3" / "Voxprint.exe").is_file() and hits == []


def test_main_from_folder_flag(tmp_path, web, capsys):
    root, base, hits = web
    _release(root, base)
    folder = tmp_path / "P"
    assert of.main(["--manifest", f"{base}/manifest.json", "--dest", str(tmp_path / "a"), "--cache", str(tmp_path / "c"), "--role", "core",
                    "--portable", str(folder), "--portable-all", "--models", "none"]) == 0
    hits.clear()
    assert of.main(["--from-folder", str(folder), "--dest", str(tmp_path / "b"), "--cache", str(tmp_path / "c2"), "--role", "core"]) == 0
    assert (tmp_path / "b" / "Voxprint.exe").is_file() and hits == []


def test_a_damaged_or_incomplete_folder_is_reported_not_installed(tmp_path, web):
    root, base, _ = web
    _release(root, base)
    folder = tmp_path / "P"
    of.run(f"{base}/manifest.json", tmp_path / "a", tmp_path / "c", St(), roles=["core"], portable=folder, keep_all=True)
    z = folder / "components" / "shell-01" / "shell-1.0.zip"
    z.write_bytes(z.read_bytes()[:-3] + b"xyz")                                          # same size, other content
    with pytest.raises(of.FetchError, match="incomplete or damaged"):
        of.run("", tmp_path / "b", tmp_path / "c2", St(), roles=["core"], portable=folder, offline=True)
    assert not (tmp_path / "b" / "Voxprint.exe").exists()
    rep = pt.verify(folder, "cpu", deep=True)
    assert rep.damaged and not rep.complete and "shell-1.0.zip" in rep.text()
    z.unlink()
    assert pt.verify(folder, "cpu").missing


def test_a_newer_version_fetches_only_the_changed_parts_and_prunes_the_old_ones(tmp_path, web):
    root, base, hits = web
    _release(root, base, "1.0")
    folder, dest = tmp_path / "P", tmp_path / "app"
    of.run(f"{base}/manifest.json", dest, tmp_path / "c", St(), roles=["core"], portable=folder, keep_all=True)
    old_manifest = json.loads((folder / "manifest.json").read_text())
    new = _release(root, base, "1.1", shell_extra=b"-new")                              # only the shell changes
    hits.clear()
    st = St()
    assert of.run(f"{base}/manifest.json", dest, tmp_path / "c", st, roles=["core"], portable=folder, keep_all=True) == 1
    fetched = [h for h in hits if h.endswith((".zip", ".whl"))]
    assert fetched == ["/shell-1.1.zip"]                                                 # nothing else was downloaded again
    assert any("newer version" in t for _, _, t in st.lines)
    assert not (folder / "components" / "shell-01" / "shell-1.0.zip").exists()           # old part pruned
    assert (folder / "components" / "shell-01" / "shell-1.1.zip").is_file()
    assert json.loads((folder / "manifest.json").read_text())["app_version"] == "1.1"
    d = pt.diff(old_manifest, new, "cpu")
    assert d.changed == ["shell-01"] and d.newer and d.fetch_bytes > 0 and "1.0 -> 1.1" in d.text()
    assert not pt.diff(new, new, "cpu").newer


def test_the_program_installs_modules_from_the_setup_folder(tmp_path, web, monkeypatch):
    root, base, hits = web
    man = _release(root, base)
    folder = tmp_path / "P"
    of.run(f"{base}/manifest.json", tmp_path / "a", tmp_path / "c", St(), roles=["core"], portable=folder, keep_all=True)
    from infra import modules

    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv(pt.ENV_DIR, str(folder))
    cfg = tmp_path / "modules.json"
    cfg.write_text(json.dumps({"manifest_url": "http://127.0.0.1:9/manifest.json"}))      # no internet
    monkeypatch.setenv("VOXPRINT_MODULES_CONFIG", str(cfg))
    man["modules"] = [{"id": "libs", "title": "Libs", "required": True, "components": ["whl-lib"]}]
    (folder / "manifest.json").write_text(json.dumps(man))
    hits.clear()
    assert modules.install(["libs"]) == 1
    assert (modules.runtime_dir() / "lib" / "__init__.py").is_file() and hits == []


def test_the_state_file_remembers_the_folder(tmp_path, web, monkeypatch):
    root, base, _ = web
    _release(root, base)
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(pt.ENV_DIR, raising=False)
    assert pt.configured_folder() is None
    folder = tmp_path / "P"
    of.run(f"{base}/manifest.json", tmp_path / "a", tmp_path / "c", St(), roles=["core"], portable=folder, keep_all=True)
    pt.set_folder(folder)
    assert pt.configured_folder() == folder and pt.find_folder() == folder
    pt.set_folder(None)
    assert pt.configured_folder() is None
    assert not pt.is_setup_folder(tmp_path / "nothing")


def test_flavor_rules(monkeypatch):
    monkeypatch.delenv("VOXPRINT_TORCH_FLAVOR")
    f = ["cu128", "cu126", "cpu"]
    assert pt.choose_flavor(f, (12, 9)) == "cu128" and pt.choose_flavor(f, (12, 6)) == "cu126" and pt.choose_flavor(f, (11, 8)) == "cpu"
    assert pt.choose_flavor(f, None) == "cpu"
    monkeypatch.setattr(pt.shutil, "which", lambda n: "nvidia-smi")
    assert pt.detect_cuda(lambda c: "| NVIDIA-SMI 570  Driver Version: 570.1  CUDA Version: 12.8 |") == (12, 8)
    assert pt.detect_vram_mb(lambda c: "24564\n8192\n") == 24564
    monkeypatch.setattr(pt.shutil, "which", lambda n: None)
    assert pt.detect_cuda() is None and pt.detect_vram_mb() == 0
    assert pt.models_for("none", 0) == [] and pt.models_for("auto", 24000)[1] == pt.TTS_LARGE and pt.models_for("auto", 6000)[1] == pt.TTS_SMALL
    auto = pt.models_for("auto", 24000)
    assert pt.ASR in auto and pt.SAGE in auto
    assert pt.TTS_LARGE in pt.models_for("all", 0)


# ------------------------------------------------------------------------------------------------ models
REPO, SHA = "Org/Small-Model", "b" * 40
MFILES = {"config.json": b'{"model_type": "x"}', "model.safetensors": b"W" * 200_000, "sub/tok.json": b"{}"}


def _h(b):
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def models(tmp_path, web, monkeypatch):
    root, base, hits = web
    # Hugging Face (resolve/<rev>/<file>) and the project's mirror, both served by the fake server
    for sub in (f"{REPO}/resolve/{SHA}", f"Me/mirror/resolve/{'m' * 40}"):
        for n, b in MFILES.items():
            p = root / sub / n
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)
    files = {n: {"size": len(b), "sha256": _h(b)} for n, b in MFILES.items()}
    mm = tmp_path / "mirrors.json"
    mm.write_text(json.dumps({"schema": 1, "models": {REPO: {"source_repo": REPO, "source_revision": SHA, "license": "MIT", "mirror_repo": "Me/mirror",
                                                            "mirror_revision": "m" * 40, "files": files}}}))
    rm = tmp_path / "release.json"
    rm.write_text(json.dumps({"schema": 1, "repo": "o/r", "tag": "models-v1", "base_url": f"{base}/gh", "models": {REPO: {
        "source_revision": SHA, "files": {n: {"asset": "m--" + n.replace("/", "--"), "size": len(b), "sha256": _h(b)} for n, b in MFILES.items()}}}}))
    (root / "gh").mkdir()
    for n, b in MFILES.items():
        (root / "gh" / ("m--" + n.replace("/", "--"))).write_bytes(b)
    monkeypatch.setenv("HF_ENDPOINT", base)
    monkeypatch.delenv("VOXPRINT_NO_MIRROR", raising=False)
    return mm, rm, hits


def test_models_come_from_github_first_into_the_program_layout(tmp_path, models):
    mm, rm, hits = models
    folder = tmp_path / "P"
    seen = []
    used = pt.fetch_models(folder, [REPO], lambda f, t: seen.append((f, t)), mm, rm, hf_fast=True)
    assert used == {REPO: "gh"} and not any("/resolve/" in h for h in hits)
    d = folder / "models" / "Org--Small-Model"
    assert (d / "model.safetensors").read_bytes() == MFILES["model.safetensors"] and (d / ".revision").read_text() == SHA
    assert pt.read_index(folder)["models"][REPO]["revision"] == SHA
    assert seen[-1][0] == 1.0
    assert pt.fetch_models(folder, [REPO], mirror_manifest=mm, release_manifest=rm, hf_fast=True) == {REPO: "cached"}


def test_models_fall_back_when_github_fails_and_when_a_file_is_corrupt(tmp_path, models, monkeypatch):
    mm, rm, hits = models
    (tmp_path / "www" / "gh" / "m--model.safetensors").write_bytes(b"X" * 200_000)       # a bad release asset (right size, wrong hash)
    used = pt.fetch_models(tmp_path / "P", [REPO], mirror_manifest=mm, release_manifest=rm, hf_fast=True)
    assert used == {REPO: "hf"}                                                          # original Hugging Face next
    assert (tmp_path / "P" / "models" / "Org--Small-Model" / "sub" / "tok.json").read_bytes() == b"{}"


def test_models_use_the_hugging_face_mirror_when_the_original_is_gone(tmp_path, models):
    mm, rm, hits = models
    import shutil
    shutil.rmtree(tmp_path / "www" / REPO.split("/")[0])
    shutil.rmtree(tmp_path / "www" / "gh")
    assert pt.fetch_models(tmp_path / "P", [REPO], mirror_manifest=mm, release_manifest=rm, hf_fast=True) == {REPO: "hfm"}


def test_a_stalled_source_is_abandoned_and_the_total_stays_stable(tmp_path, models, monkeypatch):
    mm, rm, hits = models

    def stalled(req, timeout):
        class R:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n):
                time.sleep(3)
                return b""
        if "/gh/" in req.full_url:
            return R()
        import urllib.request
        return urllib.request.urlopen(req, timeout=timeout)

    lines = []
    used = pt.fetch_models(tmp_path / "P", [REPO], lambda f, t: lines.append(t), mm, rm, opener=stalled, hf_fast=True, stall=0.5, poll=0.05)
    assert used == {REPO: "hf"}


def test_unknown_models_and_missing_space_are_errors(tmp_path, models):
    mm, rm, _ = models
    with pytest.raises(pt.PortableError, match="No pinned file list"):
        pt.fetch_models(tmp_path / "P", ["Nobody/None"], mirror_manifest=mm)


def test_run_portable_installs_even_if_the_models_fail(tmp_path, web, monkeypatch):
    root, base, _ = web
    _release(root, base)
    monkeypatch.setattr(pt, "fetch_models", lambda *a, **k: (_ for _ in ()).throw(pt.PortableError("no space")))
    monkeypatch.setattr(pt, "detect_vram_mb", lambda *a: 24000)
    st = St()
    n = of.run_portable(f"{base}/manifest.json", tmp_path / "a", tmp_path / "c", st, tmp_path / "P", roles=["core"], models="auto")
    assert n == 1 and (tmp_path / "a" / "Voxprint.exe").is_file()
    assert st.lines[-1][0] == "done" and "models are incomplete" in st.lines[-1][2]


# ------------------------------------------------------------------------------------------------ the installer script
def _iss():
    return (Path(__file__).resolve().parents[1] / "installer" / "Voxprint.iss").read_bytes().decode("utf-8-sig")


def test_the_wizard_page_and_the_downloader_agree():
    import re
    iss = _iss()
    # the option, its folder chooser and its default folder
    assert "PortableCheck" in iss and "BrowseForFolder(CustomMessage('PortablePrompt')" in iss and r"{userdocs}\Voxprint Portable" in iss
    assert "CreatePortablePage();" in iss
    # the command line the installer builds is accepted by the downloader
    ap_help = _help()
    for flag in ("--portable", "--portable-all", "--from-folder", "--models", "--role"):
        assert flag in ap_help and flag in iss
    # the file the installer writes is the file the program reads; the layout names are the same
    assert pt.STATE_FILE in iss and r"state\portable_dir.txt" in iss
    assert "manifest.json" in iss
    # no desktop shortcut
    assert not re.search(r"^\[Tasks\]", iss, re.M) and not re.search(r'^Name: "\{(common|user)desktop\}', iss, re.M)
    # strictly offline with /FromFolder, silent switches
    for sw in ("{param:Portable|0}", "{param:PortableDir|}", "{param:PortableModels|auto}", "{param:FromFolder|}"):
        assert sw in iss


def _help():
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            of.main(["--help"])
        except SystemExit:
            pass
    return buf.getvalue()


def test_all_portable_texts_exist_in_three_languages():
    import re
    iss = _iss()
    for key in ("PortablePageCaption", "PortablePageDescription", "PortableCheck", "PortableInfo", "PortableFoundInfo", "PortablePrompt",
                "PortableBadFolder", "PortableMissing"):
        for lang in ("english", "russian", "german"):
            m = re.search(rf"^{lang}\.{key}=(.+)$", iss, re.M)
            assert m and m.group(1).strip(), (lang, key)
    assert "portable" in re.search(r"^english\.PortableCheck=(.+)$", iss, re.M).group(1).lower()


from tests.test_studio import app  # noqa: E402,F401  (QApplication fixture)


def test_the_components_window_names_the_setup_folder(app, tmp_path, monkeypatch, web):
    from core import i18n
    from ui.modules_dialog import ModulesDialog

    root, base, _ = web
    _release(root, base)
    folder = tmp_path / "P"
    of.run(f"{base}/manifest.json", tmp_path / "a", tmp_path / "c", St(), roles=["core"], portable=folder, keep_all=True)
    i18n.set_language("en")
    assert str(folder) not in ModulesDialog(manifest_fn=lambda: {"modules": []}).lbl_hint.text()
    monkeypatch.setenv(pt.ENV_DIR, str(folder))
    assert str(folder) in ModulesDialog(manifest_fn=lambda: {"modules": []}).lbl_hint.text()
