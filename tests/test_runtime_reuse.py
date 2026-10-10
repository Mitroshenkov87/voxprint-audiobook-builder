"""Thin installer v2: reuse what is on the PC (PyTorch of another program), flavor choice, manifest from the lock, install from "upstream"."""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

from infra import modules as mods
from infra import runtime_reuse as rr
from tests.test_online_installer import Server
from tools import make_online_payload as mk
from tools import runtime_manifest as rm

ROOT = Path(__file__).resolve().parents[1]
LOCK = {"compat": {"torch": ">=2.8,<2.13"}, "flavors": ["cu128", "cu126", "cpu"]}
TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"     # the reuse rules accept only wheels for the running Python


def fake_site(site: Path, torch="2.11.0+cu128", audio="2.11.0+cu128", tag=TAG, plat="win_amd64", lib=True) -> Path:
    """A site-packages folder that looks like an installed PyTorch (no real code)."""
    site.mkdir(parents=True, exist_ok=True)
    for name, ver in (("torch", torch), ("torchaudio", audio)):
        if not ver:
            continue
        di = site / f"{name}-{ver}.dist-info"
        di.mkdir()
        (di / "METADATA").write_text(f"Name: {name}\nVersion: {ver}\n")
        (di / "WHEEL").write_text(f"Wheel-Version: 1.0\nTag: {tag}-{tag}-{plat}\n")
        (site / name).mkdir()
        (site / name / "__init__.py").write_text("")
    if lib:
        (site / "torch" / "lib").mkdir()
    return site


def env_with(tmp_path, name="proj", **kw) -> Path:
    prefix = tmp_path / name / ".venv"
    fake_site(prefix / "Lib" / "site-packages", **kw)
    return prefix


# ------------------------------------------------------------------------------------------------ finding and judging
def test_find_torch_reads_dist_info_without_running_anything(tmp_path):
    a, b = env_with(tmp_path, "a", torch="2.9.1+cu128", audio="2.9.1+cu128"), env_with(tmp_path, "b")
    found = rr.find_torch([a, b, tmp_path / "nothing"])
    assert [e.torch for e in found] == ["2.11.0+cu128", "2.9.1+cu128"]          # newest first
    e = found[0]
    assert (e.py_tag, e.platform, e.flavor, e.torchaudio) == (TAG, "win_amd64", "cu128", "2.11.0+cu128")
    assert Path(e.site) == b / "Lib" / "site-packages"


def test_candidate_prefixes_find_project_venvs_and_pinokio(tmp_path):
    home = tmp_path / "home"
    (home / "work" / "tts" / ".venv").mkdir(parents=True)
    (home / "pinokio" / "api" / "app1" / "env").mkdir(parents=True)
    got = rr.candidate_prefixes(environ={}, home=home)
    assert home / "work" / "tts" / ".venv" in got and home / "pinokio" / "api" / "app1" / "env" in got


@pytest.mark.parametrize("kw,driver,wanted,ok,why", [
    ({}, (12, 8), "cu128", True, "ok"),
    ({"torch": "2.11.0+cu126", "audio": "2.11.0+cu126"}, (12, 8), "cu128", True, "ok"),            # older CUDA build than the driver offers
    ({"torch": "2.11.0+cu130", "audio": "2.11.0+cu130"}, (12, 8), "cu128", False, "newer NVIDIA driver"),
    ({"torch": "2.11.0+cpu", "audio": "2.11.0+cpu"}, (12, 8), "cu128", False, "NVIDIA GPU"),
    ({"torch": "2.11.0+cpu", "audio": "2.11.0+cpu"}, None, "cpu", True, "ok"),
    ({"torch": "2.11.0+cu128", "audio": "2.11.0+cu128"}, None, "cpu", True, "ok"),                 # CUDA build, no GPU: runs on the CPU
    ({"tag": "cp310"}, (12, 8), "cu128", False, "cp310"),                                           # a wheel for another Python
    ({"torch": "2.4.0+cu128", "audio": "2.4.0+cu128"}, (12, 8), "cu128", False, "range"),
    ({"torch": "3.0.0+cu128", "audio": "3.0.0+cu128"}, (12, 8), "cu128", False, "range"),
    ({"audio": None}, (12, 8), "cu128", False, "torchaudio is not installed"),
    ({"audio": "2.10.0+cu128"}, (12, 8), "cu128", False, "does not match"),
    ({"lib": False}, (12, 8), "cu128", False, "torch/lib"),
])
def test_compatibility_rules(tmp_path, kw, driver, wanted, ok, why):
    site = fake_site(tmp_path / "s", **kw)
    ext = rr.inspect_site(site)
    got_ok, reason = rr.assess(ext, LOCK, wanted, driver, py_tag=TAG)
    assert got_ok is ok and why in reason, reason


def test_flavor_choice_follows_the_driver():
    fl = ["cu128", "cu126", "cpu"]
    assert rr.choose_flavor(fl, (13, 0)) == "cu128" and rr.choose_flavor(fl, (12, 8)) == "cu128"
    assert rr.choose_flavor(fl, (12, 7)) == "cu126" and rr.choose_flavor(fl, (12, 6)) == "cu126"
    assert rr.choose_flavor(fl, (12, 4)) == "" and rr.choose_flavor(fl, None) == ""


def test_flavor_can_be_forced(monkeypatch):
    monkeypatch.setenv("VOXPRINT_TORCH_FLAVOR", "cpu")
    assert rr.choose_flavor(["cu128", "cpu"], (12, 8)) == "cpu"


# ------------------------------------------------------------------------------------------------ decision and memory
def test_the_first_candidate_that_passes_the_check_is_used_and_remembered(tmp_path):
    bad = env_with(tmp_path, "newer", torch="2.12.0+cu128", audio="2.12.0+cu128")
    good = env_with(tmp_path, "older", torch="2.11.0+cu128", audio="2.11.0+cu128")
    never = env_with(tmp_path, "cpu", torch="2.10.0+cpu", audio="2.10.0+cpu")
    checked = []

    def verifier(ext):
        checked.append(ext.torch)
        return (ext.torch == "2.11.0+cu128", "stub")

    ext = rr.try_reuse_torch(LOCK, (12, 8), "cu128", prefixes=[bad, good, never], verifier=verifier)
    assert ext is not None and ext.torch == "2.11.0+cu128"
    assert checked == ["2.12.0+cu128", "2.11.0+cu128"]            # the CPU build was rejected by the rules, never checked
    assert rr.reused("torch").site == ext.site and rr.extra_paths() == [ext.site]


def test_nothing_suitable_means_download(tmp_path):
    only_cpu = env_with(tmp_path, "c", torch="2.11.0+cpu", audio="2.11.0+cpu")
    assert rr.try_reuse_torch(LOCK, (12, 8), "cu128", prefixes=[only_cpu], verifier=lambda e: (True, "")) is None
    failing = env_with(tmp_path, "f")
    assert rr.try_reuse_torch(LOCK, (12, 8), "cu128", prefixes=[failing], verifier=lambda e: (False, "DLL load failed")) is None
    assert rr.reused("torch") is None and rr.extra_paths() == []


def test_a_remembered_copy_is_dropped_when_it_changed_or_disappeared(tmp_path):
    p = env_with(tmp_path)
    ext = rr.try_reuse_torch(LOCK, (12, 8), "cu128", prefixes=[p], verifier=lambda e: (True, ""))
    assert rr.reused() is not None
    import shutil

    shutil.rmtree(Path(ext.site) / "torch" / "lib")
    assert rr.reused() is None                                          # half-deleted environment
    fake = fake_site(tmp_path / "x")
    rr.save_choice("torch", rr.inspect_site(fake))
    assert rr.reused() is not None
    shutil.rmtree(fake)
    assert rr.reused() is None and rr.extra_paths() == []


def test_the_search_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_NO_REUSE", "1")
    p = env_with(tmp_path)
    assert rr.try_reuse_torch(LOCK, (12, 8), "cu128", prefixes=[p], verifier=lambda e: (True, "")) is None


def test_verify_runs_a_child_with_the_candidate_on_the_path(tmp_path):
    ext = rr.inspect_site(fake_site(tmp_path / "s"))
    seen = {}

    def run(cmd, env, timeout):
        seen.update(cmd=cmd, env=env, timeout=timeout)
        return 0, "noise\nVXTORCH OK torch 2.11.0 from X on cuda:RTX\n"

    ok, detail = rr.verify(ext, run)
    assert ok and "cuda:RTX" in detail and seen["env"]["VOXPRINT_EXTRA_SITE"] == ext.site and "--probe-torch" in seen["cmd"]
    ok, detail = rr.verify(ext, lambda c, e, t: (1, "Traceback...\nVXTORCH FAIL OSError: DLL load failed"))
    assert not ok and "DLL load failed" in detail


def test_the_probe_reports_a_missing_torch_as_a_failure(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)               # import torch -> ImportError
    monkeypatch.setenv("VOXPRINT_EXTRA_SITE", "")
    assert rr.probe_torch_main() == 1 and "VXTORCH FAIL" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ manifest from the lock + install
def make_wheel(path: Path, dist: str, version: str, files=None):
    name = path.name.split("-")[0]
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{name}/__init__.py", f"# {dist}\n")
        z.writestr(f"{name}-{version}.dist-info/METADATA", f"Name: {dist}\nVersion: {version}\n")
        for k, v in (files or {}).items():
            z.writestr(k, v)


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    """A fake 'PyPI + pytorch.org' served locally, a lock pointing to it, and a shell folder."""
    www = tmp_path / "www"
    www.mkdir()
    srv = Server(www)
    wheels = []

    def add(dist, version, group, flavor=None, fname=None):
        fname = fname or f"{dist}-{version}-py3-none-any.whl"
        make_wheel(www / fname, dist, version, {"torch/lib/x.dll": "dll"} if group == "torch" else None)
        from tools.make_runtime_lock import norm  # noqa: F401

        w = {"dist": dist, "version": version, "file": fname, "url": f"{srv.url}/{fname}", "size": (www / fname).stat().st_size,
             "sha256": rm.sha256_of(www / fname), "group": group}
        if flavor:
            w["flavor"] = flavor
        wheels.append(w)

    add("scipy", "1.17.1", "audio")
    add("nagisa", "0.2.11", "text")
    add("transformers", "4.57.6", "libs")
    for fl in LOCK["flavors"]:
        for pkg in ("torch", "torchaudio"):
            add(pkg, f"2.11.0+{fl}", "torch", fl, f"{pkg}-2.11.0+{fl}-cp311-cp311-win_amd64.whl")
    lock = dict(LOCK, schema=1, python="3.11", platform="win_amd64", torch_version="2.11.0", wheels=wheels,
                group_titles={"torch": "PyTorch", "libs": "Libraries", "audio": "Audio", "text": "Text"}, shell={})
    lp = tmp_path / "lock.json"
    lp.write_text(json.dumps(lock))
    shell = tmp_path / "dist" / "Voxprint"
    (shell / "_internal").mkdir(parents=True)
    (shell / "Voxprint.exe").write_bytes(b"MZ" + os.urandom(2000))
    out = tmp_path / "release"
    mp = mk.build(shell, out, "v0.1.0-beta", "o/r", base_url=srv.url, runtime_lock=lp)
    cfg = tmp_path / "modules.json"
    cfg.write_text(json.dumps({"schema": 1, "manifest_url": mp.as_uri() if False else str(mp)}), encoding="utf-8")
    monkeypatch.setenv("VOXPRINT_MODULES_CONFIG", str(cfg))
    monkeypatch.setattr(mods, "driver_cuda", lambda: (12, 8))
    yield srv, mp, out, wheels
    srv.close()
    for p in [p for p in sys.path if "voxprint_home" in p or "proj" in p]:
        sys.path.remove(p)


def test_the_manifest_points_to_the_upstream_files_and_the_release_holds_only_the_shell(upstream):
    srv, mp, out, wheels = upstream
    man = json.loads(mp.read_text())
    assert mp.name == "manifest-thin-beta.json" and man["thin"] and man["runtime"]["flavors"] == ["cu128", "cu126", "cpu"]
    assert sorted(p.name for p in out.glob("*.zip")) == ["Voxprint-shell-01.zip"]            # nothing third-party is repacked
    core = [c for c in man["components"] if c["role"] == "core"]
    rt = [c for c in man["components"] if c["role"] == "runtime"]
    assert len(core) == 1 and len(rt) == len(wheels) and all(c["kind"] == "wheel" for c in rt)
    by_file = {w["file"]: w for w in wheels}
    for c in rt:                                                                               # the original address and hash, untouched
        assert c["url"] == by_file[c["file"]]["url"] and c["sha256"] == by_file[c["file"]]["sha256"]
    assert [m["id"] for m in man["modules"]] == ["libs", "text", "audio", "torch"]
    assert next(m for m in man["modules"] if m["id"] == "torch")["reusable"] == "torch"
    mj = json.loads((upstream[0].root.parent / "dist" / "Voxprint" / "_internal" / "modules.json").read_text())
    assert mj["manifest_url"].endswith("manifest-thin-beta.json")


def test_install_downloads_only_the_flavor_of_this_pc_from_upstream(upstream, monkeypatch):
    srv, mp, out, wheels = upstream
    monkeypatch.setenv("VOXPRINT_NO_REUSE", "1")
    n = mods.install()
    got = {p[0].lstrip("/") for p in srv.requests}
    assert "torch-2.11.0+cu128-cp311-cp311-win_amd64.whl" in got and not any("cu126" in g or "+cpu" in g for g in got)
    assert n == 3 + 2 and all(m.installed for m in mods.modules(mods.load_manifest()))
    assert (mods.runtime_dir() / "torchaudio-2.11.0+cu128.dist-info" / "METADATA").is_file()
    assert mods.install() == 0 and mods.installed_without_network()


def test_a_pc_without_a_supported_cuda_flavor_does_not_download_torch(upstream, monkeypatch):
    srv, *_ = upstream
    monkeypatch.setattr(mods, "driver_cuda", lambda: None)
    monkeypatch.setenv("VOXPRINT_NO_REUSE", "1")
    assert mods.install(["torch"]) == 0
    got = {p[0].lstrip("/") for p in srv.requests}
    assert not any("torch" in g for g in got)


def test_an_existing_pytorch_is_reused_and_not_downloaded(upstream, tmp_path, monkeypatch):
    srv, *_ = upstream
    prefix = env_with(tmp_path, "other-app")
    monkeypatch.setattr(rr, "candidate_prefixes", lambda *a, **k: [prefix])
    monkeypatch.setattr(rr, "verify", lambda ext, run=None: (True, "VXTORCH OK stub"))
    n = mods.install()
    got = {p[0].lstrip("/") for p in srv.requests}
    assert not any(g.startswith("torch") for g in got) and n == 3                          # only the small libraries were fetched
    man = mods.load_manifest()
    torch = next(m for m in mods.modules(man) if m.id == "torch")
    assert torch.installed and torch.reused and torch.size == 0
    site = str(prefix / "Lib" / "site-packages")
    assert site in sys.path and sys.path.index(site) > sys.path.index(str(mods.runtime_dir()))     # foreign copy comes LAST
    assert mods.installed_without_network()


def test_a_reused_copy_that_fails_the_check_falls_back_to_our_own(upstream, tmp_path, monkeypatch):
    srv, *_ = upstream
    prefix = env_with(tmp_path, "broken-app")
    monkeypatch.setattr(rr, "candidate_prefixes", lambda *a, **k: [prefix])
    monkeypatch.setattr(rr, "verify", lambda ext, run=None: (False, "VXTORCH FAIL OSError: DLL load failed"))
    mods.install()
    got = {p[0].lstrip("/") for p in srv.requests}
    assert "torch-2.11.0+cu128-cp311-cp311-win_amd64.whl" in got
    assert rr.reused("torch") is None and str(prefix / "Lib" / "site-packages") not in sys.path


def test_previously_downloaded_components_are_not_fetched_again(upstream, monkeypatch):
    srv, *_ = upstream
    monkeypatch.setenv("VOXPRINT_NO_REUSE", "1")
    mods.install()
    srv.requests.clear()
    mods.install()                                                 # e.g. after a re-install of the program: the runtime folder is still there
    assert srv.requests == [] or all("manifest" in r[0] for r in srv.requests)


def test_sdist_only_packages_are_built_into_a_small_wheel(tmp_path):
    lock = {"flavors": [], "wheels": [{"dist": "eng-to-ipa", "version": "0.0.2", "file": "eng-to-ipa-0.0.2.tar.gz", "url": "https://x/e.tgz",
                                       "size": 1, "sha256": "0" * 64, "group": "text", "sdist": True}]}
    built = tmp_path / "eng_to_ipa-0.0.2-py3-none-any.whl"
    make_wheel(built, "eng-to-ipa", "0.0.2")
    comps, mods_, _ = rm.components(lock, "https://rel", tmp_path, builder=lambda w, work, out: built)
    assert comps[0]["url"] == "https://rel/eng_to_ipa-0.0.2-py3-none-any.whl" and comps[0]["sha256"] == rm.sha256_of(built)
    assert comps[0]["markers"] == ["eng_to_ipa-0.0.2.dist-info/METADATA"]


def test_a_non_pure_sdist_is_refused(tmp_path):
    w = {"dist": "x", "version": "1", "file": "x-1.tar.gz", "url": "https://x", "size": 1, "sha256": "ab" * 32}
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "x-1.tar.gz").write_bytes(b"data")
    import hashlib
    w["sha256"] = hashlib.sha256(b"data").hexdigest()

    class R:
        returncode = 0
        stdout = stderr = ""

    def run(cmd, **k):
        (tmp_path / "o").mkdir(exist_ok=True)
        (tmp_path / "o" / "x-1-cp311-cp311-win_amd64.whl").write_bytes(b"")
        return R()

    with pytest.raises(SystemExit, match="not pure"):
        rm.build_sdist_wheel(w, tmp_path / "w", tmp_path / "o", fetch=lambda u, d: None, run=run)


def test_the_thin_shell_bundles_the_whole_standard_library(tmp_path):
    """The downloaded libraries import the whole stdlib; the CI end-to-end check once failed with ``No module named 'timeit'``."""
    from tools import gen_stdlib_bundle as gen

    gen.main([str(tmp_path)])
    src = (tmp_path / "_vx_stdlib.py").read_text(encoding="utf-8")
    compile(src, "_vx_stdlib.py", "exec")
    for need in ("timeit", "unittest", "doctest", "sched", "pdb", "zipfile", "sqlite3", "ctypes", "multiprocessing"):
        assert f"import {need}\n" in src, need
    assert "import tkinter\n" not in src and "import this\n" not in src
    bat = (ROOT / "build_thin.bat").read_text(encoding="utf-8")
    assert "gen_stdlib_bundle.py" in bat and "--hidden-import _vx_stdlib" in bat
