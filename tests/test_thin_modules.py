"""The THIN installer pieces: splitting the environment into runtime modules, the role filter of the downloader, the manifest of a
thin build, the module manager (infra/modules.py), the Components window and the Inno script switches."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

from infra import modules as mods
from tests.test_online_installer import Server
from tests.test_studio import app, lib, make_studio, wait_for  # noqa: F401  (fixtures)
from tools import make_online_payload as mk
from tools import make_runtime_modules as rt
from tools import online_fetch as of

ROOT = Path(__file__).resolve().parents[1]
ISS = (ROOT / "installer" / "Voxprint.iss").read_text(encoding="utf-8-sig")


def add_dist(site: Path, name: str, files: dict) -> None:
    """A fake installed distribution (METADATA + RECORD) with the given files."""
    di = site / f"{name}-1.0.dist-info"
    di.mkdir(parents=True)
    (di / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8")
    rec = []
    for rel, data in files.items():
        p = site / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        rec.append(f"{rel},,")
    rec += [f"{name}-1.0.dist-info/METADATA,,", f"{name}-1.0.dist-info/RECORD,,", "../../Scripts/tool.exe,,", f"{name}/__pycache__/x.pyc,,"]
    (di / "RECORD").write_text("\n".join(rec) + "\n", encoding="utf-8")


@pytest.fixture
def site(tmp_path):
    s = tmp_path / "site-packages"
    s.mkdir()
    add_dist(s, "torch", {"torch/__init__.py": b"# torch\n", "torch/lib/big.dll": os.urandom(60000), "torch/_C.pyd": os.urandom(20000)})
    add_dist(s, "scipy", {"scipy/__init__.py": b"# scipy\n", "scipy/signal.py": b"x = 1\n"})
    add_dist(s, "transformers", {"transformers/__init__.py": b"# tf\n"})            # not named anywhere: the "ml" module
    add_dist(s, "numpy", {"numpy/__init__.py": b"# numpy\n"})                        # shell
    add_dist(s, "PySide6", {"PySide6/__init__.py": b"# qt\n"})                       # shell
    add_dist(s, "pytest", {"pytest/__init__.py": b"# t\n"})                          # tooling
    return s


@pytest.fixture
def shell(tmp_path):
    d = tmp_path / "dist" / "Voxprint"
    (d / "_internal").mkdir(parents=True)
    (d / "Voxprint.exe").write_bytes(b"MZ" + os.urandom(20000))
    (d / "_internal" / "python311.dll").write_bytes(os.urandom(20000))
    return d


# ------------------------------------------------------------------------------------------------ splitting
def test_modules_by_distribution_name():
    assert rt.module_of("torch") == "torch" and rt.module_of("nvidia-cudnn-cu12") == "torch" and rt.module_of("SciPy") == "audio"
    assert rt.module_of("imageio-ffmpeg") == "ffmpeg" and rt.module_of("ru_normalizr") == "text"
    assert rt.module_of("transformers") == "ml" and rt.module_of("qwen-tts") == "ml"
    for stay in ("PySide6", "numpy", "soundfile", "certifi", "pytest", "pip"):
        assert rt.module_of(stay) is None


def test_files_are_split_without_overlap_and_metadata_comes_along(site):
    files = rt.dist_files(site)
    assert set(files) == {"torch", "audio", "ml"}
    assert "torch/lib/big.dll" in files["torch"] and "torch-1.0.dist-info/METADATA" in files["torch"]
    assert "scipy/signal.py" in files["audio"] and "transformers/__init__.py" in files["ml"]
    flat = [f for v in files.values() for f in v]
    assert len(flat) == len(set(flat))
    assert not any(f.startswith("..") or "__pycache__" in f or f.startswith(("numpy", "PySide6", "pytest")) for f in flat)


def test_modules_become_parts_with_roles_and_a_limit(site, tmp_path):
    out = tmp_path / "out"
    comps, mod_list = rt.build_modules(site, out, "http://x", limit_mib=0)       # 0: a part per file -> several parts
    assert {c["role"] for c in comps} == {"runtime"} and {m["id"] for m in mod_list} == {"torch", "audio", "ml"}
    torch_parts = [c for c in comps if c["module"] == "torch"]
    assert len(torch_parts) >= 3 and [m for m in mod_list if m["id"] == "torch"][0]["components"] == [c["id"] for c in torch_parts]
    for c in comps:
        p = out / c["file"]
        assert c["size"] == p.stat().st_size and c["url"] == f"http://x/{c['file']}" and c["sha256"] == rt._sha256(p)
        with zipfile.ZipFile(p) as z:
            assert all(m in z.namelist() for m in c["markers"])


def test_thin_manifest_and_shell(site, shell, tmp_path):
    out = tmp_path / "o"
    mp = mk.build(shell.parent / "Voxprint", out, "v0.1.0-beta", "o/r", base_url="http://127.0.0.1:1", limit_mib=0, runtime_site=site)
    man = json.loads(mp.read_text(encoding="utf-8"))
    assert man["thin"] is True and {m["id"] for m in man["modules"]} == {"torch", "audio", "ml"}
    roles = {c["id"]: c["role"] for c in man["components"]}
    assert {r for i, r in roles.items() if i.startswith("payload")} == {"core"} and {r for i, r in roles.items() if i.startswith("rt-")} == {"runtime"}
    cfg = json.loads((shell / "_internal" / "modules.json").read_text(encoding="utf-8"))
    assert cfg["manifest_url"] == "http://127.0.0.1:1/manifest-beta.json"
    names = [n for z in out.glob("Voxprint-payload-*.zip") for n in zipfile.ZipFile(z).namelist()]
    assert any(n.endswith("modules.json") for n in names)               # the shell carries it
    # without runtime_site nothing changes: no roles, no modules
    plain = json.loads(mk.build(shell.parent / "Voxprint", tmp_path / "p", "v0.1.0-beta", "o/r", limit_mib=0).read_text(encoding="utf-8"))
    assert "thin" not in plain and "modules" not in plain and all("role" not in c for c in plain["components"])


def test_downloader_role_filter(site, shell, tmp_path):
    out = tmp_path / "srv"
    srv = Server(out)
    try:
        mp = mk.build(shell.parent / "Voxprint", out, "v0.1.0-beta", "o/r", base_url=srv.url, limit_mib=0, runtime_site=site)
        core = tmp_path / "core"
        of.run(str(mp), core, tmp_path / "c1", of.Status(None), roles=["core"])
        assert (core / "Voxprint.exe").is_file() and not (core / "torch").exists()
        assert not any(p.startswith("/Voxprint-rt-") for p, _ in srv.requests)
        n = of.run(str(mp), tmp_path / "all", tmp_path / "c2", of.Status(None))               # no --role: everything, as before
        assert n == len(json.loads(mp.read_text(encoding="utf-8"))["components"])
    finally:
        srv.close()


# ------------------------------------------------------------------------------------------------ the module manager
@pytest.fixture
def thin(site, shell, tmp_path, monkeypatch):
    out = tmp_path / "srv"
    srv = Server(out)
    mk.build(shell.parent / "Voxprint", out, "v0.1.0-beta", "o/r", base_url=srv.url, limit_mib=0, runtime_site=site)
    cfg = tmp_path / "modules.json"
    cfg.write_text(json.dumps({"schema": 1, "manifest_url": f"{srv.url}/manifest-beta.json"}), encoding="utf-8")
    monkeypatch.setenv("VOXPRINT_MODULES_CONFIG", str(cfg))
    yield srv
    srv.close()
    for p in [p for p in sys.path if "runtime" in p and "voxprint_home" in p]:
        sys.path.remove(p)


def test_a_full_build_is_untouched(monkeypatch):
    monkeypatch.delenv("VOXPRINT_MODULES_CONFIG", raising=False)
    assert mods.is_thin() is False and mods.missing_required() == [] and mods.installed_without_network() is True
    assert mods.activate() is None


def test_install_downloads_required_modules_and_activates_them(thin):
    assert mods.is_thin() and mods.installed_without_network() is False          # nothing cached yet
    man = mods.load_manifest()
    assert [m.id for m in mods.missing_required(man)] == [m["id"] for m in man["modules"]]
    seen = []
    n = mods.install(progress=lambda f, m: seen.append(f))
    assert n == len(man["components"]) - len([c for c in man["components"] if c["role"] == "core"])
    assert seen and max(seen) <= 1.0
    assert all(m.installed for m in mods.modules(man)) and mods.installed_without_network() is True
    assert (mods.runtime_dir() / "torch" / "lib" / "big.dll").is_file()
    assert str(mods.runtime_dir()) in sys.path
    import importlib.machinery

    rt_dir = str(mods.runtime_dir())
    spec = importlib.machinery.PathFinder.find_spec("torch", [rt_dir])                       # importable from the runtime folder
    assert spec is not None and Path(spec.origin).parent.parent == mods.runtime_dir()
    others = [i for i, e in enumerate(sys.path) if e.endswith("site-packages")]
    assert not others or sys.path.index(rt_dir) < min(others)                                # and it wins over site-packages
    assert mods.install() == 0                                                                    # nothing left to do


def test_install_is_resumable_and_cancellable(thin):
    calls = []

    def cancel() -> bool:
        calls.append(1)
        return len(calls) > 3

    with pytest.raises(mods.Cancelled):
        mods.install(["torch"], cancelled=cancel)
    assert mods.install(["torch"]) >= 1 and any(m.installed for m in mods.modules(mods.load_manifest()) if m.id == "torch")


def test_offline_start_uses_the_cached_manifest(thin, monkeypatch):
    monkeypatch.setattr(of.time, "sleep", lambda s: None)
    mods.install()
    Path(os.environ["VOXPRINT_MODULES_CONFIG"]).write_text(json.dumps({"manifest_url": "http://127.0.0.1:9/m.json"}), encoding="utf-8")   # no network
    assert mods.installed_without_network() is True
    man = mods.load_manifest()                                   # falls back to the cached copy
    assert man["thin"] is True and mods.missing_required(man) == []


def test_install_error_is_a_modules_error(thin, monkeypatch):
    monkeypatch.setattr(of.time, "sleep", lambda s: None)
    monkeypatch.setenv("VOXPRINT_MODULES_CONFIG", str(Path(os.environ["VOXPRINT_MODULES_CONFIG"]).with_name("x.json")))
    assert mods.is_thin() is False
    cfg = Path(os.environ["VOXPRINT_MODULES_CONFIG"])
    cfg.write_text(json.dumps({"manifest_url": "http://127.0.0.1:9/none.json"}), encoding="utf-8")
    with pytest.raises(mods.ModulesError):
        mods.load_manifest(offline_ok=False)


def test_cli_status_and_install(thin, capsys):
    import main

    assert main._modules_cli(["x", "--install-modules", "torch"]) == 0
    out = capsys.readouterr().out
    assert "OK:" in out and re.search(r"torch\s+installed", out) and re.search(r"audio\s+missing", out)


# ------------------------------------------------------------------------------------------------ the window
def test_components_window_downloads_and_reports(thin, app):                              # noqa: F811
    from core import i18n
    from ui.modules_dialog import ModulesDialog

    i18n.set_language("en")
    d = ModulesDialog()
    ready = []
    d.ready.connect(lambda: ready.append(1))
    d.refresh()
    assert wait_for(lambda: d.modules and not d.busy, 20)
    assert d.btn_download.isEnabled() and len(d.missing()) == 3
    assert d.start_install()
    assert wait_for(lambda: not d.missing() and not d.busy and ready, 40)
    assert not d.btn_download.isEnabled() and d.lbl_status.text() == "All components are installed."
    d.shutdown()


def test_components_window_shows_a_failure_and_can_retry(app, monkeypatch):                # noqa: F811
    from core import i18n
    from ui.modules_dialog import ModulesDialog

    i18n.set_language("en")
    m = mods.Module("torch", "PyTorch", True, 3 * 1024 ** 3, 4 * 1024 ** 3, ["a"], False)
    attempts = []

    def install_fn(ids, progress, cancelled):
        attempts.append(ids)
        progress(0.5, "half")
        if len(attempts) == 1:
            raise mods.ModulesError("network down")
        m.installed = True
        return 1

    monkeypatch.setattr(mods, "modules", lambda man: [m])
    d = ModulesDialog(manifest_fn=lambda: {"modules": []}, install_fn=install_fn)
    d.refresh()
    assert wait_for(lambda: d.modules and not d.busy, 10)
    assert d.start_install()
    assert wait_for(lambda: "network down" in d.lbl_status.text() and not d.busy, 10)
    assert "Press Download to continue" in d.lbl_status.text() and d.btn_download.isEnabled()
    assert wait_for(lambda: not d.busy, 10)
    assert d.start_install()
    assert wait_for(lambda: d.lbl_status.text() == "All components are installed." and not d.busy, 10)
    assert len(attempts) == 2 and m.installed and not d.missing()
    d.shutdown()


def test_settings_has_the_components_button_only_in_a_thin_build(app, lib, thin, monkeypatch):    # noqa: F811
    s = make_studio(lib)
    try:
        assert s.settings_dialog().btn_components.isHidden() is False
    finally:
        s.shutdown()
    monkeypatch.delenv("VOXPRINT_MODULES_CONFIG")
    s = make_studio(lib)
    try:
        assert s.settings_dialog().btn_components.isHidden() is True
    finally:
        s.shutdown()


# ------------------------------------------------------------------------------------------------ the Inno script
def test_inno_thin_switch_and_vc_redist_exit_codes():
    assert "#ifdef THIN" in ISS and "' --role core'" in ISS and "Voxprint-Setup-thin" in ISS
    assert not re.search(r"^Filename: \"\{tmp\}\\vc_redist", ISS, re.M)                  # no [Run] entry any more
    body = ISS[ISS.index("procedure InstallVcRedist"):ISS.index("procedure CurStepChanged")]
    for code in ("1638", "3010", "1641"):
        assert code in body
    assert "MsgBox" not in body and "SuppressibleMsgBox" not in body                      # never shown to the user
    assert "InstallVcRedist()" in ISS[ISS.index("procedure CurStepChanged"):]


def test_thin_build_script_excludes_the_heavy_libraries():
    bat = (ROOT / "build_thin.bat").read_text(encoding="utf-8")
    for name in ("torch", "transformers", "scipy", "librosa", "qwen_tts", "huggingface_hub"):
        assert re.search(rf"\b{name}\b", bat.split("set HEAVY=")[1].split("set EXCL=")[0])
    assert "--exclude-module" in bat and "netroute" in bat and "dist\\thin" in bat


def test_try_latest_falls_back_to_pinned_and_a_missing_pinned_set_falls_back_to_latest(thin, tmp_path, monkeypatch):
    monkeypatch.setattr(of.time, "sleep", lambda s: None)
    out = tmp_path / "srv"
    man = json.loads((out / "manifest-beta.json").read_text(encoding="utf-8"))
    bad = json.loads(json.dumps(man))
    next(c for c in bad["components"] if c["id"].startswith("rt-"))["sha256"] = "0" * 64      # a "latest" set with a wrong hash
    (out / "manifest-latest.json").write_text(json.dumps(bad), encoding="utf-8")
    cfg = tmp_path / "modules.json"
    cfg.write_text(json.dumps({"schema": 1, "manifest_url": f"{thin.url}/manifest-beta.json",
                               "latest_manifest_url": f"{thin.url}/manifest-latest.json"}), encoding="utf-8")
    mods.set_prefer_latest(True)
    assert mods.install() >= 1 and all(m.installed for m in mods.modules(mods.load_manifest()) if m.required)
    # pinned manifest gone (404): the newest release's manifest is used and remembered for the module list
    (out / "manifest-latest.json").write_text(json.dumps(man), encoding="utf-8")
    cfg.write_text(json.dumps({"schema": 1, "manifest_url": f"{thin.url}/gone.json",
                               "latest_manifest_url": f"{thin.url}/manifest-latest.json"}), encoding="utf-8")
    mods.set_prefer_latest(False)
    shutil.rmtree(mods.runtime_dir())
    (mods.paths.state_dir() / mods.MANIFEST_CACHE).unlink()
    assert mods.install() >= 1
    assert mods._channel_state()["installed_from"].endswith("/manifest-latest.json")
    assert all(m.installed for m in mods.modules(mods.load_manifest()) if m.required)
