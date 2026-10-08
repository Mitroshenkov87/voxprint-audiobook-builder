"""Component updates (0.1.1 build 665 on a real PC): a rebuilt release with the same package versions is not an update, the
size shown is only what really changes, and an update is never installed without a click."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from infra import modules as mods  # noqa: E402
from tests.test_studio import app, wait_for  # noqa: F401  (fixtures)
from tools import online_fetch as of  # noqa: E402


def wheel(dist: str, version: str, size: int, sha: str = "a" * 64) -> dict:
    f = f"{dist}-{version}-py3-none-any.whl"
    return {"id": f"whl-{dist}-{version}", "file": f, "url": f"https://x/{f}", "size": size, "sha256": sha, "kind": "wheel",
            "role": "runtime", "module": "libs", "markers": [f"{dist}-{version}.dist-info/METADATA"]}


def manifest(comps: list, version: str = "0.1.1-beta", build: str = "665") -> dict:
    return {"schema": 1, "app_version": version, "build": build, "components": comps,
            "modules": [{"id": "libs", "title": "Libraries", "required": True, "components": [c["id"] for c in comps]}]}


def install_fake(comps: list, version: str = "0.1.0-beta") -> None:
    """As if these wheels were installed by an older release: state + dist-info markers in the runtime folder."""
    rd = mods.runtime_dir()
    for c in comps:
        (rd / c["markers"][0]).parent.mkdir(parents=True, exist_ok=True)
        (rd / c["markers"][0]).write_text("Metadata-Version: 2.1\n", encoding="utf-8")
    of.save_state(rd, {c["id"]: c["sha256"] for c in comps}, version, "")


def test_same_package_versions_rebuilt_are_not_an_update():
    old = [wheel("docopt", "0.6.2", 13_858, "1" * 64), wheel("sox", "1.5.0", 40_164, "2" * 64),
           wheel("transformers", "4.57.6", 12_000_000, "3" * 64)]
    install_fake(old)
    # build 665: the sdist-built wheels have other bytes, every package version is the same
    new = [wheel("docopt", "0.6.2", 13_900, "9" * 64), wheel("sox", "1.5.0", 40_200, "8" * 64), old[2]]
    (libs,) = mods.modules(manifest(new))
    assert libs.installed and not libs.update and mods.pending([libs]) == []


def test_a_new_package_version_is_an_update_of_only_that_size():
    old = [wheel("peft", "0.18.1", 500_000), wheel("transformers", "4.57.6", 12_000_000)]
    install_fake(old)
    new = [wheel("peft", "0.21.2", 600_000), old[1]]
    (libs,) = mods.modules(manifest(new))
    assert libs.update and not libs.installed and libs.size == 600_000          # not the 12.6 MB of the whole module
    assert mods.pending([libs]) == [libs]


def test_an_update_is_not_missing_but_a_missing_package_is(monkeypatch):
    old = [wheel("peft", "0.18.1", 500_000)]
    install_fake(old)
    man = manifest([wheel("peft", "0.21.2", 600_000)])
    monkeypatch.setattr(mods, "is_thin", lambda: True)
    monkeypatch.setattr(mods, "load_manifest", lambda *a, **k: man)
    (mods.paths.state_dir() / mods.MANIFEST_CACHE).parent.mkdir(parents=True, exist_ok=True)
    (mods.paths.state_dir() / mods.MANIFEST_CACHE).write_text(json.dumps(man), encoding="utf-8")
    monkeypatch.setattr(of, "validate_manifest", lambda d: d)
    assert mods.missing_required(man) == [] and mods.installed_without_network() is True   # no start-up auto-install
    man2 = manifest([wheel("peft", "0.21.2", 600_000), wheel("einops", "0.8.2", 50_000)])  # einops never installed
    (libs,) = mods.modules(man2)
    assert not libs.update and not libs.installed and libs.size == 650_000
    assert [m.id for m in mods.missing_required(man2)] == ["libs"]


def _dialog(mod_list, installs, autostart=True):
    from ui.modules_dialog import ModulesDialog

    def install_fn(ids, progress, cancelled):
        installs.append(list(ids))
        for m in mod_list:
            if m.id in ids:
                m.installed, m.update = True, False
        return len(ids)

    return ModulesDialog(manifest_fn=lambda: {"modules": []}, install_fn=install_fn, autostart=autostart)


def test_autostart_never_installs_an_update(app, monkeypatch):  # noqa: F811
    from core import i18n

    i18n.set_language("en")
    upd = mods.Module("libs", "Libraries", True, 600_000, 0, ["whl-peft-0.21.2"], installed=False, update=True)
    monkeypatch.setattr(mods, "modules", lambda man: [upd])
    installs: list = []
    d = _dialog([upd], installs)
    d.refresh()
    assert wait_for(lambda: d.modules and not d.busy, 10)
    QApplication.processEvents()
    assert installs == [] and d.lbl_status.text().startswith("Updates available: ") and d.btn_download.isEnabled()
    d.on_download_clicked()                                     # the user's click
    assert wait_for(lambda: installs == [["libs"]] and not d.busy, 10)
    d.shutdown()


def test_autostart_fetches_what_is_missing_but_not_the_update(app, monkeypatch):  # noqa: F811
    upd = mods.Module("libs", "Libraries", True, 600_000, 0, ["a"], installed=False, update=True)
    miss = mods.Module("torch", "PyTorch", True, 3 << 30, 0, ["b"], installed=False)
    monkeypatch.setattr(mods, "modules", lambda man: [upd, miss])
    installs: list = []
    d = _dialog([upd, miss], installs)
    d.refresh()
    assert wait_for(lambda: installs and not d.busy, 10)
    assert installs[0] == ["torch"] and not upd.installed
    d.shutdown()
