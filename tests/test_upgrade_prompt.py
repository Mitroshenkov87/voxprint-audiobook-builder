"""Outdated components in the USER's environment: ask first (offer_upgrade); accept -> upgrade, decline + compatible ->
reuse, decline + incompatible -> Voxprint's own copy; own environment -> automatic.  Plus Repair offer and model line."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from core import i18n
from infra import env_probe as ep, paths
from infra.updater import Updater
from infra.verified_manifest import Manifest
from infra.version_manager import (ACTION_INSTALL, ACTION_OFFER, ACTION_REUSE, ACTION_UPGRADE, decide_package,
                                   make_offers)
from ui.main_window import MainWindow
from ui.upgrade_dialog import UpgradeOfferDialog
from workers.process_worker import UpdateWorker

LANGS = list(i18n.LANGS)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ------------------------------------------------------------------------------------------ decisions
def test_external_decisions():
    d = decide_package("accelerate", "1.12.0", "1.15.0", "", external=True)
    assert (d.action, d.reason, d.outdated, d.compatible) == (ACTION_OFFER, "outdated", True, True)
    d = decide_package("transformers", "4.57.3", "4.57.6", ">=4.57.6,<5", external=True)
    assert d.action == ACTION_OFFER and d.reason == "incompatible" and not d.compatible and d.outdated
    # missing, current, newer: never a prompt
    assert decide_package("x", None, "1.0", "", external=True).action == ACTION_INSTALL
    assert decide_package("x", "1.0", "1.0", "", external=True).action == ACTION_REUSE
    assert decide_package("peft", "0.21.2", "0.18.1", "", external=True).action == ACTION_REUSE
    # too new for the constraint (not outdated): own copy, nothing to ask
    d = decide_package("transformers", "5.1.0", "4.57.6", ">=4.57.6,<5", external=True)
    assert d.action == ACTION_INSTALL and not d.compatible and not d.outdated


def test_own_environment_upgrades_automatically():
    assert decide_package("accelerate", "1.12.0", "1.15.0", "", external=False).action == ACTION_UPGRADE
    assert decide_package("transformers", "4.57.3", "4.57.6", ">=4.57.6,<5").action == ACTION_INSTALL


def test_is_own_environment(monkeypatch):
    monkeypatch.setenv("VOXPRINT_OWN_ENV", "0")
    assert not ep.is_own_environment()
    monkeypatch.setenv("VOXPRINT_OWN_ENV", "1")
    assert ep.is_own_environment()
    monkeypatch.delenv("VOXPRINT_OWN_ENV")
    assert not ep.is_own_environment()            # the test interpreter is not Voxprint's venv
    monkeypatch.setattr("sys.prefix", str(paths.app_home() / "venv"))
    assert ep.is_own_environment()


def test_probe_external_offers_and_torch_is_not_swapped_silently():
    from tests.test_env_install import fake_machine

    run, which, _ = fake_machine(smi="CUDA Version: 12.8")
    inst = {"accelerate": "1.12.0", "torch": "2.8.0+cpu", "transformers": "4.57.3"}.get
    pins = {"accelerate": "1.15.0", "transformers": "4.57.6"}
    kw = dict(scan_other_pythons=False, packages={"accelerate": "", "transformers": ">=4.57.6,<5"})
    rep = ep.probe_environment(pins, inst, run, which, external_env=True, **kw)
    act = {d.name: d.action for d in rep.decisions}
    assert act == {"accelerate": ACTION_OFFER, "transformers": ACTION_OFFER, "torch": ACTION_OFFER}
    assert rep.counts()[ACTION_OFFER] == 3 and rep.counts()[ACTION_UPGRADE] == 0
    msgs = "\n".join(ep.user_messages(rep))
    assert "1.12.0" in msgs and "1.15.0" in msgs
    own = ep.probe_environment(pins, inst, run, which, external_env=False, **kw)
    assert own.counts()[ACTION_OFFER] == 0 and own.counts()[ACTION_UPGRADE] >= 1


@pytest.mark.parametrize("lang", LANGS)
def test_env_offer_message_localized(lang):
    i18n.set_language(lang, persist=False)
    txt = i18n.tr("env.offer", name="accelerate", old="1.12.0", new="1.15.0")
    assert "accelerate" in txt and "1.15.0" in txt and "{" not in txt


# ------------------------------------------------------------------------------------------ updater flow
def _ext_updater(pkg="accelerate", old="1.12.0", new="1.15.0", constraint="", pip_ok=True, smoke_ok=True, **kw):
    log = {"pip": [], "smoke": []}

    def pip(cmd):
        log["pip"].append(cmd)
        if "--target" in cmd and pip_ok:
            site = Path(cmd[cmd.index("--target") + 1])
            (site / pkg).mkdir(parents=True, exist_ok=True)
            (site / pkg / "NEW").write_text("x")
        return (0 if pip_ok else 1), "pip output"

    def smoke(cmd):
        log["smoke"].append(cmd)
        return (0, 'SMOKE_OK {"%s": "%s"}' % (pkg, new)) if smoke_ok else (1, "ImportError")

    def fetch(url):
        return {"releases": {old: [{}], new: [{}]}} if "pypi.org" in url else {"sha": "S"}

    u = Updater(fetch_json=fetch, pip_runner=pip, smoke_runner=smoke, python_exe="python", now=lambda: 1_000_000.0,
                installed_fn={pkg: old}.get, packages={pkg: constraint}, models=(),
                manifest=Manifest(date="2026-10-02", packages={pkg: new}), external_env=True, **kw)
    return u, log


def _inplace(log):
    return [c for c in log["pip"] if "--target" not in c]


def test_accept_upgrades_in_place_without_target():
    u, log = _ext_updater()
    seen = []
    rep, res = u.check_and_apply(ask=lambda offers: seen.append(offers) or {o.name for o in offers})
    assert [o.name for o in seen[0]] == ["accelerate"] and seen[0][0].compatible
    cmds = _inplace(log)
    assert len(cmds) == 1 and "--upgrade" in cmds[0] and "accelerate==1.15.0" in cmds[0] and "--no-deps" in cmds[0]
    assert res.after == {"accelerate": "1.15.0"} and res.before == {"accelerate": "1.12.0"} and res.needs_restart
    assert not any("--target" in c for c in log["pip"]) and log["smoke"]


def test_decline_compatible_reuses_as_is():
    u, log = _ext_updater()
    _, res = u.check_and_apply(ask=lambda offers: set())
    assert log["pip"] == [] and res.after == {} and not res.needs_restart
    assert any("1.12.0" in m for m in res.messages)


def test_no_ask_callback_counts_as_decline():
    u, log = _ext_updater()
    _, res = u.check_and_apply()
    assert log["pip"] == [] and res.after == {}


def test_decline_incompatible_goes_to_own_overlay():
    u, log = _ext_updater(pkg="transformers", old="4.57.3", new="4.57.6", constraint=">=4.57.6,<5")
    _, res = u.check_and_apply(ask=lambda offers: set())
    assert _inplace(log) == [] and any("--target" in c for c in log["pip"])
    assert res.after == {"transformers": "4.57.6"}
    assert any("4.57.6" in m for m in res.messages)


def test_accept_incompatible_upgrades_in_place():
    u, log = _ext_updater(pkg="transformers", old="4.57.3", new="4.57.6", constraint=">=4.57.6,<5")
    offers_seen = []
    _, res = u.check_and_apply(ask=lambda o: offers_seen.extend(o) or {x.name for x in o})
    assert offers_seen and not offers_seen[0].compatible
    assert len(_inplace(log)) == 1 and not any("--target" in c for c in log["pip"])
    assert res.after == {"transformers": "4.57.6"}


def test_declined_offer_is_not_asked_again_for_same_target():
    u, log = _ext_updater()
    asked = []
    u.check_and_apply(ask=lambda o: asked.append(1) or set())
    u.check_and_apply(ask=lambda o: asked.append(1) or set())
    assert len(asked) == 1
    st = json.loads(u.state_file.read_text(encoding="utf-8"))
    assert st["declined_offers"] == {"accelerate": "1.15.0"}
    # a newer target is a new question
    u.manifest = lambda: Manifest(date="2026-10-03", packages={"accelerate": "1.16.0"})
    u.fetch_json = lambda url: {"releases": {"1.12.0": [{}], "1.16.0": [{}]}} if "pypi.org" in url else {"sha": "S"}
    u.check_and_apply(ask=lambda o: asked.append(1) or set())
    assert len(asked) == 2


def test_failed_smoke_rolls_back_to_old_version():
    u, log = _ext_updater(smoke_ok=False)
    _, res = u.check_and_apply(ask=lambda o: {x.name for x in o})
    cmds = _inplace(log)
    assert len(cmds) == 2 and "accelerate==1.12.0" in cmds[1] and "--upgrade" not in cmds[1]
    assert res.rolled_back and res.after == {} and not res.needs_restart
    assert any("1.12.0" in m for m in res.messages)


def test_own_environment_updater_is_automatic():
    u, log = _ext_updater()
    u.external_env = False
    called = []
    _, res = u.check_and_apply(ask=lambda o: called.append(o) or set())
    assert not called and any("--target" in c for c in log["pip"]) and res.after == {"accelerate": "1.15.0"}


def test_make_offers_env_is_reported():
    u, _ = _ext_updater()
    rep = u.check()
    offers = make_offers(rep, "C:/Python311")
    assert offers[0].env == "C:/Python311" and offers[0].target == "1.15.0"


# ------------------------------------------------------------------------------------------ worker
class _FakeUpdater:
    def __init__(self):
        self.got = None

    def should_autocheck(self):
        return True

    def check_and_apply(self, progress, ask=None):
        from infra.updater import UpdateResult
        from infra.version_manager import Offer

        self.got = ask([Offer("accelerate", "1.12.0", "1.15.0", True, "C:/Py")])
        return None, UpdateResult(after={"accelerate": "1.15.0"} if self.got else {})


def _run_worker(qapp, w):
    done = []
    w.done.connect(lambda s, c: done.append((s, c)))
    w.start()
    assert w.wait(10000)
    qapp.processEvents()
    return done


def test_worker_auto_answer(qapp):
    fu = _FakeUpdater()
    done = _run_worker(qapp, UpdateWorker(lambda: fu, auto_answer=lambda items: {i["name"] for i in items}))
    assert fu.got == {"accelerate"} and done[0][1] is True
    fu = _FakeUpdater()
    done = _run_worker(qapp, UpdateWorker(lambda: fu, auto_answer=lambda items: set()))
    assert fu.got == set() and done[0][1] is False


def test_worker_offer_signal_and_answer(qapp):
    fu = _FakeUpdater()
    w = UpdateWorker(lambda: fu)
    got = []
    # direct connection: answer from the emitting thread, the same way a queued GUI slot would call it
    w.offer.connect(lambda items: (got.append(items), w.answer([i["name"] for i in items])),
                    __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.ConnectionType.DirectConnection)
    done = _run_worker(qapp, w)
    assert got[0][0] == {"name": "accelerate", "installed": "1.12.0", "target": "1.15.0", "compatible": True,
                         "env": "C:/Py"}
    assert fu.got == {"accelerate"} and done[0][1]


def test_worker_timeout_means_decline(qapp):
    fu = _FakeUpdater()
    w = UpdateWorker(lambda: fu)
    w.ANSWER_TIMEOUT_S = 0.05
    _run_worker(qapp, w)
    assert fu.got == set()


def test_worker_supports_updaters_without_ask(qapp):
    from infra.updater import UpdateResult

    class Old:
        def should_autocheck(self):
            return True

        def check_and_apply(self, progress):
            return None, UpdateResult()

    assert _run_worker(qapp, UpdateWorker(lambda: Old()))[0][1] is False


# ------------------------------------------------------------------------------------------ dialog
OFFERS = [{"name": "accelerate", "installed": "1.12.0", "target": "1.15.0", "compatible": True, "env": "C:/Py311"},
          {"name": "transformers", "installed": "4.57.3", "target": "4.57.6", "compatible": False, "env": "C:/Py311"}]


@pytest.mark.parametrize("lang", LANGS)
def test_dialog_all_locales(qapp, lang):
    i18n.set_language(lang, persist=False)
    dlg = UpgradeOfferDialog(OFFERS)
    items = dlg.lbl_items.text()
    assert "accelerate: 1.12.0" in items and "1.15.0" in items and "4.57.6" in items
    assert "C:/Py311" in dlg.lbl_changes.text()
    assert dlg.btn_upgrade.text() == i18n.tr("upg.btn_upgrade") and dlg.btn_later.text() == i18n.tr("upg.btn_later")
    assert i18n.tr("upg.note_compatible") in items and i18n.tr("upg.note_incompatible") in items
    assert "{" not in items + dlg.lbl_changes.text() + dlg.lbl_intro.text()


def test_dialog_accept_and_reject(qapp):
    got = []
    dlg = UpgradeOfferDialog(OFFERS)
    dlg.decided.connect(got.append)
    dlg.btn_upgrade.click()
    assert got == [["accelerate", "transformers"]]
    got.clear()
    dlg = UpgradeOfferDialog(OFFERS)
    dlg.decided.connect(got.append)
    dlg.btn_later.click()
    dlg.reject()                                  # only one answer is ever emitted
    assert got == [[]]


def test_main_window_shows_dialog_and_answers_worker(qapp):
    win = MainWindow(autocheck=False, health_fn=lambda: [], model_states_fn=lambda: {})
    answers = []

    class W:
        def answer(self, names):
            answers.append(list(names))

    worker = W()                                  # the window keeps its real worker alive; here the test does
    win._show_offer(worker, OFFERS)
    assert win.upgrade_dialog is not None
    win.upgrade_dialog.btn_upgrade.click()
    assert answers == [["accelerate", "transformers"]]


# ------------------------------------------------------------------------------------------ repair + model line
def _wait_status(qapp, win):
    win.refresh_status()
    assert win.status_worker.wait(10000)
    qapp.processEvents()


@pytest.mark.parametrize("lang", LANGS)
def test_repair_banner_and_model_line_localized(qapp, lang):
    i18n.set_language(lang, persist=False)
    reasons = [i18n.tr("health.missing_module", detail="torch")]
    win = MainWindow(autocheck=False, health_fn=lambda: reasons,
                     model_states_fn=lambda: {"Qwen/Qwen3-ForcedAligner-0.6B": "ready",
                                              "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "partial",
                                              "Qwen/Other": "missing"})
    _wait_status(qapp, win)
    assert not win.health_row.isHidden() and win.btn_repair.text() == i18n.tr("ui.btn_repair")
    assert reasons[0] in win.lbl_health.text()
    line = win.lbl_models.text()
    assert not win.lbl_models.isHidden()
    for state in ("ready", "partial", "missing"):
        assert i18n.tr(f"ui.model_state_{state}") in line
    assert "Qwen3-ForcedAligner-0.6B" in line and "{" not in line
    # language switch re-renders the existing state
    other = "de" if lang != "de" else "en"
    win.set_language(other)
    assert i18n.tr("ui.model_state_ready") in win.lbl_models.text()
    assert win.btn_repair.text() == i18n.tr("ui.btn_repair")


def test_repair_banner_hidden_when_healthy(qapp):
    win = MainWindow(autocheck=False, health_fn=lambda: [], model_states_fn=lambda: {})
    _wait_status(qapp, win)
    assert win.health_row.isHidden() and win.lbl_models.isHidden()


def test_repair_button_runs_repair_fn(qapp):
    calls = []

    def repair(progress):
        progress(0.5, "step")
        calls.append(1)
        return 0, "Repair finished."

    healthy = {"v": False}
    win = MainWindow(autocheck=False, health_fn=lambda: [] if healthy["v"] else ["x"], model_states_fn=lambda: {},
                     repair_fn=repair)
    _wait_status(qapp, win)
    assert not win.health_row.isHidden()
    healthy["v"] = True
    win.btn_repair.click()
    assert win.repair_worker.wait(10000)
    qapp.processEvents()
    assert calls == [1] and win.lbl_status.text() == "Repair finished." and win.health_row.isHidden()
    win.status_worker.wait(10000)


def test_repair_install_function_and_no_uv(monkeypatch):
    from infra import install_state as ist

    rc, text = ist.repair_install(lambda a: (0, ""), lambda name: None)
    assert rc == 2 and text == i18n.tr("health.no_uv")


def test_default_health_skips_foreign_environment():
    from workers.process_worker import default_health

    assert default_health() == []                 # no Voxprint venv/manifest in the temp home


def test_default_health_reports_broken_install():
    from infra import install_state as ist
    from workers.process_worker import default_health

    ist.venv_dir().mkdir(parents=True)
    out = default_health()
    assert out and all("{" not in r for r in out)
