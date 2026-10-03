"""An installer (PyInstaller) build has no per-user venv and no install manifest: verify must not demand one, repair must not need 'uv'
(found by running the real installer on Windows: --verify-install exit 1 and --repair exit 2 right after a clean install)."""
import sys

from infra import install_state as st


def _freeze(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)


def test_verify_does_not_require_a_manifest_when_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(st, "manifest_path", lambda: tmp_path / "none.json")
    assert not st.verify_install(has_module=lambda m: True).ok            # source install: no manifest = never completed
    _freeze(monkeypatch)
    assert st.verify_install(has_module=lambda m: True).ok                # installer build: fine
    assert not st.verify_install(has_module=lambda m: False).ok           # but a missing component is still reported


def test_repair_when_frozen_reports_instead_of_needing_uv(monkeypatch, tmp_path):
    monkeypatch.setattr(st, "manifest_path", lambda: tmp_path / "none.json")
    monkeypatch.setattr(st, "_has_module", lambda m: True)
    _freeze(monkeypatch)
    rc, text = st.repair_install(lambda c: (1, "must not run"), lambda n: None)
    from core.i18n import tr
    assert rc == 0 and text == tr("health.frozen_repair")


def test_cli_verify_prints_and_main_printer_writes_a_log(monkeypatch, tmp_path):
    import main
    from infra import paths

    monkeypatch.setattr(paths, "logs_dir", lambda: tmp_path)
    p = main._cli_printer("verify_install")
    p("one")
    p("two")
    assert (tmp_path / "verify_install.txt").read_text(encoding="utf-8").splitlines() == ["one", "two"]
    p2 = main._cli_printer("verify_install")           # a new run starts the file over
    p2("fresh")
    assert (tmp_path / "verify_install.txt").read_text(encoding="utf-8").splitlines() == ["fresh"]
