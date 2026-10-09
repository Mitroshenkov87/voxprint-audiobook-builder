"""Shared models folder users file ``models/.users.json`` (spec agreed with the Movie Dubber)."""
from __future__ import annotations

import json

import pytest

from infra import models_users as mu


def test_register_keeps_other_programs_and_is_idempotent(tmp_path):
    (tmp_path / ".users.json").write_text('\ufeff{"movie-dubber": true}', encoding="utf-8")   # a BOM is accepted
    assert mu.register(root=tmp_path) == {"movie-dubber": True, "audiobook-builder": True}
    before = (tmp_path / ".users.json").stat().st_mtime_ns
    mu.register(root=tmp_path)
    assert (tmp_path / ".users.json").stat().st_mtime_ns == before          # no write when already there
    assert json.loads((tmp_path / ".users.json").read_text(encoding="utf-8")) == {
        "audiobook-builder": True, "movie-dubber": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_unregister_reports_the_other_users(tmp_path):
    mu.register(root=tmp_path)
    mu.register("movie-dubber", root=tmp_path)
    assert mu.unregister(root=tmp_path) == ["movie-dubber"]
    assert mu.read_users(tmp_path) == {"movie-dubber": True}
    assert mu.unregister("movie-dubber", root=tmp_path) == []
    assert mu.read_users(tmp_path) == {}


def test_damaged_or_missing_file_counts_as_no_users(tmp_path):
    assert mu.read_users(tmp_path) == {}
    (tmp_path / ".users.json").write_text("not json", encoding="utf-8")
    assert mu.read_users(tmp_path) == {}
    assert mu.register(root=tmp_path) == {"audiobook-builder": True}


def test_unregister_cli_writes_count_and_folder(tmp_path, monkeypatch, capsys):
    from infra import paths

    models = tmp_path / "models"
    monkeypatch.setattr(paths, "models_dir", lambda: models)
    mu.register()
    mu.register("movie-dubber")
    out = tmp_path / "out.txt"
    assert mu.unregister_cli(str(out)) == 0
    assert out.read_text(encoding="utf-8").splitlines() == ["1", str(models)]
    assert mu.read_users() == {"movie-dubber": True}
    assert "movie-dubber" in capsys.readouterr().out


def test_unregister_cli_error_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(mu, "unregister", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    out = tmp_path / "out.txt"
    assert mu.unregister_cli(str(out)) == 1 and not out.exists()


def test_main_flags(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    import main as app_main
    from infra import paths

    models = tmp_path / "m"
    monkeypatch.setattr(paths, "models_dir", lambda: models)
    assert app_main.main(["Voxprint.exe", "--register-models-user"]) == 0
    assert mu.read_users(models) == {"audiobook-builder": True}
    out = tmp_path / "o.txt"
    assert app_main.main(["Voxprint.exe", "--unregister-models-user", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").splitlines()[0] == "0" and mu.read_users(models) == {}


def test_installer_and_ci_use_the_same_file_name():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    iss = (root / "installer" / "Voxprint.iss").read_text(encoding="utf-8-sig")
    assert "'.users.json'" in iss and '"audiobook-builder"' in iss and "--unregister-models-user" in iss
    assert "UninstallModelsQuestion" in iss and "MB_DEFBUTTON2" in iss
    for lang in ("english", "russian", "german"):
        assert f"{lang}.UninstallModelsQuestion=" in iss
    assert mu.USERS_FILE == ".users.json" and mu.USER_KEY == "audiobook-builder"
