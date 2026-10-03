"""Shared pytest setup: every test gets an isolated app-data folder, a fixed language and no network/model-cache access."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Tests never write to the real application data folder."""
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "voxprint_home"))
    # Many tests compare Russian messages, so the default test language is Russian.
    # (The localization tests set VOXPRINT_LANG themselves.)
    monkeypatch.setenv("VOXPRINT_LANG", "ru")
    # The real machine's %APPDATA%/%LOCALAPPDATA% must not leak into the search for earlier installs (Windows)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata_roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata_local"))
    # Ordinary tests never touch the network or other programs' model caches (dedicated tests switch them on).
    monkeypatch.setenv("VOXPRINT_NO_EXTERNAL_MODELS", "1")
    monkeypatch.setenv("VOXPRINT_NO_MIRROR", "1")
    monkeypatch.setenv("VOXPRINT_NO_ENV_PROBE", "1")
    monkeypatch.setenv("VOXPRINT_OWN_ENV", "1")   # the test interpreter counts as Voxprint-owned (auto-upgrade); external-env tests pass external_env=True
    from core import i18n

    i18n.reset()
    yield
    i18n.reset()
