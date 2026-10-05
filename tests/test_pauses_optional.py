"""Explicit pauses are opt-in: off by default, remembered when switched."""
from core import narration as nr
from core import pauses as pz


def test_options_default_without_explicit_pauses():
    assert nr.NarrationOptions().pauses is None


def test_enabled_flag_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(pz, "_file", lambda: tmp_path / "pauses.txt")
    assert pz.load_enabled() is False
    pz.save_enabled(True)
    assert pz.load_enabled() is True
    pz.save_enabled(False)
    assert pz.load_enabled() is False
