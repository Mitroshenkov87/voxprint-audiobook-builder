"""The suite splash is the shared window. Six clicks open the egg through on_click."""
import json
from pathlib import Path

import pytest
from PySide6.QtGui import QGuiApplication

ROOT = Path(__file__).resolve().parents[1]
CAPTION = "Кури ушную серу!"


def test_easter_egg_caption_is_the_same_russian_line_in_every_catalog():
    for name in ("en", "de", "ru", "uk", "lv"):
        cat = json.loads((ROOT / "locales" / f"{name}.json").read_text(encoding="utf-8"))
        assert cat["easter_egg.caption"] == CAPTION


def test_the_egg_is_packaged():
    for name in ("build.bat", "build_thin.bat"):
        assert "easter-egg.jpg" in (ROOT / name).read_text(encoding="utf-8")
    extra = (ROOT / "tools" / "make_linux_package.py").read_text(encoding="utf-8")
    assert "assets/easter-egg.jpg" in extra
    note = (ROOT / "assets" / "ICON-LICENSE.txt").read_text(encoding="utf-8")
    assert "on_click" in note and "easter-egg.jpg" in note


def test_record_click_needs_six_inside_two_seconds():
    from ui.splash import record_click

    stamps: list[float] = []
    for moment in (0.0, 0.2, 0.4, 0.6, 0.8):
        stamps, hit = record_click(stamps, moment)
        assert hit is False
    stamps, hit = record_click(stamps, 1.0)
    assert hit is True
    late, hit = record_click([0.0], 2.1)
    assert hit is False and late == [2.1]


def test_six_clicks_open_the_egg_through_on_click_and_another_click_closes_it():
    pytest.importorskip("PySide6")
    from voxprint_common.ui.splash import SplashEvent

    from ui import splash

    assert splash.WIDTH == 960 and splash.HEIGHT == 540 and splash.IDENTITY == "#f5d76e"
    screen = splash.show()
    try:
        work = QGuiApplication.primaryScreen().availableGeometry()
        assert screen.width() in (1016, min(1016, work.width()))
        assert screen.height() in (596, min(596, work.height()))
        info = splash.build_info()
        assert info.version == "1.0.0" and info.rc == 4 and info.build == 1002
        assert info.codename == "Ometz"
        assert info.label() == '1.0.0 RC 4 build 1002 "Ometz"'
        assert not screen.egg_visible()
        for _ in range(5):
            screen._on_click(SplashEvent(1, 1, "left"))
        assert not screen.egg_visible()
        screen._on_click(SplashEvent(1, 1, "left"))
        assert screen.egg_visible()
        screen._on_click(SplashEvent(2, 2, "left"))
        assert not screen.egg_visible()
    finally:
        screen._dismiss_egg()
        screen.finish()
