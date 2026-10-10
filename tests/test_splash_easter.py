"""Splash easter egg: six clicks inside two seconds, once per launch, without holding up loading."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from core.i18n import LANGS, set_language, tr
from tests.test_icon_artwork import _jpeg_size
from ui import splash

ROOT = Path(__file__).resolve().parents[1]
CAPTION = "Кури ушную серу!"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_six_clicks_inside_two_seconds_trigger():
    stamps: list[float] = []
    hit = False
    for i in range(splash.EGG_CLICKS):
        stamps, hit = splash.record_click(stamps, i * 0.3)
    assert hit and len(stamps) == splash.EGG_CLICKS
    assert splash.EGG_WINDOW_S == 2.0 and splash.EGG_HOLD_S == 3.0


def test_five_clicks_inside_the_window_do_not_trigger():
    stamps: list[float] = []
    hit = False
    for i in range(splash.EGG_CLICKS - 1):
        stamps, hit = splash.record_click(stamps, i * 0.2)
    assert not hit and len(stamps) == 5


def test_a_click_on_the_window_edge_still_counts():
    stamps: list[float] = []
    for t in (0.0, 0.4, 0.8, 1.2, 1.6):
        stamps, hit = splash.record_click(stamps, t)
        assert not hit
    stamps, hit = splash.record_click(stamps, 2.0)          # 2.0 - 0.0 == the window, still inside
    assert hit


def test_clicks_older_than_two_seconds_drop_out_of_the_window():
    stamps: list[float] = []
    for t in (0.0, 0.4, 0.8, 1.2, 1.6):
        stamps, _hit = splash.record_click(stamps, t)
    stamps, hit = splash.record_click(stamps, 2.0 + 1e-6)   # the click at 0.0 is outside
    assert not hit and 0.0 not in stamps and len(stamps) == 5
    stamps, hit = splash.record_click(stamps, 2.2)          # the five that remain, plus this one
    assert hit and len(stamps) == 6


def test_caption_is_the_russian_joke_in_every_language():
    for lang in LANGS:
        set_language(lang)
        assert tr("easter_egg.caption") == CAPTION


def test_easter_egg_image_is_original_artwork_and_is_shipped():
    path = ROOT / "assets" / "easter-egg.jpg"
    assert path.is_file() and _jpeg_size(path) == (1280, 720)
    note = (ROOT / "assets" / "ICON-LICENSE.txt").read_text(encoding="utf-8")
    assert "easter-egg.jpg" in note and "original generated artwork" in note
    assert "no real person's likeness" in note
    for name in ("build.bat", "build_thin.bat"):
        assert "easter-egg.jpg" in (ROOT / name).read_text(encoding="utf-8")
    extra = (ROOT / "tools" / "make_linux_package.py").read_text(encoding="utf-8")
    assert "assets/easter-egg.jpg" in extra


def _burst(screen: splash.Splash, n: int) -> None:
    for _ in range(n):
        QTest.mouseClick(screen, Qt.MouseButton.LeftButton)


def test_six_clicks_show_the_overlay_once_and_a_click_closes_it(app):
    set_language("en")
    screen = splash.show()
    try:
        _burst(screen, splash.EGG_CLICKS - 1)
        assert not screen.egg_visible()
        started = time.monotonic()
        _burst(screen, 1)
        assert time.monotonic() - started < 0.5          # painting the overlay does not wait
        assert screen.egg_visible() and screen.isVisible()
        assert screen.egg_caption() == CAPTION
        assert screen._egg_timer is not None and screen._egg_timer.interval() == int(splash.EGG_HOLD_S * 1000)
        assert not screen.grab().isNull()                 # grab() runs drawContents, caption included
        _burst(screen, 1)                                # a click dismisses it
        assert not screen.egg_visible()
        _burst(screen, splash.EGG_CLICKS)                # once per launch
        assert not screen.egg_visible()
    finally:
        screen.close()


def test_finish_returns_at_once_and_waits_only_until_the_overlay_closes(app):
    screen = splash.show()
    window = QWidget()
    window.show()
    try:
        _burst(screen, splash.EGG_CLICKS)
        assert screen.egg_visible()
        started = time.monotonic()
        screen.finish(window)                            # the main window is ready; do not block
        assert time.monotonic() - started < 0.5
        assert screen.isVisible() and screen.egg_visible()
        screen.dismiss_egg()
        assert not screen.egg_visible() and not screen.isVisible()
    finally:
        screen.close()
        window.close()
