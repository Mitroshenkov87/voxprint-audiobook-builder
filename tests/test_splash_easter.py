"""The suite splash replaces the private window. The Easter-egg picture is not shown."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_easter_egg_caption_is_gone_from_every_catalog():
    for name in ("en", "de", "ru", "uk", "lv"):
        text = (ROOT / "locales" / f"{name}.json").read_text(encoding="utf-8")
        assert "easter_egg.caption" not in text


def test_the_egg_is_not_packaged():
    for name in ("build.bat", "build_thin.bat"):
        assert "easter-egg.jpg" not in (ROOT / name).read_text(encoding="utf-8")
    extra = (ROOT / "tools" / "make_linux_package.py").read_text(encoding="utf-8")
    assert "assets/easter-egg.jpg" not in extra
    note = (ROOT / "assets" / "ICON-LICENSE.txt").read_text(encoding="utf-8")
    assert "easter-egg.jpg" in note and "does not show it" in note


def test_suite_splash_uses_the_shared_window():
    pytest.importorskip("PySide6")
    from ui import splash

    assert splash.WIDTH == 960 and splash.HEIGHT == 540 and splash.IDENTITY == "#f5d76e"
    screen = splash.show()
    try:
        assert screen.width() == 1016 and screen.height() == 596
        assert not hasattr(screen, "record_click")
        info = splash.build_info()
        assert info.version == "1.0.0" and info.rc == 4 and info.build == 1002
        assert info.codename == "Ometz"
        assert '1.0.0 RC 4 build 1002 "Ometz"' == info.label()
    finally:
        screen.finish()


def test_splash_module_has_no_egg_api():
    import ui.splash as splash

    assert not hasattr(splash, "EGG_CLICKS")
    source = (ROOT / "ui" / "splash.py").read_text(encoding="utf-8")
    assert "easter" not in source.lower()
    pytest.importorskip("voxprint_common.ui.splash")
