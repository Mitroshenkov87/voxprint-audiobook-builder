"""Windows fit a 16:10 laptop at 150 % (2560x1600 -> 1707x1067 logical, ~1707x1027 without the taskbar); transparency."""
from unittest import mock

import pytest
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication, QWidget

from core import i18n
from core.voice_library import VoiceLibrary
from infra import ui_prefs
from tests.test_studio import make_studio
from ui import main_window, screen_fit

AVAIL = QRect(0, 0, 1707, 1027)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_fit_centres_and_clamp_pulls_a_window_back_on_screen(app):
    w = QWidget()
    screen_fit.fit(w, 3000, 3000, avail=AVAIL)
    assert AVAIL.contains(w.geometry()) and abs(w.geometry().center().x() - AVAIL.center().x()) <= 1
    w.setGeometry(1500, 900, 2400, 1400)                 # too big and partly off-screen
    screen_fit.clamp(w, AVAIL)
    assert AVAIL.contains(w.frameGeometry())


@pytest.mark.parametrize("lang", ["de", "ru"])           # the longest strings
def test_every_window_fits_1707x1027_and_narrate_uses_two_columns(app, tmp_path, lang):
    from ui.about_dialog import AboutDialog
    from ui.modules_dialog import ModulesDialog
    from ui.settings_dialog import SettingsDialog
    from ui.upgrade_dialog import UpgradeOfferDialog

    i18n.set_language(lang)
    s = make_studio(VoiceLibrary(tmp_path / "voices"))
    wins = [s, s.trainer, s.voices_window, s.narrate_window, s.revoice_window, SettingsDialog(s), AboutDialog(s),
            ModulesDialog(parent=s), UpgradeOfferDialog([{"name": "x", "target": "2", "installed": "1"}], s)]
    with mock.patch.object(screen_fit, "available", lambda w: AVAIL):
        for w in wins:
            m = w.minimumSizeHint().expandedTo(w.minimumSize())
            assert m.width() <= AVAIL.width() and m.height() <= AVAIL.height(), (type(w).__name__, m)
            w.resize(2000, 1400)
            w.move(-50, 600)
            w.show()                                      # the clamp runs on show in the app; here explicitly
            screen_fit.clamp(w)
            assert AVAIL.contains(w.frameGeometry()), type(w).__name__
            w.hide()
        screen_fit.fit(s, 1040, 900)
        s.show()
        s.navigate("narrate")
        app.processEvents()
        assert s.narrate_window.flow.two and AVAIL.contains(s.narrate_window.frameGeometry())
    s.shutdown()


def test_window_transparency_levels(app):
    assert ui_prefs.transparency() == "default"
    ui_prefs.set_transparency("more")
    assert ui_prefs.transparency() == "more" and main_window.ROOT_GLASS_MORE in main_window.build_style(True)
    ui_prefs.set_transparency("off")
    assert main_window.build_style(True) == main_window.build_style(False)
    with pytest.raises(ValueError):
        ui_prefs.set_transparency("max")


def test_splash_shows_a_status_and_closes_with_the_main_window(app):
    from ui import splash

    s = splash.show()
    s.status("Loading interface\u2026")
    assert s.isVisible() and s.message() == "Loading interface\u2026"
    assert s.width() <= AVAIL.width() and s.height() <= AVAIL.height() and s.width() == s.height()
    s.checking()
    assert s._fraction == 0.7 and not s.pixmap().isNull() and not s.grab().isNull()     # grab() runs drawContents
    w = QWidget()
    w.show()
    s.finish(w)
    assert not s.isVisible()


def test_transparency_switched_back_on_gives_a_shown_window_its_blur(app, monkeypatch):
    calls = []
    monkeypatch.setattr(main_window.platform_win, "apply_backdrop", lambda hwnd: calls.append(hwnd) or "acrylic")
    w = QWidget()
    w.backdrop = "plain"
    w.show()
    ui_prefs.set_transparency("off")
    with monkeypatch.context() as m:
        m.setattr(main_window.sys, "platform", "win32")
        main_window.apply_look(w)                         # shown while "off": no backdrop, solid look
        assert not calls and w.styleSheet() == main_window.build_style(False)
        ui_prefs.set_transparency("more")
        main_window.apply_look(w)                         # switched back on: the backdrop is applied now
    assert calls and w.backdrop == "acrylic" and main_window.ROOT_GLASS_MORE in w.styleSheet()
