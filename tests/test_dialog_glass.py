"""Dialogs get exactly the main window's translucent look (0.1.1 build 665 bug: Settings showed a solid grey block over a
black, backdrop-less frame while the main window was Acrylic)."""
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QScrollArea  # noqa: E402

from tests.test_studio import app, lib, make_studio  # noqa: E402,F401


def test_settings_scroll_content_is_transparent(app, lib):  # noqa: F811
    """QScrollArea.setWidget turns on an opaque palette fill: the Settings content must stay see-through like the windows."""
    s = make_studio(lib)
    d = s.settings_dialog()
    d.show()
    QApplication.processEvents()
    for sa in d.findChildren(QScrollArea):
        assert not sa.viewport().autoFillBackground()
        assert sa.widget() is None or not sa.widget().autoFillBackground()
    sa = d.findChild(QScrollArea, "settingsScroll")
    assert sa is not None and sa.widget().objectName() == "content"   # QWidget#content {background: transparent}
    d.close()
    s.shutdown()


def test_dialog_root_uses_the_window_tint_not_a_solid_override():
    from ui import main_window as mw

    for glass in (True, False):
        css = mw.build_style(glass)
        assert "QDialog#root { background" not in css.replace("  ", " ")
        assert "QTextBrowser { background: transparent" in css
    assert mw.ROOT_GLASS in mw.build_style(True) or mw.ROOT_GLASS_MORE in mw.build_style(True)


def test_settings_look_is_its_own_not_copied_from_the_hidden_train_window(app, lib):  # noqa: F811
    """Opened from the Studio, the (hidden, never shown) Train window is still plain; its stylesheet must not be copied."""
    from ui import main_window as mw

    s = make_studio(lib)
    for win in (s, s.trainer):
        win.setStyleSheet("/* plain */")
        d = win.settings_dialog()
        d.backdrop = "acrylic"           # as on Windows 11 after apply_backdrop
        d._backdrop_tried = True
        win.open_settings()              # offscreen: show() instead of exec()
        QApplication.processEvents()
        assert d.styleSheet() == mw.build_style(True), type(win).__name__
        d.close()
    s.shutdown()


def test_backdrop_is_requested_again_after_the_fade_in_and_for_a_new_native_window(app, monkeypatch):  # noqa: F811
    from ui import main_window as mw
    from ui.glass import GlassDialog

    calls = []
    monkeypatch.setattr(mw.platform_win, "apply_backdrop", lambda hwnd, dark=True: calls.append(hwnd) or "acrylic")
    d = GlassDialog()
    d.show()
    QApplication.processEvents()
    monkeypatch.setattr(mw.sys, "platform", "win32")
    try:
        d.backdrop = "acrylic"
        d.setWindowOpacity(0.0)
        mw._end_fade_in(d)                      # layered during the first request: ask again once it is opaque
        assert d.windowOpacity() == 1.0 and calls == [int(d.winId())]
        d._backdrop_tried = True
        d._backdrop_hwnd = 0                    # Qt re-created the native window
        mw.apply_look(d)
        assert calls[-1] == int(d.winId()) and d._backdrop_hwnd == int(d.winId())
        n = len(calls)
        mw.apply_look(d)                        # same window: not asked again
        assert len(calls) == n
    finally:
        monkeypatch.undo()
        d.close()
