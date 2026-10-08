"""0.1.1 fixes from the first PC test: no stray start-up window, aligned Settings, one "Check & repair" button, dialogs
with the main window's look, the window size after Narrate, the projects folder, FLAC training clips."""
import os

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QObject, QSize  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QGridLayout  # noqa: E402

from tests.test_studio import app, lib, make_studio  # noqa: E402,F401


class _Shows(QObject):
    def __init__(self):
        super().__init__()
        self.windows = []

    def eventFilter(self, obj, e):  # noqa: N802
        if e.type() == QEvent.Type.Show and obj.isWidgetType() and obj.isWindow():
            self.windows.append(type(obj).__name__ + ":" + obj.objectName())
        return False


def test_no_stray_top_level_window_while_the_studio_is_built(app, lib):  # noqa: F811
    """A button made visible before it had a parent flashed as its own blank "Voxprint" window at start-up."""
    f = _Shows()
    app.installEventFilter(f)
    try:
        s = make_studio(lib)
        s.show_studio()
        QApplication.processEvents()
    finally:
        app.removeEventFilter(f)
    assert f.windows == ["StudioWindow:root"], f.windows
    assert not s.narrate_window.btn_back.isHidden() and s.narrate_window.btn_back.parent() is not None
    s.shutdown()


def test_settings_combos_share_one_grid_and_repair_is_one_button(app, lib):  # noqa: F811
    from core import i18n

    i18n.set_language("en")
    s = make_studio(lib)
    d = s.settings_dialog()
    grids = [lay for lay in d.findChildren(QGridLayout)]
    combos = (d.cmb_lang, d.cmb_transparency, d.cmb_net, d.cmb_asr)
    assert any(all(g.indexOf(c) >= 0 for c in combos) for g in grids)
    assert all(c.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToContents for c in combos)
    assert d.btn_repair is d.btn_autorepair and d.btn_autorepair.text() == "Check && repair"
    assert "SHA-256" in d.lbl_repair_desc.text()
    s.shutdown()


def test_dialogs_use_the_glass_look(app):  # noqa: F811
    from PySide6.QtCore import Qt

    from ui.about_dialog import AboutDialog
    from ui.glass import GlassDialog
    from ui.modules_dialog import ModulesDialog
    from ui.settings_dialog import SettingsDialog
    from ui.upgrade_dialog import UpgradeOfferDialog

    for cls in (AboutDialog, ModulesDialog, SettingsDialog, UpgradeOfferDialog):
        assert issubclass(cls, GlassDialog)
    d = AboutDialog()
    assert d.objectName() == "root" and d.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    d.show()
    QApplication.processEvents()
    assert d.styleSheet()                      # apply_look ran on show
    d.close()


def test_back_from_narrate_restores_the_earlier_size(app, lib):  # noqa: F811
    s = make_studio(lib)
    s.show_studio()
    s.resize(900, 600)
    QApplication.processEvents()
    s.narrate_window.wide_width = lambda: 1300
    s.navigate("narrate")
    QApplication.processEvents()
    s.navigate("studio")
    QApplication.processEvents()
    assert s.size() == QSize(900, 600) or s.width() <= 900
    s.shutdown()


def test_accessible_names_of_the_cards(app, lib):  # noqa: F811
    s = make_studio(lib)
    assert s.card_narrate.accessibleName() and s.card_train.accessibleDescription()
    s.shutdown()


def test_projects_folder_default_choice_onedrive_and_shortcut(tmp_path, monkeypatch):
    from infra import paths, projects

    assert projects.projects_dir() == paths.app_home() / "Projects" and projects.projects_dir().is_dir()
    other = tmp_path / "big" / "Projects"
    assert projects.set_projects_dir(other) == other and projects.sub(projects.VOICES) == other / "Voices"
    assert projects.set_projects_dir(None) == paths.app_home() / "Projects"
    monkeypatch.setenv("OneDrive", str(tmp_path / "OneDrive"))
    assert projects.in_onedrive(tmp_path / "OneDrive" / "Documents" / "Voxprint")
    assert not projects.in_onedrive(tmp_path / "local")
    assert projects.documents_dir(lambda: str(tmp_path / "OneDrive" / "Documents")) == tmp_path / "OneDrive" / "Documents"
    docs = tmp_path / "docs"
    docs.mkdir()
    made = []
    link = projects.ensure_shortcut(other, docs, make_link=lambda lnk, tgt: made.append((lnk, tgt)))
    assert link is not None and link.parent == docs and made == [(link, other)]
    if os.name != "nt":
        assert projects.ensure_shortcut(other, docs) == docs / "Voxprint Projects"
        assert (docs / "Voxprint Projects").resolve() == other.resolve()


def test_training_project_goes_into_the_projects_folder(tmp_path):
    from workers.pipeline_runner import TaskRequest

    req = TaskRequest(kind="dataset", audio=tmp_path / "a" / "anna.wav", text=tmp_path / "a" / "t.txt",
                      projects_root=tmp_path / "Projects" / "Voices")
    assert req.resolved_root() == tmp_path / "Projects" / "Voices" / "anna_Voxprint"
    old = TaskRequest(kind="dataset", audio=tmp_path / "a" / "anna.wav", text=tmp_path / "a" / "t.txt")
    assert old.resolved_root().parent == (tmp_path / "a").resolve()


def test_training_clips_are_flac():
    from core.types import Segment

    assert Segment(index=7, start=0.0, end=1.0, text="x").filename == "segment_007.flac"


def test_build_number_and_codename(tmp_path, monkeypatch):
    import json

    from tools import build_number as bn

    # runs 35 (failed) and 37 (cancelled for the codename change) published nothing: offset 629 from run 38 on
    assert bn.build_number({"GITHUB_RUN_NUMBER": "38"}) == 667 and bn.build_number({}) == 0
    assert bn.build_number({"GITHUB_RUN_NUMBER": "39"}) == 668
    assert bn.build_number({"VOXPRINT_BUILD": "700", "GITHUB_RUN_NUMBER": "34"}) == 700
    name = bn.info()["codename"]
    assert name.isascii() and name.isalpha()
    c = tmp_path / "credits.json"
    c.write_text(json.dumps({"app": {"version": "0.1.1"}}), encoding="utf-8")
    bn.stamp(c, 665, "0123456789abcdef", "Tikkun")
    assert json.loads(c.read_text(encoding="utf-8"))["app"] == {"version": "0.1.1", "build": 665, "codename": "Tikkun",
                                                               "commit": "01234567"}
    from core import appinfo

    monkeypatch.setattr(appinfo, "APP_BUILD", 665)
    monkeypatch.setattr(appinfo, "APP_CODENAME", "Tikkun")
    monkeypatch.setattr(appinfo, "APP_CHANNEL", "beta")
    monkeypatch.setattr(appinfo, "APP_VERSION", "0.1.1")
    assert appinfo.version_label() == '0.1.1-beta \u00b7 build 665 "Tikkun"'
