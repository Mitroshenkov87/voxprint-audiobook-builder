import json
import re
from pathlib import Path

import pytest

from core import appinfo, i18n

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "credits.json").read_text(encoding="utf-8"))
LANGS = ("en", "de", "ru", "uk", "be")


def test_credits_json_well_formed():
    assert DATA["app"]["name"] == "Voxprint" and re.fullmatch(r"\d+\.\d+\.\d+", DATA["app"]["version"])
    assert DATA["app"]["author"] == "Aleksandr Mitroshenkov"
    comps = DATA["components"]
    ids = [c["id"] for c in comps]
    assert len(ids) == len(set(ids)) and len(comps) >= 35
    for c in comps:
        assert c["kind"] in ("model", "library", "tool", "asset"), c["id"]
        assert c["ship"] in ("bundled", "optional", "download", "build"), c["id"]
        assert c["name"] and c["license"] and c["url"].startswith(("http://", "https://")), c["id"]
        assert c["license_files"], c["id"]
        for f in c["license_files"]:
            assert (ROOT / "licenses" / f"{f}.txt").stat().st_size > 200, (c["id"], f)
        for field in ("purpose", "note"):
            if field in c:
                assert set(c[field]) == set(LANGS), (c["id"], field)
                assert all(c[field][l].strip() for l in LANGS), (c["id"], field)
        assert "purpose" in c


def test_all_required_projects_are_credited():
    names = " ".join(c["name"].lower() for c in DATA["components"])
    for needed in ("qwen3-tts", "forcedaligner", "qwen-tts", "qwen-asr", "pytorch", "transformers", "peft",
                   "accelerate", "bitsandbytes", "pyside6", "numpy", "scipy", "librosa", "soundfile", "pydub",
                   "imageio-ffmpeg", "ffmpeg", "ru-normalizr", "rutextnorm", "pymorphy3", "num2words", "tabler",
                   "pyinstaller", "inno setup", "ctc-forced-aligner", "mms-300m"):
        assert needed in names, needed


def test_license_flags():
    by = {c["id"]: c for c in DATA["components"]}
    assert "NC" in by["mms-aligner"]["license"] and by["mms-aligner"]["ship"] == "optional"
    assert "LGPL-3.0" in by["pyside6"]["license"] and "onedir" in by["pyside6"]["note"]["en"]
    assert "GPL" in by["ffmpeg"]["license"]
    assert "soynlp" not in json.dumps(DATA).lower().replace("excluded", "")


def test_soynlp_gpl_dependency_is_not_shipped():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert not [l for l in req.splitlines() if l.strip().lower().startswith("soynlp")]
    bat = (ROOT / "build.bat").read_bytes().decode("utf-8")
    assert "--exclude-module soynlp" in bat
    assert "--hidden-import soynlp" not in bat and "--collect-data soynlp" not in bat
    assert "soynlp" in (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")   # объяснение в notices


def test_notices_in_sync_with_credits_json():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen_notices", ROOT / "tools" / "gen_notices.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    expected = mod.render(DATA)
    actual = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert actual == expected, "run: python tools/gen_notices.py"
    assert mod.main(["--check"]) == 0
    appendix = mod.installed_appendix()
    assert "| Package | Version |" in appendix and "pytest" in appendix.lower()


def test_installer_and_build_ship_notices_and_keep_encoding():
    iss = (ROOT / "installer" / "Voxprint.iss").read_bytes()
    assert iss.startswith(b"\xef\xbb\xbf") and iss.count(b"\r\n") == iss.count(b"\n")
    text = iss.decode("utf-8-sig")
    assert "THIRD_PARTY_NOTICES.md" in text and "licenses\\*" in text
    assert f'#define AppVersion "{DATA["app"]["version"]}"' in text
    bat = (ROOT / "build.bat").read_bytes()
    assert bat.count(b"\r\n") == bat.count(b"\n")
    b = bat.decode("utf-8")
    for needle in ("gen_notices.py", 'locales;locales', "credits.json;.", "licenses;licenses"):
        assert needle in b, needle


def test_about_info_matches_credits():
    assert appinfo.APP_VERSION == DATA["app"]["version"] and appinfo.REPO_URL == DATA["repo_url"]
    assert len(appinfo.components()) == len(DATA["components"])
    assert appinfo.notices_path().exists() and appinfo.licenses_dir().is_dir()


# ------------------------------------------------------------------ ссылка на репозиторий
def test_repo_url_placeholder_is_hidden():
    assert "OWNER" in DATA["repo_url"]                         # пока заглушка
    assert appinfo.public_repo_url() is None
    assert appinfo.public_repo_url("https://github.com/OWNER/voxprint") is None
    assert appinfo.public_repo_url("") is None
    assert appinfo.public_repo_url("https://github.com/alex/voxprint") == "https://github.com/alex/voxprint"


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_about_dialog_hides_link_for_placeholder_and_shows_for_real_url(app):
    from PySide6.QtCore import QUrl
    from ui.about_dialog import AboutDialog
    opened = []
    d = AboutDialog(None, open_url=lambda u: opened.append(u.toString()) or True)
    d.show()
    assert d.btn_repo.isHidden() and d.repo_link is None          # заглушка OWNER -> ссылки нет
    assert d.open_repo() is False and opened == []
    d.close()
    d = AboutDialog(None, repo_url="https://github.com/alex/voxprint",
                    open_url=lambda u: opened.append(u.toString()) or True)
    d.show()
    assert not d.btn_repo.isHidden() and d.btn_repo.toolTip() == "https://github.com/alex/voxprint"
    d.btn_repo.click()
    assert opened == ["https://github.com/alex/voxprint"]          # открывается в браузере по умолчанию
    d.close()


def test_about_dialog_content_in_every_language(app):
    from ui.about_dialog import AboutDialog
    for lang in LANGS:
        i18n.set_language(lang)
        d = AboutDialog(None)
        text = d.body.toPlainText()
        assert "Aleksandr Mitroshenkov" in text and "Qwen3-TTS" in text and "PySide6" in text
        assert "26H2" in text and "16" in text and "LGPL" in text
        assert i18n.tr("about.step1") in text and i18n.tr("about.privacy") in text
        assert d.windowTitle() == i18n.tr("about.title") and d.btn_notices.text() == i18n.tr("about.btn_notices")
        d.close()


def test_about_notices_button_opens_file(app):
    from ui.about_dialog import AboutDialog
    opened = []
    d = AboutDialog(None, open_url=lambda u: opened.append(u.toLocalFile()) or True)
    assert d.open_notices() is True and opened[0].endswith("THIRD_PARTY_NOTICES.md")


def test_main_window_about_button_opens_dialog(app):
    from ui.main_window import MainWindow
    w = MainWindow(runner=lambda *a: None, autocheck=False, auto_open_folder=False)
    assert w.btn_about.isEnabled()
    w.btn_about.click()
    assert w._about.isVisible() and w._about.btn_repo.isHidden()
    w._about.close()
    w.close()
