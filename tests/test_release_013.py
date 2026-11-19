"""Build 667 fixes from the clean install of build 666: translator fetching one by one, installer behaviours.

The download machinery these fixes originally covered (partials, hub timeouts, finalize) belonged to the app's own
downloader, which is gone: VOXPrint MDL owns the store. What remains: the translator pass and the installer tests.
"""
from pathlib import Path

import pytest

from core import i18n
from core.errors import CancelledByUser, ModelDownloadError

ROOT = Path(__file__).resolve().parents[1]


def _iss() -> str:
    return (ROOT / "installer" / "Voxprint.iss").read_text(encoding="utf-8-sig")


# ------------------------------------------------------------------------------------------------ 1. WinError 5 on rename
def test_check_and_repair_asks_mdl_to_verify_and_repair(monkeypatch):
    from core.events import CancelToken
    from infra import auto_repair

    calls = []
    item = auto_repair.check_models(lambda f, m: calls.append((f, m)), CancelToken(),
                                    verify=lambda progress: calls.append(("verify", progress)) or True)
    assert item.status == auto_repair.OK and calls and calls[0][0] == "verify"


def test_check_and_repair_reports_a_failed_model_store(monkeypatch):
    from core.events import CancelToken
    from infra import auto_repair

    monkeypatch.setenv("VOXPRINT_SKIP_SUITE", "0")
    item = auto_repair.check_models(lambda f, m: None, CancelToken(), verify=lambda progress: False)
    assert item.status == auto_repair.FAILED and "Manage models" in item.detail


def test_check_and_repair_skips_the_model_store_under_suite_skip():
    from core.events import CancelToken
    from infra import auto_repair

    assert auto_repair.check_models(lambda f, m: None, CancelToken()).status == auto_repair.SKIPPED


# ------------------------------------------------------------------------------------------------ translators one by one
def test_one_failed_translator_does_not_skip_the_others(monkeypatch):
    from infra import text_models

    todo = [text_models.get(k) for k in ("opus-big-en-ru", "opus-big-ru-en", "opus-big-ru-de")]
    monkeypatch.setattr(text_models, "missing_component_extras", lambda: list(todo))
    tried = []

    def ensure(m, progress):
        tried.append(m.key)
        if m.key == "opus-big-ru-en":
            raise ModelDownloadError("WinError 5")

    with pytest.raises(ModelDownloadError) as ei:
        text_models.ensure_component_extras(ensure_fn=ensure)
    assert tried == ["opus-big-en-ru", "opus-big-ru-en", "opus-big-ru-de"]
    assert ei.value.failed == ["opus-big-ru-en"] and ei.value.fetched == ["opus-big-en-ru", "opus-big-ru-de"]
    tried.clear()

    def cancel(m, progress):
        tried.append(m.key)
        raise CancelledByUser()

    with pytest.raises(CancelledByUser):
        text_models.ensure_component_extras(ensure_fn=cancel)
    assert tried == ["opus-big-en-ru"]


def test_the_complete_set_fetches_every_integrated_text_model(monkeypatch):
    """The Full / Quick complete set lists every integrated text model; each is fetched as its MDL manifest row
    (exactly the pinned files - never a whole repository with TF / Flax / Rust copies)."""
    from infra import text_models
    from workers import pipeline_runner as pr

    monkeypatch.setattr(pr, "required_model_repos", lambda: ["Qwen/Qwen3-ForcedAligner-0.6B",
                                                              "ai-forever/sage-fredt5-distilled-95m"])
    repos = pr.all_model_repos()
    assert "ai-forever/sage-fredt5-distilled-95m" in repos
    assert {m.repo for m in text_models.REGISTRY if m.integrated and m.repo} <= set(repos)
    assert {m.key for m in text_models.missing_other_integrated()} == {"opus-ru-en", "opus-en-ru"}


# ------------------------------------------------------------------------------------------------ 5. / 6. / 8. installer
def test_uninstall_never_deletes_models_voices_or_settings():
    iss = _iss()
    assert "UninstallDataQuestion" not in iss and "DelTree(DataDir" not in iss
    # from 704 the online build removes its program folder in [Code] (DeleteProgramFolder), sparing a sibling program inside it
    assert "[UninstallDelete]" not in iss and 'filesandordirs; Name: "{app}"' not in iss
    body = iss.split("procedure DeleteProgramFolder", 1)[1].split("\nend;", 1)[0]
    assert "ExpandConstant('{app}')" in body and "IsSiblingFolder(Path)" in body and "Voxprint\\models" not in body
    # A `{ ... }` note after the last procedure is compiled as code. `{app}` inside
    # it ends the comment at the first `}`, and Inno then reports 'BEGIN' expected.
    tail = iss.split("[Code]", 1)[1].rsplit("end;", 1)[1]
    assert "{" not in tail and "Uninstall removes" not in tail


def test_the_wizard_language_becomes_the_app_language_once():
    iss = _iss()
    assert "function AppLanguageCode(): String;" in iss
    for wizard, code in (("russian", "ru"), ("german", "de")):
        assert f"ActiveLanguage = '{wizard}' then\n    Result := '{code}'" in iss
    assert "if not FileExists(StateDir + '\\language') then" in iss
    assert "SaveStringsToUTF8FileWithoutBOM(StateDir + '\\language', Lines, False);" in iss


def test_state_files_are_written_without_bom_and_read_with_or_without(monkeypatch):
    from infra import paths, setup_mode

    iss = _iss()
    assert "SaveStringsToUTF8FileWithoutBOM(StateDir + '\\install_mode.txt', Lines, False);" in iss
    assert "SaveStringsToUTF8File(StateDir + '\\install_mode.txt'" not in iss
    paths.state_dir().mkdir(parents=True, exist_ok=True)
    (paths.state_dir() / "install_mode.txt").write_bytes(b"\xef\xbb\xbfquick\r\n")
    assert setup_mode.mode() == "quick"
    (paths.state_dir() / "language").write_bytes(b"\xef\xbb\xbfde\r\n")       # an older installer's BOM: a dropped language is not offered
    assert i18n.saved_language() is None
    (paths.state_dir() / "language").write_bytes(b"en\r\n")
    assert i18n.saved_language() == "en"


# ------------------------------------------------------------------------------------------------ 7. standard buttons
def test_standard_buttons_follow_the_ui_language():
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox

    from ui import std_buttons

    app = QApplication.instance() or QApplication([])
    std_buttons.install(app)
    try:
        for lang in ("en", "ru"):                              # a dropped UI language falls back to English
            i18n.set_language(lang)
            box = QMessageBox(QMessageBox.Icon.Information, "t", "x", QMessageBox.StandardButton.Ok)
            std_buttons.localize(box)
            assert box.button(QMessageBox.StandardButton.Ok).text() == i18n.tr("qt.btn.ok")
            bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            std_buttons.localize(bb)
            assert bb.button(QDialogButtonBox.StandardButton.Cancel).text() == i18n.tr("qt.btn.cancel")
            assert QCoreApplication.translate("QPlatformTheme", "Cancel") == i18n.tr("qt.btn.cancel")
            assert all(i18n.tr(k) and i18n.tr(k) != k for k in std_buttons.KEYS.values())
        i18n.set_language("en")
        assert QCoreApplication.translate("QPlatformTheme", "OK") == "OK"
        assert QCoreApplication.translate("SomethingElse", "OK") == "OK"         # only the button texts are touched
    finally:
        i18n.set_language("en")


def test_the_privacy_notice_and_other_info_boxes_use_the_localized_buttons():
    src = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert 'std_buttons.information(self, tr("ui.privacy_title"), tr("ui.privacy_text"))' in src
    for f in ("ui/main_window.py", "ui/settings_dialog.py", "ui/voices_window.py", "ui/about_dialog.py"):
        text = (ROOT / f).read_text(encoding="utf-8")
        assert "QMessageBox.information(" not in text and "QMessageBox.question(" not in text, f
    assert "std_buttons.install(app)" in (ROOT / "main.py").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------ 9. pinned hashes
def test_every_integrated_text_model_matches_its_manifest_row():
    """The pinned sha256 of every integrated text model equals the sha256 of its MDL manifest row."""
    from voxprint_common.mdl.catalog import load_manifest

    from infra import text_models
    from infra.mdl_gate import manifest_path

    rows = {m.id: m for m in load_manifest(manifest_path()).models}

    def row_of(repo):
        folder = repo.replace("/", "--") + "/"
        return next(m for m in rows.values() if any(f.path.startswith(folder) for f in m.files))

    for m in text_models.REGISTRY:
        if not (m.integrated and m.repo):
            continue
        row = row_of(m.repo)
        by_name = {f.path.rsplit("/", 1)[-1]: f for f in row.files}
        for name, digest in m.sha256:
            assert by_name[name].sha256 == digest, (m.repo, name)


# --------------------------------------------------------------------------- added scope: train CLI, title block, work dir

def _train(tmp_path, *extra):
    import cli as user_cli

    audio = tmp_path / "clips"
    audio.mkdir(exist_ok=True)
    seen = []

    def fake_task(req, progress, cancel=None, **_kw):
        seen.append(req)
        return type("R", (), {"voice_id": "v", "adapter_path": None, "root_dir": req.out_root, "warnings": []})()
    code = user_cli.main(["train", str(audio), *extra], run_task_fn=fake_task)
    return code, seen


def test_train_cli_license_consent_language(tmp_path):
    code, seen = _train(tmp_path, "--name", "Boaz", "--license", "CC0-1.0", "--consent", "commercial",
                        "--speaker", "Mark Chulsky", "--language", "Russian")
    req = seen[0]
    assert code == 0 and req.license == "CC0-1.0" and req.consent_mode == "manual" and req.consent_scope == "commercial"
    assert req.consent_name == req.speaker == "Mark Chulsky" and req.language == "ru" and req.asr_language == "Russian"
    code, seen = _train(tmp_path, "--consent", "auto")
    assert code == 0 and seen[0].consent_mode == "auto" and seen[0].license == "" and seen[0].language == ""
    code, seen = _train(tmp_path)
    assert code == 0 and seen[0].consent_mode == "none"


@pytest.mark.parametrize("extra", [["--license", "CC0-1.0"], ["--license", "CC-BY-NC-4.0", "--consent", "private_only"],
                                   ["--license", "WTFPL", "--consent", "commercial"], ["--language", "klingon"]])
def test_train_cli_rejects_inconsistent_options(tmp_path, extra):
    code, seen = _train(tmp_path, *extra)
    assert code != 0 and not seen


def test_voice_json_takes_cli_license_and_language(tmp_path):
    from workers import pipeline_runner as pr

    req = pr.TaskRequest(kind=pr.KIND_LORA, audio=tmp_path / "a.wav", voice_display_name="Boaz", license="CC0-1.0",
                         language="ru", consent_mode="manual", consent_scope="commercial", consent_name="Reader")
    block = {"scope": "commercial", "name": "Reader", "method": "manual", "confirmed": True}
    info = pr._write_voice_json(req, tmp_path, "en", 60.0, block)
    assert info["license"] == "CC0-1.0" and info["language"] == "ru" and info["commercial_use"]


def test_train_work_dir_is_per_voice(tmp_path):
    out = tmp_path / "out"
    _, a = _train(tmp_path, "--name", "Boaz", "--out", str(out))
    _, b = _train(tmp_path, "--name", "Tirzah", "--out", str(out))
    assert a[0].out_root == out / "Boaz_Voxprint" and b[0].out_root == out / "Tirzah_Voxprint"


def test_title_block_before_first_heading_opens_chapter_one():
    from core.book_parsers import parse_txt

    book = parse_txt("Бытие\n\nГлава 1\n\nВ начале сотворил Бог небо и землю.\n\nГлава 2\n\nИ совершил Бог к седьмому дню.")
    assert [c.title for c in book.chapters] == ["Глава 1", "Глава 2"]
    assert book.chapters[0].text.startswith("Бытие\n\nВ начале") and book.title == "Бытие"
    intro = parse_txt("Это длинное вступление к книге, а не заголовок.\n\nГлава 1\n\nТекст.")
    assert len(intro.chapters) == 2                        # real prose before the first heading stays its own part
    assert len(parse_txt("Бытие\n\nВ начале сотворил Бог небо и землю.").chapters) == 1
