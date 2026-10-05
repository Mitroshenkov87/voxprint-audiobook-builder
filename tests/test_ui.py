"""Main window behaviour under the offscreen Qt platform: core buttons, Settings dialog, scrolling, voice fields, progress, errors, theme contrast."""
import os
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core import i18n
from core.errors import AudioTextMismatchError, ModelDownloadError, OutOfMemoryError_
from core.events import Stage
from workers.pipeline_runner import KIND_DATASET, KIND_LORA, TaskResult, plan_for


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        QApplication.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def make_window(runner):
    from ui.main_window import MainWindow
    return MainWindow(runner=runner, autocheck=False, auto_open_folder=False)


def test_main_window_keeps_only_core_buttons_plus_gear(app):
    w = make_window(lambda *a: None)
    w.show()
    texts = {b.text() for b in (w.btn_audio, w.btn_text, w.btn_dataset, w.btn_lora)}
    assert texts == {"Выбрать аудио", "Выбрать текст", "Создать датасет", "Создать голос (LoRA)"}
    # service items moved to the Settings dialog
    for gone in ("btn_update", "btn_about", "cmb_lang"):
        assert not hasattr(w, gone)
    assert w.btn_settings.isEnabled() and w.btn_settings.toolTip() == "Настройки"
    assert w.windowTitle() == "Voxprint AI Audiobook Builder"
    assert not w.btn_lora.isEnabled() and not w.btn_dataset.isEnabled()  # no files
    assert [s.label for s in plan_for(KIND_LORA)] == [
        "Проверка обновлений", "Загрузка модели", "Выравнивание", "Нарезка", "Обучение LoRA", "Сохранение"]
    w.close()


def test_selftest_entrypoint_runs_offscreen(app):
    import main as vox_main
    assert vox_main.main(["voxprint", "--selftest"]) == 0


def test_selftest_imports_reports_and_writes_log(capsys):
    import main as vox_main
    from infra import paths
    rc = vox_main.main(["voxprint", "--selftest-imports"])
    out = capsys.readouterr().out
    log = (paths.logs_dir() / "selftest_imports.txt").read_text(encoding="utf-8")
    assert "SELFTEST_IMPORTS" in out and log.strip() in out        # libraries may print banners to stdout
    assert (rc == 0) == ("SELFTEST_IMPORTS OK" in out)
    import importlib.util
    if importlib.util.find_spec("torch") is None:                 # a light dev box without PyTorch: the report itself is checked
        assert "FAIL  torch" in log
    else:
        assert "OK    torch" in log and "OK    qwen_tts" in log    # core libraries are present in the full dev/test env


def test_successful_run_shows_ready_and_progress(app, tmp_path):
    seen = []

    def runner(req, progress, cancel):
        for st in plan_for(req.kind):
            progress(st, 0.5, f"идёт {st.label}")
            progress(st, 1.0, f"{st.label} завершено")
        return TaskResult(req.kind, tmp_path, tmp_path / "dataset", tmp_path / "output" / "adapter_model.safetensors",
                          12, ["предупреждение"])

    w = make_window(runner)
    w.show()
    w.set_audio(tmp_path / "a.wav")
    w.set_text(tmp_path / "t.txt")
    assert w.btn_lora.isEnabled()
    w.progress.valueChanged.connect(seen.append)
    w.btn_lora.click()
    assert wait_for(lambda: not w.lbl_ready.isHidden())
    assert w.progress.value() == 100 and seen == sorted(seen)  # progress is monotonic
    assert not w.btn_open.isHidden() and w.result_dir == tmp_path
    assert "12" in w.lbl_status.text()
    assert w.btn_lora.isEnabled()
    w.close()


@pytest.mark.parametrize("exc,kind", [
    (AudioTextMismatchError("не совпадают"), "mismatch"),
    (OutOfMemoryError_(), "oom"),
    (ModelDownloadError("не скачалось", url="https://huggingface.co/X/Y"), "download"),
    (RuntimeError("boom"), "other"),
])
def test_friendly_errors(app, tmp_path, exc, kind):
    def runner(req, progress, cancel):
        raise exc

    w = make_window(runner)
    w.show()
    w.set_audio(tmp_path / "a.wav")
    w.set_text(tmp_path / "t.txt")
    w.btn_dataset.click()
    assert wait_for(lambda: bool(w.last_error_text))
    assert w.lbl_ready.isHidden()
    if kind == "oom":
        assert w._retry_cpu_button is not None
    if kind == "download":
        assert exc.url
    if kind == "other":
        assert "boom" not in w.last_error_text  # technical details are not shown in the main text
    w.close()


def test_cancel(app, tmp_path):
    def runner(req, progress, cancel):
        for _ in range(500):
            cancel.check()
            time.sleep(0.01)

    w = make_window(runner)
    w.show()
    w.set_audio(tmp_path / "a.wav")
    w.set_text(tmp_path / "t.txt")
    w.btn_dataset.click()
    assert wait_for(lambda: w.busy)
    w.btn_cancel.click()
    assert wait_for(lambda: not w.busy and w.lbl_status.text() == "Отменено.")
    w.close()


def test_drop_and_update_button(app, tmp_path):
    from infra.updater import UpdateResult

    class FakeUpdater:
        def check_and_apply(self, progress):
            return None, UpdateResult(before={"peft": "1"}, after={"peft": "2"}, needs_restart=True)

        def should_autocheck(self):
            return True

    w = make_window(lambda *a: None)
    w.updater_factory = FakeUpdater
    w.show()
    w.open_settings()
    w.settings_dialog().btn_update.click()
    assert wait_for(lambda: "peft: 1 → 2" in w.lbl_status.text())
    w.close()


def test_os_check_and_backdrop_plain_on_linux():
    from infra import platform_win as pw
    assert pw.check_os(build=26300, is_windows=True).ok
    assert pw.check_os(build=26100, is_windows=True).ok
    bad = pw.check_os(build=22631, is_windows=True)
    assert not bad.ok and "26H2" in bad.message and "22631" in bad.message
    assert not pw.check_os(is_windows=False).ok
    assert pw.apply_backdrop(0) == "plain"  # on Linux: no effect and no exceptions


def test_style_modes():
    from ui.main_window import build_style
    from ui import main_window as mw
    assert mw.ROOT_GLASS in build_style(True) and mw.ROOT_PLAIN in build_style(False)
    assert mw.CARD_GLASS in build_style(True) and mw.CARD_PLAIN in build_style(False)
    assert mw.ROOT_GLASS not in build_style(False)


def _lum(hex_color):
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(fg, bg):
    hi, lo = sorted((_lum(fg), _lum(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _over_white(rgba):
    """Composite 'rgba(r,g,b,a)' over a pure white desktop (the worst case behind an Acrylic window)."""
    r, g, b, a = [int(v) for v in rgba[5:-1].split(",")]
    return "#%02x%02x%02x" % tuple(round(c * a / 255 + 255 * (1 - a / 255)) for c in (r, g, b))


def test_theme_contrast():
    """All readable text keeps WCAG AA (4.5:1), also over the glass tint on a white backdrop."""
    from ui import main_window as mw
    for fg, bg in mw.CONTRAST_PAIRS:
        assert _contrast(fg, bg) >= 4.5, (fg, bg, _contrast(fg, bg))
    for glass in (mw.ROOT_GLASS, mw.CARD_GLASS):
        worst = _over_white(glass)
        for fg in (mw.TEXT, mw.TEXT_MUTED, mw.TEXT_FAINT):
            assert _contrast(fg, worst) >= 4.5, (fg, glass, worst)


def test_first_run_prefetch_message_and_buttons(app, tmp_path):
    from ui.main_window import MainWindow
    import threading
    calls = []
    gate = threading.Event()

    def fake_prefetch(progress):
        progress(Stage.MODEL, 0.5, "Скачиваю модель X: 50%")
        gate.wait(5)
        calls.append(1)
        return ["X"]

    w = MainWindow(runner=lambda *a: None, autocheck=False, auto_open_folder=False, prefetch_fn=fake_prefetch)
    w.set_audio(tmp_path / "a.wav")
    w.set_text(tmp_path / "t.txt")
    w.show()
    w.start_prefetch()
    assert not w.btn_lora.isEnabled()  # the buttons are locked during the first-run download
    gate.set()
    assert wait_for(lambda: "готово" in w.lbl_status.text().lower() and not w.busy)
    assert calls and w.btn_lora.isEnabled()
    w.close()


def test_first_run_prefetch_failure_is_friendly(app):
    from ui.main_window import MainWindow
    from core.errors import ModelDownloadError

    def bad(progress):
        raise ModelDownloadError("Нет интернета.", url="https://huggingface.co/X")

    w = MainWindow(runner=lambda *a: None, autocheck=False, prefetch_fn=bad)
    w.show()
    w.start_prefetch()
    assert wait_for(lambda: "Нет интернета" in w.lbl_status.text())
    w.close()


def test_privacy_footer_and_one_time_notice(app, tmp_path):
    from ui import main_window as mw
    w = mw.MainWindow(runner=lambda *a, **k: None, autocheck=False, auto_open_folder=False)
    assert "с разрешения владельца голоса" in w.lbl_privacy.text() and "только на этом компьютере" in w.lbl_privacy.text()
    assert not mw.privacy_acknowledged()
    assert w.maybe_show_privacy_notice() is False          # offscreen: the window is not shown
    assert not mw.privacy_acknowledged()
    mw.acknowledge_privacy()
    assert mw.privacy_acknowledged() and mw.privacy_marker().parent == mw.paths.state_dir()


def test_window_size_adapts_to_screen_and_content_scrolls(app):
    from ui.main_window import MIN_WINDOW_H, MIN_WINDOW_W
    w = make_window(lambda *a: None)
    assert (w.minimumWidth(), w.minimumHeight()) == (MIN_WINDOW_W, MIN_WINDOW_H)
    avail = w.screen().availableGeometry()
    assert w.width() >= MIN_WINDOW_W and w.height() >= MIN_WINDOW_H
    assert w.height() <= max(MIN_WINDOW_H, int(avail.height() * 0.90)) + 1
    # the whole UI sits in a scroll area, so it stays usable below the content's natural height
    assert w.scroll.widget() is w.content and w.scroll.widgetResizable()
    w.resize(MIN_WINDOW_W, MIN_WINDOW_H)
    w.show()
    QApplication.processEvents()
    assert w.scroll.verticalScrollBar().maximum() > 0
    w.close()


def test_settings_dialog_holds_service_items(app, monkeypatch):
    w = make_window(lambda *a: None)
    w.show()
    w.open_settings()
    dlg = w.settings_dialog()
    assert dlg.isVisible() and w.settings_dialog() is dlg          # one reused instance
    assert dlg.cmb_lang.count() == 3 and dlg.cmb_lang.currentData() == "ru"
    assert dlg.btn_update.text() == "Проверить обновления"
    dlg.cmb_lang.setCurrentIndex(dlg.cmb_lang.findData("en"))
    assert i18n.get_language() == "en" and w.btn_lora.text() == "Create voice (LoRA)"
    assert dlg.btn_update.text() == "Check for updates" and dlg.windowTitle() == "Settings"
    opened = []
    import ui.main_window as mw
    monkeypatch.setattr(mw, "open_folder", lambda p: opened.append(Path(p).name))
    dlg.btn_models.click()
    assert opened == ["models"]
    dlg.close()
    w.close()


def test_voice_fields_are_passed_to_the_task_request(app):
    seen = {}
    w = make_window(lambda req, *a, **k: seen.update(req=req) or None)
    w.set_audio(Path("/tmp/a.wav"))
    w.set_text(Path("/tmp/a.txt"))
    w.cmb_voice_type.setCurrentIndex(w.cmb_voice_type.findData("female"))
    w.edt_voice_desc.setText("  warm  alto ")
    w.start(KIND_LORA)
    assert wait_for(lambda: "req" in seen)
    assert seen["req"].voice_type == "female" and seen["req"].voice_description == "warm  alto"
    w.close()
