import os
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

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


def test_main_window_imports_and_has_five_buttons(app):
    w = make_window(lambda *a: None)
    w.show()
    texts = {b.text() for b in (w.btn_audio, w.btn_text, w.btn_dataset, w.btn_lora, w.btn_update)}
    assert texts == {"Выбрать аудио", "Выбрать текст", "Создать датасет", "Создать голос (LoRA)",
                     "Проверить обновления"}
    assert w.windowTitle() == "Voxprint"
    assert not w.btn_lora.isEnabled() and not w.btn_dataset.isEnabled()  # нет файлов
    assert [s.label for s in plan_for(KIND_LORA)] == [
        "Проверка обновлений", "Загрузка модели", "Выравнивание", "Нарезка", "Обучение LoRA", "Сохранение"]
    w.close()


def test_selftest_entrypoint_runs_offscreen(app):
    import main as vox_main
    assert vox_main.main(["voxprint", "--selftest"]) == 0


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
    assert w.progress.value() == 100 and seen == sorted(seen)  # прогресс монотонный
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
        assert "boom" not in w.last_error_text  # технические детали не показываем в основном тексте
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
    w.btn_update.click()
    assert wait_for(lambda: "peft: 1 → 2" in w.lbl_status.text())
    w.close()


def test_os_check_and_backdrop_plain_on_linux():
    from infra import platform_win as pw
    assert pw.check_os(build=26300, is_windows=True).ok
    assert pw.check_os(build=26100, is_windows=True).ok
    bad = pw.check_os(build=22631, is_windows=True)
    assert not bad.ok and "26H2" in bad.message and "22631" in bad.message
    assert not pw.check_os(is_windows=False).ok
    assert pw.apply_backdrop(0) == "plain"  # на Linux - без эффекта и без исключений


def test_style_modes():
    from ui.main_window import build_style
    assert "rgba(20,20,26,110)" in build_style(True) and "#17171c" in build_style(False)


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
    assert not w.btn_lora.isEnabled()  # во время первичной загрузки кнопки заблокированы
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
    assert "собственный голос" in w.lbl_privacy.text() and "только на этом компьютере" in w.lbl_privacy.text()
    assert not mw.privacy_acknowledged()
    assert w.maybe_show_privacy_notice() is False          # offscreen: окно не показываем
    assert not mw.privacy_acknowledged()
    mw.acknowledge_privacy()
    assert mw.privacy_acknowledged() and mw.privacy_marker().parent == mw.paths.state_dir()
