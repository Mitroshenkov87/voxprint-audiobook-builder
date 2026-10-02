"""Точка входа Voxprint (Windows 11).

Флаги: --prefetch (принудительно докачать модели; ставится установщиком), --selftest (запуск и выход).
"""
from __future__ import annotations

import logging
import logging.handlers
import sys


def _setup_logging() -> None:
    try:
        from infra import paths

        h = logging.handlers.RotatingFileHandler(paths.logs_dir() / "voxprint.log", maxBytes=2_000_000,
                                                 backupCount=3, encoding="utf-8")
        logging.basicConfig(level=logging.INFO, handlers=[h],
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    except OSError:
        logging.basicConfig(level=logging.INFO)


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    # обновлённые пакеты - до импорта тяжёлых библиотек
    from infra.updater import activate_overlay

    activate_overlay()
    _setup_logging()
    selftest = "--selftest" in argv  # запуск и автоматический выход (проверка в offscreen)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("Voxprint")
    from workers.pipeline_runner import models_missing

    first_run = False
    if not selftest:
        try:
            first_run = bool(models_missing())
        except Exception:  # noqa: BLE001 - не мешаем запуску
            first_run = False
    win = MainWindow(autocheck=not selftest, prefetch=first_run or "--prefetch" in argv)
    win.show()
    if selftest:
        QTimer.singleShot(300, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
