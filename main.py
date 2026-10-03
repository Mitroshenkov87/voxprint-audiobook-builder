"""Точка входа Voxprint (Windows 11).

Флаги: --prefetch (принудительно докачать модели; ставится установщиком), --selftest (запуск и выход),
--verify-install (проверка установки, коды причин), --repair (восстановить только собственное окружение Voxprint).
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


def _selftest_imports() -> int:
    """`--selftest-imports`: импортирует все тяжёлые библиотеки (проверка, что в сборке PyInstaller всё упаковано).
    Результат печатается и пишется в <логи>/selftest_imports.txt (в оконной сборке консоли нет). 0 - всё в порядке."""
    import importlib
    import time

    lines, bad = [], 0
    for name in ("torch", "torchaudio", "transformers", "peft", "accelerate", "safetensors", "qwen_tts", "qwen_asr",
                 "bitsandbytes", "soundfile", "librosa", "scipy.signal", "imageio_ffmpeg", "onnxruntime", "huggingface_hub",
                 "certifi", "PySide6.QtWidgets"):
        t = time.time()
        try:
            mod = importlib.import_module(name)
            lines.append(f"OK    {name} {getattr(mod, '__version__', '')} ({time.time() - t:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            if name == "bitsandbytes":          # необязателен: без него обучение идёт на обычном AdamW
                lines.append(f"WARN  {name} (optional): {type(exc).__name__}: {exc}")
                continue
            bad += 1
            lines.append(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    try:
        import torch

        lines.append(f"cuda available: {torch.cuda.is_available()}"
                     + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
        from infra import paths
        from core import i18n

        lines.append(f"app home: {paths.app_home()}; resources: {paths.resource_dir()}; ui language: {i18n.detect_language()}")
        lines.append(f"ffmpeg: {__import__('core.audio_utils', fromlist=['x']).ensure_ffmpeg()}")
    except Exception as exc:  # noqa: BLE001
        bad += 1
        lines.append(f"FAIL  environment: {type(exc).__name__}: {exc}")
    lines.append("SELFTEST_IMPORTS " + ("FAILED" if bad else "OK"))
    text = "\n".join(lines)
    try:
        from infra import paths

        (paths.logs_dir() / "selftest_imports.txt").write_text(text + "\n", encoding="utf-8")
    except OSError:
        pass
    try:
        print(text)
    except (OSError, ValueError):      # оконная сборка без stdout
        pass
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    # обновлённые пакеты - до импорта тяжёлых библиотек
    from infra.updater import activate_overlay

    activate_overlay()
    _setup_logging()
    try:   # настройки прежней установки Voxprint (копируются, если ещё нет; старая папка не меняется)
        from infra import paths

        adopted = paths.adopt_previous_settings()
        if adopted:
            logging.getLogger("voxprint").info("adopted settings from a previous install: %s", adopted)
    except Exception:  # noqa: BLE001
        pass
    if "--verify-install" in argv or "--repair" in argv:
        import shutil
        from infra import install_state
        from infra.updater import run_subprocess

        if "--repair" in argv:
            return install_state.cli_repair(run_subprocess, shutil.which)
        return install_state.cli_verify()
    if "--selftest-imports" in argv:
        return _selftest_imports()
    selftest = "--selftest" in argv  # запуск и автоматический выход (проверка в offscreen)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("Voxprint")
    try:   # значок окна/панели задач (файл поставляется в assets/, в сборке - рядом с ресурсами)
        from PySide6.QtGui import QIcon

        from infra import paths

        ico = paths.resource_dir() / "assets" / "voxprint.ico"
        if ico.is_file():
            app.setWindowIcon(QIcon(str(ico)))
    except Exception:  # noqa: BLE001 - значок необязателен
        pass
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
