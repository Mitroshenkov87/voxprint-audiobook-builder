"""Voxprint entry point (Windows 11; the code also runs on Linux for development and tests).

Flags: ``--prefetch`` (force-download the models; used by the installer), ``--selftest`` (start and quit),
``--selftest-imports`` (import every heavy library - checks that a PyInstaller build is complete),
``--verify-install`` (install check with reason codes), ``--repair`` (rebuild only Voxprint's own environment).
"""
from __future__ import annotations

import logging
import logging.handlers
import sys


def _setup_logging() -> None:
    """Log to ``logs/voxprint.log`` (rotating); fall back to the console if the file cannot be opened."""
    try:
        from infra import paths

        h = logging.handlers.RotatingFileHandler(paths.logs_dir() / "voxprint.log", maxBytes=2_000_000,
                                                 backupCount=3, encoding="utf-8")
        logging.basicConfig(level=logging.INFO, handlers=[h],
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    except OSError:
        logging.basicConfig(level=logging.INFO)


def _selftest_imports() -> int:
    """``--selftest-imports``: import all heavy libraries (checks that a PyInstaller build packed everything).

    The result is printed and also written to ``<logs>/selftest_imports.txt`` (a windowed build has no console).
    Returns 0 when everything is fine.
    """
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
            if name == "bitsandbytes":          # optional: without it training uses plain AdamW
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
    except (OSError, ValueError):      # windowed build without stdout
        pass
    return 1 if bad else 0


def main(argv=None) -> int:
    """Start the application (or run one of the CLI maintenance flags); returns the process exit code."""
    argv = list(sys.argv if argv is None else argv)
    # updated packages must be on sys.path BEFORE the heavy libraries are imported
    from infra.updater import activate_overlay

    activate_overlay()
    _setup_logging()
    try:   # settings of an earlier Voxprint install (copied if not present yet; the old folder is left untouched)
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
    selftest = "--selftest" in argv  # start and quit automatically (used for the offscreen smoke check)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from ui.studio import StudioWindow

    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("Voxprint")
    try:   # window / taskbar icon (the file ships in assets/, next to the other resources in a build)
        from PySide6.QtGui import QIcon

        from infra import paths

        ico = paths.resource_dir() / "assets" / "voxprint.ico"
        if ico.is_file():
            app.setWindowIcon(QIcon(str(ico)))
    except Exception:  # noqa: BLE001 - the icon is optional
        pass
    from workers.pipeline_runner import models_missing

    first_run = False
    if not selftest:
        try:
            first_run = bool(models_missing())
        except Exception:  # noqa: BLE001 - never get in the way of starting up
            first_run = False
    importing = False
    if not selftest:
        try:   # an "existing models folder" (installer page / Settings) is imported on the first run
            from infra import existing_models
            from workers.pipeline_runner import required_model_repos

            importing = existing_models.pending(required_model_repos())
        except Exception:  # noqa: BLE001 - never get in the way of starting up
            importing = False
    win = StudioWindow(autocheck=not selftest, prefetch=first_run or importing or "--prefetch" in argv)
    app.aboutToQuit.connect(win.shutdown)
    win.show_studio()
    if selftest:
        QTimer.singleShot(300, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
