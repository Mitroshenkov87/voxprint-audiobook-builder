"""Voxprint entry point (Windows 11; the code also runs on Linux for development and tests).

Flags: ``--prefetch`` (force-download the models; used by the installer), ``--selftest`` (start and quit),
``--selftest-imports`` (import every heavy library - checks that a PyInstaller build is complete),
``--selftest-narrate [voice]`` (narrate two sentences headless; writes logs/selftest_narrate.txt),
``--verify-install`` (install check with reason codes; also written to logs/verify_install.txt),
``--repair`` (rebuild only Voxprint's own environment; logs/repair.txt).
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
    for handler in logging.getLogger().handlers:
        handler.addFilter(_drop_sox_warning)


def _drop_sox_warning(record: logging.LogRecord) -> bool:
    """The TTS package warns "SoX could not be found" on every import (a level set on the logger is overridden by the package);
    Voxprint never uses SoX, so the line is only noise in the log."""
    return not (record.name == "sox" and "SoX could not be found" in record.getMessage())


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
    except UnicodeEncodeError:
        print(text.encode('ascii', 'backslashreplace').decode('ascii'))
    except (OSError, ValueError):      # windowed build without stdout
        pass
    return 1 if bad else 0


def _modules_cli(argv) -> int:
    """``--modules-status`` prints the modules; ``--install-modules [id ...]`` downloads the missing required ones (or those named)."""
    from infra import modules

    out = _cli_printer("modules")
    if not modules.is_thin():
        out("This is a full build: all components are included (no modules.json).")
        return 0
    try:
        if "--install-modules" in argv:
            i = argv.index("--install-modules")
            ids = [a for a in argv[i + 1:] if not a.startswith("--")] or None
            last = [-1]

            def prog(f: float, m: str = "") -> None:
                pct = int(f * 100)
                if pct != last[0]:
                    last[0] = pct
                    out(f"{pct:3d} %  {m}")

            n = modules.install(ids, prog)
            out(f"OK: {n} component(s) installed")
        for m in modules.modules(modules.load_manifest()):
            out(f"{m.id:8} {'installed' if m.installed else 'missing  '} {m.size / 2**20:9.0f} MB  {m.title}")
        return 0
    except modules.ModulesError as exc:
        out(f"ERROR: {exc}")
        return 1


def _offer_components(win, app, then_prefetch: bool) -> None:
    """Thin build, first start: show the Components window (download the runtime modules) and, when they are ready,
    start the first-run model download that was waiting for them."""
    from PySide6.QtCore import QTimer

    from ui.modules_dialog import ModulesDialog

    dlg = ModulesDialog(win.styleSheet(), win)
    win._components = dlg                    # keep a reference

    def ready() -> None:
        try:   # PyTorch is there now: detect the GPU again (the first detection ran without it)
            win.trainer._gpu = None
            win.trainer.refresh_estimate()
        except Exception:  # noqa: BLE001
            pass
        if then_prefetch:
            QTimer.singleShot(300, win.start_prefetch)

    dlg.ready.connect(ready)
    app.aboutToQuit.connect(dlg.shutdown)
    dlg.show()
    QTimer.singleShot(0, dlg.refresh)


def _cli_printer(name: str):
    """print() that also appends to ``<logs>/<name>.txt``: a windowed (PyInstaller) build has no console, so the result of
    ``--verify-install`` / ``--repair`` would otherwise be invisible.  The file is rewritten on the first line of each run."""
    state = {"first": True}

    def out(line: str) -> None:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode('ascii', 'backslashreplace').decode('ascii'))
        except Exception:  # noqa: BLE001 - no console
            pass
        try:
            from infra import paths

            f = paths.logs_dir() / f"{name}.txt"
            f.parent.mkdir(parents=True, exist_ok=True)
            with open(f, "w" if state["first"] else "a", encoding="utf-8") as fh:
                fh.write(str(line) + "\n")
            state["first"] = False
        except Exception:  # noqa: BLE001
            pass
    return out


def main(argv=None) -> int:
    """Start the application (or run one of the CLI maintenance flags); returns the process exit code."""
    argv = list(sys.argv if argv is None else argv)
    # updated packages must be on sys.path BEFORE the heavy libraries are imported
    from infra.updater import activate_overlay

    activate_overlay()
    try:   # thin build: the downloaded runtime modules (PyTorch ...) join sys.path before any heavy import (no-op otherwise)
        from infra import modules as _modules

        _modules.activate()
    except Exception:  # noqa: BLE001
        pass
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
            return install_state.cli_repair(run_subprocess, shutil.which, _cli_printer("repair"))
        return install_state.cli_verify(print_fn=_cli_printer("verify_install"))
    if "--selftest-imports" in argv:
        return _selftest_imports()
    if "--modules-status" in argv or "--install-modules" in argv:   # thin build: list / download the runtime modules (no GUI)
        return _modules_cli(argv)
    if "--selftest-text" in argv:   # headless, no GPU/models: text prep, chunking, stub translation, ffmpeg encode
        from workers import selftest_text

        return selftest_text.run()
    if "--selftest-narrate" in argv:   # headless: narrate two sentences with the first voice (argument after the flag = voice id)
        from workers import selftest_narrate

        i = argv.index("--selftest-narrate")
        return selftest_narrate.run(argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else "")
    selftest = "--selftest" in argv  # start and quit automatically (used for the offscreen smoke check)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from ui.studio import StudioWindow

    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("Voxprint")
    app.setDesktopFileName("voxprint")      # Linux: ties the window to voxprint.desktop (icon / Wayland app-id); ignored elsewhere
    try:   # window / taskbar icon (the file ships in assets/, next to the other resources in a build)
        from PySide6.QtGui import QIcon

        from infra import paths

        ico = paths.resource_dir() / "assets" / ("voxprint.ico" if sys.platform == "win32" else "voxprint.png")
        if not ico.is_file():
            ico = paths.resource_dir() / "assets" / "voxprint.ico"
        if ico.is_file():
            app.setWindowIcon(QIcon(str(ico)))
    except Exception:  # noqa: BLE001 - the icon is optional
        pass
    from workers.pipeline_runner import models_missing

    from infra import modules as _mods

    thin_wait = False        # thin build, runtime modules missing: the model download waits until they are installed
    try:
        thin_wait = (not selftest) and _mods.is_thin() and not _mods.installed_without_network()
    except Exception:  # noqa: BLE001
        thin_wait = False
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
    want_prefetch = first_run or importing or "--prefetch" in argv
    win = StudioWindow(autocheck=not selftest and not thin_wait, prefetch=want_prefetch and not thin_wait)
    app.aboutToQuit.connect(win.shutdown)
    if not selftest:
        from infra import hard_exit

        hard_exit.arm()          # closing the main window kills every helper process at once (docs/BUILDING.md)
    win.show_studio()
    if thin_wait:
        _offer_components(win, app, want_prefetch)
    if selftest:
        QTimer.singleShot(300, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
