"""Voxprint entry point (Windows 11; the code also runs on Linux for development and tests).

Flags: ``--prefetch`` (force-download the models; used by the installer), ``--selftest`` (start and quit),
``--selftest-imports`` (import every heavy library - checks that a PyInstaller build is complete),
``--selftest-narrate [voice]`` (narrate two sentences headless; writes logs/selftest_narrate.txt),
``--verify-install`` (install check with reason codes; also written to logs/verify_install.txt),
``--repair`` (rebuild only Voxprint's own environment; logs/repair.txt),
``--auto-repair`` (check every component and model by hash, fetch missing / broken parts; logs/auto_repair.txt).
User CLI subcommands (see ``cli.py`` / docs/CLI.md): ``narrate``, ``train``, ``voices``.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys


def _setup_logging() -> None:
    """Rotating ``logs/voxprint.log`` (5 MB x 6) with exception hooks (infra/diagnostics.py); the console if the file fails."""
    from infra import diagnostics

    diagnostics.setup_logging(filters=(_drop_sox_warning,))


def _drop_sox_warning(record: logging.LogRecord) -> bool:
    """The TTS package warns "SoX could not be found" on every import (a level set on the logger is overridden by the package);
    Voxprint never uses SoX, so the line is only noise in the log."""
    return not (record.name == "sox" and "SoX could not be found" in record.getMessage())


def _selftest_imports(argv=()) -> int:
    """``--selftest-imports``: import all heavy libraries (checks that a PyInstaller build packed everything).

    The result is printed and also written to ``<logs>/selftest_imports.txt`` (a windowed build has no console).
    Returns 0 when everything is fine.
    """
    import importlib
    import time

    lines, bad = [], 0
    try:                                # thin build: the downloaded runtime (PyTorch, transformers, requests ...) lives outside the shell
        from infra import modules as _m

        _m.activate()
    except Exception as exc:  # noqa: BLE001
        lines.append(f"WARN  runtime folder not activated: {type(exc).__name__}: {exc}")
    for name in ("requests", "urllib3", "tqdm", "yaml", "tokenizers", "torch", "torchaudio", "transformers", "peft", "accelerate", "safetensors", "qwen_tts", "qwen_asr",
                 "bitsandbytes", "soundfile", "librosa", "scipy.signal", "imageio_ffmpeg", "onnxruntime", "huggingface_hub",
                 "certifi", "PySide6.QtWidgets", "http.cookies", "email.mime.text", "xml.dom.minidom", "logging.config"):
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
    for stmt in ("from transformers import AutoModel, AutoTokenizer", "from huggingface_hub import snapshot_download, HfApi",
                 "from qwen_asr import Qwen3ASRModel", "from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor", "from qwen_tts import Qwen3TTSModel", "import requests.compat"):
        t = time.time()
        try:
            exec(stmt, {})
            lines.append(f"OK    {stmt} ({time.time() - t:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            bad += 1
            lines.append(f"FAIL  {stmt}: {type(exc).__name__}: {exc}")
    # every standard-library module that imports in the BUILD Python must import in this (frozen) program too
    if "--stdlib-list" in argv:
        i = argv.index("--stdlib-list")
        try:
            with open(argv[i + 1], encoding="utf-8") as fh:
                names = [l.strip() for l in fh if l.strip()]
        except (OSError, IndexError) as exc:
            names, bad = [], bad + 1
            lines.append(f"FAIL  stdlib list unreadable: {exc}")
        import warnings

        warnings.simplefilter("ignore")
        failed = []
        for n in names:
            try:
                importlib.import_module(n)
            except BaseException as exc:  # noqa: BLE001
                failed.append(f"{n} ({type(exc).__name__}: {exc})")
        bad += len(failed)
        lines.append(f"stdlib: {len(names) - len(failed)} of {len(names)} modules import" + ("" if not failed else "; MISSING: " + ", ".join(failed)))
    # every package the shell bundles must carry its dist-info (importlib.metadata), see build_thin.bat --copy-metadata
    try:
        import json as _json
        from importlib import metadata as _md

        from infra import paths as _paths

        shell = _json.loads((_paths.resource_dir() / "infra" / "runtime_lock.json").read_text(encoding="utf-8"))["shell"]
        nometa = []
        for dist in sorted(shell):
            try:
                _md.version(dist)
            except _md.PackageNotFoundError:
                nometa.append(dist)
        bad += len(nometa)
        lines.append(f"metadata: {len(shell) - len(nometa)} of {len(shell)} shell packages" + (f"; MISSING: {', '.join(nometa)}" if nometa else ""))
    except Exception as exc:  # noqa: BLE001
        bad += 1
        lines.append(f"FAIL  shell metadata check: {type(exc).__name__}: {exc}")
    try:
        import torch

        lines.append(f"cuda available: {torch.cuda.is_available()}"
                     + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
        from infra import paths
        from core import i18n

        lines.append(f"app home: {paths.app_home()}; resources: {paths.resource_dir()}; ui language: {i18n.detect_language()}")
    except Exception as exc:  # noqa: BLE001
        bad += 1
        lines.append(f"FAIL  environment: {type(exc).__name__}: {exc}")
    try:
        lines.append(f"ffmpeg: {__import__('core.audio_utils', fromlist=['x']).ensure_ffmpeg()}")
    except Exception as exc:  # noqa: BLE001 - the runner may have none; the stdlib / library imports above are what this test is for
        lines.append(f"WARN  ffmpeg: {type(exc).__name__}: {exc}")
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

            if "--own-torch" in argv:      # ignore a PyTorch found on this PC: download our own pinned copy
                from infra import runtime_reuse

                runtime_reuse.forget("torch")
                os.environ["VOXPRINT_NO_REUSE"] = "1"
            n = modules.install(ids, prog)
            out(f"OK: {n} component(s) installed")
        for m in modules.modules(modules.load_manifest()):
            how = "reused  " if m.reused else "installed" if m.installed else "missing  "
            out(f"{m.id:8} {how} {m.size / 2**20:9.0f} MB  {m.title}")
        return 0
    except modules.ModulesError as exc:
        out(f"ERROR: {exc}")
        return 1


def _offer_components(win, app, then_prefetch: bool) -> None:
    """Thin build, first start: the Components window downloads the runtime modules and SAGE without a click (step 1) and then
    starts the download of ALL heavy models (step 2, the main window's first-run prefetch), showing one live line throughout."""
    from PySide6.QtCore import QTimer

    from ui.modules_dialog import ModulesDialog, default_extras

    def models_start(hook):
        # StudioWindow.start_prefetch delegates to the Train window (it owns the status line and the busy lock).  Before
        # this existed the call raised AttributeError inside the Qt slot and the models never started after the components.
        return win.start_prefetch(hook) if then_prefetch else None

    dlg = ModulesDialog(win.styleSheet(), win, extras_fn=default_extras, models_start=models_start if then_prefetch else None,
                        autostart=True)
    win._components = dlg                    # keep a reference

    def ready() -> None:
        try:   # PyTorch is there now: detect the GPU again (the first detection ran without it)
            win.trainer._gpu = None
            win.trainer.refresh_estimate()
        except Exception:  # noqa: BLE001
            pass

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
    # User-facing headless CLI: narrate / train / voices (see cli.py, docs/CLI.md). Keep before the GUI and
    # the maintenance flags so `python main.py narrate ...` and a packaged exe work the same way.
    try:
        from cli import is_user_cli, main as user_cli_main
    except ImportError:
        user_cli_main = None  # type: ignore[assignment]
        is_user_cli = lambda _a: False  # noqa: E731
    if user_cli_main is not None and is_user_cli(argv):
        from infra import diagnostics

        diagnostics.log_startup(import_torch=False)       # quick: a headless command must not wait for PyTorch here
        return user_cli_main(argv[1:])
    if "--auto-repair" in argv:          # same job as Settings -> Auto-repair (infra/auto_repair.py)
        from infra import auto_repair

        out = _cli_printer("auto_repair")
        rep = auto_repair.run(lambda f, m: out(f"  [{int(f * 100):3d}%] {m}"))
        for it in rep.items:
            out(f"{it.status.upper():10s} {it.kind}: {it.name}" + (f" - {it.detail}" if it.detail else ""))
        return 0 if rep.ok else 1
    if "--verify-install" in argv or "--repair" in argv:
        import shutil
        from infra import install_state
        from infra.updater import run_subprocess

        if "--repair" in argv:
            return install_state.cli_repair(run_subprocess, shutil.which, _cli_printer("repair"))
        return install_state.cli_verify(print_fn=_cli_printer("verify_install"))
    if "--selftest-imports" in argv:
        return _selftest_imports(argv)
    if "--probe-torch" in argv:   # child process of the PyTorch reuse check (infra/runtime_reuse.py): import + compute, print VXTORCH OK
        from infra import modules as _m, runtime_reuse

        _m.activate()
        return runtime_reuse.probe_torch_main()
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

    app = QApplication.instance() or QApplication(argv)
    from ui import splash as splash_mod

    splash = splash_mod.show()            # the first thing on screen: before the heavy UI imports below
    splash.loading_ui()
    from ui import screen_fit
    from ui.studio import StudioWindow


    screen_fit.install(app)               # every window / dialog clamped to the screen's work area when shown
    import threading

    from infra import diagnostics

    diagnostics.install_qt_message_handler()
    # system / GPU / settings block of the log; the GPU probe imports PyTorch, so it runs beside the UI
    threading.Thread(target=diagnostics.log_startup, name="diag-startup", daemon=True).start()
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

    splash.checking()
    thin_wait = False        # thin build, runtime modules missing: the model download waits until they are installed
    try:
        thin_wait = (not selftest) and _mods.is_thin() and not _mods.installed_without_network()
    except Exception:  # noqa: BLE001
        thin_wait = False
    if not selftest:
        try:   # a "models folder" that is really a Voxprint backup becomes a restore source (never the live folder)
            from infra import existing_models as _existing

            _existing.adopt_backup_choice()
        except Exception:  # noqa: BLE001 - never get in the way of starting up
            pass
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
    if thin_wait:
        # the check above ran without PyTorch (GPU unknown, so possibly the wrong TTS size): after the components the
        # prefetch looks again and simply finds nothing to do when every model is already there
        want_prefetch = True
    win = StudioWindow(autocheck=not selftest and not thin_wait, prefetch=want_prefetch and not thin_wait)
    app.aboutToQuit.connect(win.shutdown)
    if not selftest:
        from infra import hard_exit

        hard_exit.arm()          # closing the main window kills every helper process at once (docs/BUILDING.md)
    win.show_studio()
    splash.finish(win)                    # closes once the Studio window is on screen
    if thin_wait:
        _offer_components(win, app, want_prefetch)
    if selftest:
        QTimer.singleShot(300, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
