"""Voxprint entry point (Windows 11; the code also runs on Linux for development and tests).

Flags: ``--run-mdl`` (forward the rest of the arguments to ``voxprint-mdl``), ``--selftest`` (start and quit),
``--selftest-imports`` (import every heavy library - checks that a PyInstaller build is complete),
``--selftest-speech`` (model-free TTS / ASR / aligner smoke),
``--selftest-narrate [voice]`` (narrate two sentences headless; writes logs/selftest_narrate.txt),
``--verify-install`` (install check with reason codes; also written to logs/verify_install.txt),
``--repair`` (rebuild only Voxprint's own environment; logs/repair.txt),
``--auto-repair`` (check every component and model by hash, fetch missing / broken parts; logs/auto_repair.txt),
``--register-models-user`` / ``--unregister-models-user --out FILE`` (shared ``models/.users.json``, infra/models_users.py);
``--register-runtime-user`` / ``--unregister-runtime-user --out FILE`` (``runtime/.users.json``, same format);
``--sync-suite-settings`` (installer: copy the models folder / UI language into the shared ``state/suite.json``).
``--register-install`` / ``--unregister-install`` record or drop the suite discovery entry (install path and CLI path).
User CLI subcommands (see ``cli.py`` / docs/CLI.md): ``narrate``, ``train``, ``voices``.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from typing import Sequence

from infra.gpu_requirement import enforce, startup_is_gui, startup_requires_gpu
from infra.qwen_tf_compat import install as install_qwen_tf


def _setup_logging() -> None:
    """Rotating ``logs/voxprint.log`` (5 MB x 6) with exception hooks (infra/diagnostics.py); the console if the file fails."""
    from infra import diagnostics

    diagnostics.setup_logging(filters=(_drop_sox_warning, diagnostics.drop_known_noise))
    diagnostics.quiet_known_warnings()


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
    t = time.time()
    try:                                # CUDA 12 side libraries must be registered before this import (infra/cuda12_libs.py)
        from infra.cuda12_libs import import_ctranslate2

        mod = import_ctranslate2()
        lines.append(f"OK    ctranslate2 {getattr(mod, '__version__', '')} ({time.time() - t:.1f}s)")
    except Exception as exc:  # noqa: BLE001
        bad += 1
        lines.append(f"FAIL  ctranslate2: {type(exc).__name__}: {exc}")
    install_qwen_tf()
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
            exec(stmt, {})  # nosec B102 - constant import statements of the self-test above
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
        from core import i18n
        from infra import paths

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
            how = "reused  " if m.reused else "installed" if m.installed else "update   " if m.update else "missing  "
            out(f"{m.id:8} {how} {m.size / 2**20:9.0f} MB  {m.title}")
        return 0
    except modules.ModulesError as exc:
        out(f"ERROR: {exc}")
        return 1


def _offer_components(win, app, then_prefetch: bool, autostart: bool = True) -> None:
    """Thin build, first start: the Components window downloads the runtime modules and SAGE without a click (step 1) and then
    starts the download of ALL heavy models (step 2, the main window's first-run prefetch), showing one live line throughout.
    ``autostart=False`` (Quick setup, :mod:`infra.setup_mode`): the same download is only offered and starts on the click."""
    from PySide6.QtCore import QTimer

    from ui.modules_dialog import ModulesDialog, default_extras

    holder: dict[str, ModulesDialog] = {}

    def models_start(hook):
        # StudioWindow.start_prefetch delegates to the Train window (it owns the status line and the busy lock).  Before
        # this existed the call raised AttributeError inside the Qt slot and the models never started after the components.
        # Optional models start only when the Components window's box is ticked.
        extra = bool(holder.get("dlg") is not None and holder["dlg"].include_optional())
        return win.start_prefetch(hook, everything=extra) if then_prefetch else None

    offer = 0
    essentials = 0
    optional = 0
    try:
        from infra import setup_mode

        sizes = setup_mode.full_sizes()
        offer = 0 if autostart else sizes["total"]
        essentials = sizes["runtime"]
        optional = max(0, sizes["total"] - sizes["runtime"])
    except Exception:  # noqa: BLE001 - only the size on the line is missing then
        offer = 0
    dlg = ModulesDialog(win.styleSheet(), win, extras_fn=default_extras, models_start=models_start if then_prefetch else None,
                        autostart=autostart, offer_size=offer, essentials_bytes=essentials, optional_bytes=optional)
    holder["dlg"] = dlg
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
    from infra.stdio_guard import guard_stdio

    guard_stdio()
    argv = list(sys.argv if argv is None else argv)
    if "--run-mdl" in argv:
        from voxprint_common.mdl.cli import main as mdl_main

        index = argv.index("--run-mdl")
        return int(mdl_main(argv[index + 1:]))
    # RTX 40-series (compute >= 8.9) before any heavy import. --version, help, status and diagnostics stay usable.
    if startup_requires_gpu(argv):
        blocked = enforce(gui=startup_is_gui(argv))
        if blocked:
            return blocked
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
    if "--register-runtime-user" in argv or "--unregister-runtime-user" in argv:
        # installer / uninstaller helper: runtime\.users.json, and drop our shared-runtime ref
        from infra import models_users

        root = models_users.runtime_root()
        if "--register-runtime-user" in argv:
            models_users.register_quietly(root)
            return 0
        i = argv.index("--out") if "--out" in argv else -1
        code = models_users.unregister_cli(argv[i + 1] if 0 <= i < len(argv) - 1 else None, root=root)
        try:
            from infra import suite_shared

            suite_shared.release_runtime()
        except Exception:  # noqa: BLE001 - the users file is already written; a missing manifest is normal
            logging.getLogger("voxprint").debug("shared runtime ref not released", exc_info=True)
        return code
    if "--sync-suite-settings" in argv:      # installer helper: shared state\suite.json (infra/suite_settings.py)
        from infra import suite_settings

        return suite_settings.sync_cli()
    if "--register-install" in argv or "--unregister-install" in argv:
        from infra import install_layout

        if "--unregister-install" in argv:
            install_layout.unregister_install()
        else:
            install_layout.register_current()
        return 0
    if "--install-suite" in argv or "--unregister-suite" in argv:
        from infra import suite_install

        return suite_install.cli(argv)
    if "--register-models-user" in argv or "--unregister-models-user" in argv:
        # installer / uninstaller helper for the shared models\.users.json (infra/models_users.py, agreed with the Movie Dubber)
        from infra import models_users

        if "--register-models-user" in argv:
            models_users.register_quietly()
            return 0
        i = argv.index("--out") if "--out" in argv else -1
        return models_users.unregister_cli(argv[i + 1] if 0 <= i < len(argv) - 1 else None)
    # User-facing headless CLI: narrate / train / voices (see cli.py, docs/CLI.md). Keep before the GUI and
    # the maintenance flags so `python main.py narrate ...` and a packaged exe work the same way.
    try:
        from cli import is_user_cli
        from cli import main as user_cli_main
    except ImportError:
        user_cli_main = None  # type: ignore[assignment]
        def is_user_cli(argv: Sequence[str]) -> bool:
            return False
    if user_cli_main is not None and is_user_cli(argv):
        from infra import diagnostics
        from infra.stdio_guard import install_cli_excepthook

        install_cli_excepthook()
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
        from infra import modules as _m
        from infra import runtime_reuse

        _m.activate()
        return runtime_reuse.probe_torch_main()
    if "--modules-status" in argv or "--install-modules" in argv:   # thin build: list / download the runtime modules (no GUI)
        return _modules_cli(argv)
    if "--selftest-text" in argv:   # headless, no GPU/models: text prep, chunking, stub translation, ffmpeg encode
        from workers import selftest_text

        return selftest_text.run()
    if "--selftest-speech" in argv:   # headless, no GPU/models: sine-wave TTS, fake ASR, fake aligner, WAV
        from workers import selftest_speech

        return selftest_speech.run()
    if "--selftest-narrate" in argv:   # headless: narrate two sentences with the first voice (argument after the flag = voice id)
        from workers import selftest_narrate

        i = argv.index("--selftest-narrate")
        return selftest_narrate.run(argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else "")
    selftest = "--selftest" in argv  # start and quit automatically (used for the offscreen smoke check)
    from infra import suite_install

    def _suite_note(fraction: float, message: str) -> None:
        if message:
            print(message, file=sys.stderr, flush=True)

    # A frozen app repairs before the window. --selftest does the same check and prints progress on stderr
    # so a --json result line on stdout stays one JSON object. The GUI path repairs after the splash exists.
    if selftest:
        suite_report = suite_install.launch_check(_suite_note, repair=suite_install.should_repair())
        if "--json" in argv:
            print(json.dumps({
                "type": "result", "ok": bool(suite_report.get("ok")), "command": "selftest",
                "exit_code": int(suite_report.get("exit_code", 0)),
                "outputs": [], "warnings": [], "duration_s": 0.0, "error": None, "hint": None,
                "suite": {"root": suite_report.get("root", ""), "missing": list(suite_report.get("missing") or []),
                          "repaired": bool(suite_report.get("repaired"))},
            }, ensure_ascii=False), flush=True)
    else:
        suite_report = suite_install.launch_check(repair=False)
    from infra import models_users

    models_users.register_quietly()   # this program uses the shared models folder (models\.users.json)
    models_users.register_runtime_quietly()   # runtime\.users.json, created here on the first GUI start
    from infra import install_layout

    install_layout.register_quietly()  # frozen install: manifest installs + the current-user registry key
    from infra import suite_settings

    suite_settings.migrate_quietly()  # earlier choices (models folder, UI language) into the shared state\suite.json

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(argv)
    from ui import splash as splash_mod

    splash = splash_mod.show()            # the first thing on screen: before the heavy UI imports below
    splash.loading_ui()
    if (not selftest) and suite_install.should_repair() and suite_report.get("missing"):
        def _suite_on_splash(fraction: float, message: str, splash=splash) -> None:
            if message:
                print(message, file=sys.stderr, flush=True)
            splash.status(message or splash.message(), fraction)

        suite_report = suite_install.launch_check(_suite_on_splash, repair=True)
    from ui import screen_fit
    from ui.studio import StudioWindow


    screen_fit.install(app)               # every window / dialog clamped to the screen's work area when shown
    from ui import std_buttons

    std_buttons.install(app)              # OK / Cancel / Yes / No ... in the UI language, not the system's
    import threading

    from infra import diagnostics

    diagnostics.install_qt_message_handler()
    # system / GPU / settings block of the log; the GPU probe imports PyTorch, so it runs beside the UI
    threading.Thread(target=diagnostics.log_startup, name="diag-startup", daemon=True).start()
    app.setApplicationName("Voxprint")
    app.setDesktopFileName("voxprint-audiobook-builder")  # Linux: ties the window to the desktop file; ignored elsewhere
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
    splash.checking()
    if not selftest:
        try:   # a "models folder" that is really a Voxprint backup becomes a restore source (never the live folder)
            from infra import existing_models as _existing

            _existing.adopt_backup_choice()
        except Exception:  # noqa: BLE001 - never get in the way of starting up
            pass
        from infra import mdl_gate
        # Qt stays out of --version and the headless CLI, so this dialog is imported here.
        from ui.mdl_dialog import prompt as mdl_prompt

        if not mdl_gate.models_ready():
            splash.finish(None)
            if not mdl_prompt():
                return 4
        mdl_gate.materialize()
    win = StudioWindow(autocheck=not selftest, prefetch=False)
    app.aboutToQuit.connect(win.shutdown)
    if not selftest:
        from infra import hard_exit

        hard_exit.arm()          # closing the main window kills every helper process at once (docs/BUILDING.md)
    win.show_studio()
    splash.finish(win)                    # closes once the Studio window is on screen
    if not selftest:   # "VOXPrint Projects" shortcut in the real Documents folder (infra/projects.py); never blocks the start
        from infra import projects

        threading.Thread(target=projects.ensure_shortcut, name="projects-shortcut", daemon=True).start()
    if selftest:
        QTimer.singleShot(300, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
