"""Logging set-up and the diagnostic report (Qt-free).

* :func:`setup_logging` - one rotating log ``<data folder>/logs/voxprint.log`` (5 MB x 6 files = 30 MB at most; the data folder
  is where the settings live: ``%LOCALAPPDATA%\\Voxprint`` on Windows, see :mod:`infra.paths`), plus hooks so uncaught
  exceptions of the main thread, of worker threads and Python warnings end up in it with their tracebacks.
* :func:`log_startup` - version, OS, Python, CPU / RAM, GPU / driver / CUDA / VRAM and a settings snapshot.
* :func:`write_report` - a zip with every log file, ``system_info.json`` and ``settings.json`` (Settings -> *Save diagnostic
  report...*, CLI ``voxprint diag``).

Privacy: the snapshot keeps setting values, but paths are cut to their last part and keys that look like secrets are dropped.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import platform
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, Optional

log = logging.getLogger("voxprint.diag")

LOG_FILE = "voxprint.log"
MAX_BYTES = 5_000_000
BACKUPS = 5                                      # 6 files with the current one: ~30 MB
FORMAT = "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"
_SECRET_WORDS = ("token", "secret", "password", "passwd", "apikey", "api_key", "auth", "cookie")


def setup_logging(logs_dir: Optional[Path] = None, level: int = logging.INFO,
                  filters: tuple = ()) -> Optional[Path]:
    """Attach the rotating file handler to the root logger and install the exception hooks; returns the log path (None when
    the file cannot be opened - then the console is used)."""
    root = logging.getLogger()
    root.setLevel(level)
    path = None
    try:
        if logs_dir is None:
            from infra import paths

            logs_dir = paths.logs_dir()
        path = Path(logs_dir) / LOG_FILE
        h: logging.Handler = logging.handlers.RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUPS,
                                                                  encoding="utf-8")
    except OSError:
        h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(FORMAT))
    for f in filters:
        h.addFilter(f)
    root.addHandler(h)
    logging.captureWarnings(True)                # warnings.warn(...) -> logger "py.warnings"
    install_exception_hooks()
    return path


def install_exception_hooks() -> None:
    """Uncaught exceptions of the main thread and of any thread go to the log with their traceback (then the old hooks run)."""
    old_sys, old_thread = sys.excepthook, threading.excepthook

    def sys_hook(etype, value, tb):
        if not issubclass(etype, KeyboardInterrupt):
            logging.getLogger("voxprint.crash").error("uncaught exception", exc_info=(etype, value, tb))
        old_sys(etype, value, tb)

    def thread_hook(args):
        if args.exc_type is not SystemExit:
            logging.getLogger("voxprint.crash").error("uncaught exception in thread %s",
                                                      getattr(args.thread, "name", "?"),
                                                      exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        old_thread(args)

    if getattr(sys.excepthook, "_voxprint", False):
        return
    sys_hook._voxprint = True                    # type: ignore[attr-defined]
    sys.excepthook = sys_hook
    threading.excepthook = thread_hook


def install_qt_message_handler() -> None:
    """Qt's own warnings / errors (qWarning ...) into the log."""
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    except Exception:  # noqa: BLE001
        return
    qlog = logging.getLogger("qt")
    levels = {QtMsgType.QtDebugMsg: logging.DEBUG, QtMsgType.QtInfoMsg: logging.INFO,
              QtMsgType.QtWarningMsg: logging.WARNING, QtMsgType.QtCriticalMsg: logging.ERROR,
              QtMsgType.QtFatalMsg: logging.CRITICAL}

    def handler(mode, context, message):
        qlog.log(levels.get(mode, logging.WARNING), "%s", message)

    qInstallMessageHandler(handler)


# ----------------------------------------------------------------------------------------------- system information
def gpu_info(import_torch: bool = True) -> Dict[str, object]:
    """GPU name, VRAM, CUDA (torch) and the NVIDIA driver version; empty fields when unknown."""
    out: Dict[str, object] = {"cuda_available": False}
    torch = sys.modules.get("torch")
    if torch is None and import_torch:
        try:
            import torch  # noqa: F811
        except Exception:  # noqa: BLE001
            torch = None
    if torch is not None:
        try:
            out["torch"] = torch.__version__
            out["cuda_build"] = getattr(torch.version, "cuda", None)
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                free, total = torch.cuda.mem_get_info(0)
                out.update(cuda_available=True, gpu=p.name, vram_total_gb=round(total / 1024 ** 3, 2),
                           vram_free_gb=round(free / 1024 ** 3, 2), capability=f"{p.major}.{p.minor}")
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)
    try:
        import subprocess

        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode == 0 and r.stdout.strip():
            out["driver"] = r.stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - no NVIDIA driver / tool
        pass
    return out


def system_info(import_torch: bool = True) -> Dict[str, object]:
    """Everything the start-up line and the report need."""
    from core.appinfo import APP_VERSION
    from infra import sysinfo

    total, avail = sysinfo.memory()
    return {"app_version": APP_VERSION, "os": platform.platform(), "python": sys.version.split()[0],
            "frozen": bool(getattr(sys, "frozen", False)), "cpu_logical": os.cpu_count(),
            "cpu_physical": sysinfo.physical_cores(), "ram_total_gb": round(total / 1024 ** 3, 1),
            "ram_available_gb": round(avail / 1024 ** 3, 1), "gpu": gpu_info(import_torch),
            "time": time.strftime("%Y-%m-%d %H:%M:%S %z")}


def _clean(key: str, value):
    if isinstance(value, dict):
        return {k: _clean(k, v) for k, v in value.items() if not any(w in k.lower() for w in _SECRET_WORDS)}
    if isinstance(value, list):
        return [_clean(key, v) for v in value]
    if isinstance(value, str) and ("/" in value or "\\" in value) and not value.startswith(("http://", "https://")):
        return Path(value.replace("\\", "/")).name     # a path: only its last part
    return value


def settings_snapshot(state_dir: Optional[Path] = None) -> Dict[str, object]:
    """The small settings files of ``state/`` (JSON or one-line text), secrets dropped and paths cut to basenames."""
    if state_dir is None:
        from infra import paths

        state_dir = paths.state_dir()
    out: Dict[str, object] = {}
    for f in sorted(Path(state_dir).glob("*")):
        if not f.is_file() or f.stat().st_size > 64_000 or any(w in f.name.lower() for w in _SECRET_WORDS):
            continue
        try:
            text = f.read_text(encoding="utf-8")
            data = json.loads(text) if f.suffix == ".json" else text.strip()
        except Exception:  # noqa: BLE001 - binary / broken files are not settings
            continue
        out[f.name] = _clean(f.name, data)
    return out


def log_startup(import_torch: bool = True) -> None:
    """One block at the start: system information and settings (the GPU probe imports torch: call it off the UI thread)."""
    try:
        log.info("start: %s", json.dumps(system_info(import_torch), ensure_ascii=False))
        log.info("settings: %s", json.dumps(_short_values(settings_snapshot()), ensure_ascii=False))
    except Exception:  # noqa: BLE001 - diagnostics must never stop the program
        log.exception("start-up diagnostics failed")


#: Longest settings entry written into the start-up log line (big manifests are only in the diagnostics report).
LOG_VALUE_MAX = 300


def _short_values(snap: Dict[str, object]) -> Dict[str, object]:
    """The settings snapshot for the log: an entry longer than ``LOG_VALUE_MAX`` characters is replaced by its size."""
    out: Dict[str, object] = {}
    for k, v in snap.items():
        n = len(json.dumps(v, ensure_ascii=False))
        out[k] = v if n <= LOG_VALUE_MAX else f"<{n} chars, see the diagnostics report>"
    return out


def cuda_memory() -> str:
    """``"VRAM 3.21 GB (peak 4.02)"`` when torch with CUDA is already loaded, else ``""`` (never imports torch)."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            return "VRAM %.2f GB (peak %.2f)" % (torch.cuda.memory_allocated() / 1024 ** 3,
                                                 torch.cuda.max_memory_allocated() / 1024 ** 3)
    except Exception:  # noqa: BLE001
        pass
    return ""


# ------------------------------------------------------------------------------------------------- the report
def write_report(dest: Path, logs_dir: Optional[Path] = None, state_dir: Optional[Path] = None,
                 info: Optional[Callable[[], dict]] = None) -> Path:
    """Zip every file of the logs folder plus ``system_info.json`` and ``settings.json`` into ``dest``; returns ``dest``."""
    if logs_dir is None:
        from infra import paths

        logs_dir = paths.logs_dir()
    dest = Path(dest)
    if dest.suffix.lower() != ".zip":
        dest = dest.with_name(dest.name + ".zip")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:  # noqa: BLE001
            pass
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(Path(logs_dir).rglob("*")):
            if f.is_file() and f.resolve() != dest.resolve():
                try:
                    z.write(f, "logs/" + f.relative_to(logs_dir).as_posix())
                except OSError as exc:       # a file locked by another program: skip it, say so
                    log.warning("diagnostic report: %s skipped: %s", f.name, exc)
        z.writestr("system_info.json", json.dumps((info or system_info)(), indent=1, ensure_ascii=False))
        z.writestr("settings.json", json.dumps(settings_snapshot(state_dir), indent=1, ensure_ascii=False))
    log.info("diagnostic report written: %s", dest.name)
    return dest
