"""Shared pytest setup: every test gets an isolated app-data folder, a fixed language and no network/model-cache access."""
import ctypes
import faulthandler
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Tests never write to the real application data folder."""
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "voxprint_home"))
    # Many tests compare Russian messages, so the default test language is Russian.
    # (The localization tests set VOXPRINT_LANG themselves.)
    monkeypatch.setenv("VOXPRINT_LANG", "ru")
    # The real machine's %APPDATA%/%LOCALAPPDATA% must not leak into the search for earlier installs (Windows)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata_roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata_local"))
    # Ordinary tests never touch the network or other programs' model caches (dedicated tests switch them on).
    monkeypatch.setenv("VOXPRINT_NO_EXTERNAL_MODELS", "1")
    monkeypatch.setenv("VOXPRINT_NO_MIRROR", "1")
    monkeypatch.setenv("VOXPRINT_NO_ENV_PROBE", "1")
    # The voice index is public now: windows that refresh it when they open (Narrate, My voices) must not reach the real one
    # (a late answer adds online voices to the list in the middle of a test). Tests that need an index pass their own fetch/URL.
    monkeypatch.setenv("VOXPRINT_VOICES_INDEX", "offline://tests")
    monkeypatch.setenv("VOXPRINT_OWN_ENV", "1")   # the test interpreter counts as Voxprint-owned (auto-upgrade); external-env tests pass external_env=True
    monkeypatch.delenv("VOXPRINT_NET_IFACE", raising=False)
    from infra import netroute

    netroute.reset_memory()
    from infra import modelscope_mirror

    modelscope_mirror.reset_verdict()
    from core import i18n

    i18n.reset()
    yield
    i18n.reset()


_ci_exit: int | None = None


def remember_exit_status(exitstatus: int) -> None:
    """Remember the session result so :func:`leave_before_native_shutdown` can use it."""
    global _ci_exit
    _ci_exit = int(exitstatus)


def _terminate_windows(code: int) -> bool:
    """End this process without ``DLL_PROCESS_DETACH``. ``ExitProcess`` (what ``os._exit`` calls) still runs it,
    and on Python 3.14 that detach access-violates once Qt is loaded.

    ``faulthandler`` (enabled by the CI pytest command) reports that kill as an access violation and the
    process exit code becomes the exception code. Silence it first so a green session stays exit 0.
    """
    faulthandler.disable()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    return bool(kernel.TerminateProcess(kernel.GetCurrentProcess(), code & 0xFFFFFFFF))


def _qt_loaded() -> bool:
    """True once this process has imported PySide6 (the Windows hard-exit then crashes the session)."""
    return any(name == "PySide6" or name.startswith("PySide6.") for name in sys.modules)


def leave_before_native_shutdown() -> bool:
    """On GitHub Actions, exit with the session result and skip interpreter shutdown.

    Python 3.14 and PySide6 print ``QObject: shared QObject was deleted directly`` and then
    the process dies with a bus error after a green session. The result is already decided.
    A crash during a test never reaches this function. Off CI the process exits normally.
    Streams are flushed first: ``os._exit`` does not, and the failure text would never reach the log.

    Windows with Qt loaded exits normally. ``TerminateProcess`` and ``ExitProcess`` both come back as a
    non-zero status for those sessions (``test_auto_quality`` and ``test_screen_fit``). A Windows session
    that never imported Qt still uses ``TerminateProcess``, which skips the DLL detach crash.
    """
    if os.environ.get("GITHUB_ACTIONS") != "true" or _ci_exit is None:
        return False
    code = int(_ci_exit)
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == "win32" and _qt_loaded():
        return False
    if sys.platform == "win32" and _terminate_windows(code):
        return True
    os._exit(code)


def pytest_sessionfinish(session, exitstatus):
    remember_exit_status(exitstatus)
    # The Windows parent trusts this line. Qt shutdown can still change the process exit code after a green
    # session, and that code is not the test result.
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"VOXPRINT_PYTEST_RC={int(exitstatus)}", flush=True)


def pytest_unconfigure(config):
    # After the terminal summary. Earlier than this, os._exit would hide the failure list.
    leave_before_native_shutdown()
