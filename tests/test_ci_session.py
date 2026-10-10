"""CI leaves the process after a finished pytest session so Qt shutdown cannot turn a pass into a signal."""
import os

from tests import conftest


def test_ci_records_the_pytest_status_before_shutdown(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    conftest.pytest_sessionfinish(None, 0)
    assert "VOXPRINT_PYTEST_RC=0" in capsys.readouterr().out
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    conftest.pytest_sessionfinish(None, 1)
    assert "VOXPRINT_PYTEST_RC" not in capsys.readouterr().out


def test_off_ci_the_session_exits_normally(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    conftest.remember_exit_status(0)
    assert conftest.leave_before_native_shutdown() is False
    assert os.environ.get("GITHUB_ACTIONS") is None


def test_ci_exits_with_the_session_status_after_the_summary(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(conftest.sys, "platform", "linux")
    seen = []
    monkeypatch.setattr(conftest.os, "_exit", seen.append)
    conftest.remember_exit_status(1)
    conftest.leave_before_native_shutdown()
    conftest.remember_exit_status(0)
    conftest.leave_before_native_shutdown()
    assert seen == [1, 0]


def test_windows_ci_terminates_without_dll_detach(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(conftest.sys, "platform", "win32")
    monkeypatch.setattr(conftest, "_qt_loaded", lambda: False)
    terminated, exited = [], []
    monkeypatch.setattr(conftest, "_terminate_windows", lambda code: terminated.append(code) or True)
    monkeypatch.setattr(conftest.os, "_exit", exited.append)
    conftest.remember_exit_status(0)
    conftest.leave_before_native_shutdown()
    conftest.remember_exit_status(1)
    conftest.leave_before_native_shutdown()
    assert terminated == [0, 1] and exited == []


def test_windows_qt_session_exits_normally(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(conftest.sys, "platform", "win32")
    monkeypatch.setattr(conftest, "_qt_loaded", lambda: True)
    terminated, exited = [], []
    monkeypatch.setattr(conftest, "_terminate_windows", lambda code: terminated.append(code) or True)
    monkeypatch.setattr(conftest.os, "_exit", exited.append)
    conftest.remember_exit_status(0)
    assert conftest.leave_before_native_shutdown() is False
    assert terminated == [] and exited == []


def test_windows_terminate_silences_faulthandler_before_the_kill(monkeypatch):
    order = []

    class Kernel:
        def GetCurrentProcess(self):
            order.append("proc")
            return 7

        def TerminateProcess(self, handle, code):
            order.append(("kill", handle, code))
            return 1

    monkeypatch.setattr(conftest.faulthandler, "disable", lambda: order.append("dis"))
    monkeypatch.setattr(conftest.ctypes, "WinDLL", lambda *a, **k: Kernel(), raising=False)
    assert conftest._terminate_windows(0) is True
    assert order == ["dis", "proc", ("kill", 7, 0)]
