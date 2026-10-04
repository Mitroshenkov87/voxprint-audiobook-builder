"""Closing the main window kills the whole process tree (docs/BUILDING.md, "Closing the app")."""
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from infra import hard_exit

ROOT = Path(__file__).resolve().parent.parent
posix = pytest.mark.skipif(sys.platform == "win32", reason="uses fork / SIGKILL semantics; Windows is covered by taskkill + job object")

TREE = textwrap.dedent("""
    import os, subprocess, sys, time
    # child that has a grandchild of its own, like a helper that starts ffmpeg
    code = "import subprocess,sys,time; g=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); print(g.pid, flush=True); time.sleep(120)"
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    grand = int(child.stdout.readline())
    print(child.pid, grand, flush=True)
""")


def test_descendants_are_found_and_killed():
    script = TREE + "\ntime.sleep(120)\n"
    parent = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        child, grand = (int(x) for x in parent.stdout.readline().split())
        found = hard_exit.descendants(parent.pid)
        assert child in found and grand in found
        killed = hard_exit.kill_descendants(parent.pid)
        assert set(killed) >= {child, grand}
        assert hard_exit.wait_gone([child, grand])
        assert parent.poll() is None                              # only the helpers, not the parent
    finally:
        parent.kill()
        parent.wait()


def test_fire_does_nothing_until_armed(monkeypatch):
    called = []
    monkeypatch.setattr(hard_exit, "exit_now", lambda code=0: called.append(code))
    monkeypatch.setattr(hard_exit, "_armed", False)
    hard_exit.fire()
    assert called == []
    monkeypatch.setattr(hard_exit, "_armed", True)
    hard_exit.fire()
    assert called == [0]


def test_a_real_process_that_fires_leaves_nothing_behind():
    script = TREE + textwrap.dedent("""
        sys.path.insert(0, %r)
        from infra import hard_exit
        import threading
        threading.Thread(target=lambda: time.sleep(120)).start()      # a non-daemon thread must not delay the exit
        hard_exit.arm()
        hard_exit.fire()
        print("not reached", flush=True)
    """) % str(ROOT)
    parent = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    child, grand = (int(x) for x in parent.stdout.readline().split())
    t0 = time.monotonic()
    try:
        parent.wait(timeout=15)
    finally:
        parent.kill()
    assert time.monotonic() - t0 < 10 and parent.returncode == 0
    assert hard_exit.wait_gone([child, grand]) if sys.platform != "win32" else True


def test_closing_the_studio_window_fires_the_hard_exit(monkeypatch):
    pytest.importorskip("PySide6")
    from tests.test_studio import make_studio
    from PySide6.QtWidgets import QApplication
    from core.voice_library import VoiceLibrary
    import tempfile

    app = QApplication.instance() or QApplication([])
    calls = []
    monkeypatch.setattr(hard_exit, "exit_now", lambda code=0: calls.append(code))
    monkeypatch.setattr(hard_exit, "_armed", True)
    with tempfile.TemporaryDirectory() as d:
        studio = make_studio(VoiceLibrary(Path(d) / "voices"))
        studio.show_studio()
        studio.close()
    assert calls and set(calls) == {0} and app is not None


def test_main_arms_the_hard_exit_only_for_a_real_run():
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "hard_exit.arm()" in src and src.index("if not selftest:\n        from infra import hard_exit") > 0
