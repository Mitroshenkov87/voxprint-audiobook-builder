"""CUDA 12 library path for CTranslate2: Windows DLL directories, Linux SONAME preload."""
import os
import sys

import infra.cuda12_libs as c12


def test_windows_prepare_registers_dll_directories(monkeypatch, tmp_path):
    added = []
    monkeypatch.setattr(c12, "library_dirs", lambda: [tmp_path])
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "add_dll_directory", lambda path: added.append(path), raising=False)
    assert c12.prepare() == [str(tmp_path)]
    assert added == [str(tmp_path)]


def test_linux_prepare_prepends_the_library_path(monkeypatch, tmp_path):
    monkeypatch.setattr(c12, "library_dirs", lambda: [tmp_path])
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/already")
    assert c12.prepare() == [str(tmp_path)]
    assert os.environ["LD_LIBRARY_PATH"].startswith(str(tmp_path) + os.pathsep)


def test_linux_preload_opens_sonames_only(monkeypatch, tmp_path):
    (tmp_path / "libcublas.so.12").write_bytes(b"x")
    (tmp_path / "libcublas.so.12.9.2.10").write_bytes(b"y")
    (tmp_path / "readme.txt").write_text("no", encoding="utf-8")
    opened = []

    def fake_cdll(path, mode=None):
        opened.append(path)
        return object()

    monkeypatch.setattr(c12.ctypes, "CDLL", fake_cdll)
    got = c12.preload([tmp_path])
    assert got == [str(tmp_path / "libcublas.so.12")]
    assert opened == [str(tmp_path / "libcublas.so.12")]


def test_import_preloads_on_linux_and_not_on_windows(monkeypatch):
    calls = []
    monkeypatch.setattr(c12, "prepare", lambda: calls.append("prepare") or ["/libs"])
    monkeypatch.setattr(c12, "preload", lambda dirs=None: calls.append(("preload", dirs)))
    monkeypatch.setattr(c12.importlib, "import_module", lambda name: calls.append(name) or object())
    monkeypatch.setattr(sys, "platform", "linux")
    c12.import_ctranslate2()
    assert calls[0] == "prepare" and calls[1][0] == "preload" and calls[2] == "ctranslate2"
    calls.clear()
    monkeypatch.setattr(sys, "platform", "win32")
    c12.import_ctranslate2()
    assert calls == ["prepare", "ctranslate2"]
