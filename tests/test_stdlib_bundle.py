"""The thin shell bundles the COMPLETE standard library (real PC failure: No module named 'http.cookies')."""
import os
import sys
import sysconfig
from pathlib import Path

from tools import gen_stdlib_bundle as g


def test_submodules_of_packages_are_listed_not_only_top_level_names():
    names = set(g.modules())
    for n in ("http.cookies", "http.client", "email.mime.text", "xml.dom.minidom", "logging.config", "urllib.robotparser", "timeit",
              "json.decoder", "importlib.metadata", "concurrent.futures.thread", "multiprocessing.pool", "encodings.cp1251", "unittest.mock",
              "asyncio.windows_events", "wsgiref.simple_server", "xmlrpc.server", "ctypes.wintypes"):
        assert n in names, n
    for n in ("tkinter", "tkinter.ttk", "idlelib", "test", "lib2to3", "turtledemo", "unittest.test"):
        assert n not in names, n


def test_every_pure_python_file_of_the_standard_library_is_covered():
    """Independent of the walker: look at the files of Lib/ itself."""
    lib = Path(sysconfig.get_paths()["stdlib"])
    names = set(g.modules())
    missing = []
    for root, dirs, files in os.walk(lib):
        rel = Path(root).relative_to(lib)
        dirs[:] = [d for d in dirs if d not in ("site-packages", "lib-dynload", "__pycache__", "config-3.11-x86_64-linux-gnu")
                   and not d.startswith("config-")]
        for f in files:
            if not f.endswith(".py"):
                continue
            parts = list(rel.parts) + ([] if f == "__init__.py" else [f[:-3]])
            if not parts:
                continue
            name = ".".join(parts)
            if g.skipped(name) or parts[0] not in sys.stdlib_module_names:
                continue
            if name not in names:
                missing.append(name)
    assert missing == [], missing[:20]


def test_the_generated_module_is_valid_and_imports_everything():
    src = g.render()
    compile(src, "_vx_stdlib.py", "exec")
    assert "import http.cookies" in src and "import email.mime.text" in src


def test_the_import_list_for_the_frozen_check_only_has_importable_names(tmp_path):
    names = g.modules()[:60]
    ok = g.importable(names)
    assert set(ok) <= set(names) and ok
    assert g.main([str(tmp_path)]) == 0
    listed = (tmp_path / "stdlib_modules.txt").read_text().split()
    assert "http.cookies" in listed and "os" in listed
    assert (tmp_path / "_vx_stdlib.py").read_text().count("\ntry:") >= 500  # 3.11 has ~650, newer Pythons drop some (e.g. distutils)


def test_the_build_script_and_ci_use_it():
    root = Path(__file__).resolve().parents[1]
    bat = (root / "build_thin.bat").read_text(encoding="utf-8")
    assert "gen_stdlib_bundle.py" in bat and "--hidden-import _vx_stdlib" in bat
    ci = (root / ".github" / "workflows" / "build-installer.yml").read_text(encoding="utf-8")
    assert "--selftest-imports" in ci and "--stdlib-list" in ci and "stdlib_modules.txt" in ci
    main = (root / "main.py").read_text(encoding="utf-8")
    for needle in ("--stdlib-list", "requests", "qwen_asr", "qwen_tts", "transformers", "huggingface_hub"):
        assert needle in main
