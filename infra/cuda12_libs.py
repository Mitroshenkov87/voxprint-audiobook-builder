"""Put the CUDA 12 cublas and cuDNN wheels on the library path before CTranslate2 is imported.

CTranslate2's wheels are built for CUDA 12. Torch cu130 does not ship those libraries, so
``nvidia-cublas-cu12`` and ``nvidia-cudnn-cu12`` are installed beside it and registered here.
Call :func:`prepare` (or :func:`import_ctranslate2`) once the runtime folder is on ``sys.path``.
``infra.modules.activate`` does that for a thin install.

Example::

    from infra.cuda12_libs import import_ctranslate2
    ct2 = import_ctranslate2()

Dependencies: the two NVIDIA wheels (and ``ctranslate2`` when it is imported).
Licence: Apache-2.0.
Credit: Voxprint AI Audiobook Builder https://github.com/Mitroshenkov87/voxprint-audiobook-builder
"""
from __future__ import annotations

import ctypes
import importlib
import importlib.metadata as md
import os
import re
import sys
from pathlib import Path
from typing import List, Sequence

_SONAME = re.compile(r"lib.+\.so\.\d+$")

#: cublas before cuDNN: cuDNN's loader looks up cublas.
_DISTS = ("nvidia-cublas-cu12", "nvidia-cudnn-cu12")
_DIR_NAMES = ("bin", "lib")


def library_dirs() -> List[Path]:
    """``bin`` and ``lib`` directories of the installed CUDA 12 wheels, cublas first.

    Empty when the wheels are not installed (a source checkout that has not fetched them yet).
    """
    found: List[Path] = []
    seen = set()
    for dist in _DISTS:
        try:
            files = md.files(dist) or []
        except md.PackageNotFoundError:
            continue
        for entry in files:
            parent = Path(str(entry)).parent
            if parent.name not in _DIR_NAMES:
                continue
            located = entry.locate().parent
            key = str(located)
            if key in seen or not located.is_dir():
                continue
            seen.add(key)
            found.append(located)
    return found


def _apply_windows(dirs: Sequence[Path]) -> None:
    """Register each directory with the Windows DLL search (Python 3.8+)."""
    add = getattr(os, "add_dll_directory", None)
    if add is None:
        return
    for folder in dirs:
        add(str(folder))


def _prepend_library_path(dirs: Sequence[Path]) -> None:
    """Prepend the directories to ``LD_LIBRARY_PATH`` (child processes, and the dynamic loader's hint)."""
    extra = [str(folder) for folder in dirs if folder.is_dir()]
    if not extra:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    prefix = os.pathsep.join(extra)
    os.environ["LD_LIBRARY_PATH"] = prefix if not current else prefix + os.pathsep + current


def preload(dirs: Sequence[Path] | None = None) -> List[str]:
    """Load each CUDA 12 SONAME into this process (Linux).

    ``dlopen`` does not reread ``LD_LIBRARY_PATH`` for a library imported later, so the SONAMEs
    (``libcublas.so.12``, ``libcudnn.so.9``, ...) are opened with ``RTLD_GLOBAL`` first.
    """
    opened: List[str] = []
    mode = getattr(ctypes, "RTLD_GLOBAL", None)
    for folder in dirs if dirs is not None else library_dirs():
        for lib in sorted(folder.glob("lib*.so.*")):
            if not _SONAME.fullmatch(lib.name) or not lib.is_file():
                continue
            try:
                ctypes.CDLL(str(lib), mode=mode) if mode is not None else ctypes.CDLL(str(lib))
            except OSError:
                continue
            opened.append(str(lib))
    return opened


def prepare() -> List[str]:
    """Register the CUDA 12 library directories. Safe to call more than once. Returns the directories.

    Windows: ``os.add_dll_directory``. Linux: ``LD_LIBRARY_PATH``. The Linux preload of the SONAMEs
    happens in :func:`import_ctranslate2`, not here, so starting the program does not map cuDNN.
    """
    dirs = library_dirs()
    if sys.platform == "win32":
        _apply_windows(dirs)
    else:
        _prepend_library_path(dirs)
    return [str(folder) for folder in dirs]


def import_ctranslate2():
    """Register the CUDA 12 libraries, preload them on Linux, then import ``ctranslate2``.

    The import is deferred (``importlib``) so loading this module does not load the native library first.
    """
    dirs = prepare()
    if sys.platform != "win32":
        preload([Path(d) for d in dirs])
    return importlib.import_module("ctranslate2")
