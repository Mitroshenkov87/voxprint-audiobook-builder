"""Каталоги приложения (Windows: %LOCALAPPDATA%\\Voxprint; переопределяется VOXPRINT_HOME)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "Voxprint"


def app_home() -> Path:
    env = os.environ.get("VOXPRINT_HOME")
    if env:
        p = Path(env)
    elif sys.platform == "win32":
        p = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / APP_NAME
    else:
        p = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / APP_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sub(name: str) -> Path:
    p = app_home() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    return _sub("models")


def logs_dir() -> Path:
    return _sub("logs")


def staging_dir() -> Path:
    """.staging/ - сюда ставятся обновления до проверки совместимости."""
    return _sub(".staging")


def packages_dir() -> Path:
    """Каталог с обновлёнными Python-пакетами; подключается в sys.path при старте (до импорта тяжёлых библиотек)."""
    return app_home() / "packages"


def state_dir() -> Path:
    return _sub("state")


def default_results_dir() -> Path:
    return Path.home() / "Documents" / APP_NAME


def resource_dir() -> Path:
    """Корень ресурсов, поставляемых с программой (locales/, licenses/, credits.json).
    В сборке PyInstaller - каталог распаковки (sys._MEIPASS), иначе корень проекта."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base)
    return Path(__file__).resolve().parent.parent
