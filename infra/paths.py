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


def previous_homes() -> list:
    """Папки данных более ранних установок Voxprint (только чтение): в них можно найти модели и настройки.

    Обновление через установщик сохраняет %LOCALAPPDATA%\\Voxprint (то же app_home) - это и есть основной случай.
    Дополнительно смотрим типичные прежние места и VOXPRINT_PREVIOUS_HOMES (список через os.pathsep)."""
    cur = None
    try:
        cur = app_home().resolve()
    except OSError:
        pass
    cands = []
    for part in os.environ.get("VOXPRINT_PREVIOUS_HOMES", "").split(os.pathsep):
        if part.strip():
            cands.append(Path(part.strip().strip('"')))
    if sys.platform == "win32":
        for var in ("APPDATA", "LOCALAPPDATA"):
            base = os.environ.get(var)
            if base:
                cands.append(Path(base) / APP_NAME)
                cands.append(Path(base) / APP_NAME.lower())
    try:
        home = Path.home()
        cands += [home / ".voxprint", home / ".local" / "share" / APP_NAME]
    except (RuntimeError, OSError):
        pass
    out = []
    for c in cands:
        try:
            r = c.resolve()
            if r == cur or r in [o.resolve() for o in out] or not c.is_dir():
                continue
            if any((c / n).is_dir() for n in ("models", "state", "packages")):
                out.append(c)
        except OSError:
            continue
    return out


#: Маленькие файлы настроек, которые переносим из прежней установки (копируем только если их ещё нет).
ADOPTED_STATE_FILES = ("language", "privacy_ack", "updater_state.json", "last_adapter.json")


def adopt_previous_settings() -> list:
    """Копирует настройки из прежней установки Voxprint в текущий каталог состояния (старая папка не меняется).
    Возвращает список перенесённых файлов."""
    done = []
    dst_dir = state_dir()
    for home in previous_homes():
        src_dir = home / "state"
        for name in ADOPTED_STATE_FILES:
            src, dst = src_dir / name, dst_dir / name
            try:
                if src.is_file() and not dst.exists():
                    dst.write_bytes(src.read_bytes())
                    done.append(name)
            except OSError:
                continue
    return done
