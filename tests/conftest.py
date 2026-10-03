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
    """Тесты никогда не пишут в настоящий каталог данных приложения."""
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "voxprint_home"))
    # Старые проверки сравнивают русские сообщения; язык по умолчанию в тестах - русский.
    # (Тесты локализации задают VOXPRINT_LANG сами.)
    monkeypatch.setenv("VOXPRINT_LANG", "ru")
    # Никаких обращений к сети/чужим кэшам моделей из обычных тестов (отдельные тесты включают их сами).
    monkeypatch.setenv("VOXPRINT_NO_EXTERNAL_MODELS", "1")
    monkeypatch.setenv("VOXPRINT_NO_MIRROR", "1")
    monkeypatch.setenv("VOXPRINT_NO_ENV_PROBE", "1")
    monkeypatch.setenv("VOXPRINT_OWN_ENV", "1")   # the test interpreter counts as Voxprint-owned (auto-upgrade); external-env tests pass external_env=True
    from core import i18n

    i18n.reset()
    yield
    i18n.reset()
