"""Write offscreen pictures of the shared audiobook screen and the voice-training window.

    QT_QPA_PLATFORM=offscreen python tools/grab_ui_shell.py docs/screenshots/ui-shell
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VOXPRINT_SKIP_SUITE", "1")
os.environ.setdefault("VOXPRINT_NO_EXTERNAL_MODELS", "1")
os.environ.setdefault("VOXPRINT_NO_MIRROR", "1")
os.environ.setdefault("VOXPRINT_NO_ENV_PROBE", "1")
os.environ.setdefault("VOXPRINT_ALLOW_NO_GPU", "1")
os.environ.setdefault("VOXPRINT_VOICES_INDEX", "offline://tests")
os.environ.setdefault("VOXPRINT_OWN_ENV", "1")
os.environ.setdefault("VOXPRINT_HOME", "/tmp/abb-ui-shell-grab")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication, QWidget

from core import i18n
from ui.main_window import MainWindow
from ui.shell_window import AudiobookShell
from voxprint_common.i18n import set_language


def main() -> None:
    """Grab English and Russian screens into the directory named on the command line."""
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "docs" / "screenshots" / "ui-shell")
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    _use_home(out, "simple-en")
    english = _shell(app, "en")
    _save(english.widget, out / "simple-en.png")
    english.show_progress(3, 12, "Reading chapter 3", eta_seconds=90)
    app.processEvents()
    _save(english.widget, out / "progress-en.png")
    english.widget.close()
    app.processEvents()

    _use_home(out, "expert-en")
    english_expert = _shell(app, "en")
    _show_expert(english_expert, app)
    _save(english_expert.widget, out / "expert-en.png")
    english_expert.widget.close()

    _use_home(out, "simple-ru")
    russian = _shell(app, "ru")
    _save(russian.widget, out / "simple-ru.png")
    russian.widget.close()
    app.processEvents()

    _use_home(out, "expert-ru")
    russian_expert = _shell(app, "ru")
    _show_expert(russian_expert, app)
    _save(russian_expert.widget, out / "expert-ru.png")
    russian_expert.widget.close()

    _use_home(out, "train-en")
    training = _train(app, "en")
    _save(training, out / "train-en.png")
    training.close()
    _use_home(out, "train-ru")
    training = _train(app, "ru")
    _save(training, out / "train-ru.png")
    training.close()

    homes = out / ".homes"
    if homes.exists():
        shutil.rmtree(homes)
    artifacts = Path("/opt/cursor/artifacts")
    if artifacts.is_dir():
        dest = artifacts / "screenshots" / "ui-shell"
        dest.mkdir(parents=True, exist_ok=True)
        for picture in sorted(out.glob("*.png")):
            shutil.copy2(picture, dest / picture.name)
    print("wrote", out)


def _use_home(out: Path, name: str) -> None:
    home = out / ".homes" / name
    if home.exists():
        shutil.rmtree(home)
    home.mkdir(parents=True)
    os.environ["VOXPRINT_HOME"] = str(home)


def _shell(app: QApplication, language: str) -> AudiobookShell:
    set_language(language)
    window = AudiobookShell()
    window.widget.resize(1200, 900)
    window.show()
    app.processEvents()
    return window


def _show_expert(window: AudiobookShell, app: QApplication) -> None:
    window.layout.set_value("audio.loudness", -14.0)
    window.screen.refresh()
    window.screen.set_expert(True)
    app.processEvents()


def _train(app: QApplication, language: str) -> MainWindow:
    i18n.set_language(language)
    window = MainWindow(autocheck=False, prefetch=False)
    window.resize(1200, 900)
    window.show()
    app.processEvents()
    return window


def _save(widget: QWidget, path: Path) -> None:
    picture = widget.grab()
    if picture.isNull() or not picture.save(str(path)):
        raise RuntimeError(f"could not write {path}")


if __name__ == "__main__":
    main()
