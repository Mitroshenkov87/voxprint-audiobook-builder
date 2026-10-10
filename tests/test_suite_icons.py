"""Suite action icons: the five drawings are shipped, and no legacy save glyph is referenced."""
from pathlib import Path

from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("save", "save-as", "save-copy", "export", "download")
BANNED = (
    "assets/icons/floppy.svg",
    "assets/icons/floppy-disk.svg",
    "assets/floppy.svg",
    "assets/floppy.png",
    "assets/icons/content-save.svg",
)
NEEDLES = ("floppy", "SP_DialogSaveButton", "mdi-content-save", "content-save")
SCAN_DIRS = ("ui", "assets", "locales", "docs")
SCAN_FILES = ("cli.py", "build.bat", "build_thin.bat")


def test_suite_svgs_exist_and_use_current_color():
    folder = ROOT / "assets" / "icons"
    note = (folder / "LICENSE.txt").read_text(encoding="utf-8")
    assert "Apache-2.0" in note and "currentColor" in note
    for name in NAMES:
        text = (folder / f"{name}.svg").read_text(encoding="utf-8")
        assert 'stroke="currentColor"' in text
        assert 'viewBox="0 0 22 22"' in text
    for rel in BANNED:
        assert not (ROOT / rel).exists(), rel


def test_legacy_save_glyph_is_not_referenced():
    blobs = []
    for rel in SCAN_FILES:
        blobs.append((rel, (ROOT / rel).read_text(encoding="utf-8")))
    for folder in SCAN_DIRS:
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".ico", ".gif", ".webp"}:
                continue
            blobs.append((path.relative_to(ROOT).as_posix(), path.read_text(encoding="utf-8", errors="ignore")))
    for rel, text in blobs:
        low = text.lower()
        for needle in NEEDLES:
            assert needle.lower() not in low, (rel, needle)


def test_icons_are_sharp_at_hidpi_and_follow_the_theme():
    QApplication.instance() or QApplication([])
    from ui import main_window
    from ui.suite_icons import DISABLED, HOVER, LIGHT, render_pixmap

    assert main_window.TEXT == "#f2f2f5"
    assert LIGHT == main_window.TEXT_ON_PRIMARY
    assert HOVER == main_window.ACCENT_HOVER
    assert DISABLED == main_window.TEXT_DISABLED
    dark = render_pixmap("save", main_window.TEXT, logical=22, dpr=2.0)
    light = render_pixmap("save", LIGHT, logical=22, dpr=2.0)
    assert dark.width() == 44 and dark.devicePixelRatio() == 2.0
    assert dark.toImage() != light.toImage()
    menu = render_pixmap("export", main_window.TEXT, logical=16, dpr=3.0)
    assert menu.width() == 48 and menu.devicePixelRatio() == 3.0


def test_icons_are_packed_for_windows_and_linux():
    bat = (ROOT / "build.bat").read_text(encoding="utf-8")
    thin = (ROOT / "build_thin.bat").read_text(encoding="utf-8")
    extra = (ROOT / "tools" / "make_linux_package.py").read_text(encoding="utf-8")
    assert "assets\\icons;assets\\icons" in bat
    assert "assets\\icons;assets\\icons" in thin
    for name in (*NAMES, "LICENSE.txt"):
        assert f"assets/icons/{name}" in extra
    wired = "\n".join(
        (ROOT / name).read_text(encoding="utf-8")
        for name in (
            "ui/voices_window.py", "ui/narrate_window.py", "ui/revoice_window.py", "ui/modules_dialog.py",
        )
    )
    for name in NAMES:
        assert name in wired
