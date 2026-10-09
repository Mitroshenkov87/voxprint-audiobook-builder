"""The application icon and splash are original artwork, not the old Tabler glyph."""
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    return struct.unpack(">II", data[16:24])


def _jpeg_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:2] == b"\xff\xd8"
    i = 2
    while i + 9 < len(data):
        assert data[i] == 0xFF
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return width, height
        if marker == 0xD9:
            break
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        i += 2 + length
    raise AssertionError(f"no JPEG frame in {path}")


def _ico_sizes(path: Path) -> tuple[int, ...]:
    data = path.read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    assert reserved == 0 and kind == 1
    sizes = []
    for n in range(count):
        w, h = struct.unpack_from("<BB", data, 6 + 16 * n)
        sizes.append((w or 256, h or 256))
    return tuple(sizes)


def test_icon_and_splash_are_the_square_files_the_app_loads():
    assert _png_size(ROOT / "assets" / "voxprint.png") == (1024, 1024)
    assert _jpeg_size(ROOT / "assets" / "splash.jpg") == (1024, 1024)
    assert _png_size(ROOT / "installer" / "linux" / "voxprint-256.png") == (256, 256)
    for name in ("voxprint.ico", "voxprint-setup.ico"):
        assert _ico_sizes(ROOT / "assets" / name) == tuple((s, s) for s in ICON_SIZES)


def test_tabler_glyph_is_not_shipped_or_credited():
    for gone in ("assets/voxprint.svg", "assets/alt1.png", "assets/alt2.png", "licenses/tabler-icons.txt"):
        assert not (ROOT / gone).exists(), gone
    text = (ROOT / "assets" / "ICON-LICENSE.txt").read_text(encoding="utf-8")
    assert "original artwork" in text and "splash.jpg" in text
    assert "derived from the" not in text and "Tabler Icons fingerprint" not in text
    blob = (ROOT / "credits.json").read_text(encoding="utf-8") + (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "tabler" not in blob.lower()
    extra = (ROOT / "tools" / "make_linux_package.py").read_text(encoding="utf-8")
    assert "voxprint.svg" not in extra
    credits = json.loads((ROOT / "credits.json").read_text(encoding="utf-8"))
    assert all(c["id"] != "tabler-icons" for c in credits["components"])
