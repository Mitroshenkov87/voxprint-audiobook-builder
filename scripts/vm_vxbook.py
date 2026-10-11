"""Write the short soundscape book used by scripts/vm_test.sh and scripts/vm_test.ps1.

    python3 scripts/vm_vxbook.py story.vxbook

The archive is a ``.vxbook`` (mimetype stored first) with extension ``sound/1`` and one ambience bed.
Narration leaves the soundscape off unless the setting is on and the model is installed.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

MIME = b"application/vnd.voxprint.book+zip"

BOOK = """<!-- vx:chapter id="ch01" -->
# Утро

Ещё одно короткое утро.

<!-- vx:speaker id="anna" -->
Она сказала одну строку.

<!-- vx:chapter id="ch02" -->
# Вечер

Тихий вечер в доме.
"""

SPEAKERS = {
    "format_version": "1.1",
    "speakers": [
        {"id": "narrator", "role": "narrator", "name": "Narrator", "gender": "unspecified"},
        {"id": "anna", "role": "character", "name": "Анна", "gender": "female"},
    ],
}


def _paragraph_line() -> str:
    """The first spoken line. Its fingerprint is what sound.json stores."""
    for block in BOOK.split("\n\n"):
        lines = [ln for ln in block.split("\n") if not ln.startswith("<!--")]
        if lines and not lines[-1].startswith("# "):
            return lines[-1]
    raise RuntimeError("book has no paragraph")


def _member(name: str, data: bytes) -> dict:
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def build_bytes() -> dict:
    """Name -> bytes, after the mimetype entry."""
    line = _paragraph_line()
    fp = hashlib.sha256(line.encode("utf-8")).hexdigest()[:12]
    book = BOOK.encode("utf-8")
    speakers = json.dumps(SPEAKERS, ensure_ascii=False).encode("utf-8")
    sound = json.dumps({
        "extension": "sound/1",
        "version": "1.0",
        "cues": [{
            "id": "room",
            "kind": "bed",
            "layer": "ambience",
            "prompt": "quiet room tone",
            "chapter": "ch01",
            "start_paragraph": 1,
            "end_paragraph": 1,
            "para_fp": fp,
            "end_para_fp": fp,
        }],
    }, ensure_ascii=False).encode("utf-8")
    blobs = {"book.md": book, "speakers.json": speakers, "sound.json": sound}
    chapters = [
        {"id": "ch01", "title": "Утро", "anchor": "ch01",
         "offset": book.find('<!-- vx:chapter id="ch01" -->'.encode("utf-8"))},
        {"id": "ch02", "title": "Вечер", "anchor": "ch02",
         "offset": book.find('<!-- vx:chapter id="ch02" -->'.encode("utf-8"))},
    ]
    manifest = json.dumps({
        "format_version": "1.1",
        "title": "Проверка",
        "author": "Voxprint",
        "language": "ru",
        "extensions": ["sound/1"],
        "book": "book.md",
        "speakers": "speakers.json",
        "chapters": chapters,
        "files": {name: _member(name, data) for name, data in blobs.items()},
    }, ensure_ascii=False).encode("utf-8")
    return {"manifest.json": manifest, **blobs}


def write_vxbook(path: Path) -> None:
    """Write ``path``. The mimetype entry is stored and has no extra field."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo("mimetype")
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, MIME)
        for name, data in build_bytes().items():
            zf.writestr(name, data, compress_type=zipfile.ZIP_STORED)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: vm_vxbook.py OUT.vxbook", file=sys.stderr)
        return 2
    write_vxbook(Path(argv[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
