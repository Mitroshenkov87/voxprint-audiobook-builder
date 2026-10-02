"""Генератор THIRD_PARTY_NOTICES.md из credits.json (единственный источник данных).

  python tools/gen_notices.py                      -> THIRD_PARTY_NOTICES.md в корне проекта (хранится в репозитории)
  python tools/gen_notices.py --check              -> код 1, если файл в репозитории отличается от credits.json
  python tools/gen_notices.py --with-installed --out build/notices/THIRD_PARTY_NOTICES.md
        -> добавляет таблицу лицензий ВСЕХ установленных пакетов (включая зависимости) из importlib.metadata;
           используется build.bat для файла, который попадает в установщик.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
OUT_DEFAULT = ROOT / "THIRD_PARTY_NOTICES.md"

GROUPS = [
    ("model", "Models (downloaded on first start; not part of the installer)"),
    ("library", "Libraries and programs shipped with Voxprint"),
    ("asset", "Assets"),
    ("tool", "Build tools (not shipped, except the PyInstaller bootloader and the Inno Setup installer runtime)"),
]
SHIP = {"bundled": "bundled", "optional": "optional, not bundled", "download": "downloaded on first start",
        "build": "build time"}

HEADER = """# Third-party notices

Voxprint (c) {author}, built with AI assistance. Version {version}.

Voxprint stands on the open-source projects and models listed below. Every component keeps its own licence;
the full licence texts are in the `licenses/` folder next to this file. This file is generated from
`credits.json` by `tools/gen_notices.py` - edit `credits.json`, not this file.
"""

SPECIAL = """## Important compliance notes

* **Qt / PySide6 (LGPL-3.0).** Qt for Python is used unmodified and is linked dynamically (separate DLL/shared
  files). You may replace these libraries with your own build of PySide6/Qt: use the folder-type build
  (`build.bat onedir`), where the libraries are ordinary files. The LGPL-3.0 and GPL-3.0 texts are in
  `licenses/qt-lgpl-3.0.txt` and `licenses/gpl-3.0.txt`; source code: <https://code.qt.io> and <https://pyside.org>.
* **FFmpeg (GPL-3.0 build).** The ffmpeg program bundled inside the `imageio-ffmpeg` wheel is a GPL build
  (`--enable-gpl --enable-version3`, verified in the Windows 7.1 binary of imageio-ffmpeg 0.6.0). It is a separate
  executable started as a subprocess; it is not linked into Voxprint. Source code:
  <https://ffmpeg.org/download.html#get-sources>; build scripts of the binary:
  <https://github.com/imageio/imageio-binaries>. If you do not want to ship a GPL binary, remove it and ship an
  LGPL build of ffmpeg or require ffmpeg on `PATH`.
* **soynlp (GPLv3) is intentionally excluded.** `qwen-asr` imports it only inside the Korean-language branch of its
  forced aligner. Voxprint does not install or bundle it; therefore **Korean alignment is unavailable**.
* **Non-commercial model.** The optional backup aligner model `MahmoudAshraf/mms-300m-1130-forced-aligner` is
  licensed CC-BY-NC-4.0 (non-commercial). It is used only when the user installs the optional `ctc-forced-aligner`
  package. The default Qwen models are Apache-2.0.
* **PyInstaller** is GPL-2.0-or-later with a special exception that allows distributing the embedded bootloader as
  part of programs under any licence (`licenses/pyinstaller.txt`). **Inno Setup** is distributed under its own
  permissive licence (`licenses/inno-setup.txt`).
* Where a licence is stated as verified, it was read from the package metadata (PyPI), the project's repository
  (GitHub) or the model card (Hugging Face) on 2026-10-03. Packages pulled in indirectly are listed in the
  appendix of the build-time version of this file.
"""


def _entry(c: Dict[str, Any]) -> List[str]:
    lines = [f"### {c['name']}", ""]
    lines.append(f"* Purpose: {c['purpose']['en']}")
    lines.append(f"* Licence: {c['license']}")
    lines.append(f"* Project: <{c['url']}>")
    lines.append(f"* Status: {SHIP.get(c['ship'], c['ship'])}")
    files = ", ".join(f"[`licenses/{f}.txt`](licenses/{f}.txt)" for f in c["license_files"])
    lines.append(f"* Licence text: {files}")
    if c.get("note"):
        lines.append(f"* Note: {c['note']['en']}")
    lines.append("")
    return lines


def render(data: Dict[str, Any], with_installed: bool = False) -> str:
    app = data["app"]
    out: List[str] = [HEADER.format(author=app["author"], version=app["version"])]
    comps = data["components"]
    for kind, title in GROUPS:
        items = [c for c in comps if c["kind"] == kind]
        if not items:
            continue
        out.append(f"## {title}\n")
        for c in items:
            out.extend(_entry(c))
    out.append(SPECIAL)
    if with_installed:
        out.append(installed_appendix())
    return "\n".join(out).rstrip() + "\n"


def installed_appendix() -> str:
    """Лицензии всех установленных пакетов (транзитивные зависимости) по метаданным дистрибутивов."""
    from importlib import metadata

    rows = []
    for d in metadata.distributions():
        md = d.metadata
        name = md.get("Name") or d.metadata.get("Summary") or "?"
        lic = md.get("License-Expression") or ""
        if not lic:
            raw = (md.get("License") or "").strip().splitlines()
            lic = raw[0][:80] if raw and len(raw[0]) < 80 else ""
        if not lic:
            cls = [x.split("::")[-1].strip() for x in (md.get_all("Classifier") or []) if x.startswith("License ::")]
            lic = "; ".join(cls)
        rows.append((str(name), d.version, lic or "see package metadata"))
    rows.sort(key=lambda r: r[0].lower())
    lines = ["## Appendix: all Python packages installed in the build environment", "",
             "Generated at build time from `importlib.metadata`; includes indirect dependencies.", "",
             "| Package | Version | Licence (from metadata) |", "|---|---|---|"]
    lines += [f"| {n} | {v} | {l.replace('|', '/')} |" for n, v, l in rows]
    return "\n".join(lines) + "\n"


def main(argv: List[str]) -> int:
    data = json.loads((ROOT / "credits.json").read_text(encoding="utf-8"))
    text = render(data, with_installed="--with-installed" in argv)
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else OUT_DEFAULT
    if "--check" in argv:
        cur = OUT_DEFAULT.read_text(encoding="utf-8") if OUT_DEFAULT.exists() else ""
        if cur.replace("\r\n", "\n") != text:
            print("THIRD_PARTY_NOTICES.md is out of date: run python tools/gen_notices.py")
            return 1
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print("written", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
