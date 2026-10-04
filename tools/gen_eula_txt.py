"""Plain-text copy of the End User Agreement for the installer licence page (Inno Setup shows .txt / .rtf).

``python tools/gen_eula_txt.py``  writes ``docs/legal/EULA-audiobook-builder.txt`` (CRLF) from the Markdown source; a test keeps them in sync.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "legal" / "EULA-audiobook-builder.md"
DST = ROOT / "docs" / "legal" / "EULA-audiobook-builder.txt"


def to_text(md: str) -> str:
    """Markdown -> plain text: headings in capitals, bullets as ``-``, no emphasis / code marks, quotes without ``>``."""
    out = []
    for line in md.splitlines():
        m = re.match(r"^(#+)\s+(.*)$", line)
        if m:
            title = m.group(2)
            out += ["", title.upper() if len(m.group(1)) <= 2 else title, ""]
            continue
        line = re.sub(r"^>\s?", "", line)
        line = re.sub(r"^\*\s+", "- ", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        line = line.replace("**", "").replace("`", "")
        line = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"\1", line)
        out.append(line)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip("\n")
    text = text.replace(" \u00b7 ", " - ")                     # plain ASCII: Inno reads a file without BOM as ANSI
    return text.replace("\n", "\r\n") + "\r\n"


def main() -> int:
    DST.write_bytes(to_text(SRC.read_text(encoding="utf-8")).encode("utf-8"))
    print("wrote", DST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
