#!/usr/bin/env python3
"""Pagination check for the generated PDFs (manuals, recording scripts).

For every page except the cover (first page) and the last one:
  * the filled fraction of the content area (page minus the 22 mm top/bottom margins, measured on a rendering) must be >= MIN_FILL;
  * no 1-2 line stub of a paragraph may sit at the top of a page (a widow) or at the bottom (an orphan), and no heading
    may be the last thing on a page (found with ``pdftotext -bbox-layout``).
Needs poppler-utils (pdftoppm, pdftotext) and Pillow.  Usage: python check_pagination.py [--min-fill 0.7] file.pdf ...
Exit code 0 = every file passes.
"""
from __future__ import annotations
import argparse, re, subprocess, sys, tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

MARGIN_MM = 22.0
PAGE_MM = 297.0
DPI = 40
MIN_FILL = 0.70


def fills(pdf: Path) -> list:
    """Filled fraction of the content area of every page (measured on a rendering at 40 dpi)."""
    out = []
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-r", str(DPI), "-gray", "-png", str(pdf), f"{d}/p"], check=True)
        for f in sorted(Path(d).glob("p-*.png")):
            im = Image.open(f).convert("L")
            w, h = im.size
            px = im.load()
            bg = px[2, 2]
            y0 = int(MARGIN_MM / 25.4 * DPI)
            y1 = int((PAGE_MM - MARGIN_MM) / 25.4 * DPI)
            last = y0
            for y in range(y0, min(y1, h)):
                if any(abs(px[x, y] - bg) > 24 for x in range(int(w * 0.05), int(w * 0.95), 2)):
                    last = y
            out.append((last - y0) / (y1 - y0))
    return out


def page_lines(pdf: Path) -> list:
    """For each page a list of (block_index, [lines]) with y positions and text of the body (margins excluded)."""
    xml = subprocess.run(["pdftotext", "-bbox-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    xml = re.sub(r'<!DOCTYPE[^>]*>', '', xml)
    xml = xml.replace('xmlns="http://www.w3.org/1999/xhtml"', '')
    root = ET.fromstring(xml)
    top, bottom = MARGIN_MM / 25.4 * 72 - 2, (PAGE_MM - MARGIN_MM) / 25.4 * 72 + 2
    pages = []
    for page in root.iter("page"):
        blocks = []
        for b in page.iter("block"):
            lines = []
            for ln in b.iter("line"):
                y0, y1 = float(ln.get("yMin")), float(ln.get("yMax"))
                if y0 < top or y1 > bottom:
                    continue
                text = " ".join((w.text or "") for w in ln.iter("word"))
                lines.append((y0, y1, text))
            if lines:
                blocks.append(lines)
        pages.append(blocks)
    return pages


TERMINAL = tuple(".!?:;…»”\")")


def stubs(pdf: Path) -> list:
    """Problems of the page-break positions: ``(page_number, description)``."""
    problems = []
    pages = page_lines(pdf)
    heights = sorted(l[1] - l[0] for pg in pages for b in pg for l in b)
    body_h = heights[len(heights) // 2] if heights else 12
    for i, blocks in enumerate(pages[:-1]):
        nxt = pages[i + 1]
        if i == 0 or not blocks or not nxt:
            continue                                   # the cover has its own layout
        last_block, first_next = blocks[-1], nxt[0]
        last_line = last_block[-1]
        # a heading (taller than the body text, one or two short lines) as the last thing on a page
        if len(last_block) <= 2 and (last_line[1] - last_line[0]) > body_h * 1.25 and not last_line[2].rstrip().endswith(TERMINAL):
            problems.append((i + 1, f"heading at the bottom: {last_line[2][:50]!r}"))
        continued = not last_line[2].rstrip().endswith(TERMINAL)
        first_text = first_next[0][2].lstrip()
        if continued and first_text[:1].islower() or (continued and first_text[:1] == "(" ):
            # the paragraph really continues on the next page
            frag = lambda t: t.rstrip().endswith(TERMINAL) or len(t.split()) >= 4     # a one-word list item is not a paragraph fragment
            if len(first_next) <= 2 and all(frag(l[2]) for l in first_next[:1]):
                problems.append((i + 2, f"{len(first_next)}-line stub at the top: {first_text[:50]!r}"))
            if len(last_block) <= 2 and frag(last_block[-1][2]):
                problems.append((i + 1, f"{len(last_block)}-line stub at the bottom: {last_block[0][2][:50]!r}"))
    return problems


def check(pdf: Path, min_fill: float = MIN_FILL) -> tuple:
    """(ok, report dict)."""
    fl = fills(pdf)
    low = [(i + 1, round(f, 2)) for i, f in enumerate(fl[:-1]) if i > 0 and f < min_fill]
    st = stubs(pdf)
    body = fl[1:-1] or fl
    rep = dict(pages=len(fl), min_fill=round(min(body), 2) if body else None, avg_fill=round(sum(body) / len(body), 2) if body else None,
               low_pages=low, stubs=st, last=round(fl[-1], 2))
    return (not low and not st), rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-fill", type=float, default=MIN_FILL)
    ap.add_argument("pdfs", nargs="+")
    a = ap.parse_args()
    bad = 0
    for p in a.pdfs:
        ok, rep = check(Path(p), a.min_fill)
        print(("PASS " if ok else "FAIL ") + Path(p).name, rep)
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
