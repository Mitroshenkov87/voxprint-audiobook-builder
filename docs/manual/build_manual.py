#!/usr/bin/env python3
"""Build the Voxprint user manual PDFs (en, ru, de) from docs/manual/src/manual_<lang>.md.

Toolchain: Markdown -> HTML (python-markdown) -> PDF (WeasyPrint).  Needs: pip install markdown weasyprint pillow
  * ``{{key}}``  in the sources is replaced by the UI string ``key`` of locales/<lang>.json (exactly as in the app);
  * ``![caption](@name)`` is a screenshot: img/<lang>/name.png, falling back to img/en/name.png (noted in the caption);
    ``@narrate_top`` / ``@narrate_bottom`` are the two halves of the tall Narrate screenshot.
Two themes per language (same content): ``Voxprint-Manual-<lang>.pdf`` = DARK (default, full-bleed near-black pages) and
``Voxprint-Manual-<lang>-print.pdf`` = LIGHT for printing.  The logo (mark + wordmark) is generated here as SVG - no external assets.
Contrast of every text/background pair of both palettes is checked (WCAG, see :func:`check_contrast`) before building.
Usage: python docs/manual/build_manual.py [--out DIR] [--theme dark|print|both] [lang ...]
"""
from __future__ import annotations
import argparse, json, re, sys
from datetime import date
from pathlib import Path
import markdown
from PIL import Image
from weasyprint import HTML

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LANGS = ("en", "ru", "de")
META = {
    "en": dict(title="Voxprint AI Audiobook Builder", sub="User manual", contents="Contents", page="Page",
               ver="Version 0.1.0 (beta) · manual of 3 October 2026", fallback=" (screenshot of the English interface)",
               fig="Figure", html="en"),
    "ru": dict(title="Voxprint AI Audiobook Builder", sub="Руководство пользователя", contents="Содержание", page="Стр.",
               ver="Версия 0.1.0 (бета) · руководство от 3 октября 2026 г.", fallback=" (снимок английского интерфейса)",
               fig="Рис.", html="ru"),
    "de": dict(title="Voxprint AI Audiobook Builder", sub="Benutzerhandbuch", contents="Inhalt", page="Seite",
               ver="Version 0.1.0 (Beta) · Handbuch vom 3. Oktober 2026", fallback=" (Screenshot der englischen Oberfläche)",
               fig="Abb.", html="de"),
}

# ----------------------------------------------------------------------------------------------- themes
THEMES = {
    "dark": dict(
        page_bg="#0d0f14", cover_bg="#0d0f14", text="#e6e8ee", muted="#a9afbd", accent="#7db4ff", head="#8fc0ff",
        h3="#f0f2f6", num="#8e9bb5", panel="#161a22", panel2="#1c212b", border="#3a4152", code_text="#e9edf5",
        ui_bg="#1d2a47", ui_text="#d6e5ff", th_bg="#25324f", th_text="#ffffff", warn_bg="#2b2412", warn_line="#e0a800",
        tip_bg="#12281b", tip_line="#3fb56b", img_border="#5a647a", cover_title="#ffffff", logo_bg="#7db4ff",
        logo_fg="#0d0f14", logo_alt="#cfe3ff"),
    "print": dict(
        page_bg="#ffffff", cover_bg="#ffffff", text="#1d1f24", muted="#555a66", accent="#1a54b5", head="#123a7a",
        h3="#222222", num="#5f6f8f", panel="#f1f2f5", panel2="#f6f7fa", border="#b4bac8", code_text="#1d1f24",
        ui_bg="#e8eefc", ui_text="#0b2a66", th_bg="#123a7a", th_text="#ffffff", warn_bg="#fff7e0", warn_line="#b07f00",
        tip_bg="#e8f5ec", tip_line="#2e9d57", img_border="#8b93a5", cover_title="#0b1f4a", logo_bg="#123a7a",
        logo_fg="#ffffff", logo_alt="#123a7a"),
}
SUFFIX = {"dark": "", "print": "-print"}
#: (foreground, background, minimum WCAG contrast ratio) pairs that are checked for every theme
CONTRAST_PAIRS = [("text", "page_bg", 7.0), ("muted", "page_bg", 4.5), ("accent", "page_bg", 4.5), ("head", "page_bg", 4.5),
                  ("h3", "page_bg", 4.5), ("num", "page_bg", 3.0), ("code_text", "panel", 7.0), ("code_text", "panel2", 7.0),
                  ("text", "panel", 7.0), ("text", "panel2", 7.0), ("ui_text", "ui_bg", 7.0), ("th_text", "th_bg", 7.0),
                  ("text", "warn_bg", 7.0), ("text", "tip_bg", 7.0), ("cover_title", "cover_bg", 7.0),
                  ("accent", "cover_bg", 4.5), ("muted", "cover_bg", 4.5), ("warn_line", "page_bg", 3.0),
                  ("border", "page_bg", 1.8), ("img_border", "page_bg", 3.0), ("logo_fg", "logo_bg", 4.5)]


def _lum(hex_color: str) -> float:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio of two ``#rrggbb`` colours."""
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def check_contrast(verbose: bool = False) -> list:
    """Return the list of failing (theme, fg, bg, ratio, minimum); empty = every pair is readable."""
    bad = []
    for name, t in THEMES.items():
        for fg, bg, need in CONTRAST_PAIRS:
            r = contrast(t[fg], t[bg])
            if verbose:
                print(f"{name:5} {fg:12} on {bg:10} {r:5.1f}:1 (min {need})")
            if r < need:
                bad.append((name, fg, bg, round(r, 2), need))
    return bad


def logo_svg(theme: str, size_mm: float = 0) -> str:
    """The Voxprint mark: rounded square with a voice waveform (bars) that ends in a cursor-like dot.  Pure vector."""
    t = THEMES[theme]
    bars = [(14, 22), (24, 12), (34, 4), (44, 16), (54, 24)]       # x, half-height around y=32 (a voice-like waveform)
    rects = "".join(f'<rect x="{x}" y="{32 - h}" width="6" height="{2 * h}" rx="3" fill="{t["logo_fg"]}"/>' for x, h in bars)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 76 64" width="76" height="64">'
            f'<rect x="0" y="0" width="76" height="64" rx="14" fill="{t["logo_bg"]}"/>{rects}'
            f'<circle cx="68" cy="32" r="3" fill="{t["logo_fg"]}"/></svg>')


def logo_mark_uri(theme: str) -> str:
    """Small header logo as a data URI (used in the page header, where only an image can be placed)."""
    import base64
    return "data:image/svg+xml;base64," + base64.b64encode(logo_svg(theme).encode()).decode()


def page_css(theme: str) -> str:
    """Theme variables + the @page rules: full-bleed background on every page (margins included), header with the
    logo and the product name, footer with the page number."""
    t = THEMES[theme]
    root = ":root{" + ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in t.items()) + "}"
    font = '"Noto Sans", sans-serif'
    return (root +
            f'@page {{ size: A4; margin: 22mm 18mm 22mm 18mm; background: {t["page_bg"]};'
            f' @bottom-center {{ content: counter(page); font: 9pt {font}; color: {t["muted"]}; }}'
            f' @top-left {{ content: url("{logo_mark_uri(theme)}") "  Voxprint"; font: bold 8.5pt {font}; color: {t["accent"]};'
            f' vertical-align: middle; }}'
            f' @top-right {{ content: "AI Audiobook Builder"; font: 8pt {font}; color: {t["muted"]}; }} }}'
            f'@page :first {{ margin: 0; background: {t["cover_bg"]}; @bottom-center {{ content: none; }}'
            f' @top-left {{ content: none; }} @top-right {{ content: none; }} }}')

NARRATE_SPLIT = {"en": 940, "ru": 983, "de": 946}


def make_crops(lang: str = "") -> None:
    for l in LANGS:
        src = HERE / "img" / l / "narrate.png"
        if not src.exists():
            continue
        im = Image.open(src)
        y = NARRATE_SPLIT[l]
        im.crop((0, 0, im.width, y)).save(HERE / "build" / f"narrate_top_{l}.png")
        im.crop((0, y, im.width, im.height)).save(HERE / "build" / f"narrate_bottom_{l}.png")


def shot_path(name: str, lang: str):
    """Return (relative path, is_fallback)."""
    if name.startswith("narrate_top") or name.startswith("narrate_bottom"):
        for l in (lang, "en"):
            p = HERE / "build" / f"{name}_{l}.png"
            if p.exists():
                return p, l != lang
    for l in (lang, "en"):
        p = HERE / "img" / l / f"{name}.png"
        if p.exists():
            return p, l != lang
    raise FileNotFoundError(name)


def render(lang: str, out_dir: Path, theme: str = "dark") -> Path:
    m = META[lang]
    ui = json.loads((ROOT / "locales" / f"{lang}.json").read_text(encoding="utf-8"))
    src = (HERE / "src" / f"manual_{lang}.md").read_text(encoding="utf-8")

    def ui_sub(mo):
        key = mo.group(1)
        plain = key.endswith("_plain")
        if plain:
            key = key[:-6]
        if key not in ui:
            raise KeyError(f"unknown UI key {key}")
        v = ui[key].replace("\n", " ")
        if plain:      # strings with {placeholders}: show an ellipsis instead of the value
            v = re.sub(r"\{[a-z_]+\}", "…", v)
        v = v.replace("&", "&amp;").replace("<", "&lt;")
        return f'<span class="ui">{v}</span>'
    src = re.sub(r"\{\{([a-z_0-9.]+)\}\}", ui_sub, src)

    def img_sub(mo):
        cap, name = mo.group(1), mo.group(2)
        p, fb = shot_path(name, lang)
        if fb:
            cap += m["fallback"]
        return f"![{cap}]({p.as_uri()})"
    src = re.sub(r"!\[([^\]]*)\]\(@([a-z_0-9]+)\)", img_sub, src)

    md = markdown.Markdown(extensions=["tables", "attr_list", "toc", "sane_lists", "md_in_html"],
                           extension_configs={"toc": {"permalink": False}})
    body = md.convert(src)
    # numbering of h1/h2 + collect TOC
    toc, n1, n2 = [], 0, 0
    def head(mo):
        nonlocal n1, n2
        lvl, attrs, text = int(mo.group(1)), mo.group(2), mo.group(3)
        idm = re.search(r'id="([^"]+)"', attrs)
        hid = idm.group(1) if idm else f"h{len(toc)}"
        if "nonum" in attrs:
            return mo.group(0)
        if lvl == 1:
            n1 += 1; n2 = 0; num = f"{n1}"
        elif lvl == 2:
            n2 += 1; num = f"{n1}.{n2}"
        else:
            return mo.group(0)
        toc.append((lvl, num, re.sub(r"<[^>]+>", "", text), hid))
        return f'<h{lvl}{attrs}><span class="num">{num}</span> {text}</h{lvl}>'
    body = re.sub(r"<h([12])([^>]*)>(.*?)</h\1>", head, body, flags=re.S)
    # figures: a paragraph that holds only an image becomes <figure> with a caption
    body = re.sub(r'<p>\s*<img alt="([^"]*)" src="([^"]+)"[^>]*?/?>\s*</p>',
                  lambda mo: f'<figure><img src="{mo.group(2)}" alt=""/><figcaption>{mo.group(1)}</figcaption></figure>', body)
    toc_html = ['<nav class="toc"><h1 class="nonum toc-title">%s</h1><ul>' % m["contents"]]
    for lvl, num, text, hid in toc:
        toc_html.append(f'<li class="l{lvl}"><a href="#{hid}"><span class="n">{num}</span> {text}</a></li>')
    toc_html.append("</ul></nav>")
    cover = (f'<section class="cover"><div class="logo">{logo_svg(theme)}<div class="brand">Vox<b>print</b></div></div><h1 class="nonum cover-title">{m["title"]}</h1>'
             f'<div class="cover-sub">{m["sub"]}</div><div class="cover-rule"></div><div class="cover-ver">{m["ver"]}</div>'
             f'<div class="cover-lic">Apache-2.0 · © 2026 Aleksandr Mitroshenkov</div></section>')
    css = page_css(theme) + (HERE / "manual.css").read_text(encoding="utf-8")
    html = (f'<!doctype html><html lang="{m["html"]}"><head><meta charset="utf-8"><title>{m["title"]} — {m["sub"]}</title>'
            f'<meta name="author" content="Aleksandr Mitroshenkov"><style>{css}</style></head><body>'
            f'{cover}{"".join(toc_html)}<main>{body}</main></body></html>')
    (HERE / "build" / f"manual_{lang}{SUFFIX[theme]}.html").write_text(html, encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"Voxprint-Manual-{lang}{SUFFIX[theme]}.pdf"
    HTML(string=html, base_url=str(HERE)).write_pdf(str(pdf))
    return pdf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--theme", choices=("dark", "print", "both"), default="both")
    ap.add_argument("langs", nargs="*", default=list(LANGS))
    a = ap.parse_args()
    (HERE / "build").mkdir(exist_ok=True)
    bad = check_contrast()
    if bad:
        print("contrast check FAILED:", bad)
        return 1
    make_crops("en")
    for l in a.langs:
        for th in (("dark", "print") if a.theme == "both" else (a.theme,)):
            print(render(l, Path(a.out), th))
    return 0


if __name__ == "__main__":
    sys.exit(main())
