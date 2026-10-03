#!/usr/bin/env python3
"""Build the Voxprint user manual PDFs (en, ru, de) from docs/manual/src/manual_<lang>.md.

Toolchain: Markdown -> HTML (python-markdown) -> PDF (WeasyPrint).  Needs: pip install markdown weasyprint pillow
  * ``{{key}}``  in the sources is replaced by the UI string ``key`` of locales/<lang>.json (exactly as in the app);
  * ``![caption](@name)`` is a screenshot: img/<lang>/name.png, falling back to img/en/name.png (noted in the caption);
    ``@narrate_top`` / ``@narrate_bottom`` are the two halves of the tall Narrate screenshot.
Usage: python docs/manual/build_manual.py [--out DIR] [lang ...]
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


def render(lang: str, out_dir: Path) -> Path:
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
    cover = (f'<section class="cover"><div class="brand">Voxprint</div><h1 class="nonum cover-title">{m["title"]}</h1>'
             f'<div class="cover-sub">{m["sub"]}</div><div class="cover-ver">{m["ver"]}</div>'
             f'<div class="cover-lic">Apache-2.0 · © 2026 Aleksandr Mitroshenkov</div></section>')
    css = (HERE / "manual.css").read_text(encoding="utf-8")
    html = (f'<!doctype html><html lang="{m["html"]}"><head><meta charset="utf-8"><title>{m["title"]} — {m["sub"]}</title>'
            f'<meta name="author" content="Aleksandr Mitroshenkov"><style>{css}</style></head><body>'
            f'{cover}{"".join(toc_html)}<main>{body}</main></body></html>')
    (HERE / "build" / f"manual_{lang}.html").write_text(html, encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"Voxprint-Manual-{lang}.pdf"
    HTML(string=html, base_url=str(HERE)).write_pdf(str(pdf))
    return pdf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("langs", nargs="*", default=list(LANGS))
    a = ap.parse_args()
    (HERE / "build").mkdir(exist_ok=True)
    make_crops("en")
    for l in a.langs:
        print(render(l, Path(a.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
