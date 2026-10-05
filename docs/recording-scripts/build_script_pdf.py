#!/usr/bin/env python3
"""Recording-script PDFs (dark + print) from Voxprint-RecordingScript-v6-<lang>.txt, styled like the Voxprint manuals.

Uses the theme colours of ../manual/build_manual.py, but puts the real app icon (assets/voxprint.png) on the cover
and keeps page headers/footers free of logo marks. Layout is tuned until ../manual/check_pagination.py passes
(no page below 70 % filled, no stub lines).  Needs: pip install markdown weasyprint pillow (+ poppler-utils for the check).
Usage: python build_script_pdf.py [--out DIR] [--no-fit]
"""
import argparse, html, importlib.util, re, sys
from pathlib import Path
from weasyprint import HTML

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Real window / installer fingerprint icon (not the decorative waveform mark from build_manual).
# Prefer the 256px installer asset so PDFs stay lean; fall back to assets/voxprint.png.
_APP_LOGO_CANDIDATES = (
    ROOT / "installer" / "linux" / "voxprint-256.png",
    ROOT / "assets" / "voxprint.png",
)
APP_LOGO = next(c for c in _APP_LOGO_CANDIDATES if c.is_file())
spec = importlib.util.spec_from_file_location("bm", HERE.parent / "manual" / "build_manual.py")
bm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)

WPM = 130          # reading speed, words per minute (unhurried)
PAUSE = 1.0        # seconds of silence after every line
VERSION = 6

L = {
 "ru": dict(title="Текст для записи голоса", sub="Версия 6 · русский", opt="необязательно", guide="Памятка по чтению",
            rules="Правила", tempo_h="Темп и паузы", emo_h="Эмоции", brk_h="Перерывы", dur_h="Сколько времени займёт запись",
            blocks_h="Блоки и примерное время", col=("№", "Блок", "≈ мин"), brk="Здесь можно сделать перерыв на 2–3 минуты: глоток воды, пара шагов по комнате.",
            header="Запись голоса · версия 6", foot="Личная памятка для записи · не для публикации", mins="мин",
            tempo=["Читайте неспешно, примерно 130 слов в минуту, чуть медленнее обычного разговора, как будто рассказываете что-то знакомому.",
                   "После каждой строки — вдох и около секунды тишины. После запятой — короткая остановка, после точки — полная.",
                   "Перед тихим и перед громким блоком и после них оставляйте 3 секунды тишины.",
                   "Если запнулись, помолчите секунду и прочитайте предложение заново с начала; останавливать запись не нужно."],
            emo=[("Спокойно", "чуть тише и мягче, медленнее"), ("Радость", "теплее, с улыбкой в голосе"), ("Грусть", "тише и медленнее, без нажима"),
                 ("Раздражение", "твёрже и резче, но без крика"), ("Удивление", "чуть выше и живее"), ("Шёпот", "ближе к микрофону, не дуть в него"),
                 ("Громко", "рука вытянута от микрофона, не кричать изо всех сил")],
            emo_note="Играть не нужно: достаточно слегка изменить голос. Не получается — читайте обычным голосом.",
            brk_txt="Перерыв нужен примерно каждые 4–5 минут чтения: он отмечен в тексте значком ☕. Пейте тёплую воду, не холодную; держите микрофон на одном расстоянии.",
            dur="Основной текст (блоки 1–14): около {main} мин. Необязательные блоки 15–16: ещё около {opt} мин. Фраза согласия в конце: около 0,5 мин. С паузами, перерывами и повторами планируйте {plan} мин."),
 "en": dict(title="Voice Recording Script", sub="Version 6 · English", opt="optional", guide="Reading guide",
            rules="Rules", tempo_h="Tempo and pauses", emo_h="Emotions", brk_h="Breaks", dur_h="How long the recording takes",
            blocks_h="Blocks and approximate time", col=("No.", "Block", "≈ min"), brk="A good moment for a 2–3 minute break: a sip of water, a few steps around the room.",
            header="Voice recording · version 6", foot="Personal recording aid · not for publication", mins="min",
            tempo=["Read unhurriedly, at roughly 130 words per minute, a little slower than everyday conversation, as if telling something to a friend.",
                   "After every line take a breath and leave about a second of silence. A comma is a short stop, a full stop is a full pause.",
                   "Leave 3 seconds of silence before and after the whisper and loud passages.",
                   "If you stumble, stay silent for a second and read the sentence again from its beginning; there is no need to stop the recording."],
            emo=[("Calm", "a little softer and slower"), ("Joy", "warmer, a smile in the voice"), ("Sadness", "softer and slower, no pressure"),
                 ("Irritation", "firmer and sharper, but no shouting"), ("Surprise", "a little higher and livelier"), ("Whisper", "closer to the microphone, do not blow into it"),
                 ("Loud", "an arm's length from the microphone, do not shout with all your strength")],
            emo_note="You do not have to act: a slight change of voice is enough. If it does not come, use your normal voice.",
            brk_txt="Take a break roughly every 4–5 minutes of reading; the suggested places are marked ☕ in the text. Drink warm water, not cold, and keep the microphone at the same distance.",
            dur="Main text (blocks 1–14): about {main} min. Optional blocks 15–16: about {opt} more min. The consent sentence at the end: about 0.5 min. With pauses, breaks and retakes plan for {plan} min."),
 "de": dict(title="Aufnahmetext für die Stimme", sub="Version 6 · Deutsch", opt="optional", guide="Lesehinweise",
            rules="Regeln", tempo_h="Tempo und Pausen", emo_h="Gefühle", brk_h="Pausen zum Ausruhen", dur_h="Wie lange die Aufnahme dauert",
            blocks_h="Blöcke und ungefähre Zeit", col=("Nr.", "Block", "≈ Min."), brk="Hier passt eine Pause von 2–3 Minuten: ein Schluck Wasser, ein paar Schritte durch das Zimmer.",
            header="Stimmaufnahme · Version 6", foot="Persönliche Aufnahmehilfe · nicht zur Veröffentlichung", mins="Min.",
            tempo=["Lesen Sie gemächlich, etwa 130 Wörter pro Minute, etwas langsamer als im Alltagsgespräch, so als würden Sie einem Bekannten etwas erzählen.",
                   "Nach jeder Zeile atmen Sie durch und lassen etwa eine Sekunde Stille. Ein Komma ist ein kurzes Innehalten, ein Punkt eine volle Pause.",
                   "Vor und nach den Flüster- und Lautstellen lassen Sie 3 Sekunden Stille.",
                   "Haben Sie sich verhaspelt, schweigen Sie eine Sekunde und lesen den Satz noch einmal von vorn; die Aufnahme muss nicht gestoppt werden."],
            emo=[("Ruhig", "etwas leiser, weicher und langsamer"), ("Freude", "wärmer, mit einem Lächeln in der Stimme"), ("Traurigkeit", "leiser und langsamer, ohne Druck"),
                 ("Ärger", "fester und schärfer, aber nicht schreien"), ("Überraschung", "etwas höher und lebhafter"), ("Flüstern", "näher am Mikrofon, nicht hineinpusten"),
                 ("Laut", "eine Armlänge vom Mikrofon entfernt, nicht aus voller Kraft schreien")],
            emo_note="Spielen muss man nicht: Es genügt, die Stimme leicht zu verändern. Klappt es nicht, lesen Sie mit normaler Stimme.",
            brk_txt="Machen Sie etwa alle 4–5 Minuten Lesezeit eine Pause; passende Stellen sind im Text mit ☕ markiert. Trinken Sie warmes Wasser und halten Sie den Abstand zum Mikrofon gleich.",
            dur="Haupttext (Blöcke 1–14): etwa {main} Min. Optionale Blöcke 15–16: weitere {opt} Min. Der Einwilligungssatz am Ende: etwa 0,5 Min. Mit Pausen, Erholung und Wiederholungen planen Sie {plan} Min. ein."),
}
EXTRA_CSS = """
html { font-size: 13.5pt; }
.cover-title { font-size: 34pt; }
.cover .logo img.app-logo { width: 24mm; height: 24mm; display: block; border-radius: 5.2mm; }
.sec { margin: 0 0 16pt 0; }
.sec h2 { font-size: 20pt; margin: 24pt 0 8pt 0; padding-bottom: 3pt; border-bottom: 1.2pt solid var(--border); page-break-after: avoid; }
.sec h2 .num { font-size: 20pt; color: var(--accent); margin-right: 6pt; }
.badge { font-size: 9pt; font-weight: bold; letter-spacing: .5pt; text-transform: uppercase; color: var(--ui-text); background: var(--ui-bg);
         border: 0.6pt solid var(--border); border-radius: 3pt; padding: 1pt 5pt; margin-left: 8pt; vertical-align: middle; }
p.hint { font-size: 11.5pt; color: var(--muted); border-left: 3pt solid var(--accent); background: var(--panel); padding: 6pt 10pt; margin: 4pt 0 12pt 0; font-style: italic; page-break-inside: avoid; }
p.tag { font-size: 11.5pt; color: var(--accent); font-weight: bold; margin: 14pt 0 0 0; page-break-after: avoid; }
p.line { font-size: 16.5pt; line-height: 1.68; margin: 0 0 11pt 0; color: var(--text); page-break-inside: avoid; orphans: 2; widows: 2; }
p.term { font-size: 16.5pt; line-height: 1.45; margin: 0 0 5pt 0; }
span.br { font-size: 11.5pt; color: var(--muted); font-style: italic; }
.consent h3 { font-size: 13pt; color: var(--head); margin: 14pt 0 4pt 0; page-break-after: avoid; }
.breakmark { text-align: center; color: var(--muted); font-size: 11.5pt; border-top: 0.8pt dashed var(--border); border-bottom: 0.8pt dashed var(--border); padding: 5pt 0; margin: 14pt 0 5pt 0; page-break-inside: avoid; }
.guide ul { margin: 3pt 0 8pt 0; padding-left: 16pt; } .guide li { margin: 4pt 0; font-size: 12pt; }
.guide h2 { font-size: 16pt; }
.guide table td, .guide table th { font-size: 11.5pt; }
td.r, th.r { text-align: right; white-space: nowrap; }
"""


def page_css(theme: str, header: str) -> str:
    """Theme variables + @page rules without logo marks in the header/footer (page number only in the footer)."""
    t = bm.THEMES[theme]
    root = ":root{" + ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in t.items()) + "}"
    font = '"Noto Sans", sans-serif'
    hdr = html.escape(header)
    return (root +
            f'@page {{ size: A4; margin: 20mm 17mm 20mm 17mm; background: {t["page_bg"]};'
            f' @bottom-center {{ content: counter(page); font: 9.5pt {font}; color: {t["muted"]}; }}'
            f' @top-left {{ content: "{hdr}"; font: 9pt {font}; color: {t["muted"]}; }}'
            f' @top-right {{ content: none; }} }}'
            f'@page :first {{ margin: 0; background: {t["cover_bg"]}; @bottom-center {{ content: none; }}'
            f' @top-left {{ content: none; }} @top-right {{ content: none; }} }}')


def parse(path):
    lines = path.read_text(encoding="utf-8").split("\n")
    doc, rules, blocks, cur = None, [], [], None
    for raw in lines:
        s = raw.strip()
        if s.startswith("==="):
            t = s.strip("= ").strip()
            if doc is None:
                doc = t; continue
            m = re.match(r"^\S+\s+(\d+)\.\s*(.*)$", t)
            cur = dict(num=int(m.group(1)), title=m.group(2), items=[], words=0, lines=0)
            blocks.append(cur); continue
        if not s:
            if cur is not None: cur["items"].append(("gap", ""))
            continue
        if cur is None:
            if s.startswith("[") and s.endswith("]"): rules.append(s[1:-1])
            continue
        bracket_only = s.startswith("[") and s.endswith("]") and "]" not in s[1:-1]
        if bracket_only:
            kind = "hint" if not any(k in ("line", "tag", "sub") for k, _ in cur["items"]) and not any(k == "hint" for k, _ in cur["items"]) else "tag"
            cur["items"].append((kind, s[1:-1])); continue
        if re.match(r"^[А-ЯA-ZÄÖÜ]\.\s", s) and s.endswith(":"):
            cur["items"].append(("sub", s)); continue
        cur["items"].append(("line", s))
        plain = re.sub(r"\[[^\]]*\]", " ", s)
        cur["words"] += len(plain.split()); cur["lines"] += 1
    return doc, rules, blocks

def minutes(b):
    return b["words"] / WPM + b["lines"] * PAUSE / 60

DE_TITLES = {1: "Begrüßung", 2: "Die Geschichte der Idee", 3: "Zahlen und Daten", 4: "Fließende Sätze für verschiedene Laute", 5: "Neutral",
             6: "Ruhig", 7: "Freude", 8: "Traurigkeit", 9: "Ärger", 10: "Überraschung", 11: "Flüstern und laut", 12: "Geschichte eins: Der Start",
             13: "Geschichte zwei: Ein Gespräch mit dem Programm", 14: "Geschichte drei: Das Training", 15: "Geschichte vier, ein missratener Tag",
             16: "Fachwörter", 17: "Einwilligung des Stimminhabers — ganz am Ende der Aufnahme"}


def clean_title(t, opt_word):
    m = re.match(r"^(?:НЕОБЯЗАТЕЛЬНО|OPTIONAL|OPTIONAL)\s*:\s*(.*)$", t, re.I)
    return (m.group(1), True) if m else (t, False)

def esc(s):
    s = html.escape(s)
    return re.sub(r"\[([^\]]*)\]", r'<span class="br">[\1]</span>', s)

def build(lang, theme, outdir, gap=1.0):
    t = L[lang]
    assert APP_LOGO.is_file(), f"missing app logo: {APP_LOGO}"
    doc, rules, blocks = parse(HERE / f"Voxprint-RecordingScript-v{VERSION}-{lang}.txt")
    consent = blocks[-1]; body_blocks = blocks[:-1]
    main = sum(minutes(b) for b in body_blocks if not b["title"].upper().startswith(("НЕОБЯЗАТЕЛЬНО", "OPTIONAL")))
    opt = sum(minutes(b) for b in body_blocks if b["title"].upper().startswith(("НЕОБЯЗАТЕЛЬНО", "OPTIONAL")))
    rnd = lambda x: f"{x:.0f}"
    # break markers: after the block where the cumulative reading time crosses 4.5 / 9 min (main blocks only)
    marks, cum, nxt = set(), 0.0, 4.5
    for b in body_blocks:
        if b["title"].upper().startswith(("НЕОБЯЗАТЕЛЬНО", "OPTIONAL")): continue
        cum += minutes(b)
        if cum >= nxt and b is not body_blocks[-1]:
            marks.add(b["num"]); nxt += 4.5
    plan = f"{rnd((main + opt) * 1.6)}–{rnd((main + opt) * 2.1)}"
    parts = []
    logo_uri = APP_LOGO.as_uri()
    parts.append(f'<section class="cover"><div class="logo"><img class="app-logo" src="{logo_uri}" alt="Voxprint"/>'
                 f'<div class="brand">Vox<b>print</b></div></div>'
                 f'<h1 class="nonum cover-title">{html.escape(t["title"])}</h1><div class="cover-sub">{html.escape(t["sub"])}</div>'
                 f'<div class="cover-rule"></div><div class="cover-ver">{html.escape(t["foot"])}</div></section>')
    g = [f'<section class="guide"><h1 class="nonum" style="page-break-before:auto">{html.escape(t["guide"])}</h1>']
    g.append(f'<h2>{t["rules"]}</h2><ul>' + "".join(f"<li>{esc(r)}</li>" for r in rules) + "</ul>")
    g.append(f'<h2>{t["tempo_h"]}</h2><ul>' + "".join(f"<li>{html.escape(x)}</li>" for x in t["tempo"]) + "</ul>")
    g.append(f'<h2>{t["emo_h"]}</h2><table><thead><tr><th></th><th></th></tr></thead><tbody>' +
             "".join(f"<tr><td><b>{html.escape(a)}</b></td><td>{html.escape(b)}</td></tr>" for a, b in t["emo"]) + f'</tbody></table><p>{html.escape(t["emo_note"])}</p>')
    g.append(f'<h2>{t["brk_h"]}</h2><p>{html.escape(t["brk_txt"])}</p>')
    g.append(f'<h2>{t["dur_h"]}</h2><p>{html.escape(t["dur"].format(main=rnd(main), opt=rnd(opt), plan=plan))}</p>')
    g.append(f'<h2>{t["blocks_h"]}</h2><table><thead><tr><th>{t["col"][0]}</th><th>{t["col"][1]}</th><th class="r">{t["col"][2]}</th></tr></thead><tbody>')
    for b in blocks:
        ti, isopt = clean_title(b["title"], t["opt"])
        if lang == "de": ti = DE_TITLES[b["num"]]
        g.append(f'<tr><td>{b["num"]}</td><td>{html.escape(ti[:1] + ti[1:].lower() if ti.isupper() else ti)}{" · " + t["opt"] if isopt else ""}</td><td class="r">{max(0.5, minutes(b)):.1f}</td></tr>')
    g.append("</tbody></table></section>")
    parts.append("".join(g))
    parts.append('<main>')
    for b in blocks:
        ti, isopt = clean_title(b["title"], t["opt"])
        ti = DE_TITLES[b["num"]] if lang == "de" else (ti[:1] + ti[1:].lower() if ti.isupper() else ti)
        cls = "sec consent" if b is consent else "sec"
        h = [f'<section class="{cls}"><h2><span class="num">{b["num"]}</span>{html.escape(ti)}' + (f'<span class="badge">{t["opt"]}</span>' if isopt else "") + "</h2>"]
        is_terms = b["lines"] > 20 and all(len(x[1].split()) <= 5 for x in b["items"] if x[0] == "line")
        for kind, text in b["items"]:
            if kind == "gap": continue
            if kind == "hint": h.append(f'<p class="hint">{esc(text)}</p>')
            elif kind == "tag": h.append(f'<p class="tag">{esc(text)}</p>')
            elif kind == "sub": h.append(f"<h3>{html.escape(text)}</h3>")
            else: h.append(f'<p class="{"term" if is_terms else "line"}">{esc(text)}</p>')
        h.append("</section>")
        if b["num"] in marks:
            h.append(f'<div class="breakmark">☕ {html.escape(t["brk"])}</div>')
        parts.append("".join(h))
    parts.append("</main>")
    css = page_css(theme, t["header"]) + (bm.HERE / "manual.css").read_text(encoding="utf-8") + EXTRA_CSS + (
        f"p.line{{margin-bottom:{11 * gap:.1f}pt}} p.term{{margin-bottom:{5 * gap:.1f}pt}} .sec h2{{margin-top:{24 * gap:.1f}pt}} "
        f"p.tag{{margin-top:{14 * gap:.1f}pt}} p.hint{{margin-bottom:{12 * gap:.1f}pt}}")
    htm = f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><title>{html.escape(t["title"])}</title><style>{css}</style></head><body>{"".join(parts)}</body></html>'
    out = outdir / f"Voxprint-RecordingScript-v{VERSION}-{lang}{bm.SUFFIX[theme]}.pdf"
    HTML(string=htm, base_url=str(HERE)).write_pdf(str(out))
    return out, main, opt

GAPS = (1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2, 0.75, 1.25)


def fit(lang, out):
    """First spacing factor for which both themes pass the pagination check (else the best one)."""
    import shutil, tempfile
    sys.path.insert(0, str(HERE.parent / "manual"))
    import check_pagination as cp
    best = None
    for gap in GAPS:
        with tempfile.TemporaryDirectory() as d:
            res = [build(lang, th, Path(d), gap) for th in ("dark", "print")]
            checks = [cp.check(r[0]) for r in res]
            score = (all(c[0] for c in checks), min(c[1]["min_fill"] or 0 for c in checks))
            if best is None or score > best[0]:
                best = (score, gap)
            if score[0]:
                for r in res:
                    shutil.copy(r[0], out / r[0].name)
                print(f"  {lang}: spacing x{gap}: pagination OK (min fill {score[1]})")
                return res
    print(f"  {lang}: no spacing passed, using x{best[1]} (min fill {best[0][1]})")
    return [build(lang, th, out, best[1]) for th in ("dark", "print")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(HERE)); ap.add_argument("--no-fit", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert not bm.check_contrast(), bm.check_contrast()
    assert APP_LOGO.is_file(), APP_LOGO
    for lang in L:
        res = [build(lang, th, out) for th in ("dark", "print")] if a.no_fit else fit(lang, out)
        for p, m, o in res:
            print(p, f"main={m:.1f} opt={o:.1f}")

if __name__ == "__main__":
    sys.exit(main())
