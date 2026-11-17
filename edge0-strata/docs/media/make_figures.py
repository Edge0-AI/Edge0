"""docs/media/make_figures.py - the README's "How does it work?" illustrations, in the Strata app's style: its colors
(serve/web/tokens.css), its icons (serve/web/sprite.svg) and its font (Outfit, embedded, SIL OFL 1.1).  Light and dark
follow the reader's system setting.

    python docs/media/make_figures.py        # writes docs/media/how-it-works.svg and docs/media/guess-and-check.svg
"""
import base64
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "media"
SPRITE = (ROOT / "serve" / "web" / "sprite.svg").read_text(encoding="utf-8")
FONT = base64.b64encode((ROOT / "serve" / "web" / "fonts" / "outfit-latin-wght.woff2").read_bytes()).decode()

# the app's tokens (light, then dark)
CSS = """
@font-face{font-family:"Outfit";src:url(data:font/woff2;base64,%s) format("woff2");font-weight:100 900}
text{font-family:"Outfit",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;fill:#221e1f}
.bg{fill:#f9fafb}.card{fill:#ffffff;stroke:#dee5ee;stroke-width:1}.soft{fill:#f7f7f7}
.muted{fill:#6f6f6f}.ink{fill:#221e1f}.icon{color:#2F384C}
.acc{fill:#10b981}.acc-t{fill:#0c8a60}.acc-tint{fill:#ecfdf5}.acc-line{stroke:#10b981}
.info{fill:#0f97ff}.info-t{fill:#0668c2}.info-tint{fill:#e8f4ff}
.warn{fill:#e39b0b}.warn-t{fill:#8a5a00}.warn-tint{fill:#fff6e0}
.dan{fill:#e0283f}.dan-tint{fill:#fdecee}.dan-t{fill:#b01a2e}
.dot{fill:#dee5ee}.line{stroke:#dee5ee}.arrow{stroke:#6f6f6f;fill:none}.arrowhead{fill:#6f6f6f}
.dash{stroke:#6f6f6f;fill:none;stroke-dasharray:4 4}
.on-acc{fill:#ffffff}.icon-acc{color:#0c8a60}.icon-dan{color:#b01a2e}
@media (prefers-color-scheme:dark){
text,.ink{fill:#eef1f3}.bg{fill:#0e1113}.card{fill:#161a1d;stroke:#2a3036}.soft{fill:#1d2226}
.muted{fill:#8e979f}.icon{color:#c7cdd3}
.acc{fill:#34d399}.acc-t{fill:#4ade80}.acc-tint{fill:rgba(52,211,153,.12)}.acc-line{stroke:#34d399}
.info{fill:#3aa9ff}.info-t{fill:#7cc4ff}.info-tint{fill:rgba(58,169,255,.12)}
.warn{fill:#f5b53d}.warn-t{fill:#f8c96a}.warn-tint{fill:rgba(245,181,61,.12)}
.dan{fill:#ff5a6e}.dan-tint{fill:rgba(255,90,110,.12)}.dan-t{fill:#ff8a98}
.dot{fill:#2a3036}.line{stroke:#2a3036}.arrow{stroke:#8e979f}.arrowhead{fill:#8e979f}.dash{stroke:#8e979f}
.on-acc{fill:#052e20}.icon-acc{color:#4ade80}.icon-dan{color:#ff8a98}
}
""" % FONT


def icon(name, x, y, size=28, cls="icon"):
    """A sprite icon at (x, y), `size` px, drawn in the class's color (the icons use currentColor)."""
    m = re.search(r'<symbol id="%s"[^>]*>(.*?)</symbol>' % name, SPRITE, re.S)
    return (f'<g class="{cls}" transform="translate({x},{y}) scale({size / 24:.4f})">{m.group(1)}</g>')


def text(x, y, s, size=14, weight=400, cls="ink", anchor="start"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" class="{cls}" '
            f'text-anchor="{anchor}">{s}</text>')


def rect(x, y, w, h, r=14, cls="card"):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" class="{cls}"/>'


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{title}"><title>{title}</title><style>{CSS}</style>'
            f'{rect(0, 0, w, h, 20, "bg")}{body}</svg>')


def dots(x, y, cols, rows, step, lit, lit_cls="acc", r=2.6, seed=1):
    """A grid of expert dots; `lit` of them (spread at random) in the accent color."""
    rnd = random.Random(seed)
    n = cols * rows
    on = set(rnd.sample(range(n), min(lit, n)))
    out = []
    for i in range(n):
        cx, cy = x + (i % cols) * step, y + (i // cols) * step
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" class="{lit_cls if i in on else "dot"}"/>')
    return "".join(out)


def arrow(x1, y1, x2, y2, cls="arrow"):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2 - 7}" y2="{y2}" class="{cls}" stroke-width="1.6" stroke-linecap="round"/>'
            f'<path d="M{x2} {y2} l-8 -5 v10 z" class="arrowhead"/>')


def how_it_works():
    W, H = 960, 470
    b = []
    # the headline strip: one word, 10 experts
    b.append(rect(24, 24, W - 48, 92))
    b.append(icon("i-chat", 48, 50, 36))
    b.append(text(100, 62, "The model is made of 24,576 small specialists - “experts”.", 19, 600))
    b.append(text(100, 88, "For each word it writes, it asks only 10 of them. Strata keeps the busy ones",
                  15, 400, "muted"))
    b.append(text(100, 106, "on your graphics card and the rest in RAM - so it runs on a normal PC.", 15, 400, "muted"))
    b.append(dots(W - 262, 46, 28, 4, 7.6, 10, seed=3))
    b.append(text(W - 262, 94, "10 at work for one word", 12, 500, "acc-t"))
    # three cards
    cw, ch, cy = 288, 316, 136
    xs = [24, 24 + cw + 24, 24 + 2 * (cw + 24)]
    cards = [
        ("i-gpu", "Graphics card", "Small but very fast: 12-24 GB", "acc",
         ["Runs the core of the model", "for every word, and keeps the", "~4,000 experts used most"],
         "the most-used experts", 5, 26, 26 * 5, "acc"),
        ("i-memory", "RAM + processor", "Big: 32-64 GB of memory", "info",
         ["Holds all 24,576 experts. When a", "word needs one the card doesn't", "have, the CPU does it - at once"],
         "all 24,576 experts", 10, 26, 18, "info"),
        ("i-disk", "SSD", "Holds a 29 GB lookup table", "warn",
         ["The model looks up a few", "small rows of it per word -", "no need to keep it in RAM"],
         "", 0, 0, 0, "warn"),
    ]
    for (ic, title, sub, tone, lines, dlabel, rows, cols, lit, lit_cls), x in zip(cards, xs):
        b.append(rect(x, cy, cw, ch))
        b.append(rect(x + 20, cy + 20, 48, 48, 12, tone + "-tint"))
        b.append(icon(ic, x + 30, cy + 30, 28, "icon"))
        b.append(text(x + 82, cy + 42, title, 19, 700))
        b.append(text(x + 82, cy + 62, sub, 13, 500, tone + "-t"))
        for i, ln in enumerate(lines):
            b.append(text(x + 20, cy + 104 + i * 21, ln, 15, 400, "ink"))
        if rows:
            b.append(dots(x + 26, cy + 196, cols, rows, 9.4, lit, lit_cls, seed=len(title)))
            b.append(text(x + 20, cy + 196 + rows * 9.4 + 14, dlabel, 12, 500, "muted"))
        else:                                       # the SSD: a table of rows, a few being read
            for r in range(6):
                yy = cy + 190 + r * 17
                b.append(rect(x + 20, yy, cw - 40, 11, 4, "warn" if r in (1, 4) else "soft"))
            b.append(text(x + 20, cy + 306, "a few rows per word", 12, 500, "muted"))
    return svg(W, H, "".join(b), "How Strata splits the model between your graphics card, RAM and SSD")


def guess_and_check():
    W, H = 960, 330
    b = [rect(24, 24, W - 48, H - 48)]
    b.append(icon("i-thinking", 48, 48, 30))
    b.append(text(92, 70, "Guess, then check - many words per step", 19, 700))
    b.append(text(92, 94, "A small, fast helper inside the model guesses the next words. The big model checks all the "
                          "guesses in one go", 15, 400, "muted"))
    b.append(text(92, 113, "and keeps the ones it agrees with. The big model decides every word - it just arrives 1.6-1.8x sooner.",
                  15, 400, "muted"))
    words = ["def", "reverse", "(items", "):"]
    fixed = ", key):"                                  # the model's own word where the guess was wrong
    ok = [True, True, True, False]
    # row 1: the helper's guesses (dashed chips)
    y1, y2 = 156, 238
    b.append(text(48, y1 + 24, "Helper guesses", 14, 600, "muted"))
    b.append(text(48, y2 + 24, "Big model checks", 14, 600, "muted"))
    x = 176
    xs = []
    for i, w in enumerate(words[:4]):
        wd = 34 + len(w) * 9
        xs.append((x, wd))
        b.append(f'<rect x="{x}" y="{y1}" width="{wd}" height="38" rx="10" class="dash" stroke-width="1.4"/>')
        b.append(text(x + wd / 2, y1 + 25, w, 16, 500, "ink", "middle"))
        x += wd + 14
    # row 2: the verdicts
    for i, ((x, wd), good) in enumerate(zip(xs, ok)):
        tone = "acc" if good else "dan"
        b.append(rect(x, y2, wd, 38, 10, tone + "-tint"))
        b.append(text(x + wd / 2 - 8, y2 + 25, words[i], 16, 600, tone + "-t", "middle"))
        b.append(icon("i-check" if good else "i-error", x + wd - 22, y2 + 11, 16, "icon-" + tone))
        if not good:                                   # the model's own word beside the rejected guess
            fx = x + wd + 10
            fw = 26 + len(fixed) * 9
            b.append(rect(fx, y2, fw, 38, 10, "acc-tint"))
            b.append(text(fx + fw / 2, y2 + 25, fixed, 16, 600, "acc-t", "middle"))
            xs[i] = (fx, fw)
        b.append(f'<line x1="{x + wd / 2}" y1="{y1 + 40}" x2="{x + wd / 2}" y2="{y2 - 4}" class="line" stroke-width="1.4"/>')
    # the result
    rx = xs[-1][0] + xs[-1][1] + 40
    b.append(arrow(rx, y2 + 19, rx + 30, y2 + 19))
    b.append(rect(rx + 42, y2 - 6, W - 48 - (rx + 42), 50, 12, "acc"))
    b.append(icon("i-bolt", rx + 54, y2 + 6, 26, "on-acc-icon"))
    b.append(text(rx + 90, y2 + 16, "4 words in one step", 15, 700, "on-acc"))
    b.append(text(rx + 90, y2 + 34, "3 guesses kept + its own word", 12, 500, "on-acc"))
    return svg(W, H, "".join(b).replace('class="on-acc-icon"', 'class="icon" style="color:#ffffff"'),
               "Speculative decoding: a small helper guesses words, the big model checks them at once")


if __name__ == "__main__":
    (OUT / "how-it-works.svg").write_text(how_it_works(), encoding="utf-8")
    (OUT / "guess-and-check.svg").write_text(guess_and_check(), encoding="utf-8")
    for f in ("how-it-works.svg", "guess-and-check.svg"):
        print(f, (OUT / f).stat().st_size, "bytes")
