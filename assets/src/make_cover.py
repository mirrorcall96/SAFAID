#!/usr/bin/env python3
"""Cover image of the README: the SAFAID wordmark with the letters "AI" rebuilt from tiles, the title,
the authors, three key numbers (from the run records in results/) and a QR code to the repository.

    bash assets/src/build_cover.sh      # -> assets/cover.png (2560 x 1280)

Two steps, both called by build_cover.sh:
    python assets/src/make_cover.py source    # writes wordmark_src.html (rendered to wordmark_src.png by Chrome)
    python assets/src/make_cover.py compose   # tiles the "AI" of wordmark_src.png -> cover_art.png, writes cover.html

Requires numpy and Pillow. Fonts (Archivo, IBM Plex Mono) load from Google Fonts.
"""
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import summarize_results as S  # noqa: E402

W, H = 1280, 640
SCALE = 2                                   # pixels per CSS px in the rendered images
BG, INK, SOFT, RULE, ACCENT = "#FAFBF9", "#0F1512", "#56615B", "#D5DCD7", "#0A9468"
TILES = ["#0A9468", "#12B886", "#63D2A1", "#0B6E55", "#12B886"]
SANS = "Archivo, 'Helvetica Neue', Helvetica, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, Menlo, monospace"
FONTS = ("https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900"
         "&family=IBM+Plex+Mono:wght@400;500;600&display=swap")
REPO_URL = "github.com/mirrorcall96/SAFAID"
LABEL = f"font-family: {MONO}; font-size: 11.5px; line-height: 1.4; letter-spacing: 0.07em; text-transform: uppercase"
WM_W, WM_H = 1280, 330                      # size of the wordmark source, in CSS px


def page(markup, bg=BG):
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SAFAID cover</title>
<link rel="stylesheet" href="{FONTS.replace('&', '&amp;')}">
<style>
html, body {{ margin: 0; background: {bg} }}
</style>
</head>
<body>
{markup}
</body>
</html>
"""


def wordmark_source():
    """The wordmark in black on white, with the letters to be tiled in pure red."""
    return f"""<div style="width: {WM_W}px; height: {WM_H}px; display: flex; align-items: center; background: #FFFFFF">
<div style="font-family: {SANS}; font-weight: 900; font-stretch: 78%; font-size: 318px; line-height: 1; letter-spacing: -0.012em; color: #000000; white-space: nowrap; padding-left: 4px">SAF<span style="color: #FF0000">AI</span>D</div>
</div>"""


def tile_wordmark(src_png, out_png, tile=13, seed=3):
    """Rebuild the red letters from square tiles and let a few tiles drift above and below them."""
    import numpy as np
    from PIL import Image

    def hexc(h):
        return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255

    rng = np.random.default_rng(seed)
    a = np.asarray(Image.open(src_png).convert("RGB")).astype(np.float32) / 255
    hh, ww = a.shape[:2]
    bg, ink, cols = hexc(BG), hexc(INK), [hexc(c) for c in TILES]
    red = np.clip(a[..., 0] - np.maximum(a[..., 1], a[..., 2]), 0, 1)
    dark = np.clip(1 - a.max(-1), 0, 1) * (red < 0.2)
    out = bg + (ink - bg) * dark[..., None]
    tt, gap = tile * SCALE, 2 * SCALE
    ys, xs = np.where(red > 0.5)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    for yb in range(y0 - (y0 % tt) - 14 * tt, y1 + 3 * tt, tt):
        for xb in range(x0 - (x0 % tt) - 3 * tt, x1 + 14 * tt, tt):
            if yb < 0 or xb < 0 or yb + tt > hh or xb + tt > ww:
                continue
            c3 = cols[rng.integers(len(cols))]
            if red[yb:yb + tt, xb:xb + tt].mean() > 0.42:
                if rng.random() >= 0.05:
                    out[yb + gap // 2:yb + tt - gap // 2, xb + gap // 2:xb + tt - gap // 2] = c3
                continue
            above, below = (y0 - (yb + tt)) / tt, (yb - y1) / tt
            if not (x0 - tt <= xb <= x1 + 2 * tt) or (above < 0 and below < 0):
                continue
            d = above if above >= 0 else below * 2.2
            if d < 9 and rng.random() < 0.55 * np.exp(-d / 2.4):
                size = int((tt - gap) * (1 - 0.06 * d))
                off = (tt - size) // 2
                out[yb + off:yb + off + size, xb + off:xb + off + size] = bg + (c3 - bg) * 0.95 * np.exp(-d / 6)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(out_png, optimize=True)


def average():
    runs = [r for r in S.load_runs(str(REPO / "results")) if r["role"] == "safaid"]
    return statistics.mean(S.avg19(r) for r in runs)


def stat(number, line1, line2, first=False):
    edge = "" if first else f" padding-left: 24px; border-left: 1px solid {RULE};"
    return f"""<div style="display: flex; flex-direction: column; gap: 7px;{edge}">
<div style="font-size: 48px; font-weight: 800; font-stretch: 72%; line-height: 0.88">{number}</div>
<div style="{LABEL}">{line1}<br><span style="color: {SOFT}">{line2}</span></div>
</div>"""


def body(art_src="cover_art.png", qr_src="../../poster/qr.png"):
    art_w = W - 112
    art_h = round(WM_H * art_w / WM_W)
    return f"""<div style="position: relative; width: {W}px; height: {H}px; overflow: hidden; background: {BG}; color: {INK}; font-family: {SANS}">
<div style="position: absolute; left: 56px; top: 0; width: {art_w}px; height: {H}px; box-sizing: border-box; padding: 38px 0 36px; display: flex; flex-direction: column; justify-content: space-between">
<div style="{LABEL}; color: {SOFT}">Category-free detection of AI-generated images</div>
<img src="{art_src}" alt="SAFAID" style="width: {art_w}px; height: {art_h}px; display: block">
<div style="display: flex; justify-content: space-between; align-items: flex-end; gap: 40px">
<div style="display: flex; flex-direction: column; gap: 9px; max-width: 610px">
<div style="font-size: 25px; font-weight: 750; font-stretch: 86%; line-height: 1.12; text-wrap: balance">Selective Augmentation for Fake <span style="color: {ACCENT}">AI</span>-Generated Image Detection</div>
<p style="margin: 0; font-size: 16px; line-height: 1.42; text-wrap: pretty">One forensic question and no category labels: a fine-tuned 4B <span style="white-space: nowrap">vision&#8211;language</span> model that tells real photographs from <span style="white-space: nowrap">AI-generated</span> images.</p>
<div style="display: flex; flex-direction: column; gap: 3px; padding-top: 5px">
<div style="font-size: 16.5px; font-weight: 700">Mohammad Alhadidi &#183; Rawan Ghnemat</div>
<div style="{LABEL}; color: {SOFT}">Princess Sumaya University for Technology, Amman, Jordan</div>
</div>
</div>
<div style="display: flex; gap: 24px">
{stat(f'{average():.2f}<span style="font-size: 29px">%</span>', "Average accuracy", "19 generators", first=True)}
{stat("0", "Category labels", "training or testing")}
{stat("1", "GPU", "4.44 B parameters")}
</div>
</div>
</div>
<div style="position: absolute; right: 56px; top: 30px; display: flex; align-items: center; gap: 13px">
<div style="display: flex; flex-direction: column; align-items: flex-end; gap: 3px; text-align: right">
<div style="{LABEL}; color: {SOFT}">Code and run records</div>
<div style="font-family: {MONO}; font-size: 13px; font-weight: 600">{REPO_URL}</div>
</div>
<img src="{qr_src}" alt="QR code: {REPO_URL}" style="width: 92px; height: 92px; display: block">
</div>
</div>"""


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "source":
        (HERE / "wordmark_src.html").write_text(page(wordmark_source(), "#FFFFFF"), encoding="utf-8")
        print("wrote", HERE / "wordmark_src.html")
    elif step == "compose":
        tile_wordmark(HERE / "wordmark_src.png", HERE / "cover_art.png")
        (HERE / "cover.html").write_text(page(body()), encoding="utf-8")
        print("wrote", HERE / "cover_art.png", "and", HERE / "cover.html")
    else:
        sys.exit(__doc__)
