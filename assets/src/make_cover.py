#!/usr/bin/env python3
"""Cover image of the README: the procedural iris artwork (make_cover_art.py) with the title, one line of
text and three key numbers on top. The numbers come from the run records in results/.

    bash assets/src/build_cover.sh      # artwork -> cover.html -> assets/cover.png (2560 x 1280)

Fonts (Archivo, IBM Plex Mono) load from Google Fonts.
"""
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import summarize_results as S  # noqa: E402

W, H = 1280, 640
BG, CREAM, SOFT, RULE, ACCENT = "#0A0B0E", "#F4F4F0", "#B4B5B0", "#3A3C43", "#F0431A"
SANS = "Archivo, 'Helvetica Neue', Helvetica, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, Menlo, monospace"
FONTS = ("https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900"
         "&family=IBM+Plex+Mono:wght@400;500;600&display=swap")
LABEL = f"font-family: {MONO}; font-size: 11.5px; line-height: 1.4; letter-spacing: 0.07em; text-transform: uppercase"


def average():
    runs = [r for r in S.load_runs(str(REPO / "results")) if r["role"] == "safaid"]
    return statistics.mean(S.avg19(r) for r in runs)


def stat(number, line1, line2, first=False):
    edge = "" if first else f" padding-left: 26px; border-left: 1px solid {RULE};"
    return f"""<div style="display: flex; flex-direction: column; gap: 7px;{edge}">
<div style="font-size: 50px; font-weight: 800; font-stretch: 72%; line-height: 0.88">{number}</div>
<div style="{LABEL}">{line1}<br><span style="color: {SOFT}">{line2}</span></div>
</div>"""


def body(accent=ACCENT, art_src="cover_art.png"):
    avg = average()
    return f"""<div style="position: relative; width: {W}px; height: {H}px; overflow: hidden; background: {BG}; color: {CREAM}; font-family: {SANS}">
<img src="{art_src}" alt="" style="position: absolute; left: 0; top: 0; width: {W}px; height: {H}px; display: block">
<div style="position: absolute; left: 56px; top: 0; width: 640px; height: {H}px; box-sizing: border-box; padding: 40px 0 38px; display: flex; flex-direction: column; justify-content: space-between">
<div style="display: flex; align-items: baseline; gap: 14px">
<div style="font-size: 23px; font-weight: 800; letter-spacing: 0.03em">SAFAID</div>
<div style="{LABEL}; color: {SOFT}">Selective Augmentation for Fake AI-Generated Image Detection</div>
</div>
<div style="display: flex; flex-direction: column; gap: 18px">
<h1 style="margin: 0; font-size: 100px; font-weight: 800; font-stretch: 68%; line-height: 0.89; letter-spacing: -0.005em; text-transform: uppercase">This image<br>is under<br><span style="color: {accent}">investigation.</span></h1>
<p style="margin: 0; max-width: 520px; font-size: 18.5px; line-height: 1.42; text-wrap: pretty">One forensic question and no category labels: a fine-tuned 4B <span style="white-space: nowrap">vision&#8211;language</span> model that tells real photographs from <span style="white-space: nowrap">AI-generated</span> images.</p>
</div>
<div style="display: flex; gap: 26px">
{stat(f'{avg:.2f}<span style="font-size: 30px">%</span>', "Average accuracy", "19 generators", first=True)}
{stat("0", "Category labels", "training or testing")}
{stat("1", "GPU", "4.44 B parameters")}
</div>
</div>
<div style="position: absolute; right: 44px; top: 36px; padding: 6px 11px; background: {BG}; border: 1px solid {CREAM}; {LABEL}; font-weight: 500">Real or AI-generated?</div>
</div>"""


def page(markup):
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SAFAID cover</title>
<link rel="stylesheet" href="{FONTS.replace('&', '&amp;')}">
<style>
html, body {{ margin: 0; background: {BG} }}
</style>
</head>
<body>
{markup}
</body>
</html>
"""


if __name__ == "__main__":
    out = HERE / "cover.html"
    out.write_text(page(body()), encoding="utf-8")
    print("wrote", out)
