#!/usr/bin/env python3
"""Build the SAFAID poster as one HTML page from the run records in results/.

    python poster/make_poster.py        # writes poster/poster.html
    bash poster/build.sh                # also renders assets/poster.pdf and assets/poster.png

The artwork is A3 portrait (1123 x 1587 CSS px) and scales to A1 or A0 for printing. Every number comes
from results/<run>/metrics_*.json through scripts/summarize_results.py; the only hard-coded numbers are
the published CatAID accuracies. Fonts (Archivo, IBM Plex Mono) load from Google Fonts.
"""
import argparse
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "scripts"))
import summarize_results as S  # noqa: E402

W, H = 1123, 1587
INK, PAPER, TRACK, RULE, MUTED, ACCENT = "#0E0F12", "#F4F4F0", "#DFDFD8", "#BDBDB5", "#55565A", "#F0431A"
SANS = "Archivo, 'Helvetica Neue', Helvetica, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, Menlo, monospace"
FONTS = ("https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900"
         "&family=IBM+Plex+Mono:wght@400;500;600&display=swap")
REPO_URL = "github.com/mirrorcall96/SAFAID"

LABEL = f"font-family: {MONO}; font-size: 12.5px; line-height: 1.35; letter-spacing: 0.06em; text-transform: uppercase"
SHORT = {"Glide (100-27)": "GLIDE 100-27", "Glide (50-27)": "GLIDE 50-27", "Glide (100-10)": "GLIDE 100-10",
         "LDM (200)": "LDM 200", "LDM (CFG)": "LDM CFG", "LDM (100)": "LDM 100", "DALL-E": "DALL-E"}


def load():
    runs = S.load_runs(str(REPO / "results"))
    by = {}
    for r in runs:
        by.setdefault(r["role"], []).append(r)
    abl = [by[k][0] for k in ("abl_noprompt", "abl_uniform", "abl_noaug")]
    ref = next(r for r in by["safaid"] if r["seed"] == abl[0]["seed"])
    saf = [ref] + sorted((r for r in by["safaid"] if r is not ref), key=lambda r: r["seed"])
    return saf, ref, abl


def pct(v):
    """Position on the 50-100 % accuracy axis, in percent of the track."""
    return f"{(v - 50) / 50 * 100:.2f}%"


def chart_row(name, ours, theirs, accent, bold=False):
    win = round(ours, 2) >= theirs
    weight = 600 if win else 400
    return f"""<div style="display: flex; align-items: center; gap: 10px; height: {23 if bold else 20}px">
<div style="width: 104px; text-align: right; font-family: {MONO}; font-size: 12px; font-weight: {600 if bold else 400}; color: {INK}">{name}</div>
<div style="position: relative; flex-grow: 1; height: {14 if bold else 11}px; background: {TRACK}">
<div style="width: {pct(ours)}; height: 100%; background: {accent}"></div>
<div style="position: absolute; left: calc({pct(theirs)} - 1.5px); top: -4px; width: 3px; height: {22 if bold else 19}px; background: {INK}"></div>
</div>
<div style="width: 38px; text-align: right; font-family: {MONO}; font-size: 12px; font-weight: {weight}; color: {INK if win else MUTED}">{ours:.1f}</div>
</div>"""


def group_head(text):
    return f"""<div style="display: flex; align-items: center; gap: 10px; height: 24px">
<div style="width: 104px"></div>
<div style="flex-grow: 1; {LABEL}; font-size: 11px; color: {MUTED}">{text}</div>
<div style="width: 38px"></div>
</div>"""


def abl_row(name, value, delta, accent, full=False):
    note = "" if full else f"""<span style="color: {MUTED}; font-weight: 400">{delta}</span>"""
    return f"""<div style="display: flex; flex-direction: column; gap: 5px">
<div style="display: flex; justify-content: space-between; align-items: baseline; gap: 8px">
<div style="font-size: 15px; font-weight: {700 if full else 450}">{name}</div>
<div style="font-family: {MONO}; font-size: 13px; font-weight: 600; white-space: nowrap">{value:.2f} {note}</div>
</div>
<div style="height: 11px; background: {TRACK}">
<div style="width: {pct(value)}; height: 100%; background: {accent if full else INK}"></div>
</div>
</div>"""


def body(accent=ACCENT, qr_src="qr.png"):
    saf, ref, abl = load()
    cat = S.BASELINES["CatAID"]
    mean = lambda g: statistics.mean(S.acc(r, g) for r in saf)
    sv = [S.avg19(r) for r in saf]
    avg, std = statistics.mean(sv), statistics.stdev(sv)
    gain = S.rdiff(avg, S.BASELINE_AVG["CatAID"])
    real = statistics.mean(S.acc(r, g, "real_acc") for r in saf for g in S.ORDER)
    fake = statistics.mean(S.acc(r, g, "fake_acc") for r in saf for g in S.ORDER)
    base = S.avg19(ref)
    rows_fs = "\n".join(chart_row(SHORT.get(g, g), mean(g), cat[S.ORDER.index(g)], accent) for g in S.ORDER[:S.N_FS])
    rows_ud = "\n".join(chart_row(SHORT.get(g, g), mean(g), cat[S.ORDER.index(g)], accent) for g in S.ORDER[S.N_FS:])
    ticks = "".join(
        f"""<div style="position: absolute; left: {i * 20}%; transform: translateX(-50%); font-family: {MONO}; font-size: 11px; color: {MUTED}">{50 + i * 10}</div>"""
        for i in range(6))
    chevron = f"""<svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="{INK}" stroke-width="2" stroke-linecap="square" style="flex-shrink: 0; margin-top: 34px"><path d="M6 3 L12 9 L6 15"></path></svg>"""
    sq = lambda fill, border=INK, op=1: f"""<div style="width: 26px; height: 26px; box-sizing: border-box; background: {fill}; border: 2px solid {border}; opacity: {op}"></div>"""
    step_title = "font-size: 25px; font-weight: 750; font-stretch: 84%; line-height: 1.05"
    step_text = "margin: 0; font-size: 15.5px; line-height: 1.42"
    step_no = f"{LABEL}; font-weight: 600; color: {MUTED}"

    return f"""<div style="width: {W}px; height: {H}px; box-sizing: border-box; padding: 40px 56px 34px; background: {PAPER}; color: {INK}; font-family: {SANS}; display: flex; flex-direction: column; gap: 24px; overflow: hidden">

<div style="display: flex; justify-content: space-between; align-items: baseline; gap: 24px; padding-bottom: 12px; border-bottom: 2px solid {INK}">
<div style="display: flex; align-items: baseline; gap: 14px">
<div style="font-size: 22px; font-weight: 800; letter-spacing: 0.02em">SAFAID</div>
<div style="{LABEL}; color: {MUTED}">Selective Augmentation for Fake AI-Generated Image Detection</div>
</div>
<div style="{LABEL}; color: {MUTED}">Detecting AI-generated images &#183; 2026</div>
</div>

<div style="display: flex; flex-direction: column; gap: 16px">
<h1 style="margin: 0; font-size: 113px; font-weight: 800; font-stretch: 68%; line-height: 0.9; letter-spacing: -0.005em; text-transform: uppercase">This image is under<br><span style="color: {accent}">investigation.</span></h1>
<p style="margin: 0; max-width: 880px; font-size: 21px; line-height: 1.36; text-wrap: pretty">SAFAID asks every image the same forensic question, with no category labels, and tells real photographs from AI-generated images across 19 generators with one fine-tuned 4B <span style="white-space: nowrap">vision&#8211;language</span> model.</p>
</div>

<div style="display: flex; border-top: 2px solid {INK}; border-bottom: 2px solid {INK}">
<div style="flex-grow: 1.5; flex-basis: 0; padding: 15px 24px 13px 0; display: flex; flex-direction: column; gap: 8px">
<div style="font-size: 92px; font-weight: 800; font-stretch: 72%; line-height: 0.88">{avg:.2f}<span style="font-size: 54px">%</span></div>
<div style="{LABEL}">Average accuracy on 19 generators<br><span style="color: {MUTED}">&#177; {std:.2f} over 3 runs &#183; +{gain:.2f} over CatAID</span></div>
</div>
<div style="flex-grow: 1; flex-basis: 0; padding: 15px 24px 13px 28px; border-left: 1px solid {RULE}; display: flex; flex-direction: column; gap: 8px">
<div style="font-size: 92px; font-weight: 800; font-stretch: 72%; line-height: 0.88">0</div>
<div style="{LABEL}">Category labels<br><span style="color: {MUTED}">in training and at test time</span></div>
</div>
<div style="flex-grow: 1; flex-basis: 0; padding: 15px 0 13px 28px; border-left: 1px solid {RULE}; display: flex; flex-direction: column; gap: 8px">
<div style="font-size: 92px; font-weight: 800; font-stretch: 72%; line-height: 0.88">1</div>
<div style="{LABEL}">GPU, 4.44 B parameters<br><span style="color: {MUTED}">CatAID: 9 B parameters, 8 GPUs</span></div>
</div>
</div>

<div style="display: flex; flex-direction: column; gap: 14px">
<div style="{LABEL}; font-weight: 600">How it works</div>
<div style="display: flex; gap: 18px">
<div style="flex-grow: 1; flex-basis: 0; display: flex; flex-direction: column; gap: 10px">
<div style="{step_no}">01</div>
<div style="{step_title}">Ask one question</div>
<p style="{step_text}">Every image gets the same prompt, in training and at test time. The content of the image is never named.</p>
<div style="padding: 10px 12px; border: 1.5px solid {INK}; font-family: {MONO}; font-size: 12.5px; line-height: 1.45">&#8220;This is an image that is under investigation. Is this image generated by AI, or is it a real image?&#8221;</div>
</div>
{chevron}
<div style="flex-grow: 1; flex-basis: 0; display: flex; flex-direction: column; gap: 10px">
<div style="{step_no}">02</div>
<div style="{step_title}">Augment only the fakes</div>
<p style="{step_text}">Each generated training image gets three degraded copies. Real images stay untouched, as they are in the wild.</p>
<div style="display: flex; align-items: flex-start; gap: 22px; font-family: {MONO}; font-size: 12px">
<div style="display: flex; flex-direction: column; gap: 6px">
<div style="display: flex; gap: 5px">{sq(PAPER)}</div>
<div>real</div>
</div>
<div style="display: flex; flex-direction: column; gap: 6px">
<div style="display: flex; gap: 5px">{sq(INK)}{sq(accent, accent)}{sq(accent, accent, 0.66)}{sq(accent, accent, 0.36)}</div>
<div>fake + 3 degraded copies</div>
</div>
</div>
</div>
{chevron}
<div style="flex-grow: 1; flex-basis: 0; display: flex; flex-direction: column; gap: 10px">
<div style="{step_no}">03</div>
<div style="{step_title}">Train one compact model</div>
<p style="{step_text}">All 4.44 B parameters of Qwen3-VL-4B are fine-tuned on 10,000 ProGAN-based samples. The answer is one of two sentences.</p>
<div style="display: flex; flex-direction: column; align-items: flex-start; gap: 6px; font-family: {MONO}; font-size: 12.5px">
<div style="padding: 5px 10px; border: 1.5px solid {INK}">This is a real image.</div>
<div style="padding: 5px 10px; border: 1.5px solid {INK}; background: {INK}; color: {PAPER}">This is an AI-generated image.</div>
</div>
</div>
</div>
</div>

<div style="display: flex; gap: 44px; padding-top: 18px; border-top: 2px solid {INK}">
<div style="flex-grow: 1; flex-basis: 0; display: flex; flex-direction: column; gap: 8px">
<div style="display: flex; justify-content: space-between; align-items: center; gap: 16px">
<div style="{LABEL}; font-weight: 600">Accuracy per generator (%)</div>
<div style="display: flex; align-items: center; gap: 16px; font-family: {MONO}; font-size: 11.5px">
<div style="display: flex; align-items: center; gap: 6px"><div style="width: 22px; height: 10px; background: {accent}"></div><div>SAFAID, 3 runs</div></div>
<div style="display: flex; align-items: center; gap: 6px"><div style="width: 3px; height: 16px; background: {INK}"></div><div>CatAID, published</div></div>
</div>
</div>
<div style="display: flex; flex-direction: column">
{group_head("ForenSynths &#183; GANs, deepfakes, super-resolution")}
{rows_fs}
{group_head("UniverDiffu &#183; diffusion models")}
{rows_ud}
<div style="height: 8px"></div>
{chart_row("AVERAGE", avg, S.BASELINE_AVG["CatAID"], accent, bold=True)}
<div style="display: flex; align-items: center; gap: 10px; height: 20px">
<div style="width: 104px"></div>
<div style="position: relative; flex-grow: 1; height: 12px">{ticks}</div>
<div style="width: 38px"></div>
</div>
</div>
</div>

<div style="width: 318px; flex-shrink: 0; display: flex; flex-direction: column; gap: 22px">
<div style="display: flex; flex-direction: column; gap: 12px">
<div style="{LABEL}; font-weight: 600">What matters</div>
{abl_row("SAFAID", base, "", accent, full=True)}
{abl_row("without the investigative prompt", S.avg19(abl[0]), S.signed(S.avg19(abl[0]) - base), accent)}
{abl_row("augmenting real images too", S.avg19(abl[1]), S.signed(S.avg19(abl[1]) - base), accent)}
{abl_row("no augmentation", S.avg19(abl[2]), S.signed(S.avg19(abl[2]) - base), accent)}
<div style="font-family: {MONO}; font-size: 11.5px; line-height: 1.4; color: {MUTED}">Average accuracy (%), same seed and schedule.</div>
</div>
<div style="display: flex; flex-direction: column; gap: 10px; padding-top: 18px; border-top: 1px solid {RULE}">
<div style="{LABEL}; font-weight: 600">Takeaways</div>
<p style="margin: 0; font-size: 16px; line-height: 1.38"><b>Category labels are not needed.</b> One context-free question is enough.</p>
<p style="margin: 0; font-size: 16px; line-height: 1.38"><b>Augment only the fakes.</b> Degrading real images too is worse than no augmentation.</p>
<p style="margin: 0; font-size: 16px; line-height: 1.38"><b>Mind the asymmetry.</b> {real:.1f}% of real but {fake:.1f}% of generated images are classified correctly: a &#8220;fake&#8221; verdict is the stronger evidence.</p>
</div>
</div>
</div>

<div style="display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; margin-top: auto; padding-top: 14px; border-top: 2px solid {INK}">
<div style="display: flex; align-items: center; gap: 16px">
<img src="{qr_src}" alt="QR code: {REPO_URL}" style="width: 78px; height: 78px; display: block">
<div style="display: flex; flex-direction: column; gap: 4px">
<div style="{LABEL}; color: {MUTED}">Code, data manifests and all run records</div>
<div style="font-family: {MONO}; font-size: 17px; font-weight: 600">{REPO_URL}</div>
</div>
</div>
<div style="display: flex; flex-direction: column; align-items: flex-end; gap: 4px; text-align: right">
<div style="font-size: 19px; font-weight: 700">Mohammad Alhadidi &#183; Rawan Ghnemat</div>
<div style="{LABEL}; color: {MUTED}">King Hussein School of Computing Sciences<br>Princess Sumaya University for Technology, Amman, Jordan</div>
</div>
</div>

</div>"""


def page(markup):
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SAFAID poster</title>
<link rel="stylesheet" href="{FONTS.replace('&', '&amp;')}">
<style>
@page {{ size: 297mm 420mm; margin: 0 }}
html, body {{ margin: 0; background: {PAPER}; -webkit-print-color-adjust: exact; print-color-adjust: exact }}
</style>
</head>
<body>
{markup}
</body>
</html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=HERE / "poster.html")
    a = ap.parse_args()
    a.out.write_text(page(body()), encoding="utf-8")
    print("wrote", a.out)
