#!/usr/bin/env python3
"""Render the README charts of the SAFAID release from the per-run metrics files.

Usage (from the repository root):

    python assets/make_charts.py                     # reads results/<run>/ of this repository
    python assets/make_charts.py --runs_root DIR     # any directory holding the run folders

Every plotted value is computed from results/<run>/metrics_<prompt>.json (row
accuracies and real/fake accuracies) and results/<run>/config.json (peak training
memory). The only hard-coded numbers are the published CatAID accuracies
(Cai et al., ICCVW 2025, Table 1), which are not produced by this code.

Writes, next to this script:
    per_generator_{light,dark}.png   headline_{light,dark}.png   ablations_{light,dark}.png
    qlora_{light,dark}.png           real_vs_fake_{light,dark}.png
    chart_data.csv                   (every plotted value; the table view of the charts)

Requires numpy and matplotlib only.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# --------------------------------------------------------------------------- data
# Table-1 order of the manuscript (ForenSynths: 11 GAN-based, UniverDiffu: 8 diffusion-based).
FORENSYNTHS = ["ProGAN", "CycleGAN", "BigGAN", "StyleGAN", "GauGAN", "SAN", "StarGAN",
               "DeepFakes", "SITD", "CRN", "IMLE"]
UNIVERDIFFU = ["ADM", "LDM (200)", "LDM (CFG)", "LDM (100)", "Glide (100-27)",
               "Glide (50-27)", "Glide (100-10)", "DALL-E"]
GENERATORS = FORENSYNTHS + UNIVERDIFFU

# CatAID as published (Cai et al., ICCVW 2025, Table 1); not produced by this repository.
CATAID_PUBLISHED = {
    "ProGAN": 99.86, "CycleGAN": 98.83, "BigGAN": 97.00, "StyleGAN": 96.63, "GauGAN": 96.24,
    "SAN": 57.28, "StarGAN": 99.35, "DeepFakes": 71.53, "SITD": 82.22, "CRN": 79.05,
    "IMLE": 79.87, "ADM": 78.80, "LDM (200)": 92.35, "LDM (CFG)": 88.45, "LDM (100)": 92.50,
    "Glide (100-27)": 88.80, "Glide (50-27)": 89.30, "Glide (100-10)": 89.25, "DALL-E": 91.35,
}

SAFAID_RUNS = ["r11", "r3", "r4"]   # investigative prompt, runs 1, 2, 3 (run 1 is the reference of the ablations)
CONTROL_RUNS = ["r2", "r5", "r6"]   # category prompt (CatAID-style), same backbone, 3 runs
QLORA_RUN = "r16"                    # QLoRA 4-bit, r16, alpha 16, LR 2e-4, same seed as SAFAID run 1
ABLATION_RUNS = [                    # same seed as SAFAID run 1
    ("r12", "Without investigative prompt", "vanilla prompt for training and test"),
    ("r13", "Uniform augmentation", "real and fake images augmented, 16,000 samples"),
    ("r14", "No augmentation", "4,000 original images"),
]


def find_run(root: Path, rid: str) -> Path:
    hits = sorted(p for p in root.glob(f"{rid}_*") if p.is_dir())
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one run directory {rid}_* under {root}, found {len(hits)}"
                         " (pass --runs_root to point at the directory holding the run folders)")
    return hits[0]


def load_metrics(root: Path, rid: str) -> dict:
    run = find_run(root, rid)
    files = sorted(run.glob("metrics_*.json"))
    if len(files) != 1:
        raise SystemExit(f"expected one metrics_*.json in {run.name}, found {len(files)}")
    m = json.loads(files[0].read_text())
    rows = m["rows"]
    avg = float(np.mean([rows[g]["acc"] for g in GENERATORS]))
    if abs(avg - m["average_19"]) > 0.006:
        raise SystemExit(f"{rid}: mean of the 19 rows {avg:.4f} != average_19 {m['average_19']}")
    return m


def load_config(root: Path, rid: str) -> dict:
    return json.loads((find_run(root, rid) / "config.json").read_text())


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def _betacf(a, b, x, itmax=300, eps=3e-14):
    """Continued fraction of the regularised incomplete beta function (Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc_reg(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def welch(x, y):
    """Welch's two-sample t-test: returns t, Welch-Satterthwaite df, two-sided p."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    t = (x.mean() - y.mean()) / math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    p = betainc_reg(df / 2.0, 0.5, df / (df + t * t))
    return float(t), float(df), float(p)


def collect(root: Path) -> dict:
    S = [load_metrics(root, r) for r in SAFAID_RUNS]
    C = [load_metrics(root, r) for r in CONTROL_RUNS]
    d = {"per_gen": {}, "seeds": {}}
    for g in GENERATORS:
        s_m, s_sd = mean_sd([m["rows"][g]["acc"] for m in S])
        c_m, c_sd = mean_sd([m["rows"][g]["acc"] for m in C])
        real_m, _ = mean_sd([m["rows"][g]["real_acc"] for m in S])
        fake_m, _ = mean_sd([m["rows"][g]["fake_acc"] for m in S])
        d["per_gen"][g] = dict(safaid=s_m, safaid_sd=s_sd, control=c_m, control_sd=c_sd,
                               cataid=CATAID_PUBLISHED[g], real=real_m, fake=fake_m)
    d["seeds"]["safaid"] = [m["average_19"] for m in S]
    d["seeds"]["control"] = [m["average_19"] for m in C]
    d["avg"] = dict(
        safaid=mean_sd(d["seeds"]["safaid"]),
        control=mean_sd(d["seeds"]["control"]),
        cataid=float(np.mean([CATAID_PUBLISHED[g] for g in GENERATORS])),
        real=float(np.mean([d["per_gen"][g]["real"] for g in GENERATORS])),
        fake=float(np.mean([d["per_gen"][g]["fake"] for g in GENERATORS])),
    )
    d["higher_on"] = sum(d["per_gen"][g]["safaid"] > d["per_gen"][g]["control"] for g in GENERATORS)
    d["welch"] = welch(d["seeds"]["safaid"], d["seeds"]["control"])
    ref = S[0]
    d["ref"] = dict(avg=ref["average_19"], san=ref["rows"]["SAN"]["acc"])
    for rid, _, _ in ABLATION_RUNS:   # the ablations must share the seed of the reference run
        if load_config(root, rid)["seed"] != load_config(root, SAFAID_RUNS[0])["seed"]:
            raise SystemExit(f"{rid} does not use the seed of {SAFAID_RUNS[0]}")
    q = load_metrics(root, QLORA_RUN)
    d["qlora"] = dict(avg=q["average_19"], san=q["rows"]["SAN"]["acc"])
    # full fine-tuning side of the QLoRA chart: the SAFAID run with the QLoRA seed if there is one,
    # otherwise the mean over the SAFAID runs
    q_seed = load_config(root, QLORA_RUN)["seed"]
    pair = [m for rid, m in zip(SAFAID_RUNS, S) if load_config(root, rid)["seed"] == q_seed]
    full = pair or S
    d["full"] = dict(avg=float(np.mean([m["average_19"] for m in full])),
                     san=float(np.mean([m["rows"]["SAN"]["acc"] for m in full])),
                     paired=bool(pair), n=len(full))
    d["mem"] = dict(full=float(load_config(root, SAFAID_RUNS[0])["peak_mem_gb"]),
                    qlora=float(load_config(root, QLORA_RUN)["peak_mem_gb"]))
    d["ablations"] = [(label, detail, load_metrics(root, rid)["average_19"])
                      for rid, label, detail in ABLATION_RUNS]
    return d


# --------------------------------------------------------------------------- theme
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  s1="#2a78d6", s2="#eb6834", s3="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835",
                 s1="#3987e5", s2="#d95926", s3="#199e70"),
}
PT = 0.72            # points per CSS pixel (figures are laid out at 100 dpi = 1x)
DPI = 200            # saved at 2x
FONT = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
LABELS = {
    "safaid": "SAFAID (investigative prompt)",
    "control": "Category prompt, same backbone (control)",
    "cataid": "CatAID, published (Cai et al., 2025)",
}


def op(dx, dy):
    """Offset in CSS px -> points, so offsets scale with the saved dpi."""
    return dx * PT, dy * PT


def available_fonts(names):
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    found = [n for n in names if n in have]
    return found or ["DejaVu Sans"]


class Canvas:
    """A figure laid out in CSS pixels from the top-left corner (1 unit = 1 px at 1x)."""

    def __init__(self, t: dict, w: int, h: int):
        self.t, self.w, self.h = t, w, h
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=t["surface"])
        self.overlay = self.fig.add_axes([0, 0, 1, 1], zorder=10)
        self.overlay.set_xlim(0, w)
        self.overlay.set_ylim(h, 0)
        self.overlay.axis("off")
        self.overlay.patch.set_alpha(0)
        self.renderer = self.fig.canvas.get_renderer()

    def text(self, x, y, s, size=12, color="ink", weight="normal", ha="left", va="baseline"):
        return self.overlay.text(x, y, s, fontsize=size * PT, color=self.t.get(color, color),
                                 fontweight=weight, ha=ha, va=va, family=FONT)

    def width(self, txt) -> float:
        return txt.get_window_extent(self.renderer).width

    def hline(self, x0, x1, y, color="grid"):
        self.overlay.plot([x0, x1], [y, y], color=self.t[color], lw=1 * PT,
                          solid_capstyle="butt")

    def header(self, title, subtitle_lines, x=24, y=34):
        self.text(x, y, title, size=18, weight="semibold")
        for i, line in enumerate(subtitle_lines):
            self.text(x, y + 24 + 18 * i, line, size=12.5, color="ink2")
        return y + 24 + 18 * (len(subtitle_lines) - 1)

    def legend(self, items, x, y, gap=22):
        """items: [(colour_key, label, kind)], kind in {'square', 'dot'}; y is the text baseline."""
        for key, label, kind in items:
            if kind == "dot":
                self.overlay.scatter([x + 5], [y - 4], s=(10 * PT) ** 2, color=self.t[key],
                                     edgecolors=self.t["surface"], linewidths=2 * PT, zorder=12)
            else:
                self.overlay.add_patch(FancyBboxPatch((x, y - 9), 10, 10,
                                                      boxstyle="round,pad=0,rounding_size=2",
                                                      fc=self.t[key], ec="none"))
            txt = self.text(x + 16, y, label, size=12, color="ink2")
            x += 16 + self.width(txt) + gap
        return x

    def axes(self, left, top, width, height, xlim):
        ax = self.fig.add_axes([left / self.w, 1 - (top + height) / self.h,
                                width / self.w, height / self.h])
        ax.set_xlim(*xlim)
        ax.set_ylim(height, 0)            # y in px, downward
        ax.set_facecolor(self.t["surface"])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0, pad=6, labelsize=11 * PT, labelcolor=self.t["muted"])
        for lab in ax.get_xticklabels():
            lab.set_family(FONT)
        ax.px_per_x = width / (xlim[1] - xlim[0])
        ax.plot_height = height
        return ax

    def save(self, path: Path):
        self.fig.savefig(path, dpi=DPI, facecolor=self.t["surface"], transparent=False)
        plt.close(self.fig)
        try:  # drop the (fully opaque) alpha channel; Pillow ships with matplotlib
            from PIL import Image
            with Image.open(path) as im:
                rgb = im.convert("RGB")
            rgb.save(path, optimize=True)
        except ImportError:
            pass


def grid(ax, t, ticks, baseline=None, fmt="{:g}", spans=None):
    """Hairline gridlines at the ticks (solid, recessive); `spans` limits them to y ranges (px)."""
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt.format(v) for v in ticks], family=FONT)
    spans = spans or [(0, ax.plot_height)]
    for y0, y1 in spans:
        for v in ticks:
            if v != baseline:
                ax.plot([v, v], [y0, y1], color=t["grid"], lw=1 * PT, zorder=0,
                        solid_capstyle="butt")
        if baseline is not None:
            ax.plot([baseline, baseline], [y0, y1], color=t["axis"], lw=1 * PT, zorder=1,
                    solid_capstyle="butt")


def hbar(ax, yc, x0, x1, thick, color, radius=4, zorder=3):
    """Horizontal bar from baseline x0 to x1: square at the baseline, rounded data end."""
    hh = thick / 2.0
    ry = min(radius, hh)
    rx = min(ry / ax.px_per_x, abs(x1 - x0))
    k = 0.5523
    yb, yt = yc - hh, yc + hh
    verts = [(x0, yb), (x1 - rx, yb),
             (x1 - rx + k * rx, yb), (x1, yb + ry - k * ry), (x1, yb + ry),
             (x1, yt - ry),
             (x1, yt - ry + k * ry), (x1 - rx + k * rx, yt), (x1 - rx, yt),
             (x0, yt), (x0, yb)]
    codes = [MPath.MOVETO, MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="none", zorder=zorder))


def whisker(ax, yc, lo, hi, color, cap=7):
    ax.plot([lo, hi], [yc, yc], color=color, lw=1.25 * PT, zorder=5, solid_capstyle="butt")
    for v in (lo, hi):
        ax.plot([v, v], [yc - cap / 2, yc + cap / 2], color=color, lw=1.25 * PT, zorder=5,
                solid_capstyle="butt")


def dot(ax, x, y, color, t, size=10, zorder=6):
    ax.scatter([x], [y], s=(size * PT) ** 2, color=color, edgecolors=t["surface"],
               linewidths=2 * PT, zorder=zorder)


def row_label(ax, y, s, t, color="ink", size=12, weight="normal", dx=-12):
    ax.annotate(s, xy=(0, y), xycoords=ax.get_yaxis_transform(), xytext=op(dx, 0),
                textcoords="offset points", ha="right", va="center", fontsize=size * PT,
                color=t[color], fontweight=weight, family=FONT)


def value_label(ax, x, y, s, t, color="ink", size=11.5, weight="normal", dx=6, ha="left"):
    ax.annotate(s, xy=(x, y), xytext=op(dx, 0), textcoords="offset points", ha=ha, va="center",
                fontsize=size * PT, color=t[color], fontweight=weight, family=FONT)


def minus(s: str) -> str:
    return s.replace("-", "−")


# --------------------------------------------------------------------------- charts
def grouped_layout(groups, band, hdr=34, avg_band=None):
    """Rows of the two benchmark groups plus an Average row, in px from the plot top.

    Returns rows [(y, generator)], headers [(y, text)], separators [y], grid spans
    [(y0, y1)] (the gridlines skip the header rows) and the y of the Average row.
    """
    avg_band = avg_band or band
    rows, headers, seps, spans, y = [], [], [], [], 0
    for gi, (name, gens) in enumerate(groups):
        if gi:
            seps.append(y + 6)
            y += 12
        headers.append((y + hdr * 0.55, name))
        y += hdr
        y0 = y
        for g in gens:
            rows.append((y + band / 2, g))
            y += band
        spans.append((y0 - 6, y))
    seps.append(y + 6)
    y += 12
    avg_y = y + avg_band / 2
    spans.append((y, y + avg_band))
    y += avg_band
    return rows, headers, seps, spans, avg_y, y


def group_headers(c, ax, headers, left, t):
    for yh, name in headers:
        title, _, detail = name.partition(" | ")
        txt = ax.annotate(title, xy=(0, yh), xycoords=ax.get_yaxis_transform(),
                          xytext=op(-(left - 24), 0), textcoords="offset points", ha="left",
                          va="center", fontsize=12.5 * PT, color=t["ink"], fontweight="semibold",
                          family=FONT)
        if detail:
            wpx = txt.get_window_extent(c.renderer).width
            ax.annotate(detail, xy=(0, yh), xycoords=ax.get_yaxis_transform(),
                        xytext=op(-(left - 24) + wpx + 8, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=12 * PT, color=t["ink2"], family=FONT)


def axis_title(c, x_right, y, s):
    c.text(x_right, y, s, size=11, color="muted", ha="right")


def chart_per_generator(d, t, path):
    bar, gap, band = 9, 2, 42
    W, left, right, top = 900, 150, 40, 128
    rows, headers, seps, spans, avg_y, plot_h = grouped_layout(
        [("ForenSynths | GAN-based, 11 generators", FORENSYNTHS),
         ("UniverDiffu | diffusion-based, 8 generators", UNIVERDIFFU)], band, avg_band=50)
    H = top + plot_h + 50
    c = Canvas(t, W, H)
    c.header("Accuracy per generator",
             ["Accuracy (%) on each test generator; every model is trained on ProGAN images only. "
              "Bars start at 50%, the chance level.",
              "SAFAID and the control: mean over 3 training runs, whiskers ±1 standard "
              "deviation. CatAID: single published value."])
    c.legend([("s1", LABELS["safaid"], "square"), ("s2", LABELS["control"], "square"),
              ("s3", LABELS["cataid"], "square")], 24, 104)
    xlim = (50, 103)
    ax = c.axes(left, top, W - left - right, plot_h, xlim)
    grid(ax, t, [50, 60, 70, 80, 90, 100], baseline=50, spans=spans)
    offs = [-(bar + gap), 0, bar + gap]

    def group(yc, vals, offs=offs):
        (s, s_sd), (cc, c_sd), ca = vals
        for off, v, col in zip(offs, (s, cc, ca), ("s1", "s2", "s3")):
            hbar(ax, yc + off, 50, v, bar, t[col])
        whisker(ax, yc + offs[0], s - s_sd, s + s_sd, t["ink"])
        whisker(ax, yc + offs[1], cc - c_sd, cc + c_sd, t["ink"])

    for yc, g in rows:
        p = d["per_gen"][g]
        group(yc, ((p["safaid"], p["safaid_sd"]), (p["control"], p["control_sd"]), p["cataid"]))
        row_label(ax, yc, g, t)
    group_headers(c, ax, headers, left, t)
    for ys in seps:
        c.hline(24, W - right, top + ys, color="axis")
    a = d["avg"]
    offs = [-(bar + 5), 0, bar + 5]      # a little more air so the three labels do not touch
    group(avg_y, (a["safaid"], a["control"], a["cataid"]), offs)
    row_label(ax, avg_y, "Average (19)", t, weight="semibold")
    col_x = max(a["safaid"][0] + a["safaid"][1], a["control"][0] + a["control"][1], a["cataid"])
    value_label(ax, col_x, avg_y + offs[0], f"{a['safaid'][0]:.2f} ± {a['safaid'][1]:.2f}",
                t, weight="semibold", dx=10)
    value_label(ax, col_x, avg_y + offs[1], f"{a['control'][0]:.2f} ± {a['control'][1]:.2f}",
                t, dx=10)
    value_label(ax, col_x, avg_y + offs[2], f"{a['cataid']:.2f}", t, dx=10)
    axis_title(c, left + (W - left - right) * (100 - xlim[0]) / (xlim[1] - xlim[0]), H - 12,
               "Accuracy (%)")
    c.save(path)


def chart_headline(d, t, path):
    W, left, right, top, band = 900, 270, 190, 150, 50
    series = [("safaid", "s1", "SAFAID", "investigative prompt, 3 runs"),
              ("control", "s2", "Category prompt, same backbone", "control, 3 runs"),
              ("cataid", "s3", "CatAID, published", "Cai et al., 2025, single value")]
    plot_h = band * len(series)
    H = top + plot_h + 50
    c = Canvas(t, W, H)
    tt, df, p = d["welch"]
    diff = d["avg"]["safaid"][0] - d["avg"]["control"][0]
    c.header("Average accuracy over 19 generators",
             ["Accuracy (%) averaged over the 11 ForenSynths and 8 UniverDiffu generators. "
              "Dots: training runs; vertical tick: mean.",
              minus(f"SAFAID is {diff:+.2f} points above category prompting on the same backbone "
                    f"and above the published CatAID.")])
    c.legend([("s1", LABELS["safaid"], "dot"), ("s2", LABELS["control"], "dot"),
              ("s3", LABELS["cataid"], "dot")], 24, 110)
    ax = c.axes(left, top, W - left - right, plot_h, (82, 93))
    grid(ax, t, [82, 84, 86, 88, 90, 92])
    for i, (key, col, name, sub) in enumerate(series):
        yc = band * i + band / 2
        ax.annotate(name, xy=(0, yc), xycoords=ax.get_yaxis_transform(),
                    xytext=op(-(left - 24), 1), textcoords="offset points", ha="left",
                    va="bottom", fontsize=12.5 * PT, color=t["ink"], family=FONT)
        ax.annotate(sub, xy=(0, yc), xytext=op(-(left - 24), -1),
                    xycoords=ax.get_yaxis_transform(), textcoords="offset points", ha="left",
                    va="top", fontsize=11 * PT, color=t["ink2"], family=FONT)
        if key == "cataid":
            m = d["avg"]["cataid"]
            dot(ax, m, yc, t[col], t, size=12)
            lab = f"{m:.2f}"
        else:
            m, sd = d["avg"][key]
            ax.plot([m - sd, m + sd], [yc, yc], color=t[col], lw=2 * PT, alpha=0.35, zorder=2,
                    solid_capstyle="round")
            ax.plot([m, m], [yc - 12, yc + 12], color=t[col], lw=2.5 * PT, zorder=4,
                    solid_capstyle="round")
            for v in d["seeds"][key]:
                dot(ax, v, yc, t[col], t)
            lab = f"{m:.2f} ± {sd:.2f}"
        ax.annotate(lab, xy=(1, yc), xycoords=ax.get_yaxis_transform(), xytext=op(28, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=13 * PT,
                    color=t["ink"], fontweight="semibold", family=FONT)
    c.text(W - right + 28, top - 10, "mean ± std", size=11, color="muted")
    axis_title(c, W - right, H - 12, "Average accuracy (%)")
    c.save(path)


def chart_ablations(d, t, path):
    W, left, right, top, band, bar = 900, 290, 60, 110, 52, 20
    base = d["ref"]["avg"]
    rows = [("SAFAID (run 1)", "investigative prompt, fake-only augmentation", base, None)]
    rows += [(label, detail, v, v - base) for label, detail, v in d["ablations"]]
    plot_h = band * len(rows)
    H = top + plot_h + 50
    c = Canvas(t, W, H)
    c.header("Ablations",
             ["Average accuracy over 19 generators (%); single runs with the seed of SAFAID run 1, same "
              "learning-rate schedule. Bars start at 50%, the chance level.",
              "Labels give the change from SAFAID run 1 in percentage points; the vertical rule "
              "marks run 1."])
    ax = c.axes(left, top, W - left - right, plot_h, (50, 100))
    grid(ax, t, [50, 60, 70, 80, 90, 100], baseline=50)
    ax.plot([base, base], [0, plot_h], color=t["axis"], lw=1 * PT, zorder=1,
            solid_capstyle="butt")
    for i, (label, detail, v, delta) in enumerate(rows):
        yc = band * i + band / 2
        hbar(ax, yc, 50, v, bar, t["s1"])
        ax.annotate(label, xy=(0, yc), xycoords=ax.get_yaxis_transform(),
                    xytext=op(-(left - 24), 1), textcoords="offset points", ha="left",
                    va="bottom", fontsize=12.5 * PT, color=t["ink"],
                    fontweight="semibold" if delta is None else "normal", family=FONT)
        ax.annotate(detail, xy=(0, yc), xycoords=ax.get_yaxis_transform(),
                    xytext=op(-(left - 24), -1), textcoords="offset points", ha="left",
                    va="top", fontsize=11 * PT, color=t["ink2"], family=FONT)
        lab = f"{v:.2f}" if delta is None else minus(f"{v:.2f}  ({delta:+.2f})")
        value_label(ax, max(v, base), yc, lab, t, size=12.5,
                    weight="semibold" if delta is None else "normal", dx=10)
    axis_title(c, W - right, H - 12, "Average accuracy (%)")
    c.save(path)


def chart_qlora(d, t, path):
    W, top = 900, 150
    bar, gap, band = 14, 2, 56
    plot_h = band * 2
    H = top + plot_h + 50
    c = Canvas(t, W, H)
    c.header("Full fine-tuning vs QLoRA",
             [("Same seed, data, prompt and schedule. " if d["full"]["paired"] else
               f"Same data, prompt and schedule; full fine-tuning is the mean of {d['full']['n']} runs. ")
              + "Full fine-tuning: all 4.44B parameters in bf16, LR 5e-6.",
              "QLoRA: 4-bit base, rank 16, alpha 16, 39.3M trainable parameters, LR 2e-4."])
    c.legend([("s1", "Full fine-tuning", "square"), ("s2", "QLoRA", "square")], 24, 104)
    # (a) accuracy
    la, wa = 140, 330
    ax = c.axes(la, top, wa, plot_h, (50, 100))
    grid(ax, t, [50, 60, 70, 80, 90, 100], baseline=50)
    c.text(24, top - 14, "(a) Accuracy (%)", size=12.5, weight="semibold")
    offs = [-(bar + gap) / 2, (bar + gap) / 2]
    for i, (name, full, q) in enumerate([("Average (19)", d["full"]["avg"], d["qlora"]["avg"]),
                                         ("SAN", d["full"]["san"], d["qlora"]["san"])]):
        yc = band * i + band / 2
        for off, v, col in zip(offs, (full, q), ("s1", "s2")):
            hbar(ax, yc + off, 50, v, bar, t[col])
            value_label(ax, v, yc + off, f"{v:.2f}", t, size=11.5)
        row_label(ax, yc, name, t)
    axis_title(c, la + wa, H - 12, "Accuracy (%)")
    # (b) peak memory: its own panel and axis (never a second scale on panel a)
    lb, wb = 610, 230
    bx = c.axes(lb, top, wb, plot_h, (0, 50))
    grid(bx, t, [0, 10, 20, 30, 40, 50], baseline=0)
    c.text(lb - 90, top - 14, "(b) Peak training memory (GB)", size=12.5, weight="semibold")
    for i, (name, v, col) in enumerate([("Full FT", d["mem"]["full"], "s1"),
                                        ("QLoRA", d["mem"]["qlora"], "s2")]):
        yc = band * i + band / 2
        hbar(bx, yc, 0, v, bar, t[col])
        value_label(bx, v, yc, f"{v:.1f} GB", t, size=11.5)
        row_label(bx, yc, name, t)
    axis_title(c, lb + wb, H - 12, "Peak allocated memory (GB)")
    c.save(path)


def chart_real_vs_fake(d, t, path):
    W, left, right, top, band = 900, 150, 60, 128, 30
    rows, headers, seps, spans, avg_y, plot_h = grouped_layout(
        [("ForenSynths", FORENSYNTHS), ("UniverDiffu", UNIVERDIFFU)], band, avg_band=38)
    H = top + plot_h + 50
    c = Canvas(t, W, H)
    c.header("Real vs AI-generated images (SAFAID)",
             ["Accuracy (%) on the real and on the AI-generated test images of each generator, "
              "mean over 3 runs. Axis starts at 40%.",
              "UniverDiffu generators other than ADM share the same LAION real images, "
              "so their real-image accuracy is identical."])
    c.legend([("s1", "Real images", "dot"), ("s2", "AI-generated images", "dot")], 24, 104)
    xlim = (40, 101)
    ax = c.axes(left, top, W - left - right, plot_h, xlim)
    grid(ax, t, [40, 50, 60, 70, 80, 90, 100], spans=spans)
    lowest = min(GENERATORS, key=lambda g: d["per_gen"][g]["fake"])

    def pair(yc, real, fake, size=10):
        ax.plot([fake, real], [yc, yc], color=t["muted"], lw=2 * PT, zorder=2,
                solid_capstyle="butt")
        dot(ax, real, yc, t["s1"], t, size=size)
        dot(ax, fake, yc, t["s2"], t, size=size)

    for yc, g in rows:
        p = d["per_gen"][g]
        pair(yc, p["real"], p["fake"])
        row_label(ax, yc, g, t)
        if g == lowest:
            value_label(ax, p["fake"], yc, f"{p['fake']:.1f}", t, dx=-12, ha="right")
    group_headers(c, ax, headers, left, t)
    for ys in seps:
        c.hline(24, W - right, top + ys, color="axis")
    a = d["avg"]
    pair(avg_y, a["real"], a["fake"], size=12)
    row_label(ax, avg_y, "Average (19)", t, weight="semibold")
    value_label(ax, a["fake"], avg_y, f"{a['fake']:.1f}", t, weight="semibold", dx=-14, ha="right")
    value_label(ax, a["real"], avg_y, f"{a['real']:.1f}", t, weight="semibold", dx=14)
    axis_title(c, left + (W - left - right) * (100 - xlim[0]) / (xlim[1] - xlim[0]), H - 12,
               "Accuracy (%)")
    c.save(path)


CHARTS = {
    "per_generator": chart_per_generator,
    "headline": chart_headline,
    "ablations": chart_ablations,
    "qlora": chart_qlora,
    "real_vs_fake": chart_real_vs_fake,
}


def write_table(d, path: Path):
    """The table view of every plotted value."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chart", "row", "series", "value", "sd", "note"])
        for g in GENERATORS:
            p = d["per_gen"][g]
            w.writerow(["per_generator", g, "safaid", f"{p['safaid']:.2f}", f"{p['safaid_sd']:.2f}",
                        "mean/sd over runs " + ",".join(SAFAID_RUNS)])
            w.writerow(["per_generator", g, "control", f"{p['control']:.2f}",
                        f"{p['control_sd']:.2f}", "mean/sd over runs " + ",".join(CONTROL_RUNS)])
            w.writerow(["per_generator", g, "cataid_published", f"{p['cataid']:.2f}", "",
                        "Cai et al. 2025, Table 1"])
        a = d["avg"]
        for key in ("safaid", "control"):
            for rid, v in zip(SAFAID_RUNS if key == "safaid" else CONTROL_RUNS, d["seeds"][key]):
                w.writerow(["headline", "average_19", f"{key}_{rid}", f"{v:.2f}", "", ""])
            w.writerow(["headline", "average_19", f"{key}_mean", f"{a[key][0]:.2f}",
                        f"{a[key][1]:.2f}", "sample sd (ddof=1)"])
        w.writerow(["headline", "average_19", "cataid_published", f"{a['cataid']:.2f}", "", ""])
        tt, df, p = d["welch"]
        w.writerow(["headline", "welch_safaid_vs_control", "t", f"{tt:.3f}", "",
                    f"df={df:.2f}, two-sided p={p:.3f}; SAFAID higher on {d['higher_on']}/19"])
        w.writerow(["ablations", "SAFAID run 1", SAFAID_RUNS[0], f"{d['ref']['avg']:.2f}", "", ""])
        for (rid, _, _), (label, _, v) in zip(ABLATION_RUNS, d["ablations"]):
            w.writerow(["ablations", label, rid, f"{v:.2f}", "",
                        f"change vs run 1: {v - d['ref']['avg']:+.2f}"])
        full_tag = "full_ft_same_seed" if d["full"]["paired"] else f"full_ft_mean_of_{d['full']['n']}_runs"
        w.writerow(["qlora", "average_19", full_tag, f"{d['full']['avg']:.2f}", "", ""])
        w.writerow(["qlora", "average_19", f"qlora_{QLORA_RUN}", f"{d['qlora']['avg']:.2f}", "", ""])
        w.writerow(["qlora", "SAN", full_tag, f"{d['full']['san']:.2f}", "", ""])
        w.writerow(["qlora", "SAN", f"qlora_{QLORA_RUN}", f"{d['qlora']['san']:.2f}", "", ""])
        w.writerow(["qlora", "peak_mem_gb", "full_ft", f"{d['mem']['full']:.2f}", "",
                    "peak allocated, config.json"])
        w.writerow(["qlora", "peak_mem_gb", f"qlora_{QLORA_RUN}", f"{d['mem']['qlora']:.2f}", "",
                    "peak allocated, config.json"])
        for g in GENERATORS:
            p = d["per_gen"][g]
            w.writerow(["real_vs_fake", g, "real_acc", f"{p['real']:.2f}", "", "SAFAID, mean of 3 runs"])
            w.writerow(["real_vs_fake", g, "fake_acc", f"{p['fake']:.2f}", "", "SAFAID, mean of 3 runs"])
        w.writerow(["real_vs_fake", "average_19", "real_acc", f"{a['real']:.2f}", "", ""])
        w.writerow(["real_vs_fake", "average_19", "fake_acc", f"{a['fake']:.2f}", "", ""])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs_root", type=Path, default=REPO / "results",
                    help="directory holding the run folders (default: results/)")
    ap.add_argument("--out_dir", type=Path, default=HERE, help="where to write the PNGs")
    ap.add_argument("--only", nargs="*", choices=sorted(CHARTS), help="render only these charts")
    args = ap.parse_args()
    global FONT
    FONT = available_fonts(FONT)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # semibold -> bold
    plt.rcParams.update({"font.family": FONT, "axes.unicode_minus": True,
                         "savefig.transparent": False})
    d = collect(args.runs_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in CHARTS.items():
        if args.only and name not in args.only:
            continue
        for mode, t in THEMES.items():
            out = args.out_dir / f"{name}_{mode}.png"
            fn(d, t, out)
            print("wrote", out.relative_to(args.out_dir.parent) if out.is_relative_to(
                args.out_dir.parent) else out)
    write_table(d, args.out_dir / "chart_data.csv")
    a = d["avg"]
    tt, df, p = d["welch"]
    print(f"SAFAID {a['safaid'][0]:.2f} +- {a['safaid'][1]:.2f} | control {a['control'][0]:.2f} +- "
          f"{a['control'][1]:.2f} | CatAID {a['cataid']:.2f} | higher on {d['higher_on']}/19 | "
          f"Welch t={tt:.2f} df={df:.2f} p={p:.3f} | real {a['real']:.1f} fake {a['fake']:.1f}")


if __name__ == "__main__":
    main()
