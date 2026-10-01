#!/usr/bin/env python3
"""Procedural artwork of the SAFAID cover: an iris under a scan line, organic on the left half and
rebuilt from tiles on the right half. No photograph or dataset image is used; everything is generated
from a fixed random seed.

    python assets/src/make_cover_art.py            # writes assets/src/cover_art.png (2560 x 1280)

Requires numpy and Pillow.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
S = 2                                   # pixels per CSS px
W, H = 1280 * S, 640 * S
CX, CY, R = 975 * S, 320 * S, 272 * S   # iris centre and radius
BG = np.array([10, 11, 14], np.float32) / 255
CREAM = np.array([244, 244, 240], np.float32) / 255
VERM = np.array([240, 67, 26], np.float32) / 255
rng = np.random.default_rng(11)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def iris():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx - CX, yy - CY
    r = np.hypot(dx, dy) / R
    th = np.arctan2(dy, dx)

    # radial fibres: many angular frequencies, slightly twisted along the radius
    f = np.zeros_like(r)
    for n, a in [(31, 0.9), (53, 0.85), (89, 0.8), (139, 0.7), (211, 0.6), (307, 0.5), (433, 0.4), (601, 0.3), (811, 0.2)]:
        ph, tw, wob = rng.uniform(0, 6.283), rng.uniform(-5, 5), rng.uniform(0.4, 1.6)
        f += a * np.cos(n * th + ph + tw * r + wob * np.sin(5 * th + ph) * r)
    f = (f - f.min()) / (f.max() - f.min())
    rings = 0.5 + 0.5 * np.cos(34 * r + 1.6 * np.sin(7 * th + 1.3) + 0.8 * np.sin(13 * th))
    inten = np.clip(0.18 + 1.05 * f ** 1.7 * (0.62 + 0.38 * rings), 0, 1)

    # colour ramp along the radius: bright collarette -> orange -> vermilion -> deep red -> dark limbus
    stops_r = np.array([0.26, 0.36, 0.52, 0.70, 0.88, 1.00], np.float32)
    stops_c = np.array([[1.00, 0.90, 0.70], [1.00, 0.62, 0.22], [0.96, 0.33, 0.11], [0.80, 0.18, 0.06],
                        [0.40, 0.07, 0.03], [0.10, 0.02, 0.02]], np.float32)
    col = np.stack([np.interp(r, stops_r, stops_c[:, k]) for k in range(3)], -1)
    rgb = col * (0.22 + 0.95 * inten)[..., None]
    rgb += (inten ** 7)[..., None] * CREAM * 0.35 * smoothstep(0.95, 0.4, r)[..., None]      # bright fibres
    rgb += (np.exp(-((r - 0.30) / 0.035) ** 2) * 0.35)[..., None] * np.array([1.0, 0.8, 0.5], np.float32)  # collarette

    pupil = smoothstep(0.262, 0.238, r)
    rgb = rgb * (1 - pupil)[..., None] + np.array([0.016, 0.017, 0.022], np.float32) * pupil[..., None]
    alpha = smoothstep(1.006, 0.994, r)
    img = BG * (1 - alpha)[..., None] + rgb * alpha[..., None]
    # halo just outside the iris
    img += (np.exp(-(np.clip(r - 1.0, 0, None) / 0.085) ** 2) * 0.20 * (r > 1.0))[..., None] * VERM

    # catchlight: a soft window reflection on the upper left of the pupil
    ux, uy = (dx + 0.20 * R) / (0.115 * R), (dy + 0.19 * R) / (0.075 * R)
    c, s = np.cos(-0.5), np.sin(-0.5)
    g = np.exp(-((ux * c - uy * s) ** 2 + (ux * s + uy * c) ** 2) ** 1.6)
    img = img + (CREAM - img) * (0.92 * g)[..., None]
    g2 = np.exp(-(((dx - 0.13 * R) / (0.035 * R)) ** 2 + ((dy - 0.15 * R) / (0.035 * R)) ** 2))
    img = img + (CREAM - img) * (0.45 * g2)[..., None]
    return np.clip(img, 0, 1), r


def mosaic(img, r):
    """Right half: the image rebuilt from square tiles; tiles drift away past the edge."""
    out = img.copy()
    out[:, CX:] = BG
    t, gap = 22 * S, 2 * S
    x0s = np.arange(CX + gap // 2, W, t)
    y0s = np.arange((CY % t) - t, H, t)
    for y0 in y0s:
        for x0 in x0s:
            xc, yc = x0 + t / 2, y0 + t / 2
            rr = np.hypot(xc - CX, yc - CY) / R
            u = rng.random()
            if rr <= 1.0:
                ya, yb, xa, xb = max(int(y0), 0), min(int(y0 + t), H), int(x0), min(int(x0 + t), W)
                if yb <= ya or xb <= xa:
                    continue
                m = img[ya:yb, xa:xb].reshape(-1, 3).mean(0)
                lum = float(m @ np.array([0.2126, 0.7152, 0.0722], np.float32))
                if rr < 0.25:
                    c = BG + (CREAM - BG) * 0.05                      # pupil tiles: just visible
                elif u < 0.11:
                    c = np.clip(m * 1.25, 0, 1)                       # kept in colour
                elif u < 0.16:
                    continue                                          # missing tile
                else:
                    c = BG + (CREAM - BG) * np.clip(lum * 2.1 + 0.04, 0, 1) ** 0.9
                size = t - gap
            elif rr <= 1.9 and u < 0.62 * np.exp(-(rr - 1.0) / 0.26):
                k = np.clip((rr - 1.0) / 0.9, 0, 1)
                size = int((t - gap) * (1 - 0.62 * k))
                a = 0.85 * np.exp(-(rr - 1.0) / 0.34)
                c = BG + ((VERM if rng.random() < 0.3 else CREAM) - BG) * a
            else:
                continue
            off = (t - gap - size) // 2
            ya, xa = int(y0 + gap // 2 + off), int(x0 + off)
            out[max(ya, 0):min(ya + size, H), xa:min(xa + size, W)] = c
    return out


def overlays(img, r):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # scan line through the centre: bright core and a warm glow, fading toward the top and bottom edges
    fade = smoothstep(1.32, 0.95, np.abs(yy - CY) / R) * 0.85 + 0.15
    glow = np.exp(-((xx - CX) / (15 * S)) ** 2) * 0.55 * fade
    img = img + glow[..., None] * np.array([1.0, 0.42, 0.14], np.float32)
    core = np.exp(-((xx - CX) / (1.1 * S)) ** 2) * fade
    img = img + (CREAM - img) * core[..., None]

    # reticle and faint background specks, drawn with Pillow on a transparent layer
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cream = (244, 244, 240)
    rr = 1.085 * R
    d.ellipse([CX - rr, CY - rr, CX + rr, CY + rr], outline=cream + (120,), width=S)
    for deg in range(0, 360, 3):
        a = np.deg2rad(deg)
        long = deg % 30 == 0
        r1, r2 = rr + 3 * S, rr + (13 if long else 7) * S
        d.line([CX + r1 * np.cos(a), CY + r1 * np.sin(a), CX + r2 * np.cos(a), CY + r2 * np.sin(a)],
               fill=cream + (170 if long else 80,), width=S)
    for _ in range(520):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        if np.hypot(x - CX, y - CY) < 1.2 * R:
            continue
        s = rng.choice([1, 1, 1, 2, 2, 3])
        d.ellipse([x, y, x + s, y + s], fill=cream + (int(rng.uniform(18, 95)),))
    out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    out = Image.alpha_composite(out, layer).convert("RGB")
    # soft vignette
    v = 1 - 0.42 * smoothstep(0.55, 1.25, np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)))
    out = Image.fromarray((np.asarray(out).astype(np.float32) * v[..., None]).astype(np.uint8))
    return out


if __name__ == "__main__":
    base, r = iris()
    art = overlays(mosaic(base, r), r)
    art.save(HERE / "cover_art.png", optimize=True)
    print("wrote", HERE / "cover_art.png", art.size)
