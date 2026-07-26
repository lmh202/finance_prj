"""
prepare_background.py — bake the two bitmap textures the poster sits on.

    python prepare_background.py

Writes, both under assets/:

    poster_background.jpg   3000 x 4247, the full A1 canvas wash
    headline-wall.png       1600 x 700, the texture inside the PROBLEM panel

Why baked bitmaps and not CSS
-----------------------------
Everything here could be expressed as CSS gradients plus a blur filter, and it
was, once. Chrome's --print-to-pdf pipeline responds to `filter: blur()` on a
full-page element by rasterising the *entire* canvas at print resolution: the
exported PDF went from ~3 MB to ~77 MB and every vector glyph on the poster
became part of one giant image. So the blur happens here, offline, and the CSS
only ever places a finished bitmap. Same constraint the HDBrain poster hit; see
../hdbrain_poster/export.js.

Both textures are drawn from THIS project's real data, not stock art:

  * the wash is ~140 real cumulative-return curves out of
    data/processed/wide_price_panel.parquet (2013-2023 FNSPID panel, the same
    file the risk-engine studies read), heavily blurred until no individual
    series is legible and only the density survives;
  * the headline wall is real fetched headlines out of data/news_raw.json,
    the RSS cache news_intelligence/collector.py grows.

If either file is missing (both are gitignored runtime/derived state on some
checkouts) the script falls back to a seeded synthetic stand-in and says so, so
this never becomes a hard blocker for someone rebuilding the poster.

Needs Pillow, numpy, scipy and pandas. pandas is in backend/requirements.txt;
the other three are not — install them ad hoc.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).parent
REPO = HERE.parent
OUT = HERE / "assets"

PANEL_PARQUET = REPO / "data" / "processed" / "wide_price_panel.parquet"
NEWS_JSON = REPO / "data" / "news_raw.json"

# A1 at the same pixel size as the reference poster's background. 3000 px over
# 594 mm is ~128 dpi, which would be far too coarse for type and is plenty for
# an image whose highest spatial frequency has been blurred away on purpose.
W, H = 3000, 4247

# Straight out of frontendjs/src/app/globals.css, dark theme. The poster and
# the product are meant to read as one thing, so nothing here is re-picked by
# eye.
BG0 = (0x06, 0x09, 0x0C)
ACCENT = (0xB3, 0xF3, 0x4C)  # --accent, the light green
TEAL = (0x5E, 0xEA, 0xD4)  # AuroraMark's second refraction ray
SPY = (0x8D, 0xA2, 0xFB)  # --spy
QQQ = (0xCF, 0xA3, 0xFF)  # --qqq
INK = (0xEC, 0xF3, 0xEE)  # --ink


# ---------------------------------------------------------------------------
# 1. The canvas wash
# ---------------------------------------------------------------------------
def load_curves(n: int = 140, length: int = 900) -> list[np.ndarray]:
    """Real cumulative-return paths, normalised to 0..1, one per symbol-slice."""
    rng = np.random.default_rng(20260726)

    if PANEL_PARQUET.exists():
        import pandas as pd

        panel = pd.read_parquet(PANEL_PARQUET, columns=["date", "symbol", "ret_1d"])
        panel = panel.dropna(subset=["ret_1d"])
        curves: list[np.ndarray] = []
        for _, grp in panel.groupby("symbol", sort=True):
            r = grp.sort_values("date")["ret_1d"].to_numpy(dtype=float)
            if r.size < length // 2:
                continue
            # Several overlapping windows per symbol: 21 symbols is not enough
            # density on its own, and a window is still a real price path.
            for _ in range(max(1, n // 18)):
                if r.size <= length:
                    seg = r
                else:
                    start = int(rng.integers(0, r.size - length))
                    seg = r[start : start + length]
                curves.append(np.cumsum(np.log1p(np.clip(seg, -0.5, 4.0))))
        rng.shuffle(curves)
        curves = curves[:n]
        if curves:
            print(f"  wash    {len(curves)} real return paths from {PANEL_PARQUET.name}")
            return [_norm(c) for c in curves]

    print("  wash    ! price panel not found, using seeded synthetic paths")
    curves = []
    for _ in range(n):
        drift = rng.normal(0.0004, 0.0006)
        vol = rng.uniform(0.008, 0.026)
        curves.append(np.cumsum(rng.normal(drift, vol, length)))
    return [_norm(c) for c in curves]


def _norm(c: np.ndarray) -> np.ndarray:
    lo, hi = float(c.min()), float(c.max())
    return (c - lo) / (hi - lo) if hi > lo else np.zeros_like(c)


def radial(shape: tuple[int, int], cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    """A soft elliptical falloff in 0..1, the same shape as a CSS radial-gradient."""
    h, w = shape
    y, x = np.ogrid[0:h, 0:w]
    d = np.sqrt(((x - cx * w) / (rx * w)) ** 2 + ((y - cy * h) / (ry * h)) ** 2)
    return np.clip(1.0 - d, 0.0, 1.0) ** 2


def build_wash() -> Image.Image:
    rng = random.Random(20260726)
    curves = load_curves()

    # Curves are drawn at 1/2 scale and upscaled afterwards. Blurring a
    # half-size buffer and then enlarging with LANCZOS gives a smoother,
    # cheaper falloff than blurring at full size with a huge radius — the same
    # upscale-then-blur trick the reference poster's background uses, in the
    # other order.
    sw, sh = W // 2, H // 2
    layer = Image.new("RGB", (sw, sh), (0, 0, 0))
    d = ImageDraw.Draw(layer)

    # A faint session grid under the curves, so the texture reads as a chart
    # surface rather than as noise.
    for i in range(1, 26):
        y = sh * i / 26
        d.line([(0, y), (sw, y)], fill=(10, 16, 20), width=1)

    # Weighted toward the accent green: the poster's own hue should be the
    # one the eye picks out of the field, with the AuroraMark's other
    # refraction rays only tinting it.
    palette = [ACCENT, ACCENT, TEAL, ACCENT, SPY, QQQ, INK]
    for i, c in enumerate(curves):
        # Each path gets its own band of the canvas and its own amplitude, so
        # the field fills top to bottom instead of piling up on one baseline.
        # The bands are deliberately tall and the strokes deliberately fat: at
        # this blur radius a thin line vanishes and a whole screenful of them
        # averages into flat grey. Fewer, broader ribbons keep the sense of
        # depth the panels then sit in front of.
        band_h = sh * rng.uniform(0.22, 0.62)
        top = rng.uniform(-0.10, 1.10) * sh - band_h / 2
        xs = np.linspace(-0.04 * sw, 1.04 * sw, c.size)
        ys = top + (1.0 - c) * band_h
        hue = palette[i % len(palette)]
        # Most paths sit near the floor of visibility; a handful are allowed to
        # come forward. The blur below turns that spread into depth.
        k = rng.choice([0.05, 0.07, 0.09, 0.10, 0.18])
        col = tuple(int(v * k) for v in hue)
        d.line(list(zip(xs.tolist(), ys.tolist())), fill=col, width=rng.choice([5, 7, 9, 14]))

    layer = layer.filter(ImageFilter.GaussianBlur(9.0))
    layer = layer.resize((W, H), Image.LANCZOS)

    arr = np.asarray(layer, dtype=np.float32)
    # A second, much wider blur added back on top: the curves' own light,
    # bloomed. This is what stops the field from looking like a screenshot of a
    # chart and makes it look like light behind glass.
    bloom = gaussian_filter(arr, sigma=(60, 60, 0))
    arr = arr + bloom * 2.6

    # frontendjs's body background is three radial washes over --bg0. Same three,
    # same corners, same hues — see globals.css.
    base = np.zeros((H, W, 3), dtype=np.float32)
    base += np.asarray(BG0, dtype=np.float32)
    for cx, cy, rx, ry, hue, k in [
        (0.86, -0.06, 0.74, 0.36, ACCENT, 62.0),
        (-0.08, 0.17, 0.62, 0.44, SPY, 34.0),
        (0.50, 1.14, 0.66, 0.44, ACCENT, 46.0),
        (0.14, 0.66, 0.50, 0.28, TEAL, 22.0),
        (0.92, 0.78, 0.44, 0.28, QQQ, 16.0),
    ]:
        base += radial((H, W), cx, cy, rx, ry)[..., None] * np.asarray(hue, dtype=np.float32) * (k / 255.0)

    out = base + arr * 0.24

    # Vignette, then a flat scrim. Flat rather than a full-canvas gradient:
    # large smooth dark gradients are the classic cause of banding on a poster
    # printer, and the panels supply their own contrast anyway.
    y, x = np.ogrid[0:H, 0:W]
    vig = 1.0 - 0.42 * np.clip(
        (((x - W / 2) / (W * 0.62)) ** 2 + ((y - H / 2) / (H * 0.66)) ** 2), 0, 1
    )
    out *= vig[..., None]

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# 2. The headline wall
# ---------------------------------------------------------------------------
def pick_font(size: int) -> ImageFont.FreeTypeFont:
    """A monospaced face if the machine has one; the feed should read as a feed."""
    for name in ("consola.ttf", "cour.ttf", "segoeui.ttf", "arial.ttf"):
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size)


def load_headlines(n: int = 260) -> list[str]:
    if NEWS_JSON.exists():
        raw = json.loads(NEWS_JSON.read_text(encoding="utf-8"))
        titles = [str(a.get("title", "")).strip() for a in raw if a.get("title")]
        titles = [t for t in titles if 24 <= len(t) <= 96]
        if titles:
            random.Random(7).shuffle(titles)
            print(f"  wall    {min(n, len(titles))} real headlines from {NEWS_JSON.name}")
            return (titles * ((n // max(1, len(titles))) + 1))[:n]

    print("  wall    ! news cache not found, using generic stand-in strings")
    return [f"MARKET WIRE {i:04d} — headline text unavailable offline" for i in range(n)]


def build_wall(w: int = 1600, h: int = 700) -> Image.Image:
    """Dense, unreadable-by-design headline texture with an alpha ramp.

    It has to survive being sat on: the PROBLEM panel's own body copy runs over
    the top of it, so the wall fades to nothing down the left two-thirds and
    keeps its density on the right, where the panel has no text.
    """
    heads = load_headlines()
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = pick_font(15)

    rng = random.Random(11)
    y, i = -8, 0
    while y < h:
        x = -rng.randint(0, 180)
        while x < w:
            t = heads[i % len(heads)]
            i += 1
            # Two ink levels only. A third made the field read as three
            # separate layers instead of one wall of text. The tint is --ink
            # scaled, so the wall is the same colour as the copy over it.
            g = rng.choice([0.62, 0.62, 0.82, 1.0])
            d.text((x, y), t, font=font, fill=(*(int(v * g) for v in INK), 255))
            x += int(d.textlength(t, font=font)) + rng.randint(26, 90)
        y += 21
        # every ~7th row is an accent-green line: the few stories that actually
        # matter, which is the whole point of the panel beside it.
        if rng.random() < 0.14:
            d.line([(0, y - 4), (w, y - 4)], fill=(*ACCENT, 26), width=1)

    a = np.asarray(img, dtype=np.float32)

    # Horizontal ramp: transparent at the left edge where the copy sits,
    # strongest at the right. Vertical ramp: fade out at top and bottom so the
    # texture never collides with the panel's rounded corners.
    xr = np.clip((np.arange(w) - w * 0.30) / (w * 0.62), 0, 1)[None, :] ** 1.35
    yr = np.clip(np.minimum(np.arange(h) / (h * 0.22), (h - np.arange(h)) / (h * 0.26)), 0, 1)[:, None]
    a[..., 3] *= xr * yr * 0.5

    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    wash = build_wash()
    p = OUT / "poster_background.jpg"
    wash.save(p, quality=88, subsampling=1, optimize=True)
    print(f"  wrote   {p.name}  {wash.size[0]}x{wash.size[1]}  {p.stat().st_size / 1e6:.2f} MB")

    wall = build_wall()
    p = OUT / "headline-wall.png"
    wall.save(p, optimize=True)
    print(f"  wrote   {p.name}  {wall.size[0]}x{wall.size[1]}  {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
