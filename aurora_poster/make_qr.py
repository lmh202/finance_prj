"""
make_qr.py — turn the poster's QR placeholders into real, printable codes.

    pip install segno
    python make_qr.py

Fill in TARGETS below, run it, and the SVGs land in assets/qr/. poster.html
picks them up by filename; a target left as None leaves the dashed placeholder
in place, so this is safe to run before every URL exists.

Why SVG and not PNG: the code prints at 30mm. A raster that small needs ~355px
just to reach 300dpi, and any resampling softens the module edges — which is
exactly what a phone camera needs to be crisp. SVG is resolution-free and
Chrome's PDF export keeps it vector.

Why error correction M: the poster is printed, flat and well lit, so the extra
redundancy of Q or H buys nothing and costs modules. Fewer modules means each
module is physically bigger, which is what actually drives scan distance.

Rule of thumb for scan distance: roughly 10x the code's width. At 30mm a phone
locks on from ~30cm, which is the distance someone stands to read a panel of an
A1 poster anyway. Prefer a short link to a long one — a shorter URL produces a
lower QR version, i.e. fewer and therefore larger modules.
"""

from pathlib import Path

import segno

HERE = Path(__file__).parent
OUT = HERE / "assets" / "qr"

# filename stem -> URL. None leaves the dashed placeholder in place.
#
# `repo` is the only one poster.html currently places. If the team deploys the
# Next.js frontend or records a walkthrough, add `site` / `demo` here and drop
# the matching <div class="qr"> into the close panel — the CSS for a three-up
# row is already there.
TARGETS = {
    "repo": "https://github.com/lmh202/finance_prj",
    "site": None,
    "demo": None,
}

# Dark modules on a light tile, which is the polarity the QR spec assumes.
#
# The reference poster learned this the expensive way: its first version drew
# the modules in the poster's ink on a transparent tile so the codes sat on the
# glass panel and read as part of the type. Rendered at 300dpi and fed to
# OpenCV, all three failed as printed and decoded only after the image was
# inverted. Phone cameras are often more forgiving than that, but "often" is
# not a property a printed poster can rely on.
#
# So the colours are the right way round, and both come from AURORA's own
# palette (--bg0 modules on an --ink tile) rather than pure black on pure
# white: correct polarity, ~16:1 contrast, still the poster's ink.
DARK = "#06090c"
LIGHT = "#ecf3ee"

# The physical width the code is placed at, from poster.css `.qr img`. Only
# used to print the per-module size, which is the number that decides whether
# the thing scans.
PLACED_MM = 30


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, url in TARGETS.items():
        if not url:
            print(f"  {stem:6s} skipped (no URL yet)")
            continue
        qr = segno.make(url, error="m")
        path = OUT / f"{stem}.svg"
        qr.save(path, scale=10, border=2, dark=DARK, light=LIGHT)
        modules = qr.symbol_size(border=2)[0]
        print(
            f"  {stem:6s} v{qr.version:<2d} {modules} modules  "
            f"{PLACED_MM / modules:.2f}mm per module at {PLACED_MM}mm  ->  {path.name}"
        )


if __name__ == "__main__":
    main()
