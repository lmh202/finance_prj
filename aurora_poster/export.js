/*
 * export.js — render poster.html to a print-ready A1 PDF.
 *
 *     node export.js
 *     node export.js --out proof.pdf
 *
 * Drives the Edge already installed on this machine. No puppeteer, no npm
 * install, no downloaded browser: Chrome's own --print-to-pdf honours the
 * @page rule in poster.css, which is all we need.
 *
 * Expected output: 594.1 x 841.0 mm, one page, text vector, and all four
 * JS-generated graphics present (ledger bars, risk bars, calibration rows,
 * and every [data-fig] substitution).
 *
 * Three things this depends on, all easy to break:
 *
 *   - @page { size: 594mm 841mm; margin: 0 } in poster.css sets the paper.
 *     Remove it and you get US Letter.
 *   - --virtual-time-budget must outlast the four self-hosted fonts and
 *     assets/figures.js. 9s is generous; if you add a slow asset, raise it,
 *     because a short budget prints a half-built page without any error.
 *   - The background must stay a pre-baked bitmap (see prepare_background.py).
 *     Reintroducing `filter: blur()` on it in CSS makes Chrome rasterise the
 *     whole canvas into one bitmap; on the reference poster that took the PDF
 *     from 2.6 MB to 77 MB.
 */
const { execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const EDGE =
  process.env.EDGE_PATH ||
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";

const outArg = process.argv.indexOf("--out");
const OUT = path.resolve(
  __dirname,
  outArg > -1 ? process.argv[outArg + 1] : "AURORA_poster_A1.pdf"
);
const PAGE = "file:///" + path.join(__dirname, "poster.html").replace(/\\/g, "/");

if (!fs.existsSync(EDGE)) {
  console.error("Edge not found at:", EDGE);
  console.error("Set EDGE_PATH to override.");
  process.exit(1);
}

fs.rmSync(OUT, { force: true });

try {
  execFileSync(
    EDGE,
    [
      "--headless=new",
      "--disable-gpu",
      "--no-sandbox",
      "--no-pdf-header-footer",
      "--virtual-time-budget=9000",
      `--print-to-pdf=${OUT}`,
      PAGE,
    ],
    { stdio: "ignore", timeout: 120000 }
  );
} catch (_) {
  // Edge exits non-zero / lingers on some machines even when the PDF is fine,
  // so the file check below is the real success test, not the exit code.
}

if (!fs.existsSync(OUT)) {
  console.error("No PDF produced. Try opening poster.html in a browser first.");
  process.exit(1);
}

console.log(
  "wrote %s (%s MB)",
  path.basename(OUT),
  (fs.statSync(OUT).size / 1e6).toFixed(1)
);
console.log("expected: 1 page, 594.1 x 841.0 mm");
