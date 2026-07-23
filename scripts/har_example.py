"""A worked, visual example of the HAR risk engine: what goes IN, what comes OUT.

Trains the same pooled HAR as train_risk_engine.py (identical features, same
train split), then walks through ONE stock on ONE day in plain numbers:

    inputs   6 features (log realized vol at 3 scales + 2 range estimators + |r|)
      |
    HAR      log(vol_next_5d) = b0 + b1*l_rv5 + ... + b6*l_absret
      |
    output   a forecast of next-5-day volatility, vs what actually happened

Then plots forecast-vs-realized over the test period so you can see it tracking.

Run: python scripts/har_example.py [SYMBOL]      (default NVDA)
"""

import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from train_risk_engine import FEATS_BASE, TRAIN_END, VAL_END, build_stock, log_feats, stocks

OUT = ROOT / "reports" / "risk_engine"
H = 5
SYM = (sys.argv[1] if len(sys.argv) > 1 else "NVDA").upper()

PRETTY = {
    "l_rv5":    "log realized vol, past 5d   (HAR 'weekly')",
    "l_rv22":   "log realized vol, past 22d  (HAR 'monthly')",
    "l_rv66":   "log realized vol, past 66d  (HAR 'quarterly')",
    "l_park5":  "log Parkinson range vol, 5d  (uses high/low)",
    "l_park22": "log Parkinson range vol, 22d (uses high/low)",
    "l_absret": "log |return| today           (today's shock)",
}


def main():
    print("training pooled HAR on 2013-2018 (same setup as the risk engine) …")
    panel = log_feats(pd.concat([build_stock(s) for s in stocks()], ignore_index=True))
    panel["ly"] = np.log(panel[f"y_{H}"] + 1e-6)
    d = panel.dropna(subset=FEATS_BASE + [f"y_{H}"]).copy()

    tr = d[d.date <= pd.Timestamp(TRAIN_END) - pd.Timedelta(days=30)]
    va = d[(d.date > TRAIN_END) & (d.date <= pd.Timestamp(VAL_END) - pd.Timedelta(days=30))]
    te = d[d.date > VAL_END]

    mdl = LinearRegression().fit(tr[FEATS_BASE], tr["ly"])
    corr = float(np.exp(np.var(va["ly"] - mdl.predict(va[FEATS_BASE]), ddof=1) / 2))

    # ---------------- the learned equation ----------------
    print("\n" + "=" * 74)
    print("THE TRAINED MODEL  —  log(vol over next 5 days) = ...")
    print("=" * 74)
    print(f"  intercept                                      {mdl.intercept_:+.4f}")
    for f, c in zip(FEATS_BASE, mdl.coef_):
        print(f"  {c:+.4f}  x  {PRETTY[f]}")
    print(f"\n  (then vol = exp(that) x {corr:.4f}  <- log->level smearing correction)")
    print("  Coefficients sum to ~1: the model is essentially a weighted blend of")
    print("  volatility measured over different lookbacks. That IS the HAR idea.")

    # ---------------- one concrete day ----------------
    sub = te[te.symbol == SYM].sort_values("date").reset_index(drop=True)
    if sub.empty:
        print(f"\n{SYM} has no test-period rows; try another symbol.")
        return
    row = sub.iloc[len(sub) // 2]           # a representative mid-test day

    print("\n" + "=" * 74)
    print(f"WORKED EXAMPLE  —  {SYM} on {row.date.date()}")
    print("=" * 74)
    print("INPUTS (what the model sees that morning):")
    print(f"  {'feature':<46} {'log value':>10} {'= vol/day':>10} {'contrib':>9}")
    total = mdl.intercept_
    for f, c in zip(FEATS_BASE, mdl.coef_):
        contrib = c * row[f]
        total += contrib
        print(f"  {PRETTY[f]:<46} {row[f]:>10.4f} {np.exp(row[f]):>10.4f} {contrib:>+9.4f}")
    print(f"  {'intercept':<46} {'':>10} {'':>10} {mdl.intercept_:>+9.4f}")
    print(f"  {'':<46} {'':>10} {'SUM =':>10} {total:>+9.4f}")

    fc_daily = float(np.exp(total) * corr)
    actual = float(row[f"y_{H}"])
    print("\nOUTPUT:")
    print(f"  forecast daily vol over next 5d : {fc_daily:.4f}  ({fc_daily*100:.2f}% per day, "
          f"{fc_daily*np.sqrt(252)*100:.1f}% annualized)")
    print(f"  ACTUAL  daily vol over next 5d : {actual:.4f}  ({actual*100:.2f}% per day)")
    print(f"  error                          : {(fc_daily-actual)*100:+.2f} pp/day "
          f"({(fc_daily/actual-1)*100:+.0f}% relative)")
    print(f"\n  For sizing: a 1.96-sigma band on the next 5-day move is "
          f"+/-{1.96*fc_daily*np.sqrt(H)*100:.1f}%")
    print(f"  What actually happened over those 5 days: {row[f'fret_{H}']*100:+.2f}%  "
          f"-> {'inside' if abs(row[f'fret_{H}']) <= 1.96*fc_daily*np.sqrt(H) else 'OUTSIDE'} the band")

    # ---------------- the picture ----------------
    sub["forecast"] = np.exp(mdl.predict(sub[FEATS_BASE])) * corr
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(sub.date, sub[f"y_{H}"] * 100, lw=1.2, color="#444", label="ACTUAL next-5d vol")
    ax[0].plot(sub.date, sub["forecast"] * 100, lw=1.6, color="#c0392b", label="HAR forecast")
    ax[0].axvline(row.date, color="#2980b9", ls=":", lw=1.5)
    ax[0].annotate(f"worked example\n{row.date.date()}", (row.date, ax[0].get_ylim()[1] * 0.92),
                   color="#2980b9", fontsize=8, ha="center")
    ax[0].set_ylabel("daily volatility (%)")
    ax[0].set_title(f"{SYM} — HAR risk engine on the held-out test period (2021-2023)")
    ax[0].legend(fontsize=9)

    band = 1.96 * sub["forecast"] * np.sqrt(H) * 100
    ax[1].fill_between(sub.date, -band, band, color="#c0392b", alpha=.18,
                       label="±1.96σ forecast band (5d)")
    ax[1].plot(sub.date, sub[f"fret_{H}"] * 100, lw=.9, color="#111", label="actual 5d return")
    ax[1].axhline(0, color="#888", lw=.6)
    ax[1].set_ylabel("5-day return (%)"); ax[1].set_xlabel("date")
    ax[1].legend(fontsize=8, ncol=2)
    inside = float(np.mean(np.abs(sub[f"fret_{H}"]) <= 1.96 * sub["forecast"] * np.sqrt(H)))
    ax[1].set_title(f"risk band vs reality — {inside:.1%} of moves fell inside (target 95%)",
                    fontsize=9)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"fig_har_example_{SYM}.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"\nchart -> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
