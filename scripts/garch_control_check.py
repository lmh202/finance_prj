"""Confirm the incremental news->volatility signal against a proper GARCH(1,1) control.

validate_news_metrics.py controlled for volatility clustering with a crude linear
20-day trailing vol. Clustering is really GARCH-like (persistent conditional
variance), so this re-tests the ~0.03 incremental signal against a *proper*
GARCH(1,1) forecast:

  For each stock, fit GARCH(1,1) on daily returns; at each day t take the model's
  h-step-ahead variance forecast (info up to t only) as the volatility baseline.
  Then: does news volume / disagreement at t still predict the ACTUAL realized
  forward vol over (t, t+h] AFTER partialling out the GARCH forecast?

If the partial Spearman stays positive & FDR-significant at >=2 horizons under
the GARCH control, the signal is real (not residual clustering the linear
control missed). If it collapses, the earlier 0.03 was an artifact.

Reads (read-only): data/processed/news_sentiment_scored.parquet, FNSPID prices.
Run: python scripts/garch_control_check.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from statsmodels.stats.multitest import multipletests

from build_training_dataset import trading_days
from validate_news_metrics import SHORTLIST, news_features

PRICES = ROOT / "FNSPID" / "final_dataset" / "prices"
VOL_H = [3, 5, 10, 20]


def garch_stock(sym: str) -> pd.DataFrame:
    px = pd.read_csv(PRICES / f"{sym}.csv", usecols=["date", "adj close"])
    px["date"] = pd.to_datetime(px["date"])
    c = px.sort_values("date").set_index("date")["adj close"]
    ret = c.pct_change().dropna()

    # GARCH(1,1), constant mean; returns scaled x100 for numerical stability
    res = arch_model(ret * 100.0, mean="Constant", vol="GARCH", p=1, q=1,
                     dist="normal").fit(disp="off", show_warning=False)
    fc = res.forecast(horizon=max(VOL_H), start=0, reindex=True)
    var = fc.variance  # columns h.1..h.H (forecast var for origin+k), index = dates

    d = pd.DataFrame(index=ret.index)
    d["past_vol"] = ret.rolling(20).std()                       # crude linear control
    for h in VOL_H:
        # first h forecast columns = variance forecast for the next 1..h days
        # (positional: arch zero-pads names as h.01..h.20 for horizon=20)
        d[f"garch_{h}"] = np.sqrt(var.iloc[:, :h].mean(axis=1))  # GARCH h-day-ahead vol forecast
        d[f"rv_{h}"] = ret.rolling(h).std().shift(-h)           # realized forward vol (future)
    d = d.reset_index().rename(columns={"index": "date"})
    d["symbol"] = sym
    return d


def partial(x, y, ctrl, d):
    cols = [x, y] + ctrl
    dd = d[cols].dropna()
    if len(dd) < 60:
        return np.nan, np.nan, len(dd)
    R = {c: dd[c].rank().values for c in cols}
    C = np.column_stack([np.ones(len(dd))] + [R[c] for c in ctrl])
    ex = R[x] - C @ np.linalg.lstsq(C, R[x], rcond=None)[0]
    ey = R[y] - C @ np.linalg.lstsq(C, R[y], rcond=None)[0]
    r, p = stats.pearsonr(ex, ey)
    return float(r), float(p), len(dd)


def main():
    cal = trading_days()
    nf = news_features(cal)
    print("fitting GARCH(1,1) per stock …")
    gf = pd.concat([garch_stock(s) for s in SHORTLIST], ignore_index=True)
    df = nf.merge(gf, on=["symbol", "date"], how="inner")
    dis = df[df["news_count"] >= 2]
    print(f"merged: {len(df):,} news-days (>=2 headlines: {len(dis):,})\n")

    metrics = {"log_count (volume)": ("log_count", df),
               "disagreement": ("disagreement", dis)}

    rows = []
    for label, (col, d) in metrics.items():
        for h in VOL_H:
            r_lin, p_lin, _ = partial(col, f"rv_{h}", ["past_vol"], d)
            r_g, p_g, n = partial(col, f"rv_{h}", [f"garch_{h}"], d)
            r_both, p_both, _ = partial(col, f"rv_{h}", [f"garch_{h}", "past_vol"], d)
            # per-stock consistency under GARCH control
            pos = 0
            tot = 0
            for sym, sub in d.groupby("symbol"):
                rs, ps, ns = partial(col, f"rv_{h}", [f"garch_{h}"], sub)
                if np.isfinite(rs):
                    tot += 1
                    pos += rs > 0
            rows.append(dict(metric=label, horizon=h, n=n,
                             lin_rho=r_lin, lin_p=p_lin,
                             garch_rho=r_g, garch_p=p_g,
                             both_rho=r_both, both_p=p_both,
                             stocks_pos=f"{pos}/{tot}"))
    res = pd.DataFrame(rows)
    res["garch_fdr"] = multipletests(res["garch_p"], 0.05, method="fdr_bh")[0]

    print("NEWS METRIC -> forward realized vol, partial ρ under each control")
    print(f"{'metric':>18} | {'h':>3} | {'linear ρ':>9} | {'GARCH ρ':>9} {'p':>8} {'FDR':>5} | "
          f"{'GARCH+lin ρ':>11} | {'stk+':>5} | {'n':>6}")
    print("-" * 94)
    for _, r in res.iterrows():
        print(f"{r.metric:>18} | {r.horizon:>3} | {r.lin_rho:>+9.4f} | "
              f"{r.garch_rho:>+9.4f} {r.garch_p:>8.2g} {str(bool(r.garch_fdr)):>5} | "
              f"{r.both_rho:>+11.4f} | {r.stocks_pos:>5} | {int(r.n):>6}")

    print("\n" + "=" * 70)
    print("VERDICT — does the signal survive a proper GARCH(1,1) control?")
    print("=" * 70)
    for label in metrics:
        s = res[res.metric == label]
        good = s[(s.garch_fdr) & (s.garch_rho > 0)]
        hz = sorted(int(h) for h in good.horizon)
        ok = len(good) >= 2
        print(f"  {label:>18}: {'CONFIRMED' if ok else 'NOT confirmed'} "
              f"(GARCH-controlled sig & positive at horizons {hz or 'none'}; need >=2)")
    print("=" * 70)
    print("If GARCH ρ ≈ linear ρ and stays significant -> the ~0.03 signal is real,")
    print("not an artifact of the crude linear control.")


if __name__ == "__main__":
    main()
