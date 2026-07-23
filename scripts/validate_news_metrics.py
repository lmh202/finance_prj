"""Do OTHER news metrics beat the FinBERT-polarity null? (two cheap probes)

Polarity sentiment showed no forward signal (scripts/validate_sentiment_signal.py).
Here we test two theoretically-stronger, still-cheap news metrics — computed from
data we already have, no new sources, no LLM:

  A. ATTENTION   news volume per day + abnormal volume (spike vs trailing median)
  B. DISAGREEMENT std of the day's individual FinBERT headline scores (needs >=2)

against the outcome news *should* move most: forward realized VOLATILITY (not
direction). The honest bar: news volume often just REACTS to big moves, so we
control for volatility clustering — report both the raw correlation and the
PARTIAL correlation holding past 20d realized vol fixed. A metric only "works"
if it predicts future vol BEYOND what past vol already tells you.

Pre-registered rule: a metric is a valid vol predictor iff its PARTIAL Spearman
(controlling past vol) is FDR-significant AND positive at >=2 horizons.
Also reported (expected null): the same metrics -> forward return (direction).

Scope: the 12 stocks with >=50% news coverage; no look-ahead (news shifted to
the next session via the shared trading-day logic).

Inputs (read-only): data/processed/news_sentiment_scored.parquet, FNSPID prices.
Outputs: reports/news_metrics/results.csv + figures + printed VERDICT.
Run: python scripts/validate_news_metrics.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from build_training_dataset import next_trading_day, trading_days  # shared no-look-ahead

SCORED = ROOT / "data" / "processed" / "news_sentiment_scored.parquet"
PRICES = ROOT / "FNSPID" / "final_dataset" / "prices"
OUT = ROOT / "reports" / "news_metrics"

SHORTLIST = ["QCOM", "MU", "GLD", "COST", "PEP", "TXN", "ADBE", "AMD",
             "NVDA", "INTC", "SLV", "ASML"]
VOL_H = [3, 5, 10, 20]     # realized-vol horizons (h=1 std is degenerate)
DIR_H = [1, 3, 5, 10, 20]  # direction horizons


# ----------------------------------------------------------- news-day features
def news_features(cal: np.ndarray) -> pd.DataFrame:
    sc = pd.read_parquet(SCORED)                       # per-headline: date, symbol, sentiment
    sc = sc[sc["symbol"].isin(SHORTLIST)].copy()
    sc["date"] = pd.to_datetime(sc["date"])
    sc["eff_date"] = next_trading_day(sc["date"], cal)  # shift to next session (no look-ahead)
    g = sc.groupby(["symbol", "eff_date"])["sentiment"]
    feat = g.agg(news_count="size", sent_mean="mean", disagreement="std").reset_index()
    feat = feat.rename(columns={"eff_date": "date"})
    # attention transforms, per symbol on its news-day sequence (trailing only)
    parts = []
    for sym, sub in feat.groupby("symbol"):
        sub = sub.sort_values("date").copy()
        base = sub["news_count"].rolling(20, min_periods=5).median().shift(1)
        sub["abn_vol"] = np.log(sub["news_count"] / base)     # spike vs trailing median
        sub["log_count"] = np.log(sub["news_count"])
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------- price/vol frame
def price_features() -> pd.DataFrame:
    parts = []
    for sym in SHORTLIST:
        px = pd.read_csv(PRICES / f"{sym}.csv", usecols=["date", "adj close"])
        px["date"] = pd.to_datetime(px["date"])
        c = px.sort_values("date").set_index("date")["adj close"]
        ret = c.pct_change()
        d = pd.DataFrame(index=c.index)
        d["past_vol"] = ret.rolling(20).std()                 # trailing (no look-ahead)
        for h in VOL_H:
            d[f"rv_{h}"] = ret.rolling(h).std().shift(-h)      # realized vol over (t, t+h]
        for h in DIR_H:
            d[f"fwd_{h}"] = c.shift(-h) / c - 1
        d = d.reset_index().rename(columns={"index": "date"})
        d["symbol"] = sym
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------- statistics
def spearman(x, y):
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    if len(x) < 30 or x.nunique() < 5:
        return np.nan, np.nan, len(x)
    r, p = stats.spearmanr(x, y)
    return r, p, len(x)


def partial_spearman(x, y, z):
    """Spearman(x, y) controlling for z — rank all, regress out z, correlate residuals."""
    m = x.notna() & y.notna() & z.notna()
    if m.sum() < 40:
        return np.nan, np.nan, int(m.sum())
    rx, ry, rz = x[m].rank().values, y[m].rank().values, z[m].rank().values
    Z = np.column_stack([np.ones(len(rz)), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r, p = stats.pearsonr(ex, ey)
    return float(r), float(p), int(m.sum())


def main():
    cal = trading_days()
    nf = news_features(cal)
    pf = price_features()
    df = nf.merge(pf, on=["symbol", "date"], how="inner")

    # sanity: no-look-ahead + power
    assert (df["date"] >= "2013-01-01").all()
    assert len(df) > 15000, f"pooled rows too few: {len(df)}"
    dis = df[df["news_count"] >= 2]
    print(f"news-day rows: {len(df):,} (>=2 headlines: {len(dis):,}) · {df['symbol'].nunique()} stocks")

    rows = []
    # ---- volatility target: raw + partial(control past_vol) ----
    predictors = {"log_count": df, "abn_vol": df, "disagreement": dis}
    for name, d in predictors.items():
        for h in VOL_H:
            r, p, n = spearman(d[name], d[f"rv_{h}"])
            pr, pp, pn = partial_spearman(d[name], d[f"rv_{h}"], d["past_vol"])
            rows.append(dict(metric=name, target=f"vol_{h}d", n=n,
                             raw_rho=r, raw_p=p, partial_rho=pr, partial_p=pp))
    # ---- confound exposure: predictor vs PAST vol (should be strongly +) ----
    for name, d in predictors.items():
        r, p, n = spearman(d[name], d["past_vol"])
        rows.append(dict(metric=name, target="past_vol", n=n,
                         raw_rho=r, raw_p=p, partial_rho=np.nan, partial_p=np.nan))
    # ---- direction target (expected null) ----
    for name, d in predictors.items():
        for h in DIR_H:
            r, p, n = spearman(d[name], d[f"fwd_{h}"])
            rows.append(dict(metric=name, target=f"dir_{h}d", n=n,
                             raw_rho=r, raw_p=p, partial_rho=np.nan, partial_p=np.nan))

    res = pd.DataFrame(rows)
    # FDR across the PARTIAL vol tests (the pre-registered primary family)
    prim = res["target"].str.startswith("vol_") & res["partial_p"].notna()
    res["fdr_sig"] = False
    res.loc[prim, "fdr_sig"] = multipletests(res.loc[prim, "partial_p"], 0.05, method="fdr_bh")[0]

    _report(res)
    _figures(df, dis, res)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "results.csv", index=False)
    _verdict(res)
    print(f"\nresults + figures written to {OUT.relative_to(ROOT)}/")


def _report(res):
    print("\nFORWARD VOLATILITY — raw vs partial (controlling past 20d vol)")
    print(f"{'metric':>13} | {'target':>7} | {'raw ρ':>8} {'raw p':>8} | {'partial ρ':>9} {'part p':>8} | {'FDR':>5} | {'n':>6}")
    print("-" * 78)
    for _, r in res[res.target.str.startswith("vol_")].iterrows():
        print(f"{r.metric:>13} | {r.target:>7} | {r.raw_rho:>+8.4f} {r.raw_p:>8.2g} | "
              f"{r.partial_rho:>+9.4f} {r.partial_p:>8.2g} | {str(bool(r.fdr_sig)):>5} | {int(r.n):>6}")
    print("\nCONFOUND — metric vs PAST 20d vol (high + => metric tracks vol clustering)")
    for _, r in res[res.target == "past_vol"].iterrows():
        print(f"  {r.metric:>13}: ρ = {r.raw_rho:+.4f} (p={r.raw_p:.2g})")
    print("\nDIRECTION — metric vs forward return (expected ~0)")
    for name in ["log_count", "abn_vol", "disagreement"]:
        sub = res[(res.metric == name) & res.target.str.startswith("dir_")]
        rng = f"{sub.raw_rho.min():+.4f}..{sub.raw_rho.max():+.4f}"
        sig = (sub.raw_p < 0.05).sum()
        print(f"  {name:>13}: ρ in [{rng}], {sig}/{len(sub)} horizons p<0.05 (uncorrected)")


def _figures(df, dis, res):
    OUT.mkdir(parents=True, exist_ok=True)
    v = res[res.target.str.startswith("vol_")]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name in ["log_count", "abn_vol", "disagreement"]:
        s = v[v.metric == name]
        hs = [int(t.split("_")[1][:-1]) for t in s.target]
        ax.plot(hs, s.raw_rho, marker="o", ls="--", alpha=.5, label=f"{name} (raw)")
        ax.plot(hs, s.partial_rho, marker="s", label=f"{name} (partial)")
    ax.axhline(0, color="#888", lw=.8)
    ax.set_xlabel("forward vol horizon (days)"); ax.set_ylabel("Spearman ρ")
    ax.set_title("News metrics → forward volatility (raw vs controlling past vol)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(OUT / "fig_news_vol.png", dpi=130); plt.close(fig)

    # attention quintiles -> mean forward 5d vol
    q = pd.qcut(df["log_count"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    m = df.assign(q=q).groupby("q", observed=True)["rv_5"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([1, 2, 3, 4, 5], m.values * 100, marker="o")
    ax.set_xlabel("news-volume quintile (1=quiet → 5=high attention)")
    ax.set_ylabel("mean forward 5d realized vol (%)")
    ax.set_title("Does more news today → more volatility next week?")
    fig.tight_layout(); fig.savefig(OUT / "fig_attention_quintiles.png", dpi=130); plt.close(fig)


def _verdict(res):
    v = res[res.target.str.startswith("vol_")]
    print("\n" + "=" * 70)
    print("VERDICT (pre-registered: partial ρ FDR-sig & positive at >=2 horizons)")
    print("=" * 70)
    for name in ["log_count", "abn_vol", "disagreement"]:
        s = v[v.metric == name]
        good = s[(s.fdr_sig) & (s.partial_rho > 0)]
        raw_good = s[(s.raw_p < 0.05) & (s.raw_rho > 0)]
        verdict = "VALID vol predictor" if len(good) >= 2 else "no incremental signal"
        print(f"  {name:>13}: {verdict}  "
              f"(partial-sig horizons: {sorted(int(t.split('_')[1][:-1]) for t in good.target) or 'none'}; "
              f"raw-sig: {len(raw_good)}/{len(s)})")
    print("=" * 70)
    print("Note: raw>0 but partial~0 means the metric only tracks volatility")
    print("clustering (news reacts to moves) — no predictive value of its own.")


if __name__ == "__main__":
    main()
