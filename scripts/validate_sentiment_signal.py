"""Model-agnostic validation: does FinBERT sentiment carry real signal?

Tests whether daily FinBERT sentiment has any statistically detectable
relationship with SUBSEQUENT price behavior — before (and independent of) any
predictive model. If there is no correlation in the raw data, no model can
manufacture one; if there is, the earlier tree-model null points at
representation/model instead.

Scope: the 12 stocks with >=50% news coverage, news-days only (has_news==1),
so we correlate real sentiment against outcomes, not structural zeros.

Six independent tests (see plan): (1) predictive correlation w/ FDR, (2)
quintile signal curve, (3) magnitude/volatility, (4) extreme-sentiment subset,
(5) contemporaneous "priced-in", (6) sentiment-surprise + per-stock consistency.
A PRE-REGISTERED decision rule (fixed before running) converts the numbers into
an honest verdict.

Inputs (read-only):
  data/processed/sentiment_features.parquet    per symbol-day sentiment (no look-ahead)
  FNSPID/final_dataset/prices/*.csv            adj close (all forward horizons)
Outputs:
  reports/sentiment_validation/results.csv + four PNG figures + printed VERDICT

Run:  python scripts/validate_sentiment_signal.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # console may be GBK; avoid encode errors

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
SENT = ROOT / "data" / "processed" / "sentiment_features.parquet"
PRICES = ROOT / "FNSPID" / "final_dataset" / "prices"
OUT = ROOT / "reports" / "sentiment_validation"

SHORTLIST = ["QCOM", "MU", "GLD", "COST", "PEP", "TXN", "ADBE", "AMD",
             "NVDA", "INTC", "SLV", "ASML"]
HORIZONS = [1, 3, 5, 10, 20]
STRONG = 0.5   # |sentiment| threshold for the extreme subset


# ------------------------------------------------------------ data assembly
def load_frame() -> pd.DataFrame:
    sent = pd.read_parquet(SENT)
    sent["date"] = pd.to_datetime(sent["date"])
    frames = []
    for sym in SHORTLIST:
        px = pd.read_csv(PRICES / f"{sym}.csv", usecols=["date", "adj close"])
        px["date"] = pd.to_datetime(px["date"])
        c = px.sort_values("date").set_index("date")["adj close"]
        d = pd.DataFrame(index=c.index)
        d["past_ret_20d"] = c.pct_change(20)          # trend already realized
        for h in HORIZONS:
            fr = c.shift(-h) / c - 1                    # strictly future
            d[f"fwd_{h}"] = fr
            d[f"fabs_{h}"] = fr.abs()
        d = d.reset_index().rename(columns={"index": "date"})
        d["symbol"] = sym
        s = sent[sent["symbol"] == sym].sort_values("date").copy()
        # sentiment "surprise" = level minus trailing 5-news-day mean
        s["sent_surprise"] = s["sentiment"] - s["sentiment"].rolling(5, min_periods=3).mean()
        frames.append(s.merge(d, on=["symbol", "date"], how="inner"))
    df = pd.concat(frames, ignore_index=True)
    return df


# ------------------------------------------------------------------- helpers
def fisher_ci(r: float, n: int):
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def corr(x, y):
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    n = len(x)
    if n < 10 or x.nunique() < 3:
        return dict(n=n, spearman=np.nan, sp=np.nan, pearson=np.nan, pp=np.nan,
                    ci_lo=np.nan, ci_hi=np.nan)
    sr, sp = stats.spearmanr(x, y)
    pr, pp = stats.pearsonr(x, y)
    lo, hi = fisher_ci(sr, n)
    return dict(n=n, spearman=sr, sp=sp, pearson=pr, pp=pp, ci_lo=lo, ci_hi=hi)


# --------------------------------------------------------------------- tests
def run(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # 1. predictive correlation: pooled + per-stock, all horizons
    for scope in ["POOLED"] + SHORTLIST:
        sub = df if scope == "POOLED" else df[df["symbol"] == scope]
        for h in HORIZONS:
            r = corr(sub["sentiment"], sub[f"fwd_{h}"])
            rows.append(dict(test="predictive", scope=scope, horizon=h, **r))

    # 3. magnitude / volatility: |sentiment| -> forward |return| (pooled)
    for h in HORIZONS:
        r = corr(df["sentiment"].abs(), df[f"fabs_{h}"])
        rows.append(dict(test="magnitude", scope="POOLED", horizon=h, **r))

    # 4. extreme-sentiment subset (|sentiment| >= STRONG), pooled predictive
    ex = df[df["sentiment"].abs() >= STRONG]
    for h in HORIZONS:
        r = corr(ex["sentiment"], ex[f"fwd_{h}"])
        rows.append(dict(test="extreme", scope="POOLED", horizon=h, **r))

    # 5. priced-in: sentiment vs PAST 20d return (pooled + per-stock)
    for scope in ["POOLED"] + SHORTLIST:
        sub = df if scope == "POOLED" else df[df["symbol"] == scope]
        r = corr(sub["sentiment"], sub["past_ret_20d"])
        rows.append(dict(test="priced_in", scope=scope, horizon=20, **r))

    # 6a. sentiment surprise -> forward return (pooled)
    for h in HORIZONS:
        r = corr(df["sent_surprise"], df[f"fwd_{h}"])
        rows.append(dict(test="surprise", scope="POOLED", horizon=h, **r))

    res = pd.DataFrame(rows)
    # BH-FDR across the primary predictive-pooled Spearman tests
    prim = (res["test"] == "predictive") & (res["scope"] == "POOLED") & res["sp"].notna()
    res["fdr_sig"] = False
    if prim.any():
        res.loc[prim, "fdr_sig"] = multipletests(res.loc[prim, "sp"], alpha=0.05,
                                                  method="fdr_bh")[0]
    return res


def quintiles(df: pd.DataFrame):
    """Non-parametric signal curve: mean forward return per sentiment quintile."""
    q = pd.qcut(df["sentiment"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    df = df.assign(q=q)
    means = df.groupby("q", observed=True)[[f"fwd_{h}" for h in HORIZONS]].mean()
    spread = {}
    for h in HORIZONS:
        q1 = df.loc[df["q"] == 1, f"fwd_{h}"].dropna()
        q5 = df.loc[df["q"] == 5, f"fwd_{h}"].dropna()
        t, p = stats.ttest_ind(q5, q1, equal_var=False)
        mono, _ = stats.spearmanr(means.index.astype(int), means[f"fwd_{h}"].values)
        spread[h] = dict(spread=float(q5.mean() - q1.mean()), t=float(t), p=float(p),
                         monotonic_rho=float(mono))
    return means, spread


# ------------------------------------------------------------------- figures
def figures(df, res, means):
    OUT.mkdir(parents=True, exist_ok=True)
    pool = res[(res.test == "predictive") & (res.scope == "POOLED")].set_index("horizon")
    mag = res[(res.test == "magnitude")].set_index("horizon")

    # 1. correlation by horizon (direction vs magnitude) with CI whiskers
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="#888", lw=0.8)
    ax.errorbar(HORIZONS, pool["spearman"],
                yerr=[pool["spearman"] - pool["ci_lo"], pool["ci_hi"] - pool["spearman"]],
                marker="o", capsize=4, label="sentiment → forward return (direction)")
    ax.plot(HORIZONS, mag["spearman"], marker="s", ls="--",
            label="|sentiment| → forward |return| (magnitude)")
    ax.set_xlabel("forward horizon (trading days)"); ax.set_ylabel("Spearman ρ (pooled)")
    ax.set_title("Sentiment vs future price behaviour"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_corr_by_horizon.png", dpi=130); plt.close(fig)

    # 2. quintile signal curves
    fig, ax = plt.subplots(figsize=(7, 4))
    for h in HORIZONS:
        ax.plot([1, 2, 3, 4, 5], means[f"fwd_{h}"].values * 100, marker="o", label=f"{h}d")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlabel("sentiment quintile (1=most negative → 5=most positive)")
    ax.set_ylabel("mean forward return (%)")
    ax.set_title("Quintile signal curve (monotone rising ⇒ signal)"); ax.legend(fontsize=8, title="horizon")
    fig.tight_layout(); fig.savefig(OUT / "fig_quintile_curves.png", dpi=130); plt.close(fig)

    # 3. per-stock correlation heatmap (predictive Spearman)
    per = res[(res.test == "predictive") & (res.scope != "POOLED")]
    mat = per.pivot(index="scope", columns="horizon", values="spearman").reindex(SHORTLIST)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-0.12, vmax=0.12, aspect="auto")
    ax.set_xticks(range(len(HORIZONS))); ax.set_xticklabels(HORIZONS)
    ax.set_yticks(range(len(SHORTLIST))); ax.set_yticklabels(SHORTLIST, fontsize=8)
    ax.set_xlabel("horizon (d)"); ax.set_title("Per-stock Spearman ρ (sentiment→fwd ret)")
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout()
    fig.savefig(OUT / "fig_perstock_heatmap.png", dpi=130); plt.close(fig)

    # 4. priced-in: past vs future correlation
    pin = res[(res.test == "priced_in") & (res.scope == "POOLED")]["spearman"].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["past 20d\n(already happened)"] + [f"fwd {h}d\n(future)" for h in HORIZONS]
    vals = [pin] + list(pool["spearman"].values)
    colors = ["#c94" ] + ["#39c"] * len(HORIZONS)
    ax.bar(labels, vals, color=colors); ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylabel("Spearman ρ with sentiment")
    ax.set_title("Does sentiment echo the past or predict the future?")
    fig.tight_layout(); fig.savefig(OUT / "fig_priced_in.png", dpi=130); plt.close(fig)


# -------------------------------------------------------------------- verdict
def verdict(res, spread):
    pool = res[(res.test == "predictive") & (res.scope == "POOLED")]
    sig_h = pool[(pool["fdr_sig"]) & (pool["spearman"] > 0)]["horizon"].tolist()
    mono_ok = [h for h in HORIZONS if spread[h]["monotonic_rho"] > 0.8 and spread[h]["p"] < 0.05
               and spread[h]["spread"] > 0]
    direction_signal = len(sig_h) >= 2 and len(mono_ok) >= 1

    mag = res[res.test == "magnitude"]
    mag_sig = mag[(mag["sp"] < 0.05) & (mag["spearman"] > 0)]["horizon"].tolist()
    pin = res[(res.test == "priced_in") & (res.scope == "POOLED")].iloc[0]

    print("\n" + "=" * 70)
    print("VERDICT  (pre-registered decision rule)")
    print("=" * 70)
    print(f"Directional signal PRESENT? {'YES' if direction_signal else 'NO'}")
    print(f"  - FDR-significant & positive predictive horizons: {sig_h or 'none'} (need >=2)")
    print(f"  - monotone quintile curve w/ sig Q5-Q1 spread at: {mono_ok or 'none'} (need >=1)")
    print(f"Magnitude/volatility signal? {'YES at ' + str(mag_sig) if mag_sig else 'no'}")
    print(f"Priced-in? sentiment vs PAST 20d return ρ = {pin['spearman']:+.4f} "
          f"(p={pin['sp']:.2g}) — {'echoes recent moves' if abs(pin['spearman'])>0.05 else 'weak'}")
    print("=" * 70)
    return direction_signal


def main():
    df = load_frame()
    # sanity: a full-price-coverage stock must keep ALL its news-days after the
    # inner join with prices (no drops, no duplication); pooled n well-powered.
    sent = pd.read_parquet(SENT)
    n_qcom = (df["symbol"] == "QCOM").sum()
    exp_qcom = int((sent["symbol"] == "QCOM").sum())   # QCOM has full 2013-2023 prices
    assert n_qcom == exp_qcom, f"QCOM news-days {n_qcom} != sentiment_features {exp_qcom}"
    assert len(df) > 18000, f"pooled rows too few: {len(df)}"
    print(f"analysis frame: {len(df):,} news-day rows · {df['symbol'].nunique()} stocks "
          f"· QCOM n={n_qcom} (= all its news-days, no drops)")

    res = run(df)
    means, spread = quintiles(df)
    figures(df, res, means)

    # console summary of the primary table
    pool = res[(res.test == "predictive") & (res.scope == "POOLED")]
    print("\nPREDICTIVE (pooled, news-days): sentiment → forward return")
    print(f"{'h':>4} | {'spearman':>9} | {'95% CI':>18} | {'p':>9} | {'FDR sig':>7} | {'n':>6}")
    print("-" * 66)
    for _, r in pool.iterrows():
        print(f"{int(r.horizon):>4} | {r.spearman:>+9.4f} | "
              f"[{r.ci_lo:+.4f},{r.ci_hi:+.4f}] | {r.sp:>9.2g} | "
              f"{str(bool(r.fdr_sig)):>7} | {int(r.n):>6}")

    print("\nQUINTILE Q5-Q1 forward-return spread (bp) & monotonicity")
    print(f"{'h':>4} | {'spread_bp':>10} | {'t':>7} | {'p':>9} | {'mono_rho':>9}")
    print("-" * 50)
    for h in HORIZONS:
        s = spread[h]
        print(f"{h:>4} | {s['spread']*1e4:>10.1f} | {s['t']:>7.2f} | {s['p']:>9.2g} | {s['monotonic_rho']:>+9.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "results.csv", index=False)
    verdict(res, spread)
    print(f"\nresults + figures written to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
