"""Is the news gain to the volatility forecast AMPLIFIABLE downstream?

Two read-only checks on the 12 news-covered stocks, both aimed at one product
question: the news features add a small, real improvement to the h=5 vol forecast
(QLIKE -1.4%, p=1e-4) — but is that gain worth MORE than its average once it feeds
position sizing, or does it stay ~1:1?

  PART A — SELECTION-BIAS RE-SCREEN.
    log_count / disagreement were chosen with validate_news_metrics.py, whose
    correlations used the FULL 2013-2023 sample (incl. the test period). Re-run the
    partial-correlation screen on train+val ONLY (<= 2020-12-31). If both features
    are still FDR-significant & positive at >=2 horizons, the feature choice is clean
    and the h=5 result stops being contaminated by hindsight.

  PART B — REGIME DECOMPOSITION (the amplification test).
    Fit HAR (price only) vs HAR+news on TRAIN only; on the held-out TEST set compute
    the per-day QLIKE improvement from news, then split days into terciles by
      (i)  realized next-5d vol   — what actually happened, and
      (ii) trailing 22d vol (rv22)— what the desk KNOWS at sizing time (ex-ante).
    Position sizing is convex in getting high-vol days right (avoiding a drawdown is
    worth more than the average), so:
      - gain concentrated in the HIGH realized-vol tercile  -> convex, worth > average
      - gain concentrated in the LOW ex-ante (rv22) tercile -> news warns when PRICE
        history looked calm but wasn't -> orthogonal info, highest decision value
      - gain uniform across terciles                        -> ~1:1, not amplifiable
    Plus a tail metric: R = realized_vol / forecast_vol. R>1 = you UNDER-forecast risk
    (the drawdown-causing case). Does news shrink the upper tail of R (p95/p99) by more
    than it shrinks the mean? If yes, the gain lands exactly where sizing cares.

Reads (read-only): FNSPID prices, data/processed/news_sentiment_scored.parquet.
Outputs: reports/risk_engine/news_amplification.csv + fig + printed VERDICT.
Run: python scripts/news_amplification_check.py
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
from statsmodels.stats.multitest import multipletests

from build_training_dataset import trading_days
from train_risk_engine import (EMBARGO, FEATS_BASE, FEATS_NEWS, NEWS_STOCKS,
                               TRAIN_END, VAL_END, dm_test, log_feats, qlike, qlike_series)
from validate_news_metrics import (VOL_H, news_features, partial_spearman,
                                   price_features)

PRICES = ROOT / "FNSPID" / "final_dataset" / "prices"
OUT = ROOT / "reports" / "risk_engine"
H = 5  # the horizon where news was shown to help


# --------------------------------------------------------------- HAR features
def build_har(sym: str) -> pd.DataFrame:
    """HAR feature block only (no GARCH/EGARCH) — identical construction to the
    risk engine's build_stock, kept light so this check runs in seconds."""
    px = pd.read_csv(PRICES / f"{sym}.csv")
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values("date").set_index("date")
    c, hi, lo = px["adj close"], px["high"], px["low"]
    ret = c.pct_change()

    d = pd.DataFrame(index=c.index)
    d["rv5"], d["rv22"], d["rv66"] = (ret.rolling(w).std() for w in (5, 22, 66))
    park = (np.log(hi / lo) ** 2) / (4 * np.log(2))
    d["park5"] = np.sqrt(park.rolling(5).mean())
    d["park22"] = np.sqrt(park.rolling(22).mean())
    d["absret"] = ret.abs()
    d[f"y_{H}"] = ret.rolling(H).std().shift(-H)          # realized fwd vol
    d[f"fret_{H}"] = c.shift(-H) / c - 1                   # realized fwd return
    d["symbol"] = sym
    return d.reset_index().rename(columns={"index": "date"})


def build_panel() -> pd.DataFrame:
    panel = log_feats(pd.concat([build_har(s) for s in NEWS_STOCKS], ignore_index=True))
    nf = news_features(trading_days())[["symbol", "date", "log_count", "disagreement"]]
    panel = panel.merge(nf, on=["symbol", "date"], how="left")
    panel["has_news"] = panel["log_count"].notna().astype(int)
    panel[["log_count", "disagreement"]] = panel[["log_count", "disagreement"]].fillna(0.0)
    return panel


# =============================================================== PART A
def part_a_rescreen():
    print("=" * 78)
    print("PART A — selection-bias re-screen (features chosen on <=2020 ONLY)")
    print("=" * 78)
    cal = trading_days()
    df = news_features(cal).merge(price_features(), on=["symbol", "date"], how="inner")
    full_n = len(df)
    df = df[df["date"] <= VAL_END]           # train+val only — never sees the test period
    dis = df[df["news_count"] >= 2]
    print(f"screen sample: {len(df):,} news-days (<=2020) of {full_n:,} full "
          f"({len(df)/full_n:.0%}); disagreement needs >=2 headlines: {len(dis):,}\n")

    rows = []
    for name, d in [("log_count", df), ("disagreement", dis)]:
        for h in VOL_H:
            pr, pp, n = partial_spearman(d[name], d[f"rv_{h}"], d["past_vol"])
            rows.append(dict(metric=name, horizon=h, partial_rho=pr, partial_p=pp, n=n))
    res = pd.DataFrame(rows)
    res["fdr_sig"] = multipletests(res["partial_p"], 0.05, method="fdr_bh")[0]

    print(f"{'metric':>13} | {'h':>3} | {'partial ρ':>9} {'p':>9} {'FDR':>5} | {'n':>6}")
    print("-" * 52)
    for _, r in res.iterrows():
        print(f"{r.metric:>13} | {int(r.horizon):>3} | {r.partial_rho:>+9.4f} "
              f"{r.partial_p:>9.2g} {str(bool(r.fdr_sig)):>5} | {int(r.n):>6}")

    print("\n  verdict (need FDR-sig & positive at >=2 horizons on <=2020 data):")
    ok_all = True
    for name in ["log_count", "disagreement"]:
        s = res[res.metric == name]
        good = s[(s.fdr_sig) & (s.partial_rho > 0)]
        ok = len(good) >= 2
        ok_all &= ok
        print(f"    {name:>13}: {'STILL SELECTED' if ok else 'DROPS OUT'} "
              f"(sig+positive at horizons {sorted(int(h) for h in good.horizon) or 'none'})")
    print(f"\n  => feature choice is {'CLEAN (not hindsight)' if ok_all else 'CONTAMINATED — revisit'}")
    return res, ok_all


# =============================================================== PART B
def _fit_predict(panel):
    """Fit HAR base & HAR+news on train (12 news stocks), return the TEST frame with
    per-day forecasts + per-day QLIKE for each model. Smearing corrected on VAL, as
    in the risk engine, so the comparison matches train_risk_engine exactly."""
    d = panel.dropna(subset=FEATS_BASE + [f"y_{H}", f"fret_{H}"]).copy()
    d["ly"] = np.log(d[f"y_{H}"] + 1e-6)
    emb = pd.Timedelta(days=int(EMBARGO * 1.5))
    tr = d[d.date <= pd.Timestamp(TRAIN_END) - emb]
    va = d[(d.date > TRAIN_END) & (d.date <= pd.Timestamp(VAL_END) - emb)]
    te = d[d.date > VAL_END].copy()

    def smear(resid):
        return float(np.exp(np.var(resid, ddof=1) / 2))

    m_b = LinearRegression().fit(tr[FEATS_BASE], tr["ly"])
    m_n = LinearRegression().fit(tr[FEATS_NEWS], tr["ly"])
    cb = smear(va["ly"] - m_b.predict(va[FEATS_BASE]))
    cn = smear(va["ly"] - m_n.predict(va[FEATS_NEWS]))

    y = te[f"y_{H}"].to_numpy()
    te["f_base"] = np.exp(m_b.predict(te[FEATS_BASE])) * cb
    te["f_news"] = np.exp(m_n.predict(te[FEATS_NEWS])) * cn
    te["q_base"] = qlike_series(y, te["f_base"].to_numpy())
    te["q_news"] = qlike_series(y, te["f_news"].to_numpy())
    te["gain"] = te["q_base"] - te["q_news"]          # >0 => news helped this day
    print(f"\n  TEST rows: {len(te):,}  ·  overall QLIKE  HAR {qlike(y, te.f_base):.4f}"
          f" -> HAR+news {qlike(y, te.f_news):.4f}"
          f" ({qlike(y, te.f_news)-qlike(y, te.f_base):+.4f})")
    return te


def _regime_table(te, by, label):
    q = pd.qcut(te[by], 3, labels=["low", "mid", "high"])
    total_gain = te["gain"].sum()
    print(f"\n  by {label} tercile:")
    print(f"    {'regime':>6} | {'n':>5} | {'QLIKE base':>10} {'QLIKE news':>10} | "
          f"{'mean gain':>10} | {'share of total gain':>19}")
    print("    " + "-" * 74)
    rows = []
    for reg in ["low", "mid", "high"]:
        s = te[q == reg]
        share = s["gain"].sum() / total_gain if total_gain != 0 else np.nan
        t, p = dm_test(s["q_base"].to_numpy(), s["q_news"].to_numpy(), lag=H)
        print(f"    {reg:>6} | {len(s):>5} | {qlike_from(s,'f_base'):>10.4f} "
              f"{qlike_from(s,'f_news'):>10.4f} | {s['gain'].mean():>+10.5f} | "
              f"{share:>17.0%}   {'(DM sig)' if p < 0.05 and t > 0 else ''}")
        rows.append(dict(split=label, regime=reg, n=len(s), share=share,
                         mean_gain=s["gain"].mean(), dm_t=t, dm_p=p))
    return pd.DataFrame(rows)


def qlike_from(s, col):
    return qlike(s[f"y_{H}"].to_numpy(), s[col].to_numpy())


def _tail_metric(te):
    """R = realized / forecast vol. R>1 = risk under-forecast (the drawdown case)."""
    Rb = te[f"y_{H}"] / te["f_base"]
    Rn = te[f"y_{H}"] / te["f_news"]
    print("\n  tail of R = realized_vol / forecast_vol  (R>1 = UNDER-forecast risk):")
    print(f"    {'model':>10} | {'mean R':>7} | {'p90':>6} {'p95':>6} {'p99':>6} | "
          f"{'% days R>1.5':>12}")
    print("    " + "-" * 60)
    out = {}
    for nm, R in [("HAR", Rb), ("HAR+news", Rn)]:
        out[nm] = dict(mean=R.mean(), p90=R.quantile(.9), p95=R.quantile(.95),
                       p99=R.quantile(.99), over=float((R > 1.5).mean()))
        print(f"    {nm:>10} | {R.mean():>7.3f} | {R.quantile(.9):>6.3f} "
              f"{R.quantile(.95):>6.3f} {R.quantile(.99):>6.3f} | {float((R>1.5).mean()):>11.1%}")
    dmean = (out['HAR']['mean'] - out['HAR+news']['mean']) / out['HAR']['mean']
    dp95 = (out['HAR']['p95'] - out['HAR+news']['p95']) / out['HAR']['p95']
    print(f"\n    news shrinks mean R by {dmean:+.1%}, p95 tail by {dp95:+.1%}")
    print(f"    => {'CONVEX: tail improves MORE than mean (lands where sizing cares)' if dp95 > dmean else 'flat: tail improves no more than the mean (~1:1)'}")
    return out, dp95 > dmean


def part_b_amplification(panel):
    print("\n" + "=" * 78)
    print(f"PART B — where does the news gain live? (h={H}d, {len(NEWS_STOCKS)} news stocks)")
    print("=" * 78)
    te = _fit_predict(panel)
    r1 = _regime_table(te, f"y_{H}", "realized fwd vol")
    r2 = _regime_table(te, "rv22", "ex-ante trailing vol (rv22)")
    tail, convex = _tail_metric(te)

    # figure: share of total gain by regime, both splits
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, (r, ttl) in zip(ax, [(r1, "by REALIZED fwd vol"),
                                (r2, "by EX-ANTE rv22 (decision-time)")]):
        a.bar(r["regime"], r["share"] * 100, color=["#8da2fb", "#b3f34c", "#c0392b"])
        a.axhline(100 / 3, color="#888", ls="--", lw=1, label="uniform (33%)")
        a.set_ylabel("share of total news QLIKE gain (%)")
        a.set_title(ttl, fontsize=10)
        a.legend(fontsize=8)
    fig.suptitle(f"News improvement to the h={H}d vol forecast — where it concentrates")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig_news_amplification.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    pd.concat([r1, r2]).to_csv(OUT / "news_amplification.csv", index=False)
    print(f"\n  chart -> {p.relative_to(ROOT)}")
    return r1, r2, convex


def main():
    _, clean = part_a_rescreen()
    panel = build_panel()
    r1, r2, convex = part_b_amplification(panel)

    hi_share_real = float(r1[r1.regime == "high"]["share"])
    lo_share_exante = float(r2[r2.regime == "low"]["share"])
    print("\n" + "=" * 78)
    print("OVERALL VERDICT — is the news gain amplifiable in the decision engine?")
    print("=" * 78)
    print(f"  A. feature choice clean (screened <=2020) ......... {'YES' if clean else 'NO'}")
    print(f"  B1. high REALIZED-vol tercile holds .............. {hi_share_real:.0%} of the gain "
          f"({'concentrated -> convex' if hi_share_real > 0.45 else 'not concentrated'})")
    print(f"  B2. low EX-ANTE-vol tercile holds ................ {lo_share_exante:.0%} of the gain "
          f"({'news adds info price missed' if lo_share_exante > 0.40 else 'mostly echoes price'})")
    print(f"  B3. under-forecast tail shrinks more than mean ... {'YES (convex)' if convex else 'NO (~1:1)'}")
    print("-" * 78)
    amplify = (hi_share_real > 0.45) or convex
    print(f"  => The h={H}d news gain is "
          f"{'AMPLIFIABLE: it concentrates where sizing is convex.' if amplify else 'roughly 1:1 — keep it (near-free) but do not build the product around it.'}")


if __name__ == "__main__":
    main()
