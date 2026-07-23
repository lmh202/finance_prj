"""Risk-measurement layer: turn the HAR volatility forecast into decision-ready
risk numbers (per-stock AND portfolio) and PROVE they are trustworthy.

The decision layer that consumes this is formula-based, not trained — so the
risk output must be high-quality on its own. We convert HAR's σ̂ into a full
downside suite via FILTERED HISTORICAL SIMULATION (FHS):

  1. HAR forecasts σ̂ (loaded from data/processed/risk_model.json — the model
     already trained in train_risk_engine_v2.py).
  2. Standardize historical returns by their σ̂ forecast: z = r / σ̂.
  3. Read the EMPIRICAL quantiles of z (fat-tailed, left-skewed — no Gaussian
     assumption, no extra training). VaR/ES/band are σ̂ × those quantiles.
  4. Portfolio: sample whole-day standardized-return VECTORS (preserving
     cross-stock correlation + tail co-movement), scale by current σ̂ and weights
     → empirical portfolio-return distribution → portfolio VaR/ES.

Then we validate on the held-out TEST set (2021-2023), tail calibrated on
train+val (≤2020):
  - Kupiec unconditional coverage (is the breach rate = α?)
  - Christoffersen independence (are breaches clustered?)
  - ES backtest (does mean tail loss match ES?)

Backtesting is done at the 1-DAY horizon (the statistically clean, standard VaR
setup — no overlapping windows); the h=5/h=20 measures are the decision output,
scaled from the same standardized distribution.

Run: python scripts/risk_measures.py
"""

import json
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from train_risk_engine import FEATS_BASE, HORIZONS, TRAIN_END, VAL_END, log_feats, stocks

PRICES = ROOT / "FNSPID" / "final_dataset" / "prices"
OUT = ROOT / "reports" / "risk_engine"
MODEL_PATH = ROOT / "data" / "processed" / "risk_model.json"
EPS = 1e-6
PORT_EXCLUDE = {"SLV"}   # SLV price history ends 2020 → can't be in a 2021-23 test portfolio


# ------------------------------------------------------------- schema
@dataclass
class RiskEstimate:
    symbol: str
    horizon: int
    sigma_daily: float          # HAR per-day vol forecast
    sigma_h: float              # forecast vol over the horizon
    var_95: float               # downside VaR (h-day return threshold, negative)
    var_99: float
    es_95: float                # expected shortfall / CVaR
    band_lo: float
    band_hi: float
    risk_level: float           # 0-100, σ̂ percentile vs the stock's own history
    as_of: str
    has_history: bool


@dataclass
class PortfolioRisk:
    horizon: int
    n_holdings: int
    sigma_h: float
    var_95: float
    var_99: float
    es_95: float
    diversification_ratio: float   # portfolio VaR / Σ|w_i|·VaR_i  (<1 = diversified)
    as_of: str


# ------------------------------------------------------------- panel
def build_one(sym: str) -> pd.DataFrame:
    px = pd.read_csv(PRICES / f"{sym}.csv", usecols=["date", "adj close", "high", "low"])
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
    d["ret1"] = ret                               # contemporaneous 1-day return (EWMA corr)
    d["next_ret"] = ret.shift(-1)                 # 1-day-ahead return (for the VaR backtest)
    for h in HORIZONS:
        d[f"fret_{h}"] = c.shift(-h) / c - 1       # h-day-ahead return
    d["symbol"] = sym
    return d.reset_index().rename(columns={"index": "date"})


def sigma_daily(df: pd.DataFrame, mh: dict) -> np.ndarray:
    """HAR per-day vol forecast from the serialized model (a dot product)."""
    logv = mh["intercept"] + sum(df[f] * mh["coef"][f] for f in mh["features"])
    return np.exp(logv) * mh["smearing"]


# ------------------------------------------------------------- backtests
def kupiec(n: int, x: int, p: float):
    """Unconditional-coverage LR test (breach rate == p). Returns (breach_rate, p_value)."""
    if n == 0:
        return np.nan, np.nan
    pi = x / n
    l0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    l1 = xlogy(n - x, 1 - pi) + xlogy(x, pi)
    lr = -2 * (l0 - l1)
    return pi, float(1 - stats.chi2.cdf(lr, 1))


def christoffersen_ind(breaches: np.ndarray):
    """Independence LR test (breaches not clustered). Returns p_value."""
    b = np.asarray(breaches).astype(int)
    if len(b) < 3:
        return np.nan
    prev, cur = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / len(cur)
    l1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    l0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    lr = -2 * (l0 - l1)
    return float(1 - stats.chi2.cdf(lr, 1))


# ------------------------------------------------------------- main
def main():
    print("building HAR panel + loading serialized model …")
    panel = log_feats(pd.concat([build_one(s) for s in stocks()], ignore_index=True))
    model = json.loads(MODEL_PATH.read_text())
    m5 = model["horizons"]["5"]

    # per-day σ̂ from the h=5 HAR model (the responsive per-day vol forecast)
    panel["sig1"] = sigma_daily(panel, m5)
    is_train_val = panel["date"] <= VAL_END
    is_test = panel["date"] > VAL_END

    # ---- FHS tail: standardized 1-day returns on ≤2020, pooled across stocks ----
    z1 = (panel.loc[is_train_val, "next_ret"] / panel.loc[is_train_val, "sig1"]).replace(
        [np.inf, -np.inf], np.nan).dropna()
    q = {"q05": float(np.quantile(z1, 0.05)), "q01": float(np.quantile(z1, 0.01)),
         "q025": float(np.quantile(z1, 0.025)), "q975": float(np.quantile(z1, 0.975))}
    es = {"es05": float(z1[z1 <= q["q05"]].mean()), "es01": float(z1[z1 <= q["q01"]].mean())}
    print(f"FHS 1-day standardized tail (≤2020, n={len(z1):,}): "
          f"q05={q['q05']:.2f} q01={q['q01']:.2f}  es05={es['es05']:.2f}  "
          f"(Gaussian would be -1.64 / -2.33)")

    # per-horizon standardized quantiles for the h-day OUTPUT measures
    fhs_h = {}
    for h in HORIZONS:
        mh = model["horizons"][str(h)]
        sig_h = sigma_daily(panel, mh) * np.sqrt(h)
        zh = (panel.loc[is_train_val, f"fret_{h}"] / sig_h[is_train_val.values]).replace(
            [np.inf, -np.inf], np.nan).dropna()
        fhs_h[str(h)] = {
            "q05": float(np.quantile(zh, 0.05)), "q01": float(np.quantile(zh, 0.01)),
            "q025": float(np.quantile(zh, 0.025)), "q975": float(np.quantile(zh, 0.975)),
            "es05": float(zh[zh <= np.quantile(zh, 0.05)].mean()),
            "es01": float(zh[zh <= np.quantile(zh, 0.01)].mean()),
        }

    # ================= PER-STOCK VaR BACKTEST (1-day, clean) =================
    te = panel[is_test].copy()
    te["var95"] = te["sig1"] * q["q05"]
    te["var99"] = te["sig1"] * q["q01"]
    te["es95"] = te["sig1"] * es["es05"]
    te["br95"] = (te["next_ret"] < te["var95"]).astype(int)
    te["br99"] = (te["next_ret"] < te["var99"]).astype(int)
    te = te.dropna(subset=["next_ret", "sig1"])

    n = len(te)
    r95, p95 = kupiec(n, int(te["br95"].sum()), 0.05)
    r99, p99 = kupiec(n, int(te["br99"].sum()), 0.01)
    # per-stock Christoffersen independence
    ind_pass = 0; ind_tot = 0
    for _, gsub in te.groupby("symbol"):
        pv = christoffersen_ind(gsub.sort_values("date")["br95"].values)
        if np.isfinite(pv):
            ind_tot += 1; ind_pass += pv > 0.05
    # ES check among breaches
    brk = te[te["br95"] == 1]
    es_ratio = float(brk["next_ret"].mean() / brk["es95"].mean()) if len(brk) else np.nan

    print("\n" + "=" * 74)
    print(f"PER-STOCK VaR BACKTEST — 1-day, pooled test 2021-23 (n={n:,})")
    print("=" * 74)
    print(f"  VaR95 breach rate {r95:.3%}  (target 5.0%)  Kupiec p={p95:.3g} "
          f"{'OK' if p95 > 0.05 else 'reject'}")
    print(f"  VaR99 breach rate {r99:.3%}  (target 1.0%)  Kupiec p={p99:.3g} "
          f"{'OK' if p99 > 0.05 else 'reject'}")
    print(f"  Christoffersen independence: {ind_pass}/{ind_tot} stocks pass (p>0.05, no clustering)")
    print(f"  ES check: mean tail loss / predicted ES = {es_ratio:.2f}  (≈1 = well-calibrated)")

    # ================= PORTFOLIO VaR (EWMA covariance + fat tail) =================
    # Fixed-history FHS understated portfolio risk in 2022 because correlations
    # spiked above their ≤2020 level. Fix: HAR σ̂ on the diagonal + a time-varying
    # EWMA correlation (λ=0.94, RiskMetrics), each estimated from data up to t only;
    # the fat-tail shape comes from the empirical portfolio standardized-return
    # quantile. This adapts the diversification benefit to the current regime.
    piv_sig = panel.pivot(index="date", columns="symbol", values="sig1")
    piv_r1 = panel.pivot(index="date", columns="symbol", values="ret1")
    piv_nxt = panel.pivot(index="date", columns="symbol", values="next_ret")
    hist_cov = piv_sig[piv_sig.index <= VAL_END].notna().mean()
    test_cov = piv_sig[piv_sig.index > VAL_END].notna().mean()
    port = [s for s in stocks() if s not in PORT_EXCLUDE
            and hist_cov.get(s, 0) > 0.5 and test_cov.get(s, 0) > 0.9]
    w = np.full(len(port), 1.0 / len(port))

    dates = (piv_r1[port].dropna().index
             .intersection(piv_sig[port].dropna().index)
             .intersection(piv_nxt[port].dropna().index)).sort_values()
    R1 = piv_r1.loc[dates, port].to_numpy()
    SIG = piv_sig.loc[dates, port].to_numpy()
    RP_NEXT = (piv_nxt.loc[dates, port].to_numpy() * w).sum(axis=1)   # realized next-day port ret

    lam = 0.94
    S = np.cov(R1[:60].T)                                             # warm-up second moment
    sigp = np.empty(len(dates))
    for i in range(len(dates)):
        if i > 0:
            r = R1[i - 1]
            S = lam * S + (1 - lam) * np.outer(r, r)                  # EWMA cov, info up to i-1
        dv = np.sqrt(np.diag(S))
        Rt = S / np.outer(dv, dv)                                     # correlation
        D = SIG[i]                                                    # HAR σ̂ per stock at i
        sigp[i] = np.sqrt(max(w @ (np.outer(D, D) * Rt) @ w, 1e-12))

    zp = RP_NEXT / sigp
    is_hist = dates <= pd.Timestamp(VAL_END)
    _zph = zp[is_hist & np.isfinite(zp)]
    qp05 = float(np.quantile(_zph, 0.05))                            # portfolio fat-tail quantile
    qp01 = float(np.quantile(_zph, 0.01))
    esp05 = float(_zph[_zph <= qp05].mean())                         # portfolio ES multiplier
    esp01 = float(_zph[_zph <= qp01].mean())

    tm = dates > pd.Timestamp(VAL_END)
    var95_t = sigp[tm] * qp05
    var99_t = sigp[tm] * qp01
    realized_t = RP_NEXT[tm]
    br95 = realized_t < var95_t
    br99 = realized_t < var99_t
    npv = int(tm.sum())
    pr95, pp95 = kupiec(npv, int(br95.sum()), 0.05)
    pr99, pp99 = kupiec(npv, int(br99.sum()), 0.01)
    pind = christoffersen_ind(br95.astype(int))

    # diversification (as-of last day): portfolio VaR vs weighted sum of single-name VaRs
    port_var95 = float(sigp[-1] * qp05)
    sum_var95 = float(np.sum(w * SIG[-1] * q["q05"]))
    div_ratio = port_var95 / sum_var95
    test_dates, port_rets, var95_series, p_break95 = dates[tm], realized_t, var95_t, br95

    print("\n" + "=" * 74)
    print(f"PORTFOLIO VaR BACKTEST — equal-weight {len(port)} stocks, EWMA-cov, 1-day (n={npv:,})")
    print("=" * 74)
    print(f"  VaR95 breach rate {pr95:.3%}  (target 5.0%)  Kupiec p={pp95:.3g} "
          f"{'OK' if pp95 > 0.05 else 'reject'}")
    print(f"  VaR99 breach rate {pr99:.3%}  (target 1.0%)  Kupiec p={pp99:.3g} "
          f"{'OK' if pp99 > 0.05 else 'reject'}")
    print(f"  Christoffersen independence p={pind:.3g} {'OK' if pind > 0.05 else 'reject'}")
    print(f"  diversification ratio {div_ratio:.2f}  "
          f"(portfolio 1-day VaR95 {port_var95:.3%} vs Σw·VaR {sum_var95:.3%})")

    # ---- figure: portfolio returns vs VaR95 line, breaches marked ----
    fig, ax = plt.subplots(figsize=(11, 4))
    pr = np.array(port_rets) * 100; vv = np.array(var95_series) * 100
    ax.plot(test_dates, pr, lw=.7, color="#444", label="portfolio 1-day return")
    ax.plot(test_dates, vv, lw=1.1, color="#c0392b", label="FHS VaR-95")
    br_idx = np.array(p_break95)
    ax.scatter(np.array(test_dates)[br_idx], pr[br_idx], s=18, color="#c0392b",
               zorder=5, label=f"breaches ({int(br_idx.sum())})")
    ax.axhline(0, color="#888", lw=.5); ax.set_ylabel("%"); ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"Portfolio VaR-95 backtest (2021-23) — breach rate {pr95:.1%}, target 5%")
    fig.tight_layout(); OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_var_exceedances.png", dpi=130); plt.close(fig)

    # ---- persist backtest results + extend the model artifact ----
    pd.DataFrame([
        dict(scope="per_stock", horizon="1d", var="95", breach_rate=r95, kupiec_p=p95, n=n),
        dict(scope="per_stock", horizon="1d", var="99", breach_rate=r99, kupiec_p=p99, n=n),
        dict(scope="portfolio", horizon="1d", var="95", breach_rate=pr95, kupiec_p=pp95, n=npv),
        dict(scope="portfolio", horizon="1d", var="99", breach_rate=pr99, kupiec_p=pp99, n=npv),
    ]).to_csv(OUT / "risk_measures_backtest.csv", index=False)

    model["fhs"] = {"created": date.today().isoformat(),
                    "calibrated_on": "<=2020 (train+val), pooled across stocks",
                    "one_day": {**q, **es},
                    "horizons": fhs_h,
                    "portfolio": {"q05": qp05, "q01": qp01, "es05": esp05, "es01": esp01,
                                  "ewma_lambda": lam, "calibrated_weights": "equal-weight"},
                    "risk_level": "percentile of current sigma_h vs the symbol's own history"}
    MODEL_PATH.write_text(json.dumps(model, indent=2))
    print(f"\nextended model artifact -> {MODEL_PATH.relative_to(ROOT)} (added 'fhs' block)")
    print(f"backtest table -> {(OUT/'risk_measures_backtest.csv').relative_to(ROOT)}")
    print(f"figure         -> {(OUT/'fig_var_exceedances.png').relative_to(ROOT)}")

    _demo(panel, model, port, w)


def _risk_estimate(panel, model, sym, h) -> RiskEstimate:
    mh = model["horizons"][str(h)]
    fh = model["fhs"]["horizons"][str(h)]
    sub = panel[panel.symbol == sym].dropna(subset=mh["features"]).sort_values("date")
    row = sub.iloc[-1]
    sd = float(np.exp(mh["intercept"] + sum(row[f] * mh["coef"][f] for f in mh["features"]))
               * mh["smearing"])
    sh = sd * np.sqrt(h)
    sig_hist = sigma_daily(sub, mh) * np.sqrt(h)
    level = float((sig_hist <= sh).mean() * 100)
    return RiskEstimate(
        symbol=sym, horizon=h, sigma_daily=sd, sigma_h=sh,
        var_95=sh * fh["q05"], var_99=sh * fh["q01"], es_95=sh * fh["es05"],
        band_lo=sh * fh["q025"], band_hi=sh * fh["q975"], risk_level=level,
        as_of=str(row.date.date()), has_history=True)


def _demo(panel, model, port, w):
    print("\nINFERENCE DEMO (what the backend emits from the artifact):")
    for h in HORIZONS:
        re = _risk_estimate(panel, model, "NVDA", h)
        print(f"  NVDA h={h}d: σ̂_{h}d={re.sigma_h*100:.1f}%  "
              f"VaR95={re.var_95*100:.1f}%  VaR99={re.var_99*100:.1f}%  "
              f"ES95={re.es_95*100:.1f}%  risk_level={re.risk_level:.0f}/100")


if __name__ == "__main__":
    main()
