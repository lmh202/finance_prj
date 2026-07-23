"""Risk engine: HAR volatility forecasting, benchmarked honestly.

Predicts realized volatility over the next h days — the quantity our research
showed IS predictable (unlike direction). Trained on log(RV) with a pooled HAR
model, benchmarked against the two standards it must beat.

MODELS
  naive    last 22-day realized vol (random-walk vol)
  GARCH    GARCH(1,1) per stock, parameters estimated on TRAIN ONLY then filtered
           forward with fixed params (no look-ahead in the benchmark)
  HAR      pooled OLS on log realized vol at 5/22/66d + Parkinson range
           estimators + |r_t|   <- the risk engine
  HAR+news log_count + disagreement added (evaluated on the 12 news-covered stocks)

SPLITS (time-based, identical calendar boundaries for every stock; splitting by
stock would leak — market regimes hit all names at once)
  train 2013-2018 | val 2019-2020 | test 2021-2023, with a 20-day EMBARGO at each
  seam because the h-day-ahead target otherwise reaches across the boundary.

METRICS
  QLIKE   primary (robust to proxy noise; punishes under-forecasting risk)
  RMSE/R2 on log-vol
  MZ      Mincer-Zarnowitz a,b (want 0,1) — bias & scale calibration
  cover   P(|h-day return| <= 1.96*sigma*sqrt(h)) — want ~95%, the usable-for-sizing test
  DM      Diebold-Mariano (Newey-West) vs each benchmark

Run: python scripts/train_risk_engine.py
"""

import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from build_training_dataset import trading_days
from validate_news_metrics import news_features

PRICES = ROOT / "FNSPID" / "final_dataset" / "prices"
OUT = ROOT / "reports" / "risk_engine"
HORIZONS = [5, 20]
EMBARGO = 20
TRAIN_END, VAL_END = "2018-12-31", "2020-12-31"
NEWS_STOCKS = ["QCOM", "MU", "GLD", "COST", "PEP", "TXN", "ADBE", "AMD",
               "NVDA", "INTC", "SLV", "ASML"]

FEATS_BASE = ["l_rv5", "l_rv22", "l_rv66", "l_park5", "l_park22", "l_absret"]
FEATS_NEWS = FEATS_BASE + ["log_count", "disagreement", "has_news"]


def stocks():
    return sorted(p.stem for p in PRICES.glob("*.csv") if p.stem != "SPY")


def _vol_forecast(r_pct, d, prefix, spec, simulate=False):
    """Fit a conditional-variance model on TRAIN ONLY, then filter the full series
    with those frozen params and store h-day-ahead vol forecasts (decimal units)."""
    tr = r_pct[r_pct.index <= TRAIN_END]
    try:
        p = arch_model(tr, mean="Constant", **spec).fit(disp="off", show_warning=False).params
        fixed = arch_model(r_pct, mean="Constant", **spec).fix(p)
        kw = dict(horizon=max(HORIZONS), start=0, reindex=True)
        if simulate:                      # EGARCH: analytic multi-step unavailable
            kw.update(method="simulation", simulations=200)
        fc = fixed.forecast(**kw)
        for h in HORIZONS:
            d[f"{prefix}_{h}"] = np.sqrt(fc.variance.iloc[:, :h].mean(axis=1)) / 100.0
    except Exception:
        for h in HORIZONS:
            d[f"{prefix}_{h}"] = np.nan


def build_stock(sym: str) -> pd.DataFrame:
    px = pd.read_csv(PRICES / f"{sym}.csv")
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values("date").set_index("date")
    c, hi, lo = px["adj close"], px["high"], px["low"]
    ret = c.pct_change()

    d = pd.DataFrame(index=c.index)
    # ---- HAR components (close-to-close realized vol at 3 scales) ----
    d["rv5"], d["rv22"], d["rv66"] = (ret.rolling(w).std() for w in (5, 22, 66))
    # ---- range-based (Parkinson) — more efficient than close-to-close ----
    park = (np.log(hi / lo) ** 2) / (4 * np.log(2))
    d["park5"] = np.sqrt(park.rolling(5).mean())
    d["park22"] = np.sqrt(park.rolling(22).mean())
    d["absret"] = ret.abs()

    # ---- conditional-variance models: params from TRAIN ONLY, filtered forward ----
    r_pct = (ret.dropna() * 100.0)
    _vol_forecast(r_pct, d, "garch", dict(vol="GARCH", p=1, q=1))            # symmetric
    _vol_forecast(r_pct, d, "egarch", dict(vol="EGARCH", p=1, o=1, q=1),     # asymmetric
                  simulate=True)  # EGARCH has no analytic multi-step forecast

    # ---- targets: realized vol over (t, t+h] and the h-day return ----
    for h in HORIZONS:
        d[f"y_{h}"] = ret.rolling(h).std().shift(-h)
        d[f"fret_{h}"] = c.shift(-h) / c - 1
    d["symbol"] = sym
    return d.reset_index().rename(columns={"index": "date"})


def log_feats(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    for a, b in [("rv5", "l_rv5"), ("rv22", "l_rv22"), ("rv66", "l_rv66"),
                 ("park5", "l_park5"), ("park22", "l_park22"), ("absret", "l_absret")]:
        df[b] = np.log(df[a] + eps)
    return df


def qlike(realized_vol, fc_vol):
    rv2, s2 = realized_vol ** 2, np.maximum(fc_vol, 1e-8) ** 2
    r = rv2 / s2
    return float(np.mean(r - np.log(r) - 1))


def dm_test(loss_a, loss_b, lag):
    """Diebold-Mariano with Newey-West SE. >0 => a worse than b."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[np.isfinite(d)]
    n, mu = len(d), d.mean()
    g0 = np.var(d, ddof=1)
    s = g0
    for k in range(1, lag + 1):
        cov = np.cov(d[k:], d[:-k], ddof=1)[0, 1]
        s += 2 * (1 - k / (lag + 1)) * cov
    se = np.sqrt(max(s, 1e-18) / n)
    t = mu / se
    return float(t), float(2 * (1 - stats.norm.cdf(abs(t))))


def evaluate(name, y_vol, fc_vol, fret, h):
    m = np.isfinite(y_vol) & np.isfinite(fc_vol) & (fc_vol > 0)
    y, f, fr = y_vol[m], fc_vol[m], fret[m]
    ly, lf = np.log(y + 1e-6), np.log(f + 1e-6)
    b, a = np.polyfit(lf, ly, 1)                       # MZ on logs: slope, intercept
    r2 = 1 - np.sum((ly - lf) ** 2) / np.sum((ly - ly.mean()) ** 2)
    cover = float(np.mean(np.abs(fr) <= 1.96 * f * np.sqrt(h)))
    return dict(model=name, n=int(m.sum()), qlike=qlike(y, f),
                rmse_log=float(np.sqrt(np.mean((ly - lf) ** 2))), r2_log=float(r2),
                mz_a=float(a), mz_b=float(b), coverage=cover), qlike_series(y, f)


def qlike_series(y, f):
    r = (y ** 2) / np.maximum(f, 1e-8) ** 2
    return r - np.log(r) - 1


def main():
    print("building panel (GARCH params fit on TRAIN only) …")
    panel = pd.concat([build_stock(s) for s in stocks()], ignore_index=True)
    panel = log_feats(panel)
    nf = news_features(trading_days())[["symbol", "date", "log_count", "disagreement"]]
    panel = panel.merge(nf, on=["symbol", "date"], how="left")
    panel["has_news"] = panel["log_count"].notna().astype(int)
    panel[["log_count", "disagreement"]] = panel[["log_count", "disagreement"]].fillna(0.0)

    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for h in HORIZONS:
        y, g, eg = f"y_{h}", f"garch_{h}", f"egarch_{h}"
        need = FEATS_BASE + [y, g, eg, f"fret_{h}"]
        d = panel.dropna(subset=need).copy()
        d["ly"] = np.log(d[y] + 1e-6)
        # model forecasts as GBM inputs (stacking): let the trees learn WHEN to
        # trust GARCH/EGARCH vs the HAR terms
        d["l_garch"] = np.log(d[g] + 1e-6)
        d["l_egarch"] = np.log(d[eg] + 1e-6)
        gbm_feats = FEATS_BASE + ["l_garch", "l_egarch", "log_count", "disagreement", "has_news"]

        # embargoed time splits (same calendar boundaries for every stock)
        emb = pd.Timedelta(days=int(EMBARGO * 1.5))
        tr = d[d.date <= pd.Timestamp(TRAIN_END) - emb]
        va = d[(d.date > TRAIN_END) & (d.date <= pd.Timestamp(VAL_END) - emb)]
        te = d[d.date > VAL_END]
        print(f"\nh={h}d  train {len(tr):,} ({tr.symbol.nunique()} stk) | "
              f"val {len(va):,} ({va.symbol.nunique()}) | test {len(te):,} ({te.symbol.nunique()})")

        # ---- fit pooled HAR + GBM on train (log space) ----
        mdl = LinearRegression().fit(tr[FEATS_BASE], tr["ly"])
        gbm = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, early_stopping=False, random_state=0
        ).fit(tr[gbm_feats], tr["ly"])

        # log->level smearing corrections, estimated OUT-OF-SAMPLE on validation so
        # the GBM's tight in-sample fit doesn't under-correct it (same rule for both)
        def smear(resid):
            return float(np.exp(np.var(resid, ddof=1) / 2))
        corr = smear(va["ly"] - mdl.predict(va[FEATS_BASE])) if len(va) else 1.0
        corr_gbm = smear(va["ly"] - gbm.predict(va[gbm_feats])) if len(va) else 1.0

        results, losses = [], {}
        for split_name, s in [("VAL", va), ("TEST", te)]:
            if len(s) == 0:
                continue
            fr, yv = s[f"fret_{h}"].to_numpy(), s[y].to_numpy()
            cand = {
                "naive (rv22)": s["rv22"].to_numpy(),
                "GARCH(1,1)": s[g].to_numpy(),
                "EGARCH(1,1,1)": s[eg].to_numpy(),
                "HAR": np.exp(mdl.predict(s[FEATS_BASE])) * corr,
                "GBM (trees)": np.exp(gbm.predict(s[gbm_feats])) * corr_gbm,
            }
            for nm, fc in cand.items():
                row, ls = evaluate(nm, yv, fc, fr, h)
                row["split"], row["horizon"] = split_name, h
                results.append(row)
                losses[(split_name, nm)] = ls
        res = pd.DataFrame(results)
        all_rows.append(res)

        for split_name in ["VAL", "TEST"]:
            sub = res[res.split == split_name]
            if sub.empty:
                continue
            print(f"\n  [{split_name}] h={h}d — lower QLIKE is better; MZ ideal a=0 b=1; coverage ~0.95")
            print(f"  {'model':>20} | {'QLIKE':>7} | {'RMSE_log':>8} | {'R2_log':>7} | "
                  f"{'MZ a':>6} {'MZ b':>6} | {'cover':>6} | {'n':>6}")
            print("  " + "-" * 88)
            for _, r in sub.iterrows():
                print(f"  {r.model:>20} | {r.qlike:>7.4f} | {r.rmse_log:>8.4f} | {r.r2_log:>7.3f} | "
                      f"{r.mz_a:>6.3f} {r.mz_b:>6.3f} | {r.coverage:>6.3f} | {int(r.n):>6}")
            # DM: every model vs HAR (the incumbent). t<0 => the model beats HAR.
            for other in ["naive (rv22)", "GARCH(1,1)", "EGARCH(1,1,1)", "GBM (trees)"]:
                k1, k2 = (split_name, other), (split_name, "HAR")
                if k1 in losses and k2 in losses:
                    t, p = dm_test(losses[k1], losses[k2], lag=h)
                    verdict = "beats HAR" if t < 0 else "worse than HAR"
                    print(f"    DM  {other:<14} vs HAR: t={t:+.2f} p={p:.3g}  -> {verdict}"
                          f"{' (sig)' if p < 0.05 else ''}")

        # ---- news increment, on the 12 news-covered stocks only ----
        trn = tr[tr.symbol.isin(NEWS_STOCKS)]
        ten = te[te.symbol.isin(NEWS_STOCKS)]
        if len(ten) > 100:
            m_b = LinearRegression().fit(trn[FEATS_BASE], trn["ly"])
            m_n = LinearRegression().fit(trn[FEATS_NEWS], trn["ly"])
            cb = np.exp(np.var(trn["ly"] - m_b.predict(trn[FEATS_BASE]), ddof=1) / 2)
            cn = np.exp(np.var(trn["ly"] - m_n.predict(trn[FEATS_NEWS]), ddof=1) / 2)
            fb = np.exp(m_b.predict(ten[FEATS_BASE])) * cb
            fn = np.exp(m_n.predict(ten[FEATS_NEWS])) * cn
            qb, qn = qlike(ten[y].to_numpy(), fb), qlike(ten[y].to_numpy(), fn)
            t, p = dm_test(qlike_series(ten[y].to_numpy(), fn),
                           qlike_series(ten[y].to_numpy(), fb), lag=h)
            print(f"\n  [TEST] news increment (12 stocks, n={len(ten):,}): "
                  f"QLIKE HAR {qb:.4f} -> HAR+news {qn:.4f} ({qn-qb:+.4f}); "
                  f"DM t={t:+.2f} p={p:.3g} -> {'news helps' if t < 0 and p < 0.05 else 'no significant gain'}")

    pd.concat(all_rows).to_csv(OUT / "risk_engine_results.csv", index=False)
    print(f"\nresults -> {OUT.relative_to(ROOT)}/risk_engine_results.csv")


if __name__ == "__main__":
    main()
