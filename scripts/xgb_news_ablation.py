"""Does news help the STRONG model? XGBoost-Gamma: price-only vs price+news.

The leaderboard's best 5-session models (linear_gamma, xgb_price) are PRICE-ONLY,
and the joint diagnostic already showed news into the linear Gamma model adds only
~0.08%. But the specific "xgb_price + news features" arm was never isolated. This
script closes that gap directly, reusing the harness's own building blocks so the
comparison matches the leaderboard methodology:

  - same 21-symbol OHLC/news panel        (build_historical_panel)
  - same 34 price features                (PRICE_FEATURES) + the deployable news set
  - same Gamma objective + fitter         (fit_xgb_gamma, reg:gamma)
  - same embargoed ANNUAL walk-forward    (time_fold, test years 2018-2023)
  - same stock-equal QLIKE                (qlike_loss + symbol_equal_weights)

A FIXED config is used for both arms, so any difference is the news features alone
(not tuning). Reports per-year QLIKE, mean, positive years, and a Diebold-Mariano
test on the pooled per-observation loss differential.

Run: python scripts/xgb_news_ablation.py
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
from scipy import stats

from risk_engine_optimization import (PRICE_FEATURES, build_historical_panel,
                                      fit_xgb_gamma, qlike_loss, symbol_equal_weights,
                                      time_fold)

H = 5
DIRECT_VARIANCE_SCALE = 10_000.0
EPS = 1e-6
DEVICE = "cpu"
TEST_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
# one fixed Gamma config for BOTH arms — isolates the effect of the news features
CONFIG = {"max_depth": 4, "learning_rate": 0.03, "subsample": 0.8,
          "colsample_bytree": 0.8, "min_child_weight": 5.0, "reg_lambda": 1.0}


def _target(frame):
    return np.clip(frame[f"target_variance_{H}d"].to_numpy() * DIRECT_VARIANCE_SCALE, EPS, 1e4)


def _forecast(model, frame, feats):
    return np.sqrt(np.clip(model.predict(frame[feats]), EPS, 1e4) / DIRECT_VARIANCE_SCALE)


def _dm(loss_a, loss_b, lag=H):
    """Diebold-Mariano (Newey-West). d=a-b; t<0 => a better (lower loss)."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[np.isfinite(d)]
    n, mu, g0 = len(d), d.mean(), np.var(d, ddof=1)
    s = g0
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * np.cov(d[k:], d[:-k], ddof=1)[0, 1]
    t = mu / np.sqrt(max(s, 1e-18) / n)
    return float(t), float(2 * (1 - stats.norm.cdf(abs(t))))


def main():
    print("building 21-symbol panel …")
    panel, groups, research_news, deployable_news = build_historical_panel()
    news_feats = [f for f in deployable_news if f in panel.columns]
    price_news_feats = PRICE_FEATURES + news_feats
    ph = panel.dropna(subset=[f"target_variance_{H}d"]).copy()
    print(f"panel {len(ph):,} rows · {ph.symbol.nunique()} symbols · "
          f"{len(PRICE_FEATURES)} price feats + {len(news_feats)} news feats\n")

    rows, pooled_price, pooled_news = [], [], []
    for year in TEST_YEARS:
        fold = time_fold(ph, year, H)
        tr = ph[fold.train_mask].copy()
        te = ph[fold.test_mask].copy()
        # time-based inner validation for early stopping: last train year
        val_year = year - 1
        val = tr[tr.date.dt.year == val_year]
        fit = tr[tr.date.dt.year < val_year]
        if len(val) < 200 or len(fit) < 500:      # fallback: last 15% by date
            cut = tr.date.quantile(0.85)
            fit, val = tr[tr.date <= cut], tr[tr.date > cut]

        yt = np.sqrt(te[f"target_variance_{H}d"].to_numpy())    # realized vol on test
        w = symbol_equal_weights(te)
        res = {}
        for name, feats in [("price", PRICE_FEATURES), ("price+news", price_news_feats)]:
            m = fit_xgb_gamma(fit, val, feats, _target(fit), _target(val), CONFIG, DEVICE)
            fc = _forecast(m, te, feats)
            loss = qlike_loss(yt, fc)
            res[name] = (float(np.average(loss, weights=w)), loss)
        (q_p, l_p), (q_n, l_n) = res["price"], res["price+news"]
        pooled_price.append(l_p); pooled_news.append(l_n)
        gain = (q_p - q_n) / q_p
        rows.append(dict(year=year, n=len(te), qlike_price=q_p, qlike_news=q_n, gain=gain))
        print(f"  {year}: QLIKE price {q_p:.4f}  price+news {q_n:.4f}  "
              f"news gain {gain:+.2%}")

    res = pd.DataFrame(rows)
    mp, mn = res.qlike_price.mean(), res.qlike_news.mean()
    mean_gain = (mp - mn) / mp
    pos = int((res.gain > 0).sum())
    lp, ln = np.concatenate(pooled_price), np.concatenate(pooled_news)
    t, p = _dm(ln, lp)      # news vs price: t<0 => news better

    OUT = ROOT / "reports" / "risk_engine_optimization"
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "xgb_news_ablation.csv", index=False)

    print("\n" + "=" * 70)
    print("XGBoost-Gamma  ·  PRICE-ONLY vs PRICE+NEWS  ·  verdict")
    print("=" * 70)
    print(f"  mean QLIKE   price {mp:.4f}   price+news {mn:.4f}")
    print(f"  news gain    {mean_gain:+.2%}   (positive years {pos}/{len(res)})")
    print(f"  DM (pooled)  t={t:+.2f}  p={p:.3g}  -> "
          f"{'news significantly helps' if t < 0 and p < 0.05 else ('news significantly hurts' if t > 0 and p < 0.05 else 'no significant difference')}")
    print("-" * 70)
    if abs(mean_gain) < 0.01 or not (p < 0.05 and t < 0):
        print("  => News does NOT meaningfully improve the strong XGBoost model —")
        print("     consistent with linear-Gamma+news (+0.08%). A flexible price model")
        print("     already captures what news attention proxies. This is an honest,")
        print("     defensible finding, not a failure.")
    else:
        print("  => News DOES help XGBoost — 'XGBoost-Gamma + News' both wins AND uses")
        print("     news. Re-run its VaR/ES backtest before promoting.")
    print(f"\n  -> {(OUT / 'xgb_news_ablation.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
