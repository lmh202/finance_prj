"""Walk-forward ablation: does news sentiment improve next-move prediction?

Model     gradient-boosted trees (sklearn HistGradientBoostingClassifier) —
          the standard, strong choice for tabular financial data; no deep-net
          deps, fully deterministic.
Ablation  price-only (11 features) vs price+news (+ sentiment, news_count,
          has_news).
Validation WALK-FORWARD by year (expanding window): train on all years < Y,
          test on year Y. Never a random shuffle — that would leak the future.
Metric    ROC-AUC (prevalence-robust; accuracy would be fooled by the ~60/40
          up-drift). Reported per fold, averaged, and out-of-fold (OOF).

Reads   data/processed/training_dataset.parquet
Run     python scripts/train_ablation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "training_dataset.parquet"

PRICE = ["ret_1d", "mom_20d", "mom_60d", "price_vs_sma50", "sma50_vs_sma200",
         "vol_20d", "rsi_14", "drawdown", "risk_adj_mom", "beta_60d", "rel_str_20d"]
NEWS = ["sentiment", "news_count", "has_news"]
FEATURE_SETS = {"price_only": PRICE, "price+news": PRICE + NEWS}
HORIZONS = {"5d": "label_up_5d", "20d": "label_up_20d"}

FIRST_TEST_YEAR = 2015   # train needs >=1 prior year (data starts 2013-10)


def new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, early_stopping=False, random_state=0,
    )


def run_horizon(df: pd.DataFrame, label: str, years: list[int]):
    """Return per-fold AUCs and OOF prediction arrays for both feature sets."""
    results = {fs: {"fold_auc": [], "fold_acc": []} for fs in FEATURE_SETS}
    oof = {fs: np.full(len(df), np.nan) for fs in FEATURE_SETS}
    y_all = df[label].astype(int).to_numpy()

    for test_year in years:
        tr = df["year"] < test_year
        te = df["year"] == test_year
        if tr.sum() < 500 or te.sum() < 100:
            continue
        for fs, cols in FEATURE_SETS.items():
            m = new_model()
            m.fit(df.loc[tr, cols], y_all[tr.to_numpy()])
            p = m.predict_proba(df.loc[te, cols])[:, 1]
            yte = y_all[te.to_numpy()]
            results[fs]["fold_auc"].append((test_year, roc_auc_score(yte, p)))
            results[fs]["fold_acc"].append(accuracy_score(yte, p > 0.5))
            oof[fs][te.to_numpy()] = p
    return results, oof


def main() -> None:
    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = df.sort_values("date").reset_index(drop=True)
    years = [y for y in range(FIRST_TEST_YEAR, df["year"].max() + 1)]

    print("=" * 70)
    print("WALK-FORWARD ABLATION  ·  gradient-boosted trees  ·  metric = ROC-AUC")
    print(f"data: {len(df):,} rows · {df['symbol'].nunique()} symbols · "
          f"test years {years[0]}–{years[-1]} ({len(years)} folds)")
    print("=" * 70)

    for hz, label in HORIZONS.items():
        res, oof = run_horizon(df, label, years)
        print(f"\n### Horizon: next {hz}  (target = {label})")
        print(f"{'test year':>10} | {'price_only':>11} | {'price+news':>11} | {'Δ news':>8}")
        print("-" * 50)
        po = dict(res["price_only"]["fold_auc"])
        pn = dict(res["price+news"]["fold_auc"])
        for y in sorted(po):
            d = pn[y] - po[y]
            print(f"{y:>10} | {po[y]:>11.4f} | {pn[y]:>11.4f} | {d:>+8.4f}")
        po_m = np.mean(list(po.values())); pn_m = np.mean(list(pn.values()))
        print("-" * 50)
        print(f"{'MEAN AUC':>10} | {po_m:>11.4f} | {pn_m:>11.4f} | {pn_m-po_m:>+8.4f}")

        # out-of-fold overall (pooled predictions across all folds)
        mask = ~np.isnan(oof["price_only"])
        y = df[label].astype(int).to_numpy()[mask]
        auc_po = roc_auc_score(y, oof["price_only"][mask])
        auc_pn = roc_auc_score(y, oof["price+news"][mask])
        print(f"  OOF pooled AUC : price_only {auc_po:.4f} · price+news {auc_pn:.4f} "
              f"· lift {auc_pn-auc_po:+.4f}")

    # ---- per-symbol news lift at 20d (addresses uneven news coverage) ----
    label = HORIZONS["20d"]
    _, oof = run_horizon(df, label, years)
    mask = ~np.isnan(oof["price_only"])
    sub = df.loc[mask, ["symbol", label]].copy()
    sub["po"] = oof["price_only"][mask]
    sub["pn"] = oof["price+news"][mask]
    cov = df.groupby("symbol")["has_news"].mean().mul(100)
    print("\n### Per-symbol news lift (20d, OOF AUC) — sorted by news coverage")
    print(f"{'symbol':>7} | {'news%':>6} | {'price_only':>11} | {'price+news':>11} | {'Δ':>8}")
    print("-" * 56)
    rows = []
    for s, g in sub.groupby("symbol"):
        y = g[label].astype(int)
        if y.nunique() < 2 or len(g) < 50:
            continue
        a_po, a_pn = roc_auc_score(y, g["po"]), roc_auc_score(y, g["pn"])
        rows.append((s, cov[s], a_po, a_pn, a_pn - a_po))
    for s, c, a_po, a_pn, d in sorted(rows, key=lambda r: -r[1]):
        print(f"{s:>7} | {c:>5.1f}% | {a_po:>11.4f} | {a_pn:>11.4f} | {d:>+8.4f}")
    lifts = [r[4] for r in rows]
    print("-" * 56)
    print(f"symbols where news helped: {sum(d>0 for d in lifts)}/{len(lifts)} · "
          f"mean per-symbol lift {np.mean(lifts):+.4f}")
    print("\n(AUC 0.50 = coin flip; >0.50 = predictive. News 'helps' if Δ > 0.)")


if __name__ == "__main__":
    main()
