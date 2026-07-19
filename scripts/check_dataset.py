"""Usability report for the built AURORA training dataset.

Reads data/processed/{training_dataset,sentiment_features,price_features}.parquet
and prints the checks that decide whether it's fit for walk-forward ML:
coverage, feature integrity, label balance, news-channel density, and the
sentiment distribution FinBERT produced.

Run:  python scripts/check_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"

FEATURES = [
    "ret_1d", "mom_20d", "mom_60d", "price_vs_sma50", "sma50_vs_sma200",
    "vol_20d", "rsi_14", "drawdown", "risk_adj_mom", "beta_60d", "rel_str_20d",
]


def line(c="-"):
    print(c * 66)


def main() -> None:
    df = pd.read_parquet(OUT / "training_dataset.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    n = len(df)

    line("=")
    print("AURORA TRAINING DATASET — USABILITY REPORT")
    line("=")
    print(f"rows            : {n:,}")
    print(f"symbols         : {df['symbol'].nunique()}  ({', '.join(sorted(df['symbol'].unique()))})")
    print(f"date range      : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"columns         : {len(df.columns)}")

    line()
    print("FEATURE INTEGRITY (nulls must be 0 for a trainable matrix)")
    nulls = df[FEATURES].isna().sum()
    bad = nulls[nulls > 0]
    print("  all", len(FEATURES), "features null-free" if bad.empty else f"NULLS: {dict(bad)}")

    line()
    print("LABEL BALANCE (class prior — near 50/50 is healthy)")
    for lab in ["label_up_5d", "label_up_20d"]:
        up = df[lab].mean() * 100
        print(f"  {lab:14}: {up:5.1f}% up / {100-up:5.1f}% down")

    line()
    print("NEWS CHANNEL COVERAGE (how often the optional feature fires)")
    cov = df["has_news"].mean() * 100
    print(f"  symbol-days WITH news : {df['has_news'].sum():,} / {n:,}  ({cov:.1f}%)")
    print("  per-symbol news coverage %:")
    per = df.groupby("symbol")["has_news"].mean().mul(100).sort_values(ascending=False)
    for s, v in per.items():
        print(f"      {s:5} {v:5.1f}%")

    line()
    print("SENTIMENT DISTRIBUTION (FinBERT, on news days only)")
    s = df.loc[df["has_news"] == 1, "sentiment"]
    print(f"  news-day sentiment: mean {s.mean():+.3f}  std {s.std():.3f}  "
          f"min {s.min():+.3f}  max {s.max():+.3f}")
    print(f"  positive {(s>0.1).mean()*100:4.1f}%  |  neutral {((s>=-0.1)&(s<=0.1)).mean()*100:4.1f}%  "
          f"|  negative {(s<-0.1).mean()*100:4.1f}%")

    line()
    print("NEWS COVERAGE BY YEAR (news-days / total symbol-days)")
    g = df.groupby("year")["has_news"].agg(["size", "sum"])
    for y, r in g.iterrows():
        print(f"  {y}: {int(r['sum']):>6,} / {int(r['size']):>6,}  ({r['sum']/r['size']*100:4.1f}%)")

    line()
    print("WALK-FORWARD FEASIBILITY (expanding-window folds by year)")
    yrs = sorted(df["year"].unique())
    print(f"  {len(yrs)} years available: {yrs[0]}-{yrs[-1]}")
    print("  example: train<=Y, test=Y+1 gives", len(yrs) - 1, "folds")
    rows_per_year = df.groupby("year").size()
    print(f"  rows/year: min {rows_per_year.min():,}  max {rows_per_year.max():,}  "
          f"(enough per fold: {'yes' if rows_per_year.min() > 500 else 'THIN'})")

    line("=")
    verdict = "USABLE" if bad.empty and 35 < df["label_up_20d"].mean() * 100 < 65 else "REVIEW"
    print(f"VERDICT: {verdict}")
    line("=")


if __name__ == "__main__":
    main()
