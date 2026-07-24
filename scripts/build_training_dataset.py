"""Build the AURORA training dataset from the filtered FNSPID corpus.

Pipeline (all local — NO LLM API; FinBERT runs on-device):

  1. score news    FinBERT sentiment per headline (summary if present, else title)
  2. sentiment     aggregate -> per symbol-day, shifted to the NEXT trading day
                   (no look-ahead), schema: date,symbol,sentiment,news_count,has_news
  3. price feats   per symbol-day indicators from adj-close (+ SPY beta/rel-strength)
                   and forward-return labels
  4. merge         left-join sentiment onto price features (no news -> 0/0/0)

Inputs
  FNSPID/final_dataset/news_top20_gold_silver_2013_2023.csv
  FNSPID/final_dataset/prices/*.csv          (21 tickers + SPY benchmark)

Outputs (data/processed/)
  news_sentiment_scored.parquet    per-headline FinBERT score (cached; expensive)
  sentiment_features.parquet       the frozen sentiment_features contract shape
  price_features.parquet           per symbol-day price indicators + labels
  training_dataset.parquet         the merged, trainable table

Run:  python scripts/build_training_dataset.py         (reuses cached scores)
      python scripts/build_training_dataset.py --rescore   (force re-score)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "FNSPID" / "final_dataset"
NEWS_CSV = DS / "news_top20_gold_silver_2013_2023.csv"
PRICES = DS / "prices"
OUT = ROOT / "data" / "processed"
BENCHMARK = "SPY"

SCORED = OUT / "news_sentiment_scored.parquet"
SENT_FEATS = OUT / "sentiment_features.parquet"
PRICE_FEATS = OUT / "price_features.parquet"
TRAIN = OUT / "training_dataset.parquet"
EXTENDED_NEWS = OUT / "extended_news_features.parquet"

TRADING_DAYS = 252


# --------------------------------------------------------------- 1. FinBERT
def score_news(rescore: bool = False) -> pd.DataFrame:
    if SCORED.exists() and not rescore:
        print(f"[1/4] using cached scores: {SCORED.name}")
        return pd.read_parquet(SCORED)

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("[1/4] scoring news with FinBERT …")
    news = pd.read_csv(NEWS_CSV, dtype=str, keep_default_na=False)
    summ = news["summary"].str.strip()
    title = news["title"].str.strip()
    text = summ.where(summ != "", title)          # summary if present, else headline
    news["text_used"] = text
    news["text_source"] = np.where(summ != "", "summary", "title")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = (
        AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", use_safetensors=True)
        .to(dev)
        .eval()
    )
    lbl = model.config.id2label
    pos = next(i for i, l in lbl.items() if l.lower() == "positive")
    neg = next(i for i, l in lbl.items() if l.lower() == "negative")

    texts = text.tolist()
    scores = np.empty(len(texts), dtype=np.float32)
    B = 256
    with torch.no_grad():
        for i in range(0, len(texts), B):
            enc = tok(texts[i : i + B], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(dev)
            p = F.softmax(model(**enc).logits, dim=-1)
            scores[i : i + B] = (p[:, pos] - p[:, neg]).cpu().numpy()
            if i % (B * 40) == 0:
                print(f"\r      {i:,}/{len(texts):,}", end="", flush=True)
    print(f"\r      {len(texts):,}/{len(texts):,}")

    news["sentiment"] = scores
    out = news[["date", "symbol", "sentiment", "text_source"]].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SCORED, index=False)
    print(f"      wrote {SCORED.name} ({len(out):,} rows)")
    return out


# ---------------------------------------------- market calendar (no look-ahead)
def trading_days() -> np.ndarray:
    spy = pd.read_csv(PRICES / f"{BENCHMARK}.csv", usecols=["date"])
    return np.sort(pd.to_datetime(spy["date"]).values)


def next_trading_day(dates: pd.Series, calendar: np.ndarray) -> pd.Series:
    """First trading day STRICTLY AFTER each calendar date — so a headline
    can only inform the following session, never its own."""
    d = pd.to_datetime(dates).values
    idx = np.searchsorted(calendar, d, side="right")   # strictly greater
    idx = np.clip(idx, 0, len(calendar) - 1)
    return pd.Series(calendar[idx], index=dates.index)


# --------------------------------------------------------- 2. sentiment feats
def build_sentiment_features(scored: pd.DataFrame, calendar: np.ndarray) -> pd.DataFrame:
    print("[2/4] aggregating sentiment -> next-trading-day features …")
    df = scored.copy()
    df["eff_date"] = next_trading_day(df["date"], calendar)
    agg = (
        df.groupby(["eff_date", "symbol"])
        .agg(sentiment=("sentiment", "mean"), news_count=("sentiment", "size"))
        .reset_index()
        .rename(columns={"eff_date": "date"})
    )
    agg["has_news"] = 1
    agg["sentiment"] = agg["sentiment"].round(4)
    agg = agg.sort_values(["symbol", "date"]).reset_index(drop=True)
    agg.to_parquet(SENT_FEATS, index=False)
    print(f"      wrote {SENT_FEATS.name} ({len(agg):,} symbol-days with news)")
    return agg


# ------------------------------------------------------------- 3. price feats
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_price_features(calendar: np.ndarray) -> pd.DataFrame:
    print("[3/4] engineering price features (+ SPY benchmark) …")
    closes = {}
    for f in sorted(PRICES.glob("*.csv")):
        sym = f.stem
        px = pd.read_csv(f, usecols=["date", "adj close"])
        px["date"] = pd.to_datetime(px["date"])
        closes[sym] = px.set_index("date")["adj close"].sort_index()
    wide = pd.DataFrame(closes).sort_index()

    spy = wide[BENCHMARK]
    spy_ret = spy.pct_change()
    spy_mom20 = spy.pct_change(20)

    rows = []
    for sym in wide.columns:
        if sym == BENCHMARK:
            continue
        c = wide[sym].dropna()
        if len(c) < 260:
            continue
        ret1 = c.pct_change()
        feat = pd.DataFrame(index=c.index)
        feat["ret_1d"] = ret1
        feat["mom_20d"] = c.pct_change(20)
        feat["mom_60d"] = c.pct_change(60)
        sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
        feat["price_vs_sma50"] = c / sma50 - 1
        feat["sma50_vs_sma200"] = sma50 / sma200 - 1
        feat["vol_20d"] = ret1.rolling(20).std() * np.sqrt(TRADING_DAYS)
        feat["rsi_14"] = _rsi(c)
        feat["drawdown"] = c / c.cummax() - 1
        feat["risk_adj_mom"] = feat["mom_20d"] / feat["vol_20d"].replace(0, np.nan)
        # 60d beta + relative strength vs SPY (aligned on this symbol's calendar)
        sr = spy_ret.reindex(c.index)
        cov = ret1.rolling(60).cov(sr)
        var = sr.rolling(60).var()
        feat["beta_60d"] = cov / var.replace(0, np.nan)
        feat["rel_str_20d"] = feat["mom_20d"] - spy_mom20.reindex(c.index)
        # forward-return LABELS (the only look-ahead allowed — it's the target)
        feat["fwd_ret_5d"] = c.shift(-5) / c - 1
        feat["fwd_ret_20d"] = c.shift(-20) / c - 1
        feat["label_up_5d"] = (feat["fwd_ret_5d"] > 0).astype("Int64")
        feat["label_up_20d"] = (feat["fwd_ret_20d"] > 0).astype("Int64")
        feat.insert(0, "symbol", sym)
        feat.index.name = "date"
        rows.append(feat.reset_index())

    allf = pd.concat(rows, ignore_index=True)
    feature_cols = [
        "ret_1d", "mom_20d", "mom_60d", "price_vs_sma50", "sma50_vs_sma200",
        "vol_20d", "rsi_14", "drawdown", "risk_adj_mom", "beta_60d", "rel_str_20d",
    ]
    # drop warm-up rows (need 200d SMA + 60d beta) and rows without a 20d-forward label
    allf = allf.dropna(subset=feature_cols + ["fwd_ret_20d"]).reset_index(drop=True)
    allf.to_parquet(PRICE_FEATS, index=False)
    print(f"      wrote {PRICE_FEATS.name} ({len(allf):,} symbol-days, {len(feature_cols)} features)")
    return allf, feature_cols


# ------------------------------------------------------------------ 4. merge
def merge(price: pd.DataFrame, sent: pd.DataFrame) -> pd.DataFrame:
    print("[4/4] merging price + news …")
    price["date"] = pd.to_datetime(price["date"])
    sent["date"] = pd.to_datetime(sent["date"])
    df = price.merge(sent, on=["date", "symbol"], how="left")
    df["sentiment"] = df["sentiment"].fillna(0.0)
    df["news_count"] = df["news_count"].fillna(0).astype(int)
    df["has_news"] = df["has_news"].fillna(0).astype(int)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    if EXTENDED_NEWS.exists():
        from augment_training_dataset import merge_extended_features

        extended = pd.read_parquet(EXTENDED_NEWS)
        df, stats = merge_extended_features(df, extended)
        print(
            "      merged "
            f"{len(stats['added_columns']):,} extended FNSPID/FinBERT features "
            f"from {EXTENDED_NEWS.name}"
        )
    else:
        print(
            f"      warning: {EXTENDED_NEWS.name} not found; "
            "writing the legacy three-column news contract only"
        )
    df.to_parquet(TRAIN, index=False)
    print(f"      wrote {TRAIN.name} ({len(df):,} rows)")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true", help="force FinBERT re-score")
    args = ap.parse_args()

    cal = trading_days()
    scored = score_news(args.rescore)
    sent = build_sentiment_features(scored, cal)
    price, _ = build_price_features(cal)
    merge(price, sent)
    print("done.")


if __name__ == "__main__":
    main()
