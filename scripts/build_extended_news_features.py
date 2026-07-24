"""Build richer, no-look-ahead FNSPID + FinBERT daily news features.

The original training dataset compresses each symbol-day to mean sentiment,
headline count, and a news flag. FNSPID retains much more structure: headline,
summary, publisher, URL, and cross-symbol story duplication. This builder
extracts that structure without calling another language model.

All news dated t is shifted to the first trading session strictly after t,
matching build_training_dataset.py. FNSPID timestamps are not used because more
than 99% are recorded at midnight and therefore cannot support honest
pre-market/after-hours classification.

Feature families:
  attention   count level, trailing accumulation, abnormal count
  sentiment   distribution, disagreement, tails, surprise
  source      publisher diversity / entropy, summary availability
  diffusion   unique stories, duplication, cross-symbol story breadth
  novelty     new title-token share versus the preceding five news days
  event       keyword-based event-category shares

Outputs:
  data/processed/extended_news_features.parquet
  data/processed/extended_news_feature_spec.json

Run:
  python scripts/build_extended_news_features.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_training_dataset import next_trading_day, trading_days  # noqa: E402

RAW = ROOT / "FNSPID" / "final_dataset" / "news_top20_gold_silver_2013_2023.csv"
SCORED = ROOT / "data" / "processed" / "news_sentiment_scored.parquet"
PRICES = ROOT / "FNSPID" / "final_dataset" / "prices"
OUTPUT = ROOT / "data" / "processed" / "extended_news_features.parquet"
SPEC = ROOT / "data" / "processed" / "extended_news_feature_spec.json"
EPS = 1e-6

EVENT_PATTERNS = {
    "event_earnings": r"\b(?:earnings?|revenue|sales|profit|eps|quarterly|guidance|outlook)\b",
    "event_analyst": r"\b(?:analyst|upgrade|downgrade|price target|rating|initiates?|reiterates?)\b",
    "event_corporate_action": r"\b(?:merger|acquisition|acquire[sd]?|buyback|dividend|stock split|offering)\b",
    "event_legal_regulatory": r"\b(?:lawsuit|court|regulat|antitrust|investigat|probe|settlement|sec\b|fda\b)\b",
    "event_product": r"\b(?:launch|product|patent|approval|clinical trial|drug|chip|software|platform|artificial intelligence|ai\b)\b",
    "event_macro": r"\b(?:federal reserve|fed\b|inflation|interest rate|jobs report|gdp|tariff|geopolit|war\b|oil price)\b",
    "event_management": r"\b(?:ceo\b|cfo\b|executive|chairman|resign|appoint|management)\b",
    "event_financing": r"\b(?:debt|bond|credit|loan|bankrupt|liquidity|cash flow|refinanc)\b",
}

STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "stock",
    "stocks",
    "market",
    "markets",
    "company",
    "shares",
    "said",
    "says",
    "new",
}
TOKEN_RE = re.compile(r"[a-z][a-z0-9]{2,}")


def _story_key(url: pd.Series, title: pd.Series) -> pd.Series:
    canonical = (
        url.fillna("")
        .str.lower()
        .str.replace(r"[?#].*$", "", regex=True)
        .str.rstrip("/")
    )
    fallback = (
        title.fillna("")
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )
    return canonical.where(canonical.ne(""), fallback)


def _tokens(texts: Iterable[str]) -> set[str]:
    result: set[str] = set()
    for text in texts:
        result.update(
            token
            for token in TOKEN_RE.findall(str(text).lower())
            if token not in STOPWORDS
        )
    return result


def load_articles() -> pd.DataFrame:
    raw = pd.read_csv(RAW, low_memory=False)
    scored = pd.read_parquet(SCORED)
    if len(raw) != len(scored):
        raise ValueError(f"Raw/scored row mismatch: {len(raw)} != {len(scored)}")
    raw_dates = pd.to_datetime(raw["date"])
    scored_dates = pd.to_datetime(scored["date"])
    if not raw_dates.equals(scored_dates) or not raw["symbol"].equals(scored["symbol"]):
        raise ValueError("FinBERT cache is not row-aligned with the FNSPID source.")

    articles = raw.copy()
    articles["date"] = raw_dates
    articles["sentiment"] = scored["sentiment"].astype(float)
    articles["eff_date"] = next_trading_day(articles["date"], trading_days())
    articles["story_key"] = _story_key(articles["url"], articles["title"])
    breadth = articles.groupby("story_key")["symbol"].nunique()
    articles["story_breadth"] = articles["story_key"].map(breadth).astype(float)

    articles["publisher_clean"] = (
        articles["publisher"].fillna("").str.strip().replace("", "__missing__")
    )
    articles["summary_present"] = (
        articles["summary"].fillna("").str.strip().ne("").astype(float)
    )
    articles["text_length"] = (
        articles["title"].fillna("").str.len()
        + articles["summary"].fillna("").str.len()
    ).astype(float)
    articles["sent_abs"] = articles["sentiment"].abs()
    articles["sent_positive"] = (articles["sentiment"] >= 0.5).astype(float)
    articles["sent_negative"] = (articles["sentiment"] <= -0.5).astype(float)
    articles["sent_extreme"] = (articles["sent_abs"] >= 0.8).astype(float)
    articles["firm_specific"] = (articles["story_breadth"] == 1).astype(float)
    articles["broad_story"] = (articles["story_breadth"] >= 3).astype(float)

    text = (
        articles["title"].fillna("") + " " + articles["summary"].fillna("")
    ).str.lower()
    for feature, pattern in EVENT_PATTERNS.items():
        articles[feature] = text.str.contains(pattern, regex=True).astype(float)
    return articles


def _publisher_entropy(articles: pd.DataFrame) -> pd.Series:
    keys = ["symbol", "eff_date"]
    counts = (
        articles.groupby(keys + ["publisher_clean"])
        .size()
        .rename("publisher_count")
        .reset_index()
    )
    totals = counts.groupby(keys)["publisher_count"].transform("sum")
    probability = counts["publisher_count"] / totals
    counts["_entropy_term"] = -probability * np.log(probability)
    return counts.groupby(keys)["_entropy_term"].sum()


def _daily_novelty(articles: pd.DataFrame) -> pd.DataFrame:
    daily_tokens = (
        articles.groupby(["symbol", "eff_date"])["title"]
        .apply(_tokens)
        .rename("tokens")
        .reset_index()
    )
    parts = []
    for symbol, group in daily_tokens.groupby("symbol", sort=False):
        history: deque[set[str]] = deque(maxlen=5)
        values = []
        for tokens in group.sort_values("eff_date")["tokens"]:
            previous = set().union(*history) if history else set()
            novelty = (
                len(tokens - previous) / len(tokens) if tokens else 0.0
            )
            values.append(novelty)
            history.append(tokens)
        ordered = group.sort_values("eff_date").copy()
        ordered["title_token_novelty"] = values
        parts.append(ordered.drop(columns="tokens"))
    return pd.concat(parts, ignore_index=True)


def aggregate_daily(articles: pd.DataFrame) -> pd.DataFrame:
    keys = ["symbol", "eff_date"]
    group = articles.groupby(keys)
    daily = group.agg(
        news_count=("sentiment", "size"),
        unique_story_count=("story_key", "nunique"),
        sent_mean=("sentiment", "mean"),
        sent_std=("sentiment", "std"),
        sent_min=("sentiment", "min"),
        sent_max=("sentiment", "max"),
        sent_abs_mean=("sent_abs", "mean"),
        sent_positive_share=("sent_positive", "mean"),
        sent_negative_share=("sent_negative", "mean"),
        sent_extreme_share=("sent_extreme", "mean"),
        unique_publisher_count=("publisher_clean", "nunique"),
        publisher_missing_share=(
            "publisher_clean",
            lambda value: (value == "__missing__").mean(),
        ),
        summary_share=("summary_present", "mean"),
        mean_text_length=("text_length", "mean"),
        mean_story_breadth=("story_breadth", "mean"),
        max_story_breadth=("story_breadth", "max"),
        firm_specific_share=("firm_specific", "mean"),
        broad_story_share=("broad_story", "mean"),
        **{
            f"{name}_share": (name, "mean")
            for name in EVENT_PATTERNS
        },
    ).reset_index()
    daily["sent_std"] = daily["sent_std"].fillna(0.0)
    daily["sent_range"] = daily["sent_max"] - daily["sent_min"]
    daily["duplicate_share"] = (
        1 - daily["unique_story_count"] / daily["news_count"]
    )
    entropy = _publisher_entropy(articles).rename("publisher_entropy").reset_index()
    daily = daily.merge(entropy, on=keys, how="left")
    daily = daily.merge(_daily_novelty(articles), on=keys, how="left")
    return daily.rename(columns={"eff_date": "date"})


def _price_calendar(symbol: str) -> pd.DatetimeIndex:
    path = PRICES / f"{symbol}.csv"
    frame = pd.read_csv(path, usecols=["date"])
    return pd.DatetimeIndex(pd.to_datetime(frame["date"]).sort_values().unique())


def expand_to_trading_days(
    daily: pd.DataFrame, articles: pd.DataFrame
) -> pd.DataFrame:
    feature_columns = [
        column
        for column in daily.columns
        if column not in ("date", "symbol")
    ]
    ranges = (
        articles.groupby("symbol")["eff_date"]
        .agg(["min", "max"])
        .to_dict("index")
    )
    parts = []
    for symbol in sorted(daily["symbol"].unique()):
        calendar = pd.DataFrame({"date": _price_calendar(symbol)})
        calendar["symbol"] = symbol
        sub = calendar.merge(
            daily[daily["symbol"] == symbol],
            on=["date", "symbol"],
            how="left",
        )
        active = (
            (sub["date"] >= ranges[symbol]["min"])
            & (sub["date"] <= ranges[symbol]["max"])
        )
        sub["coverage_active"] = active.astype(int)
        sub[feature_columns] = sub[feature_columns].fillna(0.0)
        sub["has_news"] = (sub["news_count"] > 0).astype(int)
        sub["log_count"] = np.log1p(sub["news_count"])

        prior_mean = (
            sub["news_count"].rolling(20, min_periods=5).mean().shift(1)
        )
        prior_std = (
            sub["news_count"].rolling(20, min_periods=5).std().shift(1)
        )
        sub["count_z20"] = (
            (sub["news_count"] - prior_mean) / (prior_std + 1.0)
        ).fillna(0.0)
        sub["count_ratio20"] = (
            np.log1p(sub["news_count"]) - np.log1p(prior_mean)
        ).fillna(0.0)
        sub["news_count_3d"] = sub["news_count"].rolling(3).sum()
        sub["news_count_5d"] = sub["news_count"].rolling(5).sum()

        news_mean = sub["sent_mean"].where(sub["has_news"] == 1)
        prior_sent = news_mean.rolling(20, min_periods=5).mean().shift(1)
        sub["sent_surprise20"] = (
            (sub["sent_mean"] - prior_sent) * sub["has_news"]
        ).fillna(0.0)
        sub["sent_abs_surprise20"] = sub["sent_surprise20"].abs()

        last_news_date = sub["date"].where(sub["has_news"] == 1).ffill()
        sub["days_since_news"] = (
            sub["date"] - last_news_date
        ).dt.days.fillna(999).clip(upper=30)
        sub.loc[sub["coverage_active"] == 0, "days_since_news"] = 30
        parts.append(sub)
    return pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"])


def main() -> None:
    print("Loading row-aligned FNSPID + FinBERT articles …")
    articles = load_articles()
    print(f"  {len(articles):,} articles · {articles['symbol'].nunique()} symbols")
    print("Aggregating distribution/source/diffusion/event/novelty features …")
    daily = aggregate_daily(articles)
    print("Expanding to each symbol's trading calendar and adding trailing features …")
    features = expand_to_trading_days(daily, articles)
    features.to_parquet(OUTPUT, index=False)

    families = {
        "attention": [
            "has_news",
            "log_count",
            "count_z20",
            "count_ratio20",
            "news_count_3d",
            "news_count_5d",
            "days_since_news",
        ],
        "sentiment_distribution": [
            "sent_mean",
            "sent_std",
            "sent_range",
            "sent_abs_mean",
            "sent_positive_share",
            "sent_negative_share",
            "sent_extreme_share",
            "sent_surprise20",
            "sent_abs_surprise20",
        ],
        "source_diffusion_novelty": [
            "unique_story_count",
            "duplicate_share",
            "unique_publisher_count",
            "publisher_entropy",
            "publisher_missing_share",
            "summary_share",
            "mean_text_length",
            "mean_story_breadth",
            "max_story_breadth",
            "firm_specific_share",
            "broad_story_share",
            "title_token_novelty",
        ],
        "event_categories": [
            f"{name}_share" for name in EVENT_PATTERNS
        ],
    }
    spec = {
        "created": pd.Timestamp.utcnow().isoformat(),
        "source_articles": len(articles),
        "symbols": sorted(articles["symbol"].unique()),
        "date_range": [
            str(features["date"].min().date()),
            str(features["date"].max().date()),
        ],
        "no_lookahead": "news date shifted strictly to next trading session",
        "timestamp_note": "FNSPID intraday timestamps excluded: >99% are midnight",
        "families": families,
    }
    SPEC.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(
        f"Wrote {OUTPUT.relative_to(ROOT)} · "
        f"{len(features):,} rows · {sum(map(len, families.values()))} candidate features"
    )
    print(f"Spec  {SPEC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
