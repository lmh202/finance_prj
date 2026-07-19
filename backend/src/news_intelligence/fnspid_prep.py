"""FNSPID preprocessing — Developer 3 (historical corpus, stage 1).

Distills the huge FNSPID news dump (FNSPID/Stock_news/All_external.csv,
~5.5 GB: full article text + 4 machine summaries per row) down to what the
sentiment pipeline actually needs: **headlines for the top-N most-covered
symbols in a year range**, in a compact Parquet.

Strategy — the file never fits in memory, so stream it twice with pandas
chunks, loading ONLY the light columns (the article/summary columns are
dropped at parse time via `usecols`):

  pass 1  count rows per symbol inside the year range -> pick top N
  pass 2  keep rows for those symbols, light columns only -> Parquet

Run (from repo root; takes several minutes per pass on the 5.5 GB file):

    python backend/src/news_intelligence/fnspid_prep.py                 # auto top 20
    python backend/src/news_intelligence/fnspid_prep.py --top 30
    python backend/src/news_intelligence/fnspid_prep.py --symbols AAPL MSFT NVDA

Output:  data/processed/fnspid_top{N}_{start}_{end}.parquet
         columns: timestamp_utc, date, symbol, title, publisher, url
Load:    pd.read_parquet(path)
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
# All_external.csv is the small subset (ends mid-2020, uneven coverage);
# nasdaq_exteral_data.csv (22 GB) is the full corpus through 2023 —
# select with --source.
SOURCE = ROOT / "FNSPID" / "Stock_news" / "All_external.csv"
OUT_DIR = ROOT / "data" / "processed"

LIGHT_COLS = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]
CHUNK_ROWS = 200_000


def _chunks(usecols: List[str], source: Path = None):
    return pd.read_csv(
        source or SOURCE,
        usecols=usecols,
        chunksize=CHUNK_ROWS,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="skip",
    )


def count_symbols(start: int, end: int, source: Path = None) -> Counter:
    """Pass 1: news rows per symbol within [start, end] (by year prefix)."""
    counts: Counter = Counter()
    seen = 0
    for chunk in _chunks(["Date", "Stock_symbol"], source):
        years = pd.to_numeric(chunk["Date"].str[:4], errors="coerce")
        in_range = chunk.loc[(years >= start) & (years <= end), "Stock_symbol"]
        counts.update(in_range[in_range != ""])
        seen += len(chunk)
        print(f"\r  pass 1: {seen:,} rows scanned", end="", flush=True)
    print()
    return counts


def extract(symbols: List[str], start: int, end: int, out_path: Path,
            source: Path = None) -> pd.DataFrame:
    """Pass 2: light columns for the chosen symbols/years -> tidy frame."""
    wanted = set(symbols)
    parts = []
    seen = kept = 0
    for chunk in _chunks(LIGHT_COLS, source):
        years = pd.to_numeric(chunk["Date"].str[:4], errors="coerce")
        mask = (years >= start) & (years <= end) & chunk["Stock_symbol"].isin(wanted)
        part = chunk.loc[mask, LIGHT_COLS]
        if not part.empty:
            parts.append(part)
            kept += len(part)
        seen += len(chunk)
        print(f"\r  pass 2: {seen:,} scanned, {kept:,} kept", end="", flush=True)
    print()

    df = pd.concat(parts, ignore_index=True)
    df.columns = ["timestamp_utc", "title", "symbol", "url", "publisher"]
    df["date"] = df["timestamp_utc"].str[:10]
    df = df[["timestamp_utc", "date", "symbol", "title", "publisher", "url"]]
    df = df.drop_duplicates(subset=["symbol", "date", "title"])
    df = df.sort_values(["symbol", "timestamp_utc"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=20, help="top-N symbols by coverage")
    ap.add_argument("--start", type=int, default=2013)
    ap.add_argument("--end", type=int, default=2023)
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="explicit symbol list (skips pass 1)")
    ap.add_argument("--source", type=Path, default=SOURCE,
                    help="FNSPID CSV to process (default: All_external.csv)")
    args = ap.parse_args()

    if not args.source.exists():
        sys.exit(f"Source not found: {args.source}")

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
        label = f"custom{len(symbols)}"
    else:
        print(f"Counting news per symbol, {args.start}-{args.end} …")
        counts = count_symbols(args.start, args.end, args.source)
        top = counts.most_common(args.top)
        symbols = [s for s, _ in top]
        label = f"top{args.top}"
        print("  most-covered symbols:")
        for s, n in top:
            print(f"    {s:6s} {n:>9,}")

    out_path = OUT_DIR / f"fnspid_{label}_{args.start}_{args.end}.parquet"
    print(f"Extracting {len(symbols)} symbols -> {out_path.name} …")
    df = extract(symbols, args.start, args.end, out_path, args.source)

    print(f"\nDone: {len(df):,} headlines, {df['symbol'].nunique()} symbols, "
          f"{df['date'].min()} … {df['date'].max()}")
    print(f"File:  {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f'Load:  pd.read_parquet(r"{out_path}")')


if __name__ == "__main__":
    main()
