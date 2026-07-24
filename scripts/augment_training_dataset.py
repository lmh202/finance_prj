"""Merge extended FNSPID/FinBERT news features into the training dataset.

The merge is one-to-one on (date, symbol). The original labels and price
features are preserved, while extended-news-owned columns are refreshed from
``extended_news_features.parquet``. When input and output are the same file,
the parquet is replaced atomically after an optional first-run backup.

Run:
    python scripts/augment_training_dataset.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DEFAULT_TRAIN = PROCESSED / "training_dataset.parquet"
DEFAULT_EXTENDED = PROCESSED / "extended_news_features.parquet"
DEFAULT_BACKUP = PROCESSED / "training_dataset.before_extended_news.parquet"

KEYS = ["date", "symbol"]
CONTRACT_COLUMNS = {"news_count", "has_news"}
LABEL_COLUMNS = [
    "fwd_ret_5d",
    "fwd_ret_20d",
    "label_up_5d",
    "label_up_20d",
]


def _normalise_keys(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [column for column in KEYS if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing key columns: {missing}")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["symbol"] = result["symbol"].astype(str)
    duplicate_count = int(result.duplicated(KEYS).sum())
    if duplicate_count:
        raise ValueError(
            f"{name} contains {duplicate_count:,} duplicate (date, symbol) keys"
        )
    return result


def _validate_contract(base: pd.DataFrame, extended: pd.DataFrame) -> None:
    missing = CONTRACT_COLUMNS.difference(base.columns)
    missing.update(CONTRACT_COLUMNS.difference(extended.columns))
    if missing:
        raise ValueError(f"news contract columns are missing: {sorted(missing)}")

    contract = base[KEYS + sorted(CONTRACT_COLUMNS)].merge(
        extended[KEYS + sorted(CONTRACT_COLUMNS)],
        on=KEYS,
        how="left",
        suffixes=("_base", "_extended"),
        validate="one_to_one",
    )
    for column in sorted(CONTRACT_COLUMNS):
        left = contract[f"{column}_base"].to_numpy()
        right = contract[f"{column}_extended"].to_numpy()
        equal = np.isclose(left, right, equal_nan=True)
        if not equal.all():
            raise ValueError(
                f"{column} disagrees on {int((~equal).sum()):,} matched rows"
            )

    # Legacy sentiment is rounded to four decimals. Keep it for compatibility,
    # but verify that it agrees with the richer unrounded daily mean.
    if "sentiment" in base.columns and "sent_mean" in extended.columns:
        sentiment = base[KEYS + ["sentiment", "has_news"]].merge(
            extended[KEYS + ["sent_mean"]],
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
        news_rows = sentiment["has_news"].eq(1)
        max_difference = (
            sentiment.loc[news_rows, "sentiment"]
            - sentiment.loc[news_rows, "sent_mean"]
        ).abs().max()
        if pd.notna(max_difference) and max_difference > 5.1e-5:
            raise ValueError(
                "sentiment and sent_mean disagree beyond four-decimal rounding: "
                f"max absolute difference={max_difference:.8g}"
            )


def merge_extended_features(
    base: pd.DataFrame,
    extended: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return the augmented table plus merge statistics."""
    base = _normalise_keys(base, "training dataset")
    extended = _normalise_keys(extended, "extended news dataset")

    coverage = base[KEYS].merge(
        extended[KEYS],
        on=KEYS,
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    missing_count = int(coverage["_merge"].ne("both").sum())
    if missing_count:
        raise ValueError(
            f"extended news dataset does not cover {missing_count:,} training rows"
        )

    _validate_contract(base, extended)

    feature_columns = [
        column
        for column in extended.columns
        if column not in KEYS and column not in CONTRACT_COLUMNS
    ]
    added_columns = [column for column in feature_columns if column not in base.columns]
    refreshed_columns = [
        column for column in feature_columns if column in base.columns
    ]

    original_columns = list(base.columns)
    original_keys = base[KEYS].copy()
    original_labels = base[
        [column for column in LABEL_COLUMNS if column in base.columns]
    ].copy()

    result = base.drop(columns=refreshed_columns).merge(
        extended[KEYS + feature_columns],
        on=KEYS,
        how="left",
        sort=False,
        validate="one_to_one",
    )

    missing_before_fill = int(result[feature_columns].isna().sum().sum())
    result[feature_columns] = result[feature_columns].fillna(0.0)
    residual_missing = int(result[feature_columns].isna().sum().sum())
    if residual_missing:
        raise ValueError(
            f"{residual_missing:,} missing extended-feature values remain"
        )

    if len(result) != len(base):
        raise AssertionError("row count changed during extended-news merge")
    pd.testing.assert_frame_equal(
        result[KEYS].reset_index(drop=True),
        original_keys.reset_index(drop=True),
        check_dtype=True,
    )
    if not original_labels.empty:
        pd.testing.assert_frame_equal(
            result[original_labels.columns].reset_index(drop=True),
            original_labels.reset_index(drop=True),
            check_dtype=True,
        )

    protected_columns = [
        column for column in original_columns if column not in refreshed_columns
    ]
    result = result[protected_columns + feature_columns]

    stats: dict[str, object] = {
        "rows": len(result),
        "columns": len(result.columns),
        "added_columns": added_columns,
        "refreshed_columns": refreshed_columns,
        "filled_missing_values": missing_before_fill,
    }
    return result, stats


def _write_atomic(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.",
        suffix=".tmp.parquet",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        check = pd.read_parquet(temporary)
        if check.shape != frame.shape or list(check.columns) != list(frame.columns):
            raise IOError("temporary parquet failed shape/schema verification")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--extended", type=Path, default=DEFAULT_EXTENDED)
    parser.add_argument("--output", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument(
        "--backup",
        type=Path,
        default=DEFAULT_BACKUP,
        help="first-run backup; an existing file is never overwritten",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="skip creation of the first-run backup",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    extended_path = args.extended.resolve()
    output_path = args.output.resolve()
    backup_path = args.backup.resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not extended_path.exists():
        raise FileNotFoundError(extended_path)
    if not args.no_backup and backup_path == output_path:
        raise ValueError("backup path must differ from output path")

    base = pd.read_parquet(input_path)
    extended = pd.read_parquet(extended_path)
    augmented, stats = merge_extended_features(base, extended)

    if not args.no_backup:
        if backup_path.exists():
            print(f"preserving existing backup: {backup_path}")
        else:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, backup_path)
            print(f"created backup: {backup_path}")

    _write_atomic(augmented, output_path)
    print(f"wrote: {output_path}")
    print(f"shape: {stats['rows']:,} rows x {stats['columns']:,} columns")
    print(f"added columns: {len(stats['added_columns']):,}")
    print(f"refreshed columns: {len(stats['refreshed_columns']):,}")
    print(f"filled initial rolling-window values: {stats['filled_missing_values']:,}")


if __name__ == "__main__":
    main()
