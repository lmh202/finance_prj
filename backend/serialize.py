"""JSON helpers for the AURORA API.

The engine contract (src/interfaces.py) uses stdlib dataclasses that may hold
pandas DataFrames and datetimes — neither survives starlette's json.dumps
(allow_nan=False). Everything returned by a router goes through one of:

    df_records(df)  list of row dicts (NaN -> null)         tables
    df_split(df)    {index, columns, data} keeping the index backtest curves,
                                                            correlation matrix
    as_dict(obj)    dataclass -> JSON-safe dict, recursing into nested
                    dataclasses / lists / dicts / DataFrames
"""

import dataclasses
import json
import math
from typing import Any

import pandas as pd


def df_records(df: pd.DataFrame) -> list:
    # to_json round-trip converts NaN/NaT -> null and Timestamps -> ISO
    return json.loads(df.to_json(orient="records", date_format="iso"))


def df_split(df: pd.DataFrame) -> dict:
    return json.loads(df.to_json(orient="split", date_format="iso"))


def _convert(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return as_dict(value)
    if isinstance(value, pd.DataFrame):
        return df_split(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _convert(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_convert(v) for v in value]
    return value


def as_dict(obj: Any) -> dict:
    return {f.name: _convert(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
