"""Local portfolio storage for the Streamlit app.

The backend is stateless — it keeps no portfolio, so each client owns its
own and sends it with every request. The Next.js app uses localStorage; this
app is a single-user tool running on the developer's own machine, so a file
in the data directory is the natural equivalent and keeps the existing UX
(the portfolio is still there tomorrow) exactly as it was.

Deliberately dependency-free of `backend/`: nothing under frontend/ may
import `src` (see CLAUDE.md, module boundaries). This module only reads and
writes files — every normalisation rule still lives server-side and is
reached through api_client.
"""

import json
import os
from pathlib import Path
from typing import List

import pandas as pd

# Mirrors backend/src/config.py so `AURORA_DATA_DIR` sandboxes both halves.
DATA_DIR = Path(
    os.environ.get("AURORA_DATA_DIR", Path(__file__).resolve().parents[1] / "data")
)
PORTFOLIO_CSV = DATA_DIR / "portfolio.csv"
SETTINGS_JSON = DATA_DIR / "settings.json"

COLUMNS: List[str] = ["symbol", "name", "shares", "buy_price"]


def empty_holdings() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def load_holdings() -> pd.DataFrame:
    if not PORTFOLIO_CSV.exists():
        return empty_holdings()
    try:
        df = pd.read_csv(PORTFOLIO_CSV)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return empty_holdings()
    for column in COLUMNS:
        if column not in df.columns:
            df[column] = "" if column == "name" else 0.0
    return df[COLUMNS]


def save_holdings(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame = df.copy() if not df.empty else empty_holdings()
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = "" if column == "name" else 0.0
    frame[COLUMNS].to_csv(PORTFOLIO_CSV, index=False)


def load_cash() -> float:
    if SETTINGS_JSON.exists():
        try:
            return float(json.loads(SETTINGS_JSON.read_text()).get("cash", 0.0))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return 0.0


def save_cash(cash: float) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_JSON.write_text(json.dumps({"cash": float(cash)}))
