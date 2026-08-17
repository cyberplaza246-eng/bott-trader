"""Loader for the daily-bar continuous-contract CSVs (Version A, raw)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "daily_trend_data" / "raw"


def load_daily_ohlcv(symbol: str) -> pd.DataFrame:
    csv_path = DATA_PATH / f"{symbol}_continuous_1d.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No daily data for {symbol}: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["date"] = df["date"].dt.tz_localize("UTC")
    df = df.set_index("date")
    return df[["open", "high", "low", "close", "volume"]]
