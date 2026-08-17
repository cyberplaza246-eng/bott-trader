"""
CSV OHLCV loader for MNQ, NQ, MES.

NQ has no standalone data file: it tracks the same Nasdaq-100 index at the
same price level as MNQ (both tick_size=0.25), differing only in contract
multiplier/margin/commission. NQ bars are derived by reusing MNQ's price
series unchanged; the $-per-tick difference is applied later by the backtest
engine via the instrument spec, not here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import DATA_DIR, DERIVED_SYMBOLS

DATA_PATH = Path(__file__).resolve().parent.parent.parent / DATA_DIR


def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV bars for a symbol/timeframe as a DataFrame indexed by datetime."""
    source_symbol = DERIVED_SYMBOLS.get(symbol, symbol)
    csv_path = DATA_PATH / f"{source_symbol}_{timeframe}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No data file for {symbol} ({timeframe}): {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").drop_duplicates(subset="datetime").reset_index(drop=True)
    df = df.set_index("datetime")
    return df[["open", "high", "low", "close", "volume"]]


def slice_range(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    tz = df.index.tz
    if start:
        df = df[df.index >= pd.Timestamp(start, tz=tz)]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz=tz)]
    return df
