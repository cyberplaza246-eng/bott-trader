#!/usr/bin/env python3
"""
Download raw daily continuous-contract OHLCV for the 8-market daily trend
universe from Databento (GLBX.MDP3). Saves UNMODIFIED to
daily_trend_data/raw/{SYMBOL}_continuous_1d.csv -- this is Version A (raw),
per the locked research sequence: never edited, always re-derivable.

Usage:
    python scripts/download_daily_trend_data.py --start 2010-06-06 --end 2026-08-16 --confirm
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import databento as db
import pandas as pd
from dotenv import load_dotenv

from src.daily_trend.instruments import REGISTRY

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("DATABENTO_API_KEY")
OUT_DIR = ROOT / "daily_trend_data" / "raw"


def download_symbol(client: db.Historical, symbol: str, db_symbol: str, start: str, end: str, max_retries: int = 4) -> None:
    print(f"  Downloading {symbol} ({db_symbol}) daily bars {start} -> {end} ...", flush=True)
    last_exc = None
    data = None
    for attempt in range(1, max_retries + 1):
        try:
            data = client.timeseries.get_range(
                dataset="GLBX.MDP3", symbols=[db_symbol], schema="ohlcv-1d",
                start=start, end=end, stype_in="continuous",
            )
            break
        except db.common.error.BentoError as e:
            last_exc = e
            print(f"    attempt {attempt}/{max_retries} failed ({e}); retrying...", flush=True)
            time.sleep(3 * attempt)
    if data is None:
        raise last_exc

    df = data.to_df().reset_index()

    if "ts_event" in df.columns:
        df["date"] = pd.to_datetime(df["ts_event"], utc=True).dt.date
    elif df.index.name == "ts_event":
        df["date"] = pd.to_datetime(df.index, utc=True).dt.date

    if df["open"].iloc[0] > 1_000_000:
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] / 1e9

    out = df[["date", "open", "high", "low", "close", "volume", "instrument_id"]].copy()
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    out_path = OUT_DIR / f"{symbol}_continuous_1d.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved {out_path.name}: {len(out)} rows ({out['date'].iloc[0]} -> {out['date'].iloc[-1]})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", nargs="+", default=list(REGISTRY))
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit("DATABENTO_API_KEY not set in .env")
    if not args.confirm:
        raise SystemExit("Pass --confirm to actually download and spend.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = db.Historical(API_KEY)

    for symbol in args.symbols:
        spec = REGISTRY[symbol]
        out_path = OUT_DIR / f"{symbol}_continuous_1d.csv"
        if out_path.exists():
            print(f"  Skipping {symbol}: already downloaded ({out_path.name})")
            continue
        download_symbol(client, symbol, spec.databento_continuous_symbol, args.start, args.end)

    print("\nDone. Raw files in daily_trend_data/raw/ -- treat as read-only (Version A).")


if __name__ == "__main__":
    main()
