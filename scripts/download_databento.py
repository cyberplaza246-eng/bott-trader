#!/usr/bin/env python3
"""
Download real 1m OHLCV bars from Databento (CME GLBX.MDP3) for MNQ/MES/NQ and
derive 5m bars locally. Saves to data/{SYM}_1m.csv and data/{SYM}_5m.csv in
the schema the backtester expects: datetime,open,high,low,close,volume.

By default this only checks and prints the billable cost — it does NOT spend
money or download anything until you pass --confirm.

Usage:
    python scripts/download_databento.py --start 2024-01-01 --end 2026-03-12
    python scripts/download_databento.py --start 2024-01-01 --end 2026-03-12 --confirm
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import databento as db
import pandas as pd
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("DATABENTO_API_KEY")
DATA_DIR = ROOT / "data"

SYMBOL_MAP = {
    "MNQ": "MNQ.c.0",
    "MES": "MES.c.0",
    "NQ": "NQ.c.0",
}


def check_cost(db_symbol: str, start: str, end: str) -> float:
    client = db.Historical(API_KEY)
    return client.metadata.get_cost(
        dataset="GLBX.MDP3",
        symbols=[db_symbol],
        schema="ohlcv-1m",
        start=start,
        end=end,
        stype_in="continuous",
    )


def download_symbol(symbol: str, db_symbol: str, start: str, end: str) -> bool:
    out_1m = DATA_DIR / f"{symbol}_1m.csv"
    out_5m = DATA_DIR / f"{symbol}_5m.csv"

    client = db.Historical(API_KEY)
    print(f"  Downloading {symbol} ({db_symbol}) 1m bars {start} -> {end} ...", flush=True)
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=[db_symbol],
        schema="ohlcv-1m",
        start=start,
        end=end,
        stype_in="continuous",
    )
    df = data.to_df().reset_index()
    if df.empty:
        print(f"  No data returned for {symbol}")
        return False

    if "ts_event" in df.columns:
        df["datetime"] = pd.to_datetime(df["ts_event"], utc=True)
    elif df.index.name == "ts_event":
        df["datetime"] = pd.to_datetime(df.index, utc=True)

    if df["open"].iloc[0] > 1_000_000:
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] / 1e9

    out = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    out = out.sort_values("datetime").drop_duplicates("datetime", keep="last")
    out = out[out["volume"] > 0].reset_index(drop=True)

    for csv_path in (out_1m, out_5m):
        if csv_path.exists():
            bak = csv_path.with_suffix(csv_path.suffix + ".bak.pre_extended")
            if not bak.exists():
                shutil.copy2(csv_path, bak)

    out.to_csv(out_1m, index=False)
    print(f"  Saved {out_1m.name}: {len(out)} rows ({out['datetime'].iloc[0]} -> {out['datetime'].iloc[-1]})")

    out_5m_df = (
        out.set_index("datetime")
        .resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    out_5m_df = out_5m_df[out_5m_df["volume"] > 0].reset_index()
    out_5m_df.to_csv(out_5m, index=False)
    print(f"  Saved {out_5m.name}: {len(out_5m_df)} rows")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD (exclusive)")
    parser.add_argument("--symbols", nargs="+", default=list(SYMBOL_MAP), choices=list(SYMBOL_MAP))
    parser.add_argument("--confirm", action="store_true", help="Actually spend/download. Without this, only cost is checked.")
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit("DATABENTO_API_KEY not set in .env")

    total_cost = 0.0
    costs = {}
    for symbol in args.symbols:
        db_symbol = SYMBOL_MAP[symbol]
        try:
            cost = check_cost(db_symbol, args.start, args.end)
        except Exception as e:
            print(f"  Cost check failed for {symbol}: {e}")
            continue
        costs[symbol] = cost
        total_cost += cost
        print(f"  {symbol} ({db_symbol}): ${cost:.4f}")

    print(f"\nTotal estimated cost: ${total_cost:.4f}")

    if not args.confirm:
        print("\nDry run only (no download). Re-run with --confirm to actually download and spend this amount.")
        return

    print("\nDownloading...")
    for symbol in args.symbols:
        download_symbol(symbol, SYMBOL_MAP[symbol], args.start, args.end)


if __name__ == "__main__":
    main()
