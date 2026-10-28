#!/usr/bin/env python3
"""Download real 1M (and optionally 5M) OHLCV data from Databento for MES/MNQ.

Uses the CME GLBX venue. Downloads front-month continuous contracts.
Saves to data/{SYM}_1m.csv and data/{SYM}_5m.csv in the same format
the backtest engine expects: datetime,open,high,low,close,volume
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
from datetime import datetime
from dotenv import load_dotenv

import databento as db
import pandas as pd

load_dotenv()

API_KEY = os.getenv("DATABENTO_API_KEY")
if not API_KEY:
    print("ERROR: DATABENTO_API_KEY not set in .env")
    sys.exit(1)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Map our symbols to Databento continuous front-month symbols
# Format: ROOT.ROLL_RULE.RANK  (c = calendar roll, 0 = front month)
SYMBOL_MAP = {
    "MES": "MES.c.0",
    "MNQ": "MNQ.c.0",
}

# Date range matching our existing 5M data
START = "2025-12-29"
END   = "2026-03-12"   # exclusive end, so we get through Mar 11

# Databento only has ohlcv-1s, ohlcv-1m, ohlcv-1h, ohlcv-1d
# We download 1M bars and aggregate to 5M ourselves
TIMEFRAMES_TO_DOWNLOAD = ["1m"]  # download 1M, derive 5M


def download_ohlcv(symbol: str, db_symbol: str):
    """Download 1M OHLCV bars and derive 5M from Databento."""
    out_csv_1m = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    out_csv_5m = os.path.join(DATA_DIR, f"{symbol}_5m.csv")

    print(f"\n{'─'*50}")
    print(f"  {symbol} — {db_symbol}")
    print(f"  Range: {START} → {END}")
    print(f"{'─'*50}")

    client = db.Historical(API_KEY)

    # Check cost first
    print("  Checking cost...", flush=True)
    try:
        cost = client.metadata.get_cost(
            dataset="GLBX.MDP3",
            symbols=[db_symbol],
            schema="ohlcv-1m",
            start=START,
            end=END,
            stype_in="continuous",
        )
        print(f"  Estimated cost: ${cost:.4f}", flush=True)
    except Exception as e:
        print(f"  Cost check note: {e}", flush=True)

    print("  Downloading 1M bars...", flush=True)
    try:
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[db_symbol],
            schema="ohlcv-1m",
            start=START,
            end=END,
            stype_in="continuous",
        )
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return False

    # Convert to DataFrame
    df = data.to_df()
    if df.empty:
        print(f"  ❌ No data returned")
        return False

    print(f"  Raw rows: {len(df)}", flush=True)

    # Databento OHLCV columns: open, high, low, close, volume
    # Index is ts_event (nanosecond timestamp)
    df = df.reset_index()

    # Normalize column names
    if 'ts_event' in df.columns:
        df['datetime'] = pd.to_datetime(df['ts_event'], utc=True)
    elif df.index.name == 'ts_event':
        df['datetime'] = pd.to_datetime(df.index, utc=True)
    else:
        # Try first datetime-like column
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                df['datetime'] = pd.to_datetime(df[col], utc=True)
                break

    if 'datetime' not in df.columns:
        print(f"  ❌ Could not find timestamp column. Columns: {list(df.columns)}")
        return False

    # Databento prices are in fixed-point (multiply by 1e-9 for some schemas)
    # For ohlcv schema they should already be in normal price units
    # But check if values look like raw fixed-point
    sample_open = df['open'].iloc[0]
    if sample_open > 1_000_000:
        # Likely fixed-point encoding — divide by 1e9
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] / 1e9
        print(f"  Applied 1e-9 price scaling (raw was {sample_open})", flush=True)

    # Keep only what we need
    out = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    out = out.sort_values('datetime').drop_duplicates('datetime', keep='last')

    # Filter trading hours only (remove weekend gaps etc.)
    out = out[out['volume'] > 0].reset_index(drop=True)

    days = (out['datetime'].iloc[-1] - out['datetime'].iloc[0]).days
    print(f"  1M: {len(out)} bars over {days} days")
    print(f"  Range: {out['datetime'].iloc[0]} → {out['datetime'].iloc[-1]}")

    # Backup existing files
    for csv_path in [out_csv_1m, out_csv_5m]:
        bak = csv_path + ".bak.pre_databento"
        if os.path.exists(csv_path) and not os.path.exists(bak):
            shutil.copy2(csv_path, bak)
            print(f"  Backed up → {os.path.basename(bak)}")

    # Save 1M
    out.to_csv(out_csv_1m, index=False)
    print(f"  ✅ Saved {symbol}_1m.csv ({len(out)} rows)")

    # Aggregate to 5M
    out_5m = out.set_index('datetime').resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna().reset_index()
    out_5m = out_5m[out_5m['volume'] > 0].reset_index(drop=True)
    out_5m.to_csv(out_csv_5m, index=False)
    print(f"  ✅ Saved {symbol}_5m.csv ({len(out_5m)} rows)")

    return True


def main():
    print(f"\n{'='*60}")
    print(f"  DATABENTO DATA DOWNLOAD")
    print(f"  Symbols: {list(SYMBOL_MAP.keys())}")
    print(f"  Range: {START} → {END}")
    print(f"{'='*60}")

    results = {}
    for symbol, db_sym in SYMBOL_MAP.items():
        ok = download_ohlcv(symbol, db_sym)
        results[symbol] = ok

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for k, v in results.items():
        status = "✅" if v else "❌"
        print(f"  {status} {k}")
    print()


if __name__ == "__main__":
    main()
