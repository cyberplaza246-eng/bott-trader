#!/usr/bin/env python3
"""
Download raw daily OHLCV for EVERY individual contract month (not the
continuous series) for the 8-market daily trend universe. This is what the
roll audit needs to compare old-contract vs new-contract prices on the same
calendar day, to distinguish a real market move from a continuous-series
splice artifact.

Saves to daily_trend_data/raw/{SYMBOL}_contracts_1d.csv (unmodified).

Usage:
    python scripts/download_daily_trend_contracts.py --start 2010-06-06 --end 2026-08-16 --confirm
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


def download_contracts(client: db.Historical, symbol: str, root: str, start: str, end: str, max_retries: int = 4) -> None:
    parent_symbol = f"{root}.FUT"
    print(f"  Downloading {symbol} ({parent_symbol}, all contract months) {start} -> {end} ...", flush=True)
    last_exc = None
    data = None
    for attempt in range(1, max_retries + 1):
        try:
            data = client.timeseries.get_range(
                dataset="GLBX.MDP3", symbols=[parent_symbol], schema="ohlcv-1d",
                start=start, end=end, stype_in="parent",
            )
            break
        except db.common.error.BentoError as e:
            last_exc = e
            print(f"    attempt {attempt}/{max_retries} failed ({e}); retrying...", flush=True)
            time.sleep(3 * attempt)
    if data is None:
        raise last_exc

    df = data.to_df().reset_index()
    print(f"    columns: {list(df.columns)}", flush=True)

    if "ts_event" in df.columns:
        df["date"] = pd.to_datetime(df["ts_event"], utc=True).dt.date
    elif df.index.name == "ts_event":
        df["date"] = pd.to_datetime(df.index, utc=True).dt.date

    if df["open"].iloc[0] > 1_000_000:
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] / 1e9

    keep_cols = [c for c in ["date", "symbol", "instrument_id", "open", "high", "low", "close", "volume"] if c in df.columns]
    out = df[keep_cols].copy()
    out = out.sort_values(["date", "symbol"] if "symbol" in out.columns else ["date"]).reset_index(drop=True)

    out_path = OUT_DIR / f"{symbol}_contracts_1d.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved {out_path.name}: {len(out)} rows, {out['symbol'].nunique() if 'symbol' in out.columns else '?'} distinct contracts")


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
        out_path = OUT_DIR / f"{symbol}_contracts_1d.csv"
        if out_path.exists():
            print(f"  Skipping {symbol}: already downloaded ({out_path.name})")
            continue
        root = spec.databento_continuous_symbol.split(".")[0]
        download_contracts(client, symbol, root, args.start, args.end)


if __name__ == "__main__":
    main()
