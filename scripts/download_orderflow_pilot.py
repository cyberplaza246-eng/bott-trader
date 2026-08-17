#!/usr/bin/env python3
"""
Order-flow pilot download: tick-level trades for one symbol, one month, to
verify the data actually contains what we need (aggressor side) and
prototype an order-flow-imbalance calculation before spending more on a
larger pull.

Usage:
    python scripts/download_orderflow_pilot.py --symbol NQ --start 2026-06-01 --end 2026-07-01 --confirm
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import databento as db
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("DATABENTO_API_KEY")
OUT_DIR = ROOT / "orderflow_data" / "raw"

SYMBOL_MAP = {"MES": "MES.c.0", "MNQ": "MNQ.c.0", "NQ": "NQ.c.0"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, choices=list(SYMBOL_MAP))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--schema", default="trades", choices=["trades", "tbbo", "mbp-1"])
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit("DATABENTO_API_KEY not set in .env")
    if not args.confirm:
        raise SystemExit("Pass --confirm to actually download and spend.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = db.Historical(API_KEY)
    db_symbol = SYMBOL_MAP[args.symbol]

    print(f"Downloading {args.symbol} ({db_symbol}) {args.schema} {args.start} -> {args.end} ...", flush=True)
    last_exc = None
    data = None
    for attempt in range(1, 5):
        try:
            data = client.timeseries.get_range(
                dataset="GLBX.MDP3", symbols=[db_symbol], schema=args.schema,
                start=args.start, end=args.end, stype_in="continuous",
            )
            break
        except db.common.error.BentoError as e:
            last_exc = e
            print(f"  attempt {attempt}/4 failed ({e}); retrying...", flush=True)
            time.sleep(3 * attempt)
    if data is None:
        raise last_exc

    df = data.to_df().reset_index()
    print(f"  columns: {list(df.columns)}")
    print(f"  rows: {len(df)}")
    print(df.head(10).to_string())

    out_path = OUT_DIR / f"{args.symbol}_{args.schema}_{args.start}_{args.end}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
