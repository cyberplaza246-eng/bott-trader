#!/usr/bin/env python
"""
Cost/coverage check ONLY (no download, no spend) for daily OHLCV bars on the
8-market daily trend universe. Reports estimated Databento cost and, where
available, the dataset's actual coverage range for each symbol so we don't
assume history depth that isn't really there.

Usage:
    python scripts/check_daily_trend_cost.py --start 2005-01-01 --end 2026-08-17
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import databento as db
from dotenv import load_dotenv

from src.daily_trend.instruments import REGISTRY

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("DATABENTO_API_KEY")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit("DATABENTO_API_KEY not set in .env")

    client = db.Historical(API_KEY)

    print(f"Requested range: {args.start} -> {args.end}\n")
    total_cost = 0.0
    for symbol, spec in REGISTRY.items():
        try:
            cost = client.metadata.get_cost(
                dataset="GLBX.MDP3", symbols=[spec.databento_continuous_symbol],
                schema="ohlcv-1d", start=args.start, end=args.end, stype_in="continuous",
            )
            print(f"  {symbol} ({spec.databento_continuous_symbol}): ${cost:.4f}")
            total_cost += cost
        except Exception as e:
            print(f"  {symbol} ({spec.databento_continuous_symbol}): cost check failed -> {e}")

    print(f"\nTotal estimated cost for daily bars, {args.start}..{args.end}: ${total_cost:.4f}")

    try:
        dataset_range = client.metadata.get_dataset_range(dataset="GLBX.MDP3")
        print(f"\nGLBX.MDP3 dataset overall available range: {dataset_range}")
    except Exception as e:
        print(f"\nCould not fetch dataset range: {e}")


if __name__ == "__main__":
    main()
