#!/usr/bin/env python3
"""
MBP-10 order-book pilot: streams records via DBNStore.replay() and
aggregates DIRECTLY into 1-minute bar features, never materializing the
full raw tick stream (same OOM lesson as the MBP-1 pilot).

Saves the RAW per-bar ingredients needed for exactly 3 pre-defined MBP-10-
specific feature families (see orderbook_feature_battery_mbp10.py for the
feature construction + Stage-1 test):
  1. depth-weighted imbalance across all 10 levels
  2. book slope imbalance (size-weighted average resting level, bid vs ask)
  3. behind-top-of-book depth change (levels 1-9 only, excluding the level
     MBP-1 already covers)

Usage:
    python scripts/download_orderbook_mbp10_pilot.py --symbol MNQ --start 2026-06-01 --end 2026-06-08 --confirm
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import databento as db
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("DATABENTO_API_KEY")
OUT_DIR = ROOT / "orderflow_data" / "raw"

SYMBOL_MAP = {"MES": "MES.c.0", "MNQ": "MNQ.c.0", "NQ": "NQ.c.0"}
N_LEVELS = 10


class MinuteBarAccumulatorMBP10:
    def __init__(self):
        self.bars: dict[pd.Timestamp, dict] = {}
        self.prev_behind_bid_sum = None
        self.prev_behind_ask_sum = None
        self.n_records = 0

    def _bar(self, minute_ts: pd.Timestamp) -> dict:
        bar = self.bars.get(minute_ts)
        if bar is None:
            bar = {
                "bid_top10_sum_last": 0.0, "ask_top10_sum_last": 0.0,
                "bid_avg_level_last": 0.0, "ask_avg_level_last": 0.0,
                "behind_bid_sum_last": 0.0, "behind_ask_sum_last": 0.0,
                "behind_depth_added": 0.0, "behind_depth_removed": 0.0,
                "n_quotes": 0, "first_mid": None, "last_mid": None,
            }
            self.bars[minute_ts] = bar
        return bar

    def __call__(self, record) -> None:
        self.n_records += 1
        ts_event = getattr(record, "ts_event", None)
        bid_px_0 = getattr(record, "bid_px_00", None)
        ask_px_0 = getattr(record, "ask_px_00", None)
        if ts_event is None or bid_px_0 is None or ask_px_0 is None or bid_px_0 <= 0 or ask_px_0 <= 0:
            return

        minute_ts = pd.Timestamp(ts_event, unit="ns", tz="UTC").floor("1min")
        bar = self._bar(minute_ts)

        bid_szs = [float(getattr(record, f"bid_sz_{i:02d}", 0) or 0) for i in range(N_LEVELS)]
        ask_szs = [float(getattr(record, f"ask_sz_{i:02d}", 0) or 0) for i in range(N_LEVELS)]

        bid_total = sum(bid_szs)
        ask_total = sum(ask_szs)
        bar["bid_top10_sum_last"] = bid_total
        bar["ask_top10_sum_last"] = ask_total

        # Size-weighted average resting level (0=best price ... 9=deepest).
        # Low value = size concentrated near top (steep book); high value =
        # size spread deeper (flat book). Cheap per-record proxy for book
        # slope, not a full per-record OLS (250M+ records over the week).
        bid_avg_level = sum(i * s for i, s in enumerate(bid_szs)) / bid_total if bid_total > 0 else 0.0
        ask_avg_level = sum(i * s for i, s in enumerate(ask_szs)) / ask_total if ask_total > 0 else 0.0
        bar["bid_avg_level_last"] = bid_avg_level
        bar["ask_avg_level_last"] = ask_avg_level

        # Behind-top-of-book depth (levels 1-9 only, excludes level 0 which
        # MBP-1 already fully captures and which was already tested there).
        behind_bid = sum(bid_szs[1:])
        behind_ask = sum(ask_szs[1:])
        bar["behind_bid_sum_last"] = behind_bid
        bar["behind_ask_sum_last"] = behind_ask

        if self.prev_behind_bid_sum is not None:
            delta_bid = behind_bid - self.prev_behind_bid_sum
            delta_ask = behind_ask - self.prev_behind_ask_sum
            for delta in (delta_bid, delta_ask):
                if delta > 0:
                    bar["behind_depth_added"] += delta
                else:
                    bar["behind_depth_removed"] += -delta
        self.prev_behind_bid_sum, self.prev_behind_ask_sum = behind_bid, behind_ask

        bar["n_quotes"] += 1
        mid = (bid_px_0 + ask_px_0) / 2e9
        if bar["first_mid"] is None:
            bar["first_mid"] = mid
        bar["last_mid"] = mid

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for ts, bar in sorted(self.bars.items()):
            rows.append({"minute": ts, **bar})
        return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, choices=list(SYMBOL_MAP))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit("DATABENTO_API_KEY not set in .env")
    if not args.confirm:
        raise SystemExit("Pass --confirm to actually download and spend.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = db.Historical(API_KEY)
    db_symbol = SYMBOL_MAP[args.symbol]

    cost = client.metadata.get_cost(
        dataset="GLBX.MDP3", symbols=[db_symbol], schema="mbp-10",
        start=args.start, end=args.end, stype_in="continuous",
    )
    print(f"Confirmed cost for this request: ${cost:.2f}", flush=True)

    print(f"Streaming {args.symbol} ({db_symbol}) mbp-10 {args.start} -> {args.end} ...", flush=True)
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3", symbols=[db_symbol], schema="mbp-10",
        start=args.start, end=args.end, stype_in="continuous",
    )

    acc = MinuteBarAccumulatorMBP10()
    data.replay(acc)
    print(f"  Processed {acc.n_records:,} raw records -> {len(acc.bars):,} 1-minute bars", flush=True)

    df = acc.to_dataframe()
    out_path = OUT_DIR / f"{args.symbol}_orderbook_mbp10_1min_{args.start}_{args.end}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved {out_path}")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
