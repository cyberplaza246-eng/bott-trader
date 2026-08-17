#!/usr/bin/env python3
"""
MBP-1 order-book pilot: streams records via DBNStore.replay() and
aggregates DIRECTLY into 1-minute bar features, never materializing the
full raw tick stream as a DataFrame (a week of MBP-1 for an actively
quoted contract is 100M+ rows -- converting that to a single pandas
DataFrame at once is what crashed the first attempt with an OOM error).
Output is one row per minute (thousands of rows), not one row per book
update (hundreds of millions).

Usage:
    python scripts/download_orderbook_pilot.py --symbol MNQ --start 2026-06-01 --end 2026-06-08 --confirm
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


class MinuteBarAccumulator:
    def __init__(self):
        self.bars: dict[pd.Timestamp, dict] = {}
        self.prev_bid_sz = None
        self.prev_ask_sz = None
        self.n_records = 0

    def _bar(self, minute_ts: pd.Timestamp) -> dict:
        bar = self.bars.get(minute_ts)
        if bar is None:
            bar = {
                "bid_sz_last": 0.0, "ask_sz_last": 0.0,
                "bid_sz_sum": 0.0, "ask_sz_sum": 0.0, "n_quotes": 0,
                "buy_trade_vol": 0.0, "sell_trade_vol": 0.0,
                "depth_added": 0.0, "depth_removed": 0.0,
                "first_mid": None, "last_mid": None,
            }
            self.bars[minute_ts] = bar
        return bar

    def __call__(self, record) -> None:
        self.n_records += 1
        ts_event = getattr(record, "ts_event", None)
        bid_px = getattr(record, "bid_px_00", None)
        ask_px = getattr(record, "ask_px_00", None)
        if ts_event is None or bid_px is None or ask_px is None or bid_px <= 0 or ask_px <= 0:
            return

        minute_ts = pd.Timestamp(ts_event, unit="ns", tz="UTC").floor("1min")
        bar = self._bar(minute_ts)

        bid_sz = float(getattr(record, "bid_sz_00", 0) or 0)
        ask_sz = float(getattr(record, "ask_sz_00", 0) or 0)
        bar["bid_sz_last"], bar["ask_sz_last"] = bid_sz, ask_sz
        bar["bid_sz_sum"] += bid_sz
        bar["ask_sz_sum"] += ask_sz
        bar["n_quotes"] += 1

        mid = (bid_px + ask_px) / 2e9
        if bar["first_mid"] is None:
            bar["first_mid"] = mid
        bar["last_mid"] = mid

        if self.prev_bid_sz is not None:
            delta_bid = bid_sz - self.prev_bid_sz
            delta_ask = ask_sz - self.prev_ask_sz
            for delta in (delta_bid, delta_ask):
                if delta > 0:
                    bar["depth_added"] += delta
                else:
                    bar["depth_removed"] += -delta
        self.prev_bid_sz, self.prev_ask_sz = bid_sz, ask_sz

        action = getattr(record, "action", None)
        side = getattr(record, "side", None)
        if action == "T":
            size = float(getattr(record, "size", 0) or 0)
            if side == "A":
                bar["buy_trade_vol"] += size
            elif side == "B":
                bar["sell_trade_vol"] += size

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

    print(f"Streaming {args.symbol} ({db_symbol}) mbp-1 {args.start} -> {args.end} ...", flush=True)
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3", symbols=[db_symbol], schema="mbp-1",
        start=args.start, end=args.end, stype_in="continuous",
    )

    acc = MinuteBarAccumulator()
    data.replay(acc)
    print(f"  Processed {acc.n_records:,} raw records -> {len(acc.bars):,} 1-minute bars", flush=True)

    df = acc.to_dataframe()
    out_path = OUT_DIR / f"{args.symbol}_orderbook_1min_{args.start}_{args.end}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved {out_path}")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
