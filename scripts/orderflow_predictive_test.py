#!/usr/bin/env python3
"""
Predictive-information test (NOT a trading strategy): does short-term
order-flow imbalance predict forward returns?

Pipeline: trades -> signed volume (side='A' buyer-initiated = +size,
side='B' seller-initiated = -size) -> bar-level imbalance -> forward
1/3/5/10-bar returns -> chronological train/test split -> correlation +
OLS regression, both raw and volatility-normalized (z-scored forward
return by trailing realized vol, controlling for the fact that imbalance
and volatility are themselves correlated).

Reports whether the relationship (if any) survives out-of-sample. Does not
build a P&L backtest -- that's the next gate, only if this one passes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

BAR_FREQ = "1min"
HORIZONS = [1, 3, 5, 10]
TRAIN_FRACTION = 0.7


def build_bars(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades[trades["side"].isin(["A", "B"])].copy()
    trades["signed_volume"] = np.where(trades["side"] == "A", trades["size"], -trades["size"])
    trades["ts"] = pd.to_datetime(trades["ts_event"], utc=True)
    trades = trades.set_index("ts")

    bars = trades.resample(BAR_FREQ).agg(
        close=("price", "last"),
        volume=("size", "sum"),
        buy_volume=("signed_volume", lambda s: s[s > 0].sum()),
        sell_volume=("signed_volume", lambda s: -s[s < 0].sum()),
        n_trades=("price", "count"),
    )
    bars = bars.dropna(subset=["close"])
    bars["imbalance"] = (bars["buy_volume"] - bars["sell_volume"]) / bars["volume"].replace(0, np.nan)
    bars["imbalance"] = bars["imbalance"].fillna(0.0)
    return bars


def add_forward_returns_and_vol(bars: pd.DataFrame) -> pd.DataFrame:
    bars["ret_1"] = bars["close"].pct_change()
    bars["realized_vol"] = bars["ret_1"].rolling(30).std()
    for h in HORIZONS:
        fwd_ret = bars["close"].shift(-h) / bars["close"] - 1
        bars[f"fwd_ret_{h}"] = fwd_ret
        bars[f"fwd_ret_{h}_zscored"] = fwd_ret / (bars["realized_vol"] * np.sqrt(h)).replace(0, np.nan)
    return bars


def evaluate(bars: pd.DataFrame) -> None:
    bars = bars.dropna(subset=["realized_vol"])
    split_idx = int(len(bars) * TRAIN_FRACTION)
    train, test = bars.iloc[:split_idx], bars.iloc[split_idx:]
    print(f"Total bars: {len(bars)} | train: {len(train)} | test: {len(test)}\n")

    for h in HORIZONS:
        print(f"=== Horizon: {h} minute(s) ===")
        for label, kind in [("raw", f"fwd_ret_{h}"), ("vol-normalized", f"fwd_ret_{h}_zscored")]:
            for split_name, split_df in [("TRAIN", train), ("TEST (OOS)", test)]:
                sub = split_df.dropna(subset=["imbalance", kind])
                if len(sub) < 30:
                    continue
                corr, corr_p = stats.pearsonr(sub["imbalance"], sub[kind])
                slope, intercept, r, p, se = stats.linregress(sub["imbalance"], sub[kind])
                print(f"  [{label:14s}] {split_name:10s} n={len(sub):>6} corr={corr:+.4f} (p={corr_p:.4f}) "
                      f"slope={slope:+.6f} (p={p:.4f})")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="Path to trades parquet file")
    args = parser.parse_args()

    trades = pd.read_parquet(args.file)
    bars = build_bars(trades)
    bars = add_forward_returns_and_vol(bars)
    evaluate(bars)


if __name__ == "__main__":
    main()
