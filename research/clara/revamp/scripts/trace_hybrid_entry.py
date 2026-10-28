#!/usr/bin/env python
"""Dry trace: hybrid entry with aggressive flow burst (no broker)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SCALP_AGGRESSIVE", "true")
os.environ.setdefault("SCALP_ADX_MIN", "15")
os.environ.setdefault("SCALP_MOMENTUM_BURST", "true")
os.environ.setdefault("SCALP_MOMENTUM_BURST_ADX", "15")
os.environ.setdefault("SCALP_TREND_MODE", "vwap")

from src.strategy.scalp_hybrid import ScalpHybridState, check_hybrid_entry, hybrid_block_summary


def _row(close, o, h, l, adx=18, vwap=100.0):
    return pd.Series({
        "datetime": pd.Timestamp("2026-06-14 20:30:00", tz="UTC"),
        "open": o, "high": h, "low": l, "close": close,
        "volume": 500, "adx": adx, "vwap": vwap, "ema_20": close,
        "atr": 4.0, "avg_body": 1.0,
    })


def main():
    # Chop: close == vwap -> trend dir=0
    row_5m = _row(100.0, 99.5, 100.5, 99.0, adx=16, vwap=100.0)
    prev_5m = _row(99.8, 99.0, 100.0, 98.5, adx=15, vwap=100.0)
    row_1m = _row(100.1, 99.8, 100.3, 99.7, adx=16, vwap=100.0)
    prev_30s = _row(99.9, 99.8, 100.0, 99.7)
    row_30s = _row(100.2, 99.9, 100.4, 99.8)  # bullish 30s break
    flow = {"delta": 75, "buy_pct": 0.62, "sell_pct": 0.38}

    st = ScalpHybridState()
    params = dict(
        adx_min_pullback=15,
        adx_min_continuation=20,
        pullback_atr=1.0,
        trend_mode="vwap",
        continuation_volume_strict=False,
        momentum_burst_enabled=True,
        momentum_burst_adx=15,
        aggressive_mode=True,
        flow_burst_delta_min=50,
    )
    summary = hybrid_block_summary(
        row_1m, row_5m, prev_5m, row_30s, prev_30s, st,
        flow_snap=flow, **params,
    )
    signal, st2 = check_hybrid_entry(
        "MNQ", row_1m, row_5m, prev_5m, row_30s, prev_30s, st,
        flow_snap=flow, sl_pts=8, tp_pts=15, **params,
    )
    print("block_summary:", summary)
    if signal:
        print("SIGNAL:", signal["direction"].upper(), "@", signal["entry"], "mode=", signal.get("scalp_mode"))
    else:
        print("SIGNAL: none")
        sys.exit(1)
    print("OK — aggressive flow burst fires on dir=0 chop")


if __name__ == "__main__":
    main()
