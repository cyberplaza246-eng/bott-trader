#!/usr/bin/env python3
"""Backtest TrendCatcher-style MNQ recipe = locked L6 (Variant B + 1m ATR×1.5)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_official as official


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    stop_cfg = next(c for c in resp.STOP_CONFIGS if c["id"] == "1m_atr1.5_6_25")
    out = os.path.join("data", "trendcatcher_style_l6_backtest.json")

    print("TRENDCATCHER-STYLE / L6 BACKTEST")
    print("Variant B: 15m+5m agree, quiet 1m entry (body<=4)")
    print(f"Stop: {stop_cfg['label']}")
    print("TP: 1R | flatten 15:50 | max 20/day | 2 lots | $2/pt + $1.24 RT")
    print("IS/OOS: before Jun 2026 / Jun 2026+")
    print()

    df = official.load_1m(csv)
    print(f"bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    recs = resp.simulate(df, variant="B", stop_cfg=stop_cfg)
    rep = resp.full_report("B", stop_cfg, recs)
    resp.print_full_variant_report(rep)

    payload = {
        "recipe": "L6_responsive_15_5_1m_atr",
        "maps_from": "TrendCatcher-style: trend + confirm + timed entry + SL/TP bracket",
        "variant": "B",
        "stop": stop_cfg,
        "csv": csv,
        "bars": len(df),
        "range": [str(df["datetime"].iloc[0]), str(df["datetime"].iloc[-1])],
        "report": rep,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
