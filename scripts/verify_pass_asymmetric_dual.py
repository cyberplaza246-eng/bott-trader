#!/usr/bin/env python3
"""Verify asymmetric dual: morning LONG + afternoon SHORT (lower TF only)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_official as official
import scripts.find_real_1m_edge as edge
import scripts.pass_hunt_shorts as hunt

LONG = hunt.LONG_WINNER

# Best short standalone PASS (924-candidate search)
SHORT = {
    "stack": "15_5",
    "loop": "window_loop",
    "entry": "quiet",
    "win": "w3_pm",
    "stop": "atr1_1.5_6_25",
    "tp": "2R",
    "side": "short",
    "sep": False,
    "cooldown": 0,
    "max_day": 4,
    "max_lots": 1,
    "morning": False,
    "vwap": "with_side",
    "sep_short_min": None,
}


def report(label, trades):
    ins, oos = edge.split_is_oos(trades)
    is_m, oos_m = resp.group_metrics(ins), resp.group_metrics(oos)
    print(f"\n{label}")
    print(f"  IS  n={is_m['trades']} PF={is_m['profit_factor']:.2f} ${is_m['total_pnl']:.0f} WR={is_m['win_rate']:.1f}%")
    print(f"  OOS n={oos_m['trades']} PF={oos_m['profit_factor']:.2f} ${oos_m['total_pnl']:.0f} Exp=${oos_m['expectancy']:.2f} DD=${oos_m['max_drawdown']:.0f}")
    return is_m, oos_m


def main():
    df = official.load_1m(os.path.join("data", "MNQ_1m.csv"))
    a = hunt.prepare(df)
    lt = hunt.simulate_ext(a, LONG)
    st = hunt.simulate_ext(a, SHORT)
    ct = lt + st

    print("=" * 90)
    print("ASYMMETRIC DUAL (15m+5m+1m — NO daily/60m)")
    print("=" * 90)
    print("LONG  9:35+11:00 | 15/5/1 | quiet 1m | 1m ATR stop | 1R")
    print("SHORT 12:00+13:30+14:30 | 15/5 | quiet 1m | below VWAP | 1m ATR stop | 2R")

    li, lo = report("LONG (morning)", lt)
    si, so = report("SHORT (afternoon)", st)
    ci, co = report("COMBINED", ct)

    print(f"\nLong PASS:  {hunt.is_pass(li, lo)}")
    print(f"Short PASS: {hunt.is_pass(si, so)}")
    print(f"Dual PASS:  {hunt.is_pass(ci, co)}")

    out = os.path.join("data", "pass_asymmetric_dual_verified.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"long": LONG, "short": SHORT, "pass": hunt.is_pass(ci, co)}, f, indent=2, default=str)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
