#!/usr/bin/env python3
"""Verify dual long+short lower-TF system."""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_official as official
import scripts.find_real_1m_edge as edge
import scripts.pass_hunt_shorts as hunt

ET = resp.ET

LONG = hunt.LONG_WINNER
SHORT = {
    "stack": "15_5_1",
    "loop": "window_loop",
    "entry": "1m_pullback",
    "win": "w2_open",
    "stop": "atr1_1.5_6_25",
    "tp": "2R",
    "side": "short",
    "sep": False,
    "cooldown": 0,
    "max_day": 4,
    "max_lots": 1,
    "morning": False,
    "vwap": "none",
    "sep_short_min": None,
}


def leg_report(label: str, trades) -> dict:
    ins, oos = edge.split_is_oos(trades)
    is_m = resp.group_metrics(ins)
    oos_m = resp.group_metrics(oos)
    print(f"\n{label}")
    print(
        f"  IS  n={is_m['trades']} PF={is_m['profit_factor']:.2f} ${is_m['total_pnl']:.0f} "
        f"WR={is_m['win_rate']:.1f}%"
    )
    print(
        f"  OOS n={oos_m['trades']} PF={oos_m['profit_factor']:.2f} ${oos_m['total_pnl']:.0f} "
        f"Exp=${oos_m['expectancy']:.2f} DD=${oos_m['max_drawdown']:.0f}"
    )
    return {"is": is_m, "oos": oos_m}


def main() -> int:
    df = official.load_1m(os.path.join("data", "MNQ_1m.csv"))
    a = hunt.prepare(df)
    long_t = hunt.simulate_ext(a, LONG)
    short_t = hunt.simulate_ext(a, SHORT)
    combined = long_t + short_t

    print("=" * 90)
    print("DUAL LOWER-TF SYSTEM (15m+5m+1m — separate long & short recipes)")
    print("=" * 90)
    print("LONG: 15/5/1 agree + quiet 1m | 9:35+11:00 loop | 1m ATR stop | 1R")
    print("SHORT: 15/5/1 agree + 1m pullback | same windows loop | 1m ATR stop | 2R")

    lr = leg_report("LONG leg", long_t)
    sr = leg_report("SHORT leg", short_t)
    cr = leg_report("COMBINED", combined)

    ok = hunt.is_pass(cr["is"], cr["oos"])
    print(f"\nCombined VERDICT: {'PASS' if ok else 'FAIL'}")
    print(f"Short standalone PASS (n>=30): {hunt.is_pass(sr['is'], sr['oos'], min_oos=30)}")
    print(f"Short THIN (n>=20): {hunt.is_pass(sr['is'], sr['oos'], min_oos=20)}")

    out = os.path.join("data", "pass_dual_lower_tf_verified.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {"long": LONG, "short": SHORT, "long_stats": lr, "short_stats": sr, "combined": cr, "pass": ok},
            f,
            indent=2,
            default=str,
        )
    print(f"Saved {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
