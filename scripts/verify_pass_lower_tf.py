#!/usr/bin/env python3
"""Verify lower-TF PASS winner with full breakdown."""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_official as official
import scripts.find_real_1m_edge as edge
import scripts.pass_hunt_lower_tf as hunt

ET = resp.ET


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    df = official.load_1m(csv)
    a = hunt.prepare(df)

    cfg = {
        "stack": "15_5_1",
        "loop": "window_loop",
        "entry": "quiet",
        "win": "w2_open",
        "stop": "atr1_1.5_6_25",
        "tp": "1R",
        "side": "long",
        "sep": False,
        "cooldown": 0,
        "max_day": 4,
        "max_lots": 1,
        "morning": False,
    }
    trades = hunt.simulate(
        a,
        stack=cfg["stack"],
        loop=cfg["loop"],
        entry=cfg["entry"],
        winset_name=cfg["win"],
        stop_key=cfg["stop"],
        tp_key=cfg["tp"],
        side_mode=cfg["side"],
        sep_noon=cfg["sep"],
        cooldown_min=cfg["cooldown"],
        max_day=cfg["max_day"],
        max_lots=cfg["max_lots"],
        morning_only=cfg["morning"],
    )
    overall = resp.group_metrics(trades)
    ins, oos = edge.split_is_oos(trades)
    is_m = resp.group_metrics(ins)
    oos_m = resp.group_metrics(oos)
    ok = hunt.is_pass(is_m, oos_m)

    print("=" * 90)
    print("LOWER-TF PASS (15m + 5m + 1m only — NO daily / NO 60m)")
    print("=" * 90)
    print("LOOP: window_loop — after a trade closes, same window can fire again same day")
    print("STACK: completed 15m EMA8/21 + 5m EMA8/21 + 1m EMA8/21 all LONG")
    print("WINDOWS: 9:35 and 11:00 ET")
    print("ENTRY: quiet 1m (body<=4, close with trend)")
    print("STOP: 1m ATR x1.5 clipped 6-25 pts")
    print("TP: 1R | max 4 trades/day | 1 MNQ | long only")
    print()
    print(f"ALL  n={overall['trades']} WR={overall['win_rate']:.1f}% PF={overall['profit_factor']:.2f} ${overall['total_pnl']:.0f} Exp=${overall['expectancy']:.2f}")
    print(f"IS   n={is_m['trades']} PF={is_m['profit_factor']:.2f} ${is_m['total_pnl']:.0f}")
    print(f"OOS  n={oos_m['trades']} PF={oos_m['profit_factor']:.2f} ${oos_m['total_pnl']:.0f} Exp=${oos_m['expectancy']:.2f} DD=${oos_m['max_drawdown']:.0f}")
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}")

    print("\nBy month:")
    by_month = {}
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        k = ts.tz_convert(ET).strftime("%Y-%m")
        by_month.setdefault(k, []).append(t)
    for k in sorted(by_month):
        m = resp.group_metrics(by_month[k])
        print(f"  {k}  n={m['trades']:3d} PF={m['profit_factor']:.2f} ${m['total_pnl']:.0f}")

    out = os.path.join("data", "pass_lower_tf_verified.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "overall": overall, "is": is_m, "oos": oos_m, "pass": ok}, f, indent=2, default=str)
    print(f"\nSaved {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
