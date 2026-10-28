#!/usr/bin/env python3
"""Re-verify the PASS hunter winner with full report."""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_official as official
import scripts.find_real_1m_edge as edge
import scripts.pass_hunt_mnq as hunt

ET = resp.ET


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    df = official.load_1m(csv)
    a = hunt.prepare(df)

    cfg = {
        "stack": "A_d15_60",
        "win": "w2_open",
        "stop": "fix25",
        "tp": "eod500",
        "side": "long",
        "quiet": False,
        "sep": True,
    }
    name = (
        f"{cfg['stack']}|{cfg['win']}|{cfg['stop']}|{cfg['tp']}|"
        f"{cfg['side']}|q{int(cfg['quiet'])}|sep{int(cfg['sep'])}"
    )
    trades = hunt.simulate(
        a,
        stack=cfg["stack"],
        winset=hunt.WINSETS[cfg["win"]],
        stop_cfg=hunt.STOPS[cfg["stop"]],
        tp_cfg=hunt.TPS[cfg["tp"]],
        side_mode=cfg["side"],
        quiet=cfg["quiet"],
        sep_noon=cfg["sep"],
    )
    overall = resp.group_metrics(trades)
    ins, oos = edge.split_is_oos(trades)
    is_m = resp.group_metrics(ins)
    oos_m = resp.group_metrics(oos)
    ok = hunt.is_pass(is_m, oos_m)

    print("=" * 90)
    print("PASS RECIPE (completed bars — no forming-15m leak)")
    print("=" * 90)
    print(name)
    print("Rules:")
    print("  - Direction: daily EMA20 + completed 15m EMA8/21 + completed 60m EMA8/21 ALL long")
    print("  - Windows: 9:35 and 11:00 ET only")
    print("  - Side: LONG only")
    print("  - Entry: first bar in window (no quiet-1m filter)")
    print("  - Stop: fixed 25 pts")
    print("  - Target: TP cap 500 pts / flatten EOD (15:50 in sim)")
    print("  - Max 2 overlapping lots if sep>=0.45")
    print()
    print(
        f"ALL  n={overall['trades']} WR={overall['win_rate']:.1f}% "
        f"PF={overall['profit_factor']:.2f} Net=${overall['total_pnl']:.1f} "
        f"Exp=${overall['expectancy']:.2f} DD=${overall['max_drawdown']:.1f}"
    )
    print(
        f"IS   n={is_m['trades']} WR={is_m['win_rate']:.1f}% "
        f"PF={is_m['profit_factor']:.2f} Net=${is_m['total_pnl']:.1f} "
        f"Exp=${is_m['expectancy']:.2f} DD=${is_m['max_drawdown']:.1f}"
    )
    print(
        f"OOS  n={oos_m['trades']} WR={oos_m['win_rate']:.1f}% "
        f"PF={oos_m['profit_factor']:.2f} Net=${oos_m['total_pnl']:.1f} "
        f"Exp=${oos_m['expectancy']:.2f} DD=${oos_m['max_drawdown']:.1f}"
    )
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}")

    print("\nBy month:")
    buckets = {}
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        key = ts.tz_convert(ET).strftime("%Y-%m")
        buckets.setdefault(key, []).append(t)
    by_month = {}
    for k in sorted(buckets):
        m = resp.group_metrics(buckets[k])
        by_month[k] = m
        print(
            f"  {k}  n={m['trades']:3d} WR={m['win_rate']:5.1f}% "
            f"PF={m['profit_factor']:5.2f} ${m['total_pnl']:8.1f}"
        )

    print("\nBy window:")
    by_win = {}
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        et = ts.tz_convert(ET)
        m = et.hour * 60 + et.minute
        label = "09:35" if m < 10 * 60 else "11:00"
        by_win.setdefault(label, []).append(t)
    for label in ("09:35", "11:00"):
        if label not in by_win:
            continue
        m = resp.group_metrics(by_win[label])
        print(
            f"  {label}  n={m['trades']:3d} WR={m['win_rate']:5.1f}% "
            f"PF={m['profit_factor']:5.2f} ${m['total_pnl']:8.1f}"
        )

    out = os.path.join("data", "pass_recipe_verified.json")
    payload = {
        "name": name,
        "config": cfg,
        "rules": {
            "stack": "daily + completed 15m EMA8/21 + completed 60m EMA8/21",
            "windows": ["09:35", "11:00"],
            "side": "long_only",
            "stop_pts": 25,
            "tp": "cap_500_or_eod",
            "quiet_1m": False,
            "completed_bars_only": True,
        },
        "overall": overall,
        "is": is_m,
        "oos": oos_m,
        "by_month": by_month,
        "pass": ok,
        "gates": "OOS n>=30 PF>=1.15 $>0 | IS PF>=1.05 $>0",
        "csv": csv,
        "range": [str(df["datetime"].iloc[0]), str(df["datetime"].iloc[-1])],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
