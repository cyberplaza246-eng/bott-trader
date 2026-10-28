#!/usr/bin/env python3
"""Clean forward gate on locked PASS recipe + overnight break recipe."""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from typing import List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_official as official
import scripts.pass_hunt_mnq as hunt

ET = resp.ET


def win(trades, start: date, end: Optional[date] = None):
    out = []
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        d = ts.tz_convert(ET).date()
        if d < start:
            continue
        if end and d >= end:
            continue
        out.append(t)
    return out


def show(label, trades):
    m = resp.group_metrics(trades)
    print(
        f"{label:14} n={m['trades']:4} WR={m['win_rate']:5.1f}% "
        f"PF={m['profit_factor']:5.2f} Net=${m['total_pnl']:8.1f} "
        f"Exp=${m['expectancy']:6.2f} DD=${m['max_drawdown']:7.1f}"
    )
    return m


def clean_forward(is_m, val_m, aug_m) -> bool:
    return (
        is_m["trades"] >= 40
        and is_m["profit_factor"] >= 1.15
        and is_m["total_pnl"] > 0
        and val_m["trades"] >= 15
        and val_m["profit_factor"] >= 1.15
        and val_m["total_pnl"] > 0
        and aug_m["trades"] >= 10
        and aug_m["profit_factor"] >= 1.10
        and aug_m["total_pnl"] > 0
    )


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
    print("=" * 80)
    print("CLEAN FORWARD: A_d15_60 | w2_open | fix25 | eod500 | long | sep")
    print("Costs: $1.24 RT (official). No extra slip in this engine.")
    print("=" * 80)
    is_m = show("IS_<Jun", win(trades, date(2020, 1, 1), date(2026, 6, 1)))
    val_m = show("VAL_JunJul", win(trades, date(2026, 6, 1), date(2026, 8, 1)))
    aug_m = show("AUG", win(trades, date(2026, 8, 1), date(2026, 9, 1)))
    sep_m = show("SEP", win(trades, date(2026, 9, 1)))
    oos_m = show("OOS_Jun+", win(trades, date(2026, 6, 1)))
    ok = clean_forward(is_m, val_m, aug_m)
    print(f"\nCLEAN_FORWARD (IS+Val+Aug): {'PASS' if ok else 'FAIL'}")
    print(f"Legacy is_pass: {hunt.is_pass(is_m, oos_m)}")

    # Slip stress: subtract 0.5pt * $2 * 2 sides = $2 per trade approx
    slip_cost = 2.0  # $ per RT for 0.25pt/side
    def stress(m, n):
        if n == 0:
            return m
        net = m["total_pnl"] - slip_cost * n
        # approximate PF stress: reduce wins equally hard — report net/exp only
        return {"n": n, "net": round(net, 2), "exp": round(net / n, 2)}

    print("\nSlip stress (+$2/trade RT):")
    for label, m in [("IS", is_m), ("VAL", val_m), ("AUG", aug_m)]:
        s = stress(m, m["trades"])
        print(f"  {label}: net=${s['net']} exp=${s['exp']}")

    out = {
        "recipe": cfg,
        "rules": {
            "stack": "daily EMA20 + completed 15m EMA8/21 + completed 60m EMA8/21 ALL long",
            "windows": ["09:35", "11:00"],
            "side": "long_only",
            "stop_pts": 25,
            "tp": "cap_500_or_eod_1550",
            "quiet_1m": False,
            "sep_lots": "max 2 if sep>=0.45",
        },
        "is": is_m,
        "val": val_m,
        "aug": aug_m,
        "sep": sep_m,
        "oos": oos_m,
        "clean_forward": ok,
        "legacy_pass": hunt.is_pass(is_m, oos_m),
    }
    path = os.path.join("data", "clean_forward_path.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
