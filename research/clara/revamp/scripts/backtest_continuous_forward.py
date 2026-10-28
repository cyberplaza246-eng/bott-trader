#!/usr/bin/env python3
"""Backtest live continuous recipe: quiet 1m, cd=15, max 8/day, fix25, eod500."""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_official as official
import scripts.pass_hunt_mnq as hunt
import scripts.verify_continuous_forward as v

ET = resp.ET


def win(trades, start: date, end: Optional[date] = None):
    return v.win(trades, start, end)


def show(label, trades):
    m = resp.group_metrics(trades)
    print(
        f"{label:14} n={m['trades']:4} WR={m['win_rate']:5.1f}% "
        f"PF={m['profit_factor']:5.2f} Net=${m['total_pnl']:8.1f} "
        f"Exp=${m['expectancy']:6.2f} DD=${m['max_drawdown']:7.1f}"
    )
    return m


def main() -> int:
    df = official.load_1m(os.path.join("data", "MNQ_1m.csv"))
    a = hunt.prepare(df)
    trades = v.simulate_continuous(
        a, quiet=True, cooldown_min=15, max_day=8, sep_noon=True,
    )

    print("=" * 80)
    print("CONTINUOUS RTH BACKTEST (live recipe)")
    print("daily+15m+60m long | quiet 1m entry | fix25 | eod500 | cd=15m | max 8/day")
    print("Fees $1.24 RT | data/MNQ_1m.csv")
    print("=" * 80)

    is_m = show("IS_<Jun", win(trades, date(2020, 1, 1), date(2026, 6, 1)))
    val_m = show("VAL_JunJul", win(trades, date(2026, 6, 1), date(2026, 8, 1)))
    aug_m = show("AUG", win(trades, date(2026, 8, 1), date(2026, 9, 1)))
    sep_m = show("SEP", win(trades, date(2026, 9, 1)))
    oos_m = show("OOS_Jun+", win(trades, date(2026, 6, 1)))
    all_m = show("ALL", trades)

    ok = (
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
    print()
    print("CLEAN_FORWARD (IS+Val+Aug):", "PASS" if ok else "FAIL")

    print()
    print("By month (2026):")
    buckets = {}
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        key = ts.tz_convert(ET).strftime("%Y-%m")
        if key.startswith("2026"):
            buckets.setdefault(key, []).append(t)
    for k in sorted(buckets):
        m = resp.group_metrics(buckets[k])
        print(
            f"  {k}  n={m['trades']:3} WR={m['win_rate']:5.1f}% "
            f"PF={m['profit_factor']:5.2f} ${m['total_pnl']:8.1f}"
        )

    holds = []
    reasons = {}
    for t in trades:
        h = getattr(t, "hold_seconds", None)
        if h is None:
            et = pd.Timestamp(t.entry_time)
            xt = pd.Timestamp(t.exit_time)
            if et.tzinfo is None:
                et = et.tz_localize("UTC")
            if xt.tzinfo is None:
                xt = xt.tz_localize("UTC")
            h = (xt - et).total_seconds()
        holds.append(h / 60.0)
        r = getattr(t, "exit_reason", "?")
        reasons[r] = reasons.get(r, 0) + 1
    hs = np.array(holds) if holds else np.array([0.0])
    print()
    print(
        f"Hold (min): median={float(np.median(hs)):.0f} mean={float(np.mean(hs)):.0f} "
        f"| exits={reasons}"
    )

    out = {
        "recipe": {
            "stack": "A_d15_60",
            "side": "long",
            "quiet_1m": True,
            "continuous": True,
            "cooldown_min": 15,
            "max_day": 8,
            "stop": "fix25",
            "tp": "eod500",
        },
        "is": is_m,
        "val": val_m,
        "aug": aug_m,
        "sep": sep_m,
        "oos": oos_m,
        "all": all_m,
        "clean_forward": ok,
        "exits": reasons,
        "hold_median_min": round(float(np.median(hs)), 1),
        "hold_mean_min": round(float(np.mean(hs)), 1),
    }
    path = os.path.join("data", "continuous_forward_backtest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
