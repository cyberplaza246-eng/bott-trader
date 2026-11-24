#!/usr/bin/env python3
"""
Test 1: Dynamic morning-only vs locked morning long (same data, same costs).

A) Dynamic — 15m structure+EMA+VWAP, 5m confirm, 1m quiet trigger, long+short,
   continuous scan 9:35–12:00 ET.

B) Locked — LONG_WINNER recipe: 15/5/1 EMA stack, quiet 1m, long only,
   discrete windows 9:35 and 11:00 (w2_open).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_official as official
import scripts.backtest_dynamic_15_5_1_rth as dyn
import scripts.pass_hunt_shorts as hunt
from scripts.backtest_dynamic_15_5_1_rth import (
    MORNING_END,
    full_report,
    print_report,
    simulate_dynamic,
)


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    out = os.path.join("data", "morning_dynamic_vs_locked.json")

    print("TEST 1 — Morning dynamic vs locked morning long")
    print(f"Window: 9:35–12:00 ET | costs ${official.PT}/pt + ${official.COMM} RT | OOS split Jun 2026")
    print()

    df = official.load_1m(csv)
    a = dyn.prepare_arrays(df)
    print(f"bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    tr_dynamic, st_dynamic, _ = simulate_dynamic(
        a,
        pattern="A",
        sep_noon=False,
        morning_only=True,
    )
    ha = hunt.prepare(df)
    tr_locked = hunt.simulate_ext(ha, hunt.LONG_WINNER)

    rep_dynamic = full_report(
        "A) Dynamic morning (strict 15m/5m/1m, long+short, 9:35–12:00 scan)",
        tr_dynamic,
        st_dynamic,
    )
    rep_locked = full_report(
        "B) Locked morning long (15/5/1 EMA, quiet 1m, 9:35+11:00 windows)",
        tr_locked,
    )

    for r in (rep_dynamic, rep_locked):
        print_report(r)

    print(f"\n{'=' * 90}")
    print("HEAD-TO-HEAD (OOS) — does dynamic entry engine beat locked long recipe?")
    print(f"{'=' * 90}")
    for r in (rep_dynamic, rep_locked):
        oo = r["oos"]
        print(
            f"  {r['label'][:55]:<55} PF={oo['profit_factor']:5.2f} "
            f"${oo['net_pnl']:7.0f} n={oo['trades']:3d} WR={oo['win_rate']:4.1f}%"
        )

    dyn_oos = rep_dynamic["oos"]
    lock_oos = rep_locked["oos"]
    if lock_oos["profit_factor"] > dyn_oos["profit_factor"] and lock_oos["net_pnl"] > dyn_oos["net_pnl"]:
        verdict = (
            "Locked morning long recipe outperforms dynamic morning engine on OOS. "
            "The edge likely comes from the locked EMA stack + discrete windows + long-only filter, "
            "not from strict 15m structure logic."
        )
    elif dyn_oos["profit_factor"] > lock_oos["profit_factor"] and dyn_oos["net_pnl"] > lock_oos["net_pnl"]:
        verdict = (
            "Dynamic morning engine outperforms locked long recipe on OOS. "
            "Strict structure stack may add value; short morning trades warrant separate review."
        )
    else:
        verdict = (
            "No clear winner on OOS — results are mixed or both near breakeven. "
            "Neither entry engine clearly dominates in the morning slice alone."
        )

    print(f"\nVerdict: {verdict}")

    payload = {
        "test": "morning_dynamic_vs_locked",
        "morning_window": f"9:35–{MORNING_END // 60:02d}:00 ET",
        "csv": csv,
        "dynamic_morning": rep_dynamic,
        "locked_morning_long": rep_locked,
        "verdict": verdict,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
