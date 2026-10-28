#!/usr/bin/env python3
"""
Step 4: Controlled test — dynamic baseline vs baseline minus long entries 14:00–15:00 ET.

Single change only. Same data, costs, execution, Pattern A, sep filter, stops, targets.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_official as official
from scripts.backtest_dynamic_15_5_1_rth import (
    full_report,
    prepare_arrays,
    print_report,
    simulate_dynamic,
)

BLOCK_START = 14 * 60
BLOCK_END = 15 * 60


def delta(before: dict, after: dict, key: str) -> str:
    b, a = before[key], after[key]
    if isinstance(b, float):
        d = round(a - b, 2)
        return f"{d:+.2f}"
    return f"{a - b:+d}"


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    out = os.path.join("data", "dynamic_block_14_15_longs.json")

    print("STEP 4 — Dynamic baseline vs no longs 14:00–15:00 ET")
    print(f"Costs ${official.PT}/pt + ${official.COMM} RT | OOS split Jun 2026")
    print()

    df = official.load_1m(csv)
    a = prepare_arrays(df)

    tr_base, st_base, _ = simulate_dynamic(a, pattern="A", sep_noon=True)
    tr_mod, st_mod, _ = simulate_dynamic(
        a,
        pattern="A",
        sep_noon=True,
        block_long_minutes=(BLOCK_START, BLOCK_END),
    )

    rep_base = full_report("BASELINE dynamic (Pattern A)", tr_base, st_base)
    rep_mod = full_report(
        "MODIFIED — baseline minus long entries 14:00–15:00 ET",
        tr_mod,
        st_mod,
    )

    for r in (rep_base, rep_mod):
        print_report(r)

    bo, mo = rep_base["oos"], rep_mod["oos"]
    print(f"\n{'=' * 90}")
    print("OOS DELTA (modified − baseline)")
    print(f"{'=' * 90}")
    print(f"  Trades:     {delta(bo, mo, 'trades')}")
    print(f"  Win rate:   {delta(bo, mo, 'win_rate')} pp")
    print(f"  PF:         {delta(bo, mo, 'profit_factor')}")
    print(f"  Net P&L:    ${delta(bo, mo, 'net_pnl')}")
    print(f"  Expectancy: ${delta(bo, mo, 'expectancy')}")
    print(f"  Max DD:     ${delta(bo, mo, 'max_drawdown')}")
    print(f"  Blocked long signals (14–15): {st_mod.filtered_entry_block}")

    improved = (
        mo["profit_factor"] > bo["profit_factor"]
        and mo["net_pnl"] > bo["net_pnl"]
    )
    passes = rep_mod["pass"]

    if passes:
        verdict = (
            "Modified spec PASSES OOS gates. Removing 14–15 longs materially improved OOS; "
            "validate on a fresh period before stacking any further change."
        )
    elif improved and mo["net_pnl"] > 0:
        verdict = (
            "OOS improved materially but still below PASS gates (PF≥1.15, IS constraints). "
            "The 14–15 long block helps; not sufficient alone for production."
        )
    elif improved:
        verdict = (
            "OOS improved vs baseline but remains net-negative. "
            "The block removes a real losing slice but does not rescue the dynamic system."
        )
    else:
        verdict = (
            "OOS did not improve. The 14–15 long block is not supported on holdout "
            "(possible overfit to decomposition slice)."
        )

    print(f"\nVerdict: {verdict}")

    payload = {
        "test": "dynamic_block_14_15_longs",
        "change": "prohibit long entries when 14:00 <= ET minute < 15:00",
        "blocked_signals": st_mod.filtered_entry_block,
        "baseline": rep_base,
        "modified": rep_mod,
        "oos_delta": {
            "trades": mo["trades"] - bo["trades"],
            "win_rate": round(mo["win_rate"] - bo["win_rate"], 2),
            "profit_factor": round(mo["profit_factor"] - bo["profit_factor"], 2),
            "net_pnl": round(mo["net_pnl"] - bo["net_pnl"], 2),
            "expectancy": round(mo["expectancy"] - bo["expectancy"], 2),
            "max_drawdown": round(mo["max_drawdown"] - bo["max_drawdown"], 2),
        },
        "verdict": verdict,
        "modified_pass": passes,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
