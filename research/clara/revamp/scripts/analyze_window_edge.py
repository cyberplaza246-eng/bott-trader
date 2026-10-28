#!/usr/bin/env python3
"""
Window / side / alignment expectancy for the locked ema15_eod recipe.

Runs the 9:35 / 11:00 / 13:30 path on data/MNQ_1m.csv (IS+OOS and OOS-only).
Writes data/window_edge_analysis.json for the desk. Never writes mnq_profit_config.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.window_edge import (
    CANON_WINDOWS,
    build_snapshot,
    simulate_locked_ema15,
    write_snapshot,
)


def _pf(stats: dict) -> str:
    pf = stats.get("pf")
    return "n/a" if pf is None else f"{pf:.2f}"


def _dump_windows(label: str, tables: dict) -> None:
    print(f"\n{label}")
    ov = tables.get("overall") or {}
    print(
        f"  overall n={ov.get('n')}  PF={_pf(ov)}  exp=${ov.get('expectancy_usd')}  "
        f"${ov.get('total_pnl_usd')}"
    )
    for name in CANON_WINDOWS:
        s = (tables.get("by_window") or {}).get(name) or {}
        print(
            f"  {name:<6} n={s.get('n', 0):3d}  PF={_pf(s):>6}  "
            f"WR={s.get('win_rate')}  exp=${s.get('expectancy_usd')}  "
            f"${s.get('total_pnl_usd')}"
        )
    for name in ("long", "short"):
        s = (tables.get("by_side") or {}).get(name) or {}
        print(f"  {name:<6} n={s.get('n', 0):3d}  PF={_pf(s):>6}  ${s.get('total_pnl_usd')}")
    for name in ("aligned", "not aligned"):
        s = (tables.get("by_alignment") or {}).get(name) or {}
        print(f"  {name:<12} n={s.get('n', 0):3d}  PF={_pf(s):>6}  ${s.get('total_pnl_usd')}")
    dist = tables.get("mfe_mae") or {}
    print(
        f"  MFE winners median={dist.get('median_mfe_winners')}  "
        f"le20={dist.get('pct_winners_mfe_le_20')}%  ge80={dist.get('pct_winners_mfe_ge_80')}%  "
        f"MAE losers median={dist.get('median_mae_losers')}  behavior={dist.get('behavior')}"
    )
    print(f"  winner MFE buckets: {dist.get('winner_mfe_buckets')}")
    print(f"  winner exits: {dist.get('winner_exits')}")


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    print("LOCKED ema15_eod window-edge analysis (advisory only)")
    print(f"file={csv}")
    from scripts.backtest_ema15_official import load_1m

    df = load_1m(csv)
    print(f"bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")
    rows = simulate_locked_ema15(df)
    snap = build_snapshot(rows, csv=csv)
    path = write_snapshot(snap)
    _dump_windows("ALL (IS+OOS)", snap["all"])
    _dump_windows("IS", snap["is"])
    _dump_windows("OOS", snap["oos"])
    print("\nADVISORY (all):")
    for line in snap["all"].get("advisory") or []:
        print(f"  - {line}")
    print("\nADVISORY (OOS):")
    for line in snap["oos"].get("advisory") or []:
        print(f"  - {line}")
    print(f"\nwrote {path}  n_all={snap['n_all']} n_oos={snap['n_oos']}")
    print("Does not write mnq_profit_config / ENTRY_WINDOWS / stops / MAX_TRADES.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
