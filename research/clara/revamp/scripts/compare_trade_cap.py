#!/usr/bin/env python3
"""Compare capped vs unlimited daily trades on frozen combined spec."""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import scripts.backtest_combined_bidirectional as combo
import scripts.backtest_ema15_official as official
import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.find_real_1m_edge as edge
import scripts.pass_hunt_shorts as hunt
import scripts.pass_hunt_short_rth_all_windows as srth


def et_date(t):
    ts = pd.Timestamp(t.entry_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/New_York").date()


def report(label, trades):
    oos = edge.split_is_oos(trades)[1]
    om = resp.group_metrics(oos)
    days = Counter(et_date(t) for t in trades)
    counts = list(days.values()) if days else [0]
    print(label)
    print(f"  ALL n={len(trades)}  OOS n={om['trades']} PF={om['profit_factor']:.2f} Net=${om['total_pnl']:.0f}")
    print(
        f"  max/day={max(counts)} median/day={sorted(counts)[len(counts)//2]} "
        f"avg_active_day={len(trades)/len(days):.1f}"
    )
    print(f"  days with 5+ trades: {sum(1 for c in counts if c >= 5)}")
    print()


def main():
    df = official.load_1m("data/MNQ_1m.csv")
    a = combo.prepare(df)

    w6 = combo.simulate_combined_active(a)
    report("LONG w6_all + SHORT w6_all (unlimited, one position)", w6)

    w2 = combo.simulate_combined_long_w2(a)
    report("LONG w2_open + SHORT w6_all (unlimited, one position)", w2)


if __name__ == "__main__":
    main()
