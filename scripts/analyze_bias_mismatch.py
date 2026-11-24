#!/usr/bin/env python3
"""
What happens when morning is short / afternoon is long?

Compares:
  A) Locked asymmetric: morning LONG only, afternoon SHORT only
  B) Flipped: morning SHORT only, afternoon LONG only
  C) Adaptive: each slot follows 15m bias (long or short) with side-appropriate rules
  D) Count how often bias disagrees with the locked schedule
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_official as official
import scripts.find_real_1m_edge as edge
import scripts.pass_hunt_shorts as hunt
from src.strategy.mnq_15m_ema_eod import ENTRY_WINDOWS, window_index

ET = resp.ET

LONG_CFG = hunt.LONG_WINNER
SHORT_CFG = {
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

# Morning short = same as short rules but morning windows
MORNING_SHORT = {
    **SHORT_CFG,
    "win": "w2_open",
    "side": "short",
}
# Afternoon long = same as long rules but PM windows
AFTERNOON_LONG = {
    **LONG_CFG,
    "win": "w3_pm",
    "side": "long",
}


def score(trades):
    ins, oos = edge.split_is_oos(trades)
    return {
        "all": resp.group_metrics(trades),
        "is": resp.group_metrics(ins),
        "oos": resp.group_metrics(oos),
        "pass": hunt.is_pass(resp.group_metrics(ins), resp.group_metrics(oos)),
    }


def dump(label, s):
    o, oo = s["all"], s["oos"]
    flag = "PASS" if s["pass"] else "FAIL"
    print(
        f"  [{flag}] {label:<40} "
        f"OOS n={oo['trades']:3d} PF={oo['profit_factor']:.2f} ${oo['total_pnl']:7.0f} | "
        f"ALL PF={o['profit_factor']:.2f} ${o['total_pnl']:.0f}"
    )


def bias_mismatch_stats(a: Dict) -> None:
    """How often 15m bias conflicts with locked schedule at window opens."""
    morning_wins = {0, 2}  # 9:35, 11:00
    pm_wins = {3, 4, 5}
    am_short = am_long = pm_short = pm_long = 0
    seen = set()
    for i in range(hunt.WARMUP, len(a["cl"])):
        if a["dow"][i] >= 5:
            continue
        m = int(a["mins"][i])
        w = window_index(m)
        if w is None:
            continue
        day = a["dates"][i]
        key = (day, w)
        if key in seen:
            continue
        # first bar of window only
        start, _ = ENTRY_WINDOWS[w]
        if m != start:
            continue
        seen.add(key)
        s15 = int(a["t15"][i])
        if w in morning_wins:
            if s15 < 0:
                am_short += 1
            elif s15 > 0:
                am_long += 1
        if w in pm_wins:
            if s15 > 0:
                pm_long += 1
            elif s15 < 0:
                pm_short += 1

    print("\n15m bias at window open (how often schedule mismatches):")
    print(f"  Morning slots bullish (LONG schedule OK):  {am_long}")
    print(f"  Morning slots bearish (LONG sits out):     {am_short}")
    print(f"  Afternoon slots bearish (SHORT OK):        {pm_short}")
    print(f"  Afternoon slots bullish (SHORT sits out):  {pm_long}")
    am_total = am_long + am_short
    pm_total = pm_long + pm_short
    if am_total:
        print(f"  Morning sit-out rate: {100*am_short/am_total:.0f}% of morning window opens")
    if pm_total:
        print(f"  Afternoon sit-out rate: {100*pm_long/pm_total:.0f}% of afternoon window opens")


def adaptive_simulate(a: Dict) -> List:
    """
    Each window: if 15m long use LONG recipe rules; if 15m short use SHORT recipe rules.
    Morning windows use long recipe params when long; short recipe when short.
    Afternoon same.
    """
    # Run long recipe unrestricted to both am+pm windows, then filter by side at stack
    # Simpler: simulate long on all 6 windows, short on all 6, merge (no double entry same bar)
    long_all = {
        **LONG_CFG,
        "win": "w6_all",
        "side": "long",
        "max_day": 6,
    }
    short_all = {
        **SHORT_CFG,
        "win": "w6_all",
        "side": "short",
        "max_day": 6,
    }
    # Need w6_all in WINSETS - already there
    lt = hunt.simulate_ext(a, long_all)
    st = hunt.simulate_ext(a, short_all)
    # Deduplicate: if same entry minute, keep one (prefer matching bias)
    by_key = {}
    for t in lt + st:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        key = (ts.tz_convert(ET).date(), ts.tz_convert(ET).strftime("%H:%M"), t.direction)
        by_key[key] = t
    # Also prevent long+short same minute
    by_min = {}
    for t in by_key.values():
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        k = (ts.tz_convert(ET).date(), ts.tz_convert(ET).strftime("%H:%M"))
        if k not in by_min:
            by_min[k] = t
    return list(by_min.values())


def main() -> int:
    df = official.load_1m(os.path.join("data", "MNQ_1m.csv"))
    a = hunt.prepare(df)

    print("WHAT IF MORNING IS SHORT / AFTERNOON IS LONG?")
    print("=" * 90)

    bias_mismatch_stats(a)

    print("\nBacktest comparison:")
    locked = hunt.simulate_ext(a, LONG_CFG) + hunt.simulate_ext(a, SHORT_CFG)
    dump("A Locked (AM long / PM short)", score(locked))

    flipped = hunt.simulate_ext(a, MORNING_SHORT) + hunt.simulate_ext(a, AFTERNOON_LONG)
    dump("B Flipped (AM short / PM long)", score(flipped))

    adaptive = adaptive_simulate(a)
    dump("C Adaptive (follow 15m bias any slot)", score(adaptive))

    print("\nInterpretation:")
    print("  Locked: on a short morning / long afternoon the bot stays FLAT those slots")
    print("          (does NOT take the wrong side — just skips).")
    print("  Flipped: trades those regimes; check if that still PASSes.")
    print("  Adaptive: takes whichever side 15m says, with that side's entry rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
