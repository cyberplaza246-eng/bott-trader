#!/usr/bin/env python3
"""Find configs with >=6 trades/weekday (Jun-Aug) and report P&L."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import date, timedelta
from itertools import product
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_combined_bidirectional as combo
import scripts.backtest_ema15_official as official
import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.pass_hunt_lower_tf as base
import scripts.pass_hunt_short_rth_all_windows as srth
import scripts.pass_hunt_shorts as hunt

JUNE = pd.Timestamp("2026-06-01", tz="UTC")
ET = resp.ET


def prep_trade(t):
    ts = pd.Timestamp(t.entry_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def jun_trades(trades: List) -> List:
    return [t for t in trades if prep_trade(t) >= JUNE]


def weekday_stats(trades: List, end_d: date) -> Dict[str, Any]:
    trades = jun_trades(trades)
    days = Counter(prep_trade(t).tz_convert(ET).date() for t in trades)
    cal = []
    d = date(2026, 6, 1)
    while d <= end_d:
        if d.weekday() < 5:
            cal.append(d)
        d += timedelta(days=1)
    n_cal = len(cal)
    n_active = len(days)
    total = len(trades)
    counts = list(days.values()) if days else []
    avg_all = total / n_cal if n_cal else 0.0
    avg_active = total / n_active if n_active else 0.0
    days_ge6 = sum(1 for c in counts if c >= 6)
    pct_ge6 = days_ge6 / n_cal if n_cal else 0.0
    return {
        "total": total,
        "weekdays": n_cal,
        "active": n_active,
        "avg_all_weekdays": round(avg_all, 2),
        "avg_active": round(avg_active, 2),
        "max_day": max(counts) if counts else 0,
        "days_ge6": days_ge6,
        "pct_weekdays_ge6": round(pct_ge6 * 100, 1),
        "median_active": sorted(counts)[len(counts) // 2] if counts else 0,
    }


def pnl_block(trades: List) -> Dict[str, Any]:
    trades = jun_trades(trades)
    if not trades:
        return {"trades": 0, "pf": 0.0, "net": 0.0}
    m = resp.group_metrics(trades)
    return {
        "trades": m["trades"],
        "pf": round(m["profit_factor"], 2),
        "net": round(m["total_pnl"], 1),
    }


def make_long(win: str, entry: str, max_day, max_lots: int) -> Dict:
    c = dict(hunt.LONG_WINNER)
    c["win"] = win
    c["entry"] = entry
    c["max_day"] = max_day
    c["max_lots"] = max_lots
    return c


def make_short(win: str, entry: str, structure: str | None, max_day, max_lots: int) -> Dict:
    c = dict(combo.SHORT_RECIPE)
    c["win"] = win
    c["entry"] = entry
    c["max_day"] = max_day
    c["max_lots"] = max_lots
    if structure is None:
        c.pop("structure", None)
    else:
        c["structure"] = structure
    return c


def main() -> int:
    csv = os.path.join("data", "MNQ_1m.csv")
    df = official.load_1m(csv)
    ha = hunt.prepare(df)
    a = combo.prepare(df)
    end_d = df["datetime"].iloc[-1].tz_convert(ET).date()

    wins = ("w6_all", "w4_no_noon")
    entries = ("quiet", "1m_pullback", "quiet2")
    max_days = (99, 12, 8)
    max_lots_opts = (1, 2)
    structures = ("lh_ll", None)
    modes = ("stacked", "combined_w6", "combined_v4")

    rows = []
    for mode in modes:
        for win in wins:
            for ent_l in entries:
                for ent_s in entries:
                    for struct in structures:
                        for md in max_days:
                            for ml in max_lots_opts:
                                if mode == "stacked":
                                    lc = make_long(win, ent_l, md, ml)
                                    sc = make_short(win, ent_s, struct, md, ml)
                                    trades = hunt.simulate_ext(ha, lc) + srth.simulate_short(a, sc)
                                    name = f"stack L={win}/{ent_l} S={win}/{ent_s} st={struct} md={md} ml={ml}"
                                elif mode == "combined_w6":
                                    if win != "w6_all" or struct != "lh_ll":
                                        continue
                                    saved_l, saved_s = combo.LONG_RECIPE, combo.SHORT_RECIPE
                                    combo.LONG_RECIPE = make_long(win, ent_l, md, 1)
                                    combo.SHORT_RECIPE = make_short(win, ent_s, struct, md, 1)
                                    trades = combo.simulate_combined(a)
                                    combo.LONG_RECIPE, combo.SHORT_RECIPE = saved_l, saved_s
                                    name = f"combo_w6 L={ent_l} S={ent_s} md={md}"
                                else:
                                    if win != "w4_no_noon":
                                        continue
                                    saved_l, saved_s = combo.LONG_RECIPE, combo.SHORT_RECIPE
                                    combo.LONG_RECIPE = make_long(win, ent_l, md, 1)
                                    combo.SHORT_RECIPE = make_short("w6_all", ent_s, struct, md, 1)
                                    trades = combo.simulate_combined(a)
                                    combo.LONG_RECIPE, combo.SHORT_RECIPE = saved_l, saved_s
                                    name = f"combo_v4 L={ent_l} S={ent_s} md={md}"

                                act = weekday_stats(trades, end_d)
                                pnl = pnl_block(trades)
                                rows.append({"name": name, **act, **pnl})

    hit6_avg = [r for r in rows if r["avg_all_weekdays"] >= 6.0]
    hit6_days = [r for r in rows if r["pct_weekdays_ge6"] >= 50.0]
    hit6_avg.sort(key=lambda r: (-r["pf"], -r["net"], -r["avg_all_weekdays"]))
    hit6_days.sort(key=lambda r: (-r["pf"], -r["pct_weekdays_ge6"], -r["avg_all_weekdays"]))

    all_sorted = sorted(rows, key=lambda r: (-r["avg_all_weekdays"], -r["pf"]))

    print("TOP 15 by avg trades/weekday (Jun-Aug):")
    for r in all_sorted[:15]:
        print(
            f"  avg={r['avg_all_weekdays']:4.2f} max={r['max_day']} ge6d={r['days_ge6']:2d} "
            f"n={r['trades']:4d} PF={r['pf']:4.2f} ${r['net']:7.0f} | {r['name'][:70]}"
        )

    print("\nConfigs with avg >= 6 trades/weekday:")
    if not hit6_avg:
        print("  NONE")
    else:
        for r in hit6_avg[:20]:
            print(
                f"  avg={r['avg_all_weekdays']:4.2f} PF={r['pf']:4.2f} ${r['net']:7.0f} "
                f"ge6={r['pct_weekdays_ge6']:.0f}% | {r['name']}"
            )

    profitable6 = [r for r in hit6_avg if r["pf"] >= 1.0 and r["net"] > 0]
    print("\nAvg>=6 AND profitable:")
    if not profitable6:
        print("  NONE")
    else:
        for r in profitable6[:10]:
            print(f"  avg={r['avg_all_weekdays']:.2f} PF={r['pf']:.2f} ${r['net']:.0f} | {r['name']}")

    out = os.path.join("data", "six_trades_day_search.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "hit6_avg": hit6_avg[:30],
                "top_activity": all_sorted[:30],
                "profitable6": profitable6[:15],
            },
            f,
            indent=2,
        )
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
