#!/usr/bin/env python3
"""Backtest pullback hybrid v2 from June 2026 through latest bar in MNQ_1m.csv."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_combined_bidirectional as combo
import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_official as official
import scripts.backtest_scalp_momentum as scalp_bt

JUNE_START = pd.Timestamp("2026-06-01", tz="UTC")
ET = resp.ET


def entry_ts(t: scalp_bt.ScalpTrade) -> pd.Timestamp:
    ts = pd.Timestamp(t.entry_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def filter_from(trades: List[scalp_bt.ScalpTrade], start: pd.Timestamp) -> List[scalp_bt.ScalpTrade]:
    return [t for t in trades if entry_ts(t) >= start]


def monthly(trades: List[scalp_bt.ScalpTrade]) -> dict:
    buckets: dict = {}
    for t in trades:
        k = entry_ts(t).tz_convert(ET).strftime("%Y-%m")
        buckets.setdefault(k, []).append(t)
    return {k: resp.group_metrics(v) for k, v in sorted(buckets.items())}


def session_split(trades: List[scalp_bt.ScalpTrade]) -> dict:
    morning, midday, afternoon = [], [], []
    for t in trades:
        et = entry_ts(t).tz_convert(ET)
        m = et.hour * 60 + et.minute
        if m < 12 * 60:
            morning.append(t)
        elif m < 14 * 60:
            midday.append(t)
        else:
            afternoon.append(t)
    return {
        "morning": resp.group_metrics(morning),
        "midday": resp.group_metrics(midday),
        "afternoon": resp.group_metrics(afternoon),
    }


def tpd(trades: List[scalp_bt.ScalpTrade]) -> dict:
    days = Counter(entry_ts(t).tz_convert(ET).date() for t in trades)
    if not days:
        return {"max_day": 0, "avg_active": 0.0, "days_5plus": 0}
    c = list(days.values())
    return {
        "max_day": max(c),
        "avg_active": round(len(trades) / len(days), 2),
        "days_5plus": sum(1 for x in c if x >= 5),
        "active_days": len(days),
    }


def metrics_block(trades: List[scalp_bt.ScalpTrade]) -> dict:
    if not trades:
        z = resp.group_metrics([])
        return {
            "trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "net_pnl": 0.0, "expectancy": 0.0, "max_drawdown": 0.0,
        }
    m = resp.group_metrics(trades)
    return {
        "trades": m["trades"],
        "win_rate": m["win_rate"],
        "profit_factor": m["profit_factor"],
        "net_pnl": round(m["total_pnl"], 2),
        "expectancy": round(m["expectancy"], 2),
        "max_drawdown": round(m["max_drawdown"], 2),
    }


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    out = os.path.join("data", "june_today_v2_backtest.json")

    df = official.load_1m(csv)
    a = combo.prepare(df)
    all_trades = combo.simulate_combined_recommended(a)
    trades = filter_from(all_trades, JUNE_START)

    end = df["datetime"].iloc[-1]
    longs = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]

    rep = {
        "spec": "pullback_hybrid_v2",
        "period": f"{JUNE_START.date()} through {end}",
        "data_bars": len(df),
        "long_recipe": combo.LONG_RECIPE_V4,
        "short_recipe": combo.SHORT_RECIPE,
        "combined": metrics_block(trades),
        "long": metrics_block(longs),
        "short": metrics_block(shorts),
        "activity": tpd(trades),
        "monthly": {
            k: {
                "trades": v["trades"],
                "profit_factor": v["profit_factor"],
                "net_pnl": round(v["total_pnl"], 2),
            }
            for k, v in monthly(trades).items()
        },
        "session": {
            k: {
                "trades": v["trades"],
                "profit_factor": v["profit_factor"],
                "net_pnl": round(v["total_pnl"], 2),
            }
            for k, v in session_split(trades).items()
        },
    }

    print("PULLBACK HYBRID V2 — June 2026 through latest data")
    print(f"Data: {csv} | bars={len(df):,} | end={end}")
    print(f"Period trades: {rep['combined']['trades']} (entries >= {JUNE_START.date()})")
    c = rep["combined"]
    print(
        f"  WR={c['win_rate']:.1f}% PF={c['profit_factor']:.2f} "
        f"Net=${c['net_pnl']:.1f} Exp=${c['expectancy']:.2f} DD=${c['max_drawdown']:.1f}"
    )
    lo, so = rep["long"], rep["short"]
    print(f"  Long: n={lo['trades']} PF={lo['profit_factor']:.2f} ${lo['net_pnl']:.0f}")
    print(f"  Short: n={so['trades']} PF={so['profit_factor']:.2f} ${so['net_pnl']:.0f}")
    act = rep["activity"]
    print(f"  Activity: max {act['max_day']}/day avg {act['avg_active']}/active day ({act['active_days']} days)")
    print("\nMonthly:")
    for month, m in rep["monthly"].items():
        sign = "+" if m["net_pnl"] > 0 else "-"
        print(f"  {month}: n={m['trades']:3d} PF={m['profit_factor']:.2f} ${m['net_pnl']:7.0f} ({sign})")
    print("\nSession:")
    for name, m in rep["session"].items():
        sign = "+" if m["net_pnl"] > 0 else "-"
        print(f"  {name:10s} n={m['trades']:3d} PF={m['profit_factor']:.2f} ${m['net_pnl']:7.0f} ({sign})")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, default=str)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
