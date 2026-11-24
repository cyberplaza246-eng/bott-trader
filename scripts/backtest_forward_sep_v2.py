#!/usr/bin/env python3
"""Forward score pullback_hybrid_v2 from 2026-09-01 — frozen recipes, no param changes."""
from __future__ import annotations
import json, os, sys
from collections import Counter
from typing import List
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.backtest_combined_bidirectional as combo
import scripts.backtest_ema15_15_5_1_responsive as resp
import scripts.backtest_ema15_official as official
import scripts.backtest_scalp_momentum as scalp_bt

SEP_START = pd.Timestamp("2026-09-01", tz="UTC")
JUNE_START = pd.Timestamp("2026-06-01", tz="UTC")
ET = resp.ET

def entry_ts(t):
    ts = pd.Timestamp(t.entry_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts

def filter_from(trades, start):
    return [t for t in trades if entry_ts(t) >= start]

def metrics_block(trades):
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "net_pnl": 0.0, "expectancy": 0.0, "max_drawdown": 0.0}
    m = resp.group_metrics(trades)
    return {
        "trades": m["trades"], "win_rate": m["win_rate"], "profit_factor": m["profit_factor"],
        "net_pnl": round(m["total_pnl"], 2), "expectancy": round(m["expectancy"], 2),
        "max_drawdown": round(m["max_drawdown"], 2),
    }

def tpd(trades):
    days = Counter(entry_ts(t).tz_convert(ET).date() for t in trades)
    if not days:
        return {"max_day": 0, "avg_active": 0.0, "days_5plus": 0, "active_days": 0}
    c = list(days.values())
    return {"max_day": max(c), "avg_active": round(len(trades)/len(days), 2), "days_5plus": sum(1 for x in c if x >= 5), "active_days": len(days)}

def monthly(trades):
    buckets = {}
    for t in trades:
        k = entry_ts(t).tz_convert(ET).strftime("%Y-%m")
        buckets.setdefault(k, []).append(t)
    return {k: {"trades": v["trades"], "profit_factor": v["profit_factor"], "net_pnl": round(v["total_pnl"], 2)}
            for k, v in ((k, resp.group_metrics(ts)) for k, ts in sorted(buckets.items()))}

def gate(m):
    # frozen_mnq_combined_spec forward_validation
    pf, n, net = m["profit_factor"], m["trades"], m["net_pnl"]
    if n == 0:
        return "NO_TRADES"
    if pf < 1.0:
        return "FAIL"
    if 1.0 <= pf < 1.2:
        return "BORDERLINE"
    if pf >= 1.2 and net > 0:
        return "GOOD"
    return "BORDERLINE"

def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    df = official.load_1m(csv)
    a = combo.prepare(df)
    all_trades = combo.simulate_combined_recommended(a)
    end = df["datetime"].iloc[-1]
    jun = filter_from(all_trades, JUNE_START)
    sep = filter_from(all_trades, SEP_START)
    rep = {
        "spec": "pullback_hybrid_v2",
        "data_bars": len(df),
        "data_end": str(end),
        "long_recipe": combo.LONG_RECIPE_V4,
        "short_recipe": combo.SHORT_RECIPE,
        "june_through_latest": {
            "period": f"{JUNE_START.date()} through {end}",
            "combined": metrics_block(jun),
            "long": metrics_block([t for t in jun if t.direction=="long"]),
            "short": metrics_block([t for t in jun if t.direction=="short"]),
            "activity": tpd(jun),
            "monthly": monthly(jun),
        },
        "forward_sep_plus": {
            "period": f"{SEP_START.date()} through {end}",
            "combined": metrics_block(sep),
            "long": metrics_block([t for t in sep if t.direction=="long"]),
            "short": metrics_block([t for t in sep if t.direction=="short"]),
            "activity": tpd(sep),
            "monthly": monthly(sep),
            "frozen_gate": gate(metrics_block(sep)),
            "gate_rules": {"good": "PF>=1.2 and net>0", "borderline": "PF 1.0-1.19", "fail": "PF<1.0"},
        },
    }
    out = os.path.join("data", "forward_sep_v2_backtest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, default=str)
    fs = rep["forward_sep_plus"]["combined"]
    jl = rep["june_through_latest"]["combined"]
    print("=== DATA ===")
    print(f"bars={len(df):,} end={end}")
    print("=== JUN through latest (frozen V2) ===")
    print(f"n={jl['trades']} WR={jl['win_rate']:.1f}% PF={jl['profit_factor']:.2f} Net=${jl['net_pnl']:.2f} Exp=${jl['expectancy']:.2f} DD=${jl['max_drawdown']:.2f}")
    print("monthly", rep["june_through_latest"]["monthly"])
    print("=== FORWARD Sep+ (frozen V2, no retune) ===")
    print(f"n={fs['trades']} WR={fs['win_rate']:.1f}% PF={fs['profit_factor']:.2f} Net=${fs['net_pnl']:.2f} Exp=${fs['expectancy']:.2f} DD=${fs['max_drawdown']:.2f}")
    print("activity", rep["forward_sep_plus"]["activity"])
    print("long", rep["forward_sep_plus"]["long"])
    print("short", rep["forward_sep_plus"]["short"])
    print("GATE:", rep["forward_sep_plus"]["frozen_gate"])
    print("Saved", out)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
