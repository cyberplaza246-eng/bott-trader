#!/usr/bin/env python3
"""
Test 2: Decompose dynamic afternoon trades (entry >= 12:00 ET) to find failure regimes.

Uses the full-day dynamic spec (Pattern A) and tags each afternoon entry with:
  direction, hour, hour block, 15m structure, VWAP/EMA alignment, trigger type,
  sep/ATR quality, win/loss, R-multiple.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_official as official
import scripts.backtest_scalp_momentum as scalp_bt
from scripts.backtest_dynamic_15_5_1_rth import (
    MORNING_END,
    prepare_arrays,
    r_multiple,
    simulate_dynamic,
)


def hour_block(m: int) -> str:
    if m < 13 * 60:
        return "12:00-13:00"
    if m < 14 * 60:
        return "13:00-14:00"
    if m < 15 * 60:
        return "14:00-15:00"
    return "15:00+"



def atr_bucket(atr: float) -> str:
    if atr != atr or atr <= 0:
        return "unknown"
    if atr < 8:
        return "atr_lt_8"
    if atr < 12:
        return "atr_8_12"
    if atr < 18:
        return "atr_12_18"
    return "atr_ge_18"


def enrich(trades: List[scalp_bt.ScalpTrade], meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for t, m in zip(trades, meta):
        r = round(r_multiple(t), 2)
        rows.append({
            **m,
            "direction": t.direction,
            "pnl": round(t.pnl, 2),
            "win": t.pnl > 0,
            "r_mult": r,
            "hour_block": hour_block(int(m["entry_minute"])),
            "sep_bucket": "sep_high" if m["sep_high"] else "sep_low",
            "atr_bucket": atr_bucket(float(m["atr1"])),
            "vwap_label": "aligned" if m["vwap_aligned"] else "conflict",
            "ema_label": "aligned" if m["ema_aligned"] else "conflict",
        })
    return rows


def afternoon_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if int(r["entry_minute"]) >= MORNING_END]


def bucket_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "net_pnl": 0.0, "mean_r": 0.0}
    wins = sum(r["pnl"] for r in rows if r["pnl"] > 0)
    losses = abs(sum(r["pnl"] for r in rows if r["pnl"] <= 0))
    pf = wins / losses if losses > 0 else (999.0 if wins > 0 else 0.0)
    r_vals = [r["r_mult"] for r in rows]
    return {
        "trades": len(rows),
        "win_rate": round(100.0 * sum(1 for r in rows if r["win"]) / len(rows), 1),
        "profit_factor": round(pf, 2),
        "net_pnl": round(sum(r["pnl"] for r in rows), 2),
        "mean_r": round(float(np.mean(r_vals)), 2) if r_vals else 0.0,
    }


def decompose(rows: List[Dict[str, Any]], key_fn: Callable) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(key_fn(r))].append(r)
    return {k: bucket_metrics(v) for k, v in sorted(groups.items())}


def print_section(title: str, data: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for k, m in data.items():
        print(
            f"  {k:20s} n={m['trades']:4d} WR={m['win_rate']:5.1f}% "
            f"PF={m['profit_factor']:5.2f} Net=${m['net_pnl']:8.1f} meanR={m['mean_r']:5.2f}"
        )


def worst_buckets(data: Dict[str, Dict[str, Any]], min_n: int = 20) -> List[Tuple[str, Dict[str, Any]]]:
    eligible = [(k, v) for k, v in data.items() if v["trades"] >= min_n]
    return sorted(eligible, key=lambda kv: kv[1]["net_pnl"])[:5]


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    out = os.path.join("data", "afternoon_decomposition.json")

    print("TEST 2 — Afternoon trade decomposition (full dynamic spec, Pattern A)")
    df = official.load_1m(csv)
    a = prepare_arrays(df)
    trades, stats, meta = simulate_dynamic(a, pattern="A", sep_noon=True, capture_meta=True)
    assert meta is not None

    all_rows = enrich(trades, meta)
    aft = afternoon_rows(all_rows)
    print(f"Total dynamic trades: {len(trades)} | Afternoon entries (>=12:00): {len(aft)}")
    print(f"Afternoon aggregate: {bucket_metrics(aft)}")

    sections = {
        "by_direction": decompose(aft, lambda r: r["direction"]),
        "by_hour": decompose(aft, lambda r: f"{int(r['hour']):02d}:xx"),
        "by_hour_block": decompose(aft, lambda r: r["hour_block"]),
        "by_structure": decompose(aft, lambda r: r["struct_label"]),
        "by_vwap": decompose(aft, lambda r: r["vwap_label"]),
        "by_ema": decompose(aft, lambda r: r["ema_label"]),
        "by_trigger": decompose(aft, lambda r: r["pattern"]),
        "by_sep": decompose(aft, lambda r: r["sep_bucket"]),
        "by_atr": decompose(aft, lambda r: r["atr_bucket"]),
        "by_direction_x_hour_block": decompose(
            aft, lambda r: f"{r['direction']}|{r['hour_block']}"
        ),
        "by_direction_x_vwap": decompose(
            aft, lambda r: f"{r['direction']}|{r['vwap_label']}"
        ),
        "by_structure_x_direction": decompose(
            aft, lambda r: f"{r['struct_label']}|{r['direction']}"
        ),
    }

    for title, data in sections.items():
        print_section(title.replace("_", " ").title(), data)

    print(f"\n{'=' * 90}")
    print("WORST AFTERNOON REGIMES (n>=20, sorted by net P&L)")
    print(f"{'=' * 90}")
    findings: List[str] = []
    for sec_name, data in sections.items():
        for k, m in worst_buckets(data, min_n=20):
            if m["net_pnl"] < -100:
                findings.append(f"{sec_name}/{k}: n={m['trades']} PF={m['profit_factor']} Net=${m['net_pnl']}")

    for f in findings[:12]:
        print(f"  {f}")

    # R-mult distribution for afternoon
    r_vals = [r["r_mult"] for r in aft]
    r_buckets = {
        "lt_0": sum(1 for r in r_vals if r < 0),
        "0_0.5": sum(1 for r in r_vals if 0 <= r < 0.5),
        "0.5_1": sum(1 for r in r_vals if 0.5 <= r < 1.0),
        "1_2": sum(1 for r in r_vals if 1.0 <= r < 2.0),
        "ge_2": sum(1 for r in r_vals if r >= 2.0),
    }

    # Synthesize one structural failure hypothesis
    dir_hour = sections["by_direction_x_hour_block"]
    short_13_14 = dir_hour.get("short|13:00-14:00", {})
    short_14_15 = dir_hour.get("short|14:00-15:00", {})
    long_13_14 = dir_hour.get("long|13:00-14:00", {})
    vwap_conflict = sections["by_vwap"].get("conflict", {})
    hour_13 = sections["by_hour_block"].get("13:00-14:00", {})

    hypotheses = []
    if short_13_14.get("trades", 0) >= 15 and short_13_14.get("profit_factor", 99) < 0.85:
        hypotheses.append(
            f"Afternoon shorts 13:00–14:00 are a primary bleed "
            f"(n={short_13_14['trades']}, PF={short_13_14['profit_factor']}, ${short_13_14['net_pnl']})."
        )
    if short_14_15.get("trades", 0) >= 15 and short_14_15.get("profit_factor", 99) < 0.85:
        hypotheses.append(
            f"Afternoon shorts 14:00–15:00 also fail "
            f"(n={short_14_15['trades']}, PF={short_14_15['profit_factor']}, ${short_14_15['net_pnl']})."
        )
    if long_13_14.get("trades", 0) >= 20 and long_13_14.get("profit_factor", 99) < 0.9:
        hypotheses.append(
            f"Longs are not safe either in 13:00–14:00 "
            f"(n={long_13_14['trades']}, PF={long_13_14['profit_factor']}, ${long_13_14['net_pnl']})."
        )
    if hour_13.get("trades", 0) >= 30 and hour_13.get("profit_factor", 99) < 0.85:
        hypotheses.append(
            f"13:00–14:00 hour block is the worst overall "
            f"(n={hour_13['trades']}, PF={hour_13['profit_factor']}, ${hour_13['net_pnl']})."
        )
    if vwap_conflict.get("trades", 0) >= 30:
        aligned = sections["by_vwap"].get("aligned", {})
        if vwap_conflict.get("profit_factor", 99) < aligned.get("profit_factor", 0):
            hypotheses.append(
                f"VWAP conflict trades underperform aligned "
                f"(conflict PF={vwap_conflict['profit_factor']} vs aligned PF={aligned.get('profit_factor', 0)})."
            )

    primary = hypotheses[0] if hypotheses else "No single regime dominates; afternoon losses are spread across buckets."

    print(f"\nPrimary structural failure (data-driven, not a filter prescription):")
    print(f"  {primary}")
    if len(hypotheses) > 1:
        for h in hypotheses[1:]:
            print(f"  Also: {h}")

    payload = {
        "test": "afternoon_decomposition",
        "spec": "dynamic_15_5_1_rth pattern A full day",
        "afternoon_trades": len(aft),
        "aggregate": bucket_metrics(aft),
        "sections": sections,
        "r_multiples": {
            "mean": round(float(np.mean(r_vals)), 2) if r_vals else 0.0,
            "median": round(float(np.median(r_vals)), 2) if r_vals else 0.0,
            "buckets": r_buckets,
        },
        "sim_stats": {
            "signals_stack": stats.signals_stack,
            "signals_pattern": stats.signals_pattern,
            "filtered_sep": stats.filtered_sep,
            "entries": stats.entries,
        },
        "hypotheses": hypotheses,
        "primary_failure": primary,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
