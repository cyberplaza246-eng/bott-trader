#!/usr/bin/env python3
"""
Compare variant D with/without entry_quality filters on cached Rithmic data.

Usage:
    python scripts/backtest_entry_quality.py
    python scripts/backtest_entry_quality.py --session extended
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backtest_adaptive_variants import (
    AdaptiveBacktester,
    StrategyVariant,
    apply_base_cfg,
    resample_15m,
)
from scripts.backtest_nq_rithmic import resample_5m
from scripts.backtest_rithmic_adaptive import load_profit_cfg, variant_from_cfg
from src.ai.entry_quality import parse_entry_quality


def load_cached(symbol: str) -> pd.DataFrame:
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    path = os.path.join(data_dir, f"{symbol}_1m_rithmic.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def run_one(symbol: str, df_1m: pd.DataFrame, cfg: Dict[str, Any], variant: StrategyVariant) -> Dict[str, Any]:
    df_5m = resample_5m(df_1m)
    df_15m = resample_15m(df_1m)
    apply_base_cfg(cfg)
    variant.min_rr = cfg.get("min_rr", 1.0)
    bt = AdaptiveBacktester(symbol, variant, cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        stats = bt.run_variant(df_1m, df_5m, df_15m)
    if stats.get("error"):
        return {"error": stats["error"]}
    return {
        "trades": stats["total_trades"],
        "wr": stats["win_rate"],
        "pf": stats["profit_factor"],
        "pnl": stats["total_pnl"],
        "dd": stats["max_drawdown_pct"],
    }


ENTRY_QUALITY_CANDIDATES: List[Dict[str, Any]] = [
    {"name": "none", "entry_quality": {"enabled": False}},
    {
        "name": "strict_short",
        "entry_quality": {
            "enabled": True,
            "short_below_ema9": True,
            "short_macd_negative": True,
            "short_block_green_bars": 2,
        },
    },
    {
        "name": "strict_short_di",
        "entry_quality": {
            "enabled": True,
            "short_below_ema9": True,
            "short_macd_negative": True,
            "short_block_green_bars": 2,
            "short_di_margin": 5.0,
        },
    },
    {
        "name": "full_balanced",
        "entry_quality": {
            "enabled": True,
            "short_below_ema9": True,
            "long_above_ema9": True,
            "short_macd_negative": True,
            "long_macd_positive": True,
            "short_block_green_bars": 2,
            "long_block_red_bars": 2,
            "short_di_margin": 5.0,
            "long_di_margin": 5.0,
            "extended_adx_min": 22,
            "short_rsi_falling_max": 60,
        },
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="extended", choices=["rth", "extended"])
    parser.add_argument("--symbol", default="both", choices=["NQ", "MNQ", "both"])
    args = parser.parse_args()

    base_cfg = load_profit_cfg()
    base_cfg["live_like"] = True
    base_cfg["session_mode"] = args.session
    variant = variant_from_cfg(base_cfg)
    symbols = ["NQ", "MNQ"] if args.symbol == "both" else [args.symbol]

    print(f"\n{'='*78}")
    print("  ENTRY QUALITY A/B — Rithmic cached, variant D")
    print(f"  Session: {args.session} | Base: {base_cfg.get('name')}")
    print(f"{'='*78}\n")

    results: Dict[str, Any] = {"session": args.session, "base": base_cfg, "candidates": []}
    best_name = None
    best_score = -999.0

    for cand in ENTRY_QUALITY_CANDIDATES:
        cfg = copy.deepcopy(base_cfg)
        cfg["entry_quality"] = cand["entry_quality"]
        row: Dict[str, Any] = {"name": cand["name"], "entry_quality": cand["entry_quality"], "symbols": {}}
        min_pf = 999.0
        min_trades = 9999
        score = 0.0

        for sym in symbols:
            df_1m = load_cached(sym)
            stats = run_one(sym, df_1m, cfg, variant)
            row["symbols"][sym] = stats
            if "error" not in stats:
                min_pf = min(min_pf, stats["pf"])
                min_trades = min(min_trades, stats["trades"])
                score += stats["pf"] * 10 + stats["wr"] * 0.1

        row["score"] = score
        results["candidates"].append(row)

        sym_parts = []
        for sym in symbols:
            s = row["symbols"][sym]
            if "error" in s:
                sym_parts.append(f"{sym}=ERR")
            else:
                sym_parts.append(
                    f"{sym} {s['trades']}tr WR={s['wr']:.0f}% PF={s['pf']:.2f} ${s['pnl']:,.0f}"
                )
        mark = ">>" if score > best_score and min_pf >= 1.0 and min_trades >= 40 else "  "
        if score > best_score and min_pf >= 1.0 and min_trades >= 40:
            best_score = score
            best_name = cand["name"]
        print(f"{mark} {cand['name']:<18} | {' | '.join(sym_parts)}")

    baseline = next(c for c in results["candidates"] if c["name"] == "none")
    winner = next((c for c in results["candidates"] if c["name"] == best_name), baseline)
    results["winner"] = winner

    print(f"\nBaseline (no entry_quality):")
    for sym in symbols:
        s = baseline["symbols"][sym]
        print(f"  {sym}: {s['trades']} tr | WR {s['wr']:.1f}% | PF {s['pf']:.2f} | ${s['pnl']:,.0f}")

    if best_name and best_name != "none":
        print(f"\nWinner: {best_name}")
        for sym in symbols:
            s = winner["symbols"][sym]
            b = baseline["symbols"][sym]
            d_pf = s["pf"] - b["pf"]
            d_wr = s["wr"] - b["wr"]
            d_tr = s["trades"] - b["trades"]
            print(
                f"  {sym}: {s['trades']} tr ({d_tr:+d}) | WR {s['wr']:.1f}% ({d_wr:+.1f}) | "
                f"PF {s['pf']:.2f} ({d_pf:+.2f}) | ${s['pnl']:,.0f}"
            )

    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "entry_quality_backtest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
