#!/usr/bin/env python3
"""
Before/after backtest for profitability guards (flow block, strict DI, bearish 1M bar).

Usage:
    python scripts/backtest_profitability_guards.py
    python scripts/backtest_profitability_guards.py --session extended
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backtest_adaptive_variants import (
    AdaptiveBacktester,
    StrategyVariant,
    apply_base_cfg,
    load_frames,
    merge_run_cfg,
    resample_15m,
)
from scripts.backtest_rithmic_adaptive import load_profit_cfg, variant_from_cfg


def avg_hold_minutes(trades) -> float:
    if not trades:
        return 0.0
    mins = []
    for t in trades:
        et = getattr(t, "exit_time", None)
        it = getattr(t, "entry_time", None)
        if et is None or it is None:
            continue
        delta = (et - it).total_seconds() / 60.0
        if delta >= 0:
            mins.append(delta)
    return sum(mins) / len(mins) if mins else 0.0


def run_symbol(symbol: str, df_1m, df_5m, df_15m, cfg: Dict[str, Any], variant: StrategyVariant) -> Dict[str, Any]:
    apply_base_cfg(cfg)
    variant.min_rr = float(cfg.get("min_rr", 1.0))
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
        "sl_exits": stats.get("sl_exits", 0),
        "avg_hold_min": round(avg_hold_minutes(bt.trades), 1),
    }


def baseline_cfg(live: Dict[str, Any], session: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(live)
    cfg["live_like"] = True
    cfg["session_mode"] = session
    cfg["use_flow_proxy"] = True
    cfg["flow_proxy_bars"] = 5
    # Revert new guards for baseline comparison
    cfg.pop("flow_entry_guard", None)
    eq = dict(cfg.get("entry_quality") or {})
    eq.pop("long_block_bearish_bar", None)
    cfg["entry_quality"] = eq
    if "di_tol" in cfg:
        cfg["di_tol"] = 3.0
    return cfg


def guarded_cfg(live: Dict[str, Any], session: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(live)
    cfg["live_like"] = True
    cfg["session_mode"] = session
    cfg["use_flow_proxy"] = True
    cfg["flow_proxy_bars"] = 5
    cfg["di_tol"] = 0.0
    cfg["di_flow_tol"] = 8.0
    cfg["flow_entry_guard"] = {
        "enabled": True,
        "long_buy_pct_min": 0.48,
        "short_buy_pct_max": 0.52,
    }
    eq = dict(cfg.get("entry_quality") or {})
    eq["enabled"] = True
    eq["long_block_bearish_bar"] = True
    cfg["entry_quality"] = eq
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="extended", choices=["rth", "extended"])
    args = parser.parse_args()

    live = load_profit_cfg()
    variant = variant_from_cfg(live)
    df_1m, df_5m = load_frames()
    df_15m = resample_15m(df_1m)

    before_cfg = baseline_cfg(live, args.session)
    after_cfg = guarded_cfg(live, args.session)

    print(f"\n{'='*78}")
    print("  PROFITABILITY GUARDS — before vs after (variant D, extended session)")
    print(f"  Config: {live.get('name')} | strong_trend_skip_macd={live.get('strong_trend_skip_macd')}")
    print(f"  Bars: {len(df_1m):,} × 1m")
    print(f"{'='*78}\n")

    results: Dict[str, Any] = {"session": args.session, "before": {}, "after": {}}
    for label, cfg in (("before", before_cfg), ("after", after_cfg)):
        for sym in ("MNQ", "NQ"):
            stats = run_symbol(sym, df_1m, df_5m, df_15m, cfg, variant)
            results[label][sym] = stats
            if "error" in stats:
                print(f"  {label:<6} {sym}: {stats['error']}")
            else:
                print(
                    f"  {label:<6} {sym}: {stats['trades']:>3}tr "
                    f"WR={stats['wr']:>4.0f}% PF={stats['pf']:.2f} "
                    f"PnL=${stats['pnl']:>7,.0f} SL={stats['sl_exits']} "
                    f"hold={stats['avg_hold_min']:.1f}m"
                )

    mnq_b = results["before"]["MNQ"]
    mnq_a = results["after"]["MNQ"]
    nq_b = results["before"]["NQ"]
    nq_a = results["after"]["NQ"]
    if "error" not in mnq_a and "error" not in nq_a:
        print(
            f"\n  Delta MNQ: PF {mnq_a['pf'] - mnq_b['pf']:+.2f}, "
            f"WR {mnq_a['wr'] - mnq_b['wr']:+.1f}%, "
            f"trades {mnq_a['trades'] - mnq_b['trades']:+d}, "
            f"hold {mnq_a['avg_hold_min'] - mnq_b['avg_hold_min']:+.1f}m"
        )
        print(
            f"  Delta NQ:  PF {nq_a['pf'] - nq_b['pf']:+.2f}, "
            f"WR {nq_a['wr'] - nq_b['wr']:+.1f}%, "
            f"trades {nq_a['trades'] - nq_b['trades']:+d}, "
            f"hold {nq_a['avg_hold_min'] - nq_b['avg_hold_min']:+.1f}m"
        )
        ok = mnq_a["pf"] >= 1.2 and nq_a["pf"] >= 1.2
        print(f"\n  Verdict: {'PASS — both PF >= 1.2' if ok else 'REVIEW — PF below 1.2 on one symbol'}")

    out = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "profitability_guards_backtest.json",
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
