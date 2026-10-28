#!/usr/bin/env python3
"""Sweep hybrid scalp exit + entry params for fast in/out + profitability."""
from __future__ import annotations

import itertools
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_scalp_momentum as scalp_bt
from scripts.backtest_scalp_hybrid import (
    HybridParams,
    HybridVariant,
    add_30s_body_stats,
    run_hybrid_variant,
)


@dataclass
class FastScalpParams(HybridParams):
    """Relaxed live-style entry params with tunable ADX floors."""

    adx_min_pullback: int = 15
    adx_min_continuation: int = 20
    momentum_burst_adx: int = 15

    @classmethod
    def fast_live(cls) -> "FastScalpParams":
        return cls(
            pullback_atr=0.75,
            setup_window_sec=90,
            trend_mode="vwap",
            continuation_volume_strict=False,
            chase_body_mult=2.0,
            chase_ema_atr=1.25,
            momentum_burst_enabled=True,
            momentum_burst_adx=15,
            adx_min_pullback=15,
            adx_min_continuation=20,
        )

    @classmethod
    def ultra_fast(cls) -> "FastScalpParams":
        """More entries: wider pullback, lower ADX, shorter setup window."""
        return cls(
            pullback_atr=0.85,
            setup_bars=2,
            setup_window_sec=60,
            trend_mode="vwap",
            continuation_volume_strict=False,
            chase_body_mult=2.5,
            chase_ema_atr=1.5,
            momentum_burst_enabled=True,
            momentum_burst_adx=15,
            adx_min_pullback=15,
            adx_min_continuation=18,
        )


def score(r: Dict[str, Any]) -> float:
    """Higher = better: PF-weighted PnL with trade count + hold-time bonus."""
    if r["trades"] < 50 or r["profit_factor"] < 1.0:
        return -999.0
    hold_bonus = max(0.0, 60.0 - r["avg_hold_sec"]) / 60.0
    return r["total_pnl"] * r["profit_factor"] * (1.0 + 0.3 * hold_bonus)


def main() -> None:
    df_1m_raw, df_5m_raw, df_30s_raw, data_note = scalp_bt.load_data(None, None, None)
    df_1m = scalp_bt.add_1m_indicators(df_1m_raw)
    df_5m = scalp_bt.add_5m_indicators(df_5m_raw)
    df_30s = add_30s_body_stats(df_30s_raw)

    variant = HybridVariant("hybrid", pullback_enabled=True, continuation_enabled=True)
    param_sets = [
        ("fast_live", FastScalpParams.fast_live()),
        ("ultra_fast", FastScalpParams.ultra_fast()),
    ]

    sl_vals = [6.0, 8.0]
    tp_vals = [10.0, 12.0, 14.0]
    hold_vals = [30, 45, 60]

    results: List[Dict[str, Any]] = []
    total = len(param_sets) * len(sl_vals) * len(tp_vals) * len(hold_vals)
    n = 0
    for (pname, params), sl, tp, hold in itertools.product(param_sets, sl_vals, tp_vals, hold_vals):
        if tp < sl * 1.2:
            continue
        n += 1
        print(f"[{n}/{total}] {pname} SL={sl} TP={tp} hold={hold}s", flush=True)
        stats = run_hybrid_variant(
            variant, df_1m, df_5m, df_30s, "MNQ", sl, tp, hold, params,
        )
        row = {
            "param_set": pname,
            "sl_pts": sl,
            "tp_pts": tp,
            "max_hold_sec": hold,
            "score": round(score(stats), 2),
            **{k: stats[k] for k in (
                "trades", "win_rate", "profit_factor", "expectancy",
                "total_pnl", "max_drawdown", "avg_hold_sec",
                "tp_exits", "sl_exits", "max_hold_exits",
            )},
            "entry_params": asdict(params) if hasattr(params, "__dataclass_fields__") else params.__dict__,
        }
        results.append(row)

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:15]

    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "scalp_fast_sweep_results.json")
    payload = {
        "data_note": data_note,
        "period": {
            "start": str(df_1m["datetime"].iloc[0]),
            "end": str(df_1m["datetime"].iloc[-1]),
            "bars_1m": len(df_1m),
        },
        "top15": top,
        "all_count": len(results),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'=' * 100}")
    print("TOP 10 FAST SCALP CONFIGS (score = PF × PnL × hold bonus, PF>=1 only)")
    print(f"{'=' * 100}")
    hdr = f"{'Rank':<5} {'Set':<12} {'SL':>4} {'TP':>4} {'Hold':>5} {'Trades':>7} {'WR%':>6} {'PF':>6} {'PnL$':>9} {'Hold s':>7}"
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(top[:10], 1):
        print(
            f"{i:<5} {r['param_set']:<12} {r['sl_pts']:>4.0f} {r['tp_pts']:>4.0f} "
            f"{r['max_hold_sec']:>5} {r['trades']:>7} {r['win_rate']:>6.1f} "
            f"{r['profit_factor']:>6.2f} {r['total_pnl']:>9.2f} {r['avg_hold_sec']:>7.1f}"
        )
    print(f"\nSaved: {out}")
    if top:
        print(f"\nWinner: {top[0]['param_set']} SL={top[0]['sl_pts']} TP={top[0]['tp_pts']} hold={top[0]['max_hold_sec']}s")


if __name__ == "__main__":
    main()
