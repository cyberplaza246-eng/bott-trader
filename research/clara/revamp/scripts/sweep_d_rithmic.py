#!/usr/bin/env python3
"""Sweep tuned variant-D configs on cached Rithmic NQ+MNQ (live_like)."""
from __future__ import annotations

import itertools
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backtest_adaptive_variants import StrategyVariant, apply_base_cfg, resample_15m
from scripts.backtest_rithmic_adaptive import load_cached, load_profit_cfg, run_symbol


def base_d_cfg() -> Dict[str, Any]:
    cfg = load_profit_cfg()
    cfg["live_like"] = True
    cfg["strategy_mode"] = "full_adaptive"
    cfg["use_15m_bias"] = True
    cfg["bear_adaptive"] = True
    cfg["bull_adaptive"] = True
    return cfg


def make_variant(cfg: Dict[str, Any]) -> StrategyVariant:
    return StrategyVariant(
        name="D",
        label="full_adaptive",
        use_15m_bias=True,
        bear_adaptive=True,
        bull_adaptive=True,
        counter_trend_shorts=True,
        min_rr=cfg.get("min_rr", 1.0),
    )


def score_pair(nq: Dict[str, Any], mnq: Dict[str, Any]) -> float:
    if nq.get("profit_factor", 0) < 1.0 or mnq.get("profit_factor", 0) < 1.0:
        return -999.0
    if nq.get("total_pnl", 0) <= 0 or mnq.get("total_pnl", 0) <= 0:
        return -999.0
    return min(nq["profit_factor"], mnq["profit_factor"]) * 100 + min(nq["total_pnl"], mnq["total_pnl"]) / 500


def run_combo(
    nq_1m,
    mnq_1m,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    variant = make_variant(cfg)
    apply_base_cfg(cfg)
    nq = run_symbol("NQ", nq_1m, cfg, variant)
    mnq = run_symbol("MNQ", mnq_1m, cfg, variant)
    return {"nq": nq, "mnq": mnq, "score": score_pair(nq, mnq)}


def main() -> None:
    nq_1m, _ = load_cached("NQ")
    mnq_1m, _ = load_cached("MNQ")
    print(f"NQ  {len(nq_1m):,} bars  {nq_1m['datetime'].iloc[0]} -> {nq_1m['datetime'].iloc[-1]}")
    print(f"MNQ {len(mnq_1m):,} bars  {mnq_1m['datetime'].iloc[0]} -> {mnq_1m['datetime'].iloc[-1]}\n")

    # Baseline D (current defaults)
    base = base_d_cfg()
    base_r = run_combo(nq_1m, mnq_1m, dict(base))
    print(
        f"BASE D: NQ PF={base_r['nq']['profit_factor']:.2f} ${base_r['nq']['total_pnl']:,.0f} | "
        f"MNQ PF={base_r['mnq']['profit_factor']:.2f} ${base_r['mnq']['total_pnl']:,.0f}\n"
    )

    # Phase 1: focused grid (~48 combos, ~20 min)
    grid: List[Dict[str, Any]] = []
    for strong, vwap_mode, soft15, min_rr, tp, adx in itertools.product(
        [35, 40],
        ["off", "adx40", "adx35"],
        [False, True],
        [0.8, 1.0],
        [1.3, 1.5],
        [17, 18],
    ):
        c = dict(base)
        c["strong_trend_adx"] = strong
        c["soft_15m_bias"] = soft15
        c["min_rr"] = min_rr
        c["tp"] = tp
        c["adx"] = adx
        if vwap_mode == "off":
            c["vwap_required"] = False
            c["vwap_adx_min"] = strong
        elif vwap_mode == "adx40":
            c["vwap_required"] = True
            c["vwap_adx_min"] = 40
        else:
            c["vwap_required"] = True
            c["vwap_adx_min"] = 35
        grid.append(c)

    print(f"Sweeping {len(grid)} D-tuned configs...")
    results: List[Dict[str, Any]] = []
    for i, cfg in enumerate(grid):
        r = run_combo(nq_1m, mnq_1m, cfg)
        entry = {
            "strong_trend_adx": cfg["strong_trend_adx"],
            "vwap_adx_min": cfg["vwap_adx_min"],
            "vwap_required": cfg["vwap_required"],
            "soft_15m_bias": cfg["soft_15m_bias"],
            "min_rr": cfg["min_rr"],
            "tp": cfg["tp"],
            "adx": cfg["adx"],
            "nq_pf": r["nq"]["profit_factor"],
            "nq_pnl": r["nq"]["total_pnl"],
            "nq_trades": r["nq"]["total_trades"],
            "mnq_pf": r["mnq"]["profit_factor"],
            "mnq_pnl": r["mnq"]["total_pnl"],
            "mnq_trades": r["mnq"]["total_trades"],
            "score": r["score"],
        }
        results.append(entry)
        if (i + 1) % 50 == 0:
            viable = [x for x in results if x["score"] > 0]
            print(f"  {i + 1}/{len(grid)} ... viable so far: {len(viable)}")

    viable = [x for x in results if x["score"] > 0]
    viable.sort(key=lambda x: x["score"], reverse=True)
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\nViable (PF>=1 & PnL>0 on BOTH): {len(viable)} / {len(results)}")
    print("\nTOP 12:")
    for r in results[:12]:
        mark = "OK" if r["score"] > 0 else "--"
        print(
            f"{mark} strong={r['strong_trend_adx']} vwap@{r['vwap_adx_min']} "
            f"soft15={r['soft_15m_bias']} vwap_req={r['vwap_required']} "
            f"min_rr={r['min_rr']} tp={r['tp']} adx={r['adx']} | "
            f"NQ PF={r['nq_pf']:.2f} ${r['nq_pnl']:,.0f} | "
            f"MNQ PF={r['mnq_pf']:.2f} ${r['mnq_pnl']:,.0f}"
        )

    best = viable[0] if viable else results[0]
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "d_tuned_sweep_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"baseline_d": base_r, "best": best, "viable_count": len(viable), "results": results[:50]}, f, indent=2)
    print(f"\nSaved: {out_path}")
    if viable:
        print(f"\nWINNER: strong={best['strong_trend_adx']} vwap@{best['vwap_adx_min']} "
              f"vwap_req={best['vwap_required']} soft15={best['soft_15m_bias']} "
              f"min_rr={best['min_rr']} tp={best['tp']} adx={best['adx']}")
    else:
        print("\nWARN: No config profitable on BOTH symbols")


if __name__ == "__main__":
    main()
