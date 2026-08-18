#!/usr/bin/env python3
"""
Backtest MNQ + NQ on shared index data, sweep params until both are profitable.

NQ uses the same OHLC as MNQ (identical index price); only point_value differs.
Also validates on recent Rithmic NQ bars when available.

Usage:
    python scripts/backtest_both_profit.py
    python scripts/backtest_both_profit.py --fetch-nq   # refresh Rithmic NQ first
"""
from __future__ import annotations

import io
import contextlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import scripts.backtest_mtf_scalping as mtf


def load_mnq_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    df_1m = pd.read_csv(os.path.join(data_dir, "MNQ_1m.csv"), parse_dates=["datetime"])
    df_5m = pd.read_csv(os.path.join(data_dir, "MNQ_5m.csv"), parse_dates=["datetime"])
    df_1m = df_1m.sort_values("datetime").reset_index(drop=True)
    df_5m = df_5m.sort_values("datetime").reset_index(drop=True)
    return df_1m, df_5m


def run_one(symbol: str, df_1m: pd.DataFrame, df_5m: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    mtf.TP_MULT = cfg["tp"]
    mtf.ADX_THRESHOLD = cfg["adx"]
    mtf.VOLUME_RATIO_THRESHOLD = cfg["vol"]
    mtf.DI_TOLERANCE = cfg["di_tol"]
    mtf.TP_BUFFER_ATR_MULT = cfg["tp_buffer"]
    mtf.MAX_TRADES_PER_DAY = cfg.get("max_tr", 6)
    bt = mtf.MultiTimeframeBacktester(symbol)
    bt.atr_mult = cfg.get("atr", 1.2)
    with contextlib.redirect_stdout(io.StringIO()):
        stats = bt.run(df_1m, df_5m)
    return {
        "trades": stats["total_trades"],
        "wr": stats["win_rate"],
        "pf": stats["profit_factor"],
        "pnl": stats["total_pnl"],
        "dd": stats["max_drawdown_pct"],
    }


def score_pair(mnq: Dict, nq: Dict) -> float:
    if mnq["pf"] < 1.0 or nq["pf"] < 1.0:
        return -999.0
    if mnq["pnl"] <= 0 or nq["pnl"] <= 0:
        return -999.0
    return min(mnq["pf"], nq["pf"]) * 100 + (mnq["wr"] + nq["wr"]) / 2 + min(mnq["pnl"], nq["pnl"]) / 1000


CONFIGS = [
    {"name": "current", "atr": 1.2, "tp": 1.2, "adx": 16, "vol": 0.4, "di_tol": 3.0, "tp_buffer": 0.5},
    {"name": "tp13_buf05", "atr": 1.2, "tp": 1.3, "adx": 16, "vol": 0.4, "di_tol": 3.0, "tp_buffer": 0.5},
    {"name": "tp14_buf05", "atr": 1.2, "tp": 1.4, "adx": 16, "vol": 0.4, "di_tol": 3.0, "tp_buffer": 0.5},
    {"name": "tp15_buf05", "atr": 1.2, "tp": 1.5, "adx": 16, "vol": 0.4, "di_tol": 3.0, "tp_buffer": 0.5},
    {"name": "tp13_buf08", "atr": 1.2, "tp": 1.3, "adx": 16, "vol": 0.4, "di_tol": 3.0, "tp_buffer": 0.8},
    {"name": "tp14_buf03", "atr": 1.2, "tp": 1.4, "adx": 17, "vol": 0.4, "di_tol": 3.0, "tp_buffer": 0.3},
    {"name": "tp13_adx17", "atr": 1.2, "tp": 1.3, "adx": 17, "vol": 0.4, "di_tol": 3.0, "tp_buffer": 0.5},
    {"name": "tp12_buf08", "atr": 1.2, "tp": 1.2, "adx": 16, "vol": 0.4, "di_tol": 3.0, "tp_buffer": 0.8},
    {"name": "tp15_adx18", "atr": 1.2, "tp": 1.5, "adx": 18, "vol": 0.4, "di_tol": 3.0, "tp_buffer": 0.5},
    {"name": "tp13_vol035", "atr": 1.2, "tp": 1.3, "adx": 16, "vol": 0.35, "di_tol": 3.0, "tp_buffer": 0.5},
]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-nq", action="store_true", help="Refresh NQ from Rithmic before validate")
    args = parser.parse_args()

    df_1m, df_5m = load_mnq_frames()
    print(f"\n{'='*72}")
    print(f"  DUAL BACKTEST — MNQ + NQ (same index OHLC, {len(df_1m):,} × 1m bars)")
    print(f"  Range: {df_1m['datetime'].iloc[0]} → {df_1m['datetime'].iloc[-1]}")
    print(f"{'='*72}\n")

    results: List[Dict] = []
    for cfg in CONFIGS:
        mnq = run_one("MNQ", df_1m, df_5m, cfg)
        nq = run_one("NQ", df_1m, df_5m, cfg)
        row = {**cfg, "mnq": mnq, "nq": nq, "score": score_pair(mnq, nq)}
        results.append(row)
        ok = "✅" if row["score"] > 0 else "❌"
        print(
            f"{ok} {cfg['name']:<14} | MNQ: {mnq['trades']:>3} tr WR={mnq['wr']:>4.0f}% PF={mnq['pf']:.2f} ${mnq['pnl']:>7,.0f}"
            f" | NQ: PF={nq['pf']:.2f} ${nq['pnl']:>8,.0f}",
            flush=True,
        )

    viable = [r for r in results if r["score"] > 0]
    viable.sort(key=lambda x: x["score"], reverse=True)
    if not viable:
        print("\n⚠️  No config profitable on BOTH — keeping best MNQ config")
        best = max(results, key=lambda x: x["mnq"]["pf"])
    else:
        best = viable[0]
        print(f"\n🏆 WINNER: {best['name']} (profitable on MNQ + NQ)")

    mnq_f = best["mnq"]
    nq_f = best["nq"]
    print(
        f"\n   MNQ: {mnq_f['trades']} trades | WR {mnq_f['wr']:.1f}% | PF {mnq_f['pf']:.2f} | PnL ${mnq_f['pnl']:,.0f}"
    )
    print(
        f"   NQ:  {nq_f['trades']} trades | WR {nq_f['wr']:.1f}% | PF {nq_f['pf']:.2f} | PnL ${nq_f['pnl']:,.0f}"
    )

    out_cfg = {
        "name": best["name"],
        "atr": best["atr"],
        "tp": best["tp"],
        "adx": best["adx"],
        "vol": best["vol"],
        "di_tol": best["di_tol"],
        "tp_buffer": best["tp_buffer"],
        "max_tr": 6,
        "session": "all",
        "trades_mnq": mnq_f["trades"],
        "wr_mnq": mnq_f["wr"],
        "pf_mnq": mnq_f["pf"],
        "pnl_mnq": mnq_f["pnl"],
        "trades_nq": nq_f["trades"],
        "wr_nq": nq_f["wr"],
        "pf_nq": nq_f["pf"],
        "pnl_nq": nq_f["pnl"],
        "data_bars_1m": len(df_1m),
        "data_start": str(df_1m["datetime"].iloc[0]),
        "data_end": str(df_1m["datetime"].iloc[-1]),
    }
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    with open(os.path.join(data_dir, "mnq_profit_config.json"), "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2)
    with open(os.path.join(data_dir, "dual_symbol_backtest.json"), "w", encoding="utf-8") as f:
        json.dump({"winner": out_cfg, "all_results": results}, f, indent=2, default=str)

    if args.fetch_nq:
        print("\n--- Rithmic NQ recent validation ---")
        os.system(f'"{sys.executable}" "{os.path.join(os.path.dirname(__file__), "backtest_nq_rithmic.py")}" --symbol NQ')

    print(f"\n📄 Config → data/mnq_profit_config.json")


if __name__ == "__main__":
    main()
