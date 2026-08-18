#!/usr/bin/env python3
"""Fetch MNQ + NQ from Rithmic, run MTF backtest on both, save results."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from scripts.backtest_nq_rithmic import apply_profit_config, fetch_rithmic_bars, resample_5m, save_csv
import scripts.backtest_mtf_scalping as mtf


def backtest_symbol(symbol: str, bars: int = 20000) -> dict:
    df_1m = fetch_rithmic_bars(symbol, bars, chunked=True)
    df_5m = resample_5m(df_1m)
    save_csv(symbol, df_1m, df_5m)
    apply_profit_config()
    stats = mtf.MultiTimeframeBacktester(symbol).run(df_1m, df_5m)
    mtf.print_results(stats)
    return {
        "symbol": symbol,
        "bars_1m": len(df_1m),
        "start": str(df_1m["datetime"].iloc[0]),
        "end": str(df_1m["datetime"].iloc[-1]),
        "total_trades": stats["total_trades"],
        "win_rate": stats["win_rate"],
        "profit_factor": stats["profit_factor"],
        "total_pnl": stats["total_pnl"],
        "max_drawdown_pct": stats["max_drawdown_pct"],
    }


def main():
    results = {}
    for sym in ("MNQ", "NQ"):
        print(f"\n{'#'*70}\n  RITHMIC BACKTEST — {sym}\n{'#'*70}")
        results[sym] = backtest_symbol(sym, bars=20000)

    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rithmic_dual_backtest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}\n  RITHMIC SUMMARY\n{'='*70}")
    for sym, r in results.items():
        ok = r["profit_factor"] >= 1.0 and r["total_pnl"] > 0
        mark = "✅" if ok else "❌"
        print(
            f"{mark} {sym}: {r['bars_1m']:,} bars | {r['total_trades']} tr | "
            f"WR {r['win_rate']:.1f}% | PF {r['profit_factor']:.2f} | PnL ${r['total_pnl']:,.0f}"
        )
    print(f"\n📄 {out}")


if __name__ == "__main__":
    main()
