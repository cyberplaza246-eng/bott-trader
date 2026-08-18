#!/usr/bin/env python3
"""Sweep A/B/C/BC/D on cached Rithmic CSV with live-like rules."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backtest_adaptive_variants import VARIANTS
from scripts.backtest_rithmic_adaptive import load_cached, load_profit_cfg, run_symbol


def main():
    symbols = [s for s in sys.argv[1:] if not s.startswith("-")] or ["NQ", "MNQ"]
    cfg = load_profit_cfg()
    cfg["live_like"] = True
    all_best = {}

    for symbol in symbols:
        try:
            df_1m, _ = load_cached(symbol)
        except FileNotFoundError as e:
            print(f"SKIP {symbol}: {e}")
            continue

        print(f"\n{symbol} Rithmic live-like sweep ({len(df_1m):,} bars)")
        print(f"{df_1m['datetime'].iloc[0]} -> {df_1m['datetime'].iloc[-1]}\n")

        rows = []
        for v in VARIANTS:
            r = run_symbol(symbol, df_1m, cfg, v)
            ok = r.get("profit_factor", 0) >= 1.0 and r.get("total_pnl", 0) > 0
            mark = "OK" if ok else "--"
            print(
                f"{mark} {v.name:3} {v.label:20} | {r['total_trades']:3} tr "
                f"WR={r['win_rate']:4.1f}% PF={r['profit_factor']:.2f} "
                f"PnL=${r['total_pnl']:,.0f} DD={r['max_drawdown_pct']:.1f}%"
            )
            rows.append({"variant": v.name, "label": v.label, **r})

        viable = [x for x in rows if x.get("profit_factor", 0) >= 1.0 and x.get("total_pnl", 0) > 0]
        best = max(viable, key=lambda x: x["profit_factor"]) if viable else max(rows, key=lambda x: x.get("total_pnl", -1e9))
        print(f"\nBest: {best['variant']} ({best['label']}) PF={best['profit_factor']:.2f} PnL=${best['total_pnl']:,.0f}")
        all_best[symbol] = best

        out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f"rithmic_variant_sweep_{symbol}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"symbol": symbol, "results": rows, "best": best}, f, indent=2, default=str)
        print(f"Saved: {out}")

    if all_best:
        print("\n=== SUMMARY ===")
        for sym, b in all_best.items():
            print(f"  {sym}: variant {b['variant']} PF={b['profit_factor']:.2f} PnL=${b['total_pnl']:,.0f}")


if __name__ == "__main__":
    main()
