#!/usr/bin/env python3
"""Fast focused MNQ profit sweep — 24 combos."""
import os, sys, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.optimize_mtf_profit import _run_fast_v2, session_ok
from scripts.backtest_mtf_scalping import load_data, add_indicators_1m, add_indicators_5m

def main():
    df_1m, df_5m = load_data("MNQ")
    df_1m = add_indicators_1m(df_1m)
    df_5m = add_indicators_5m(df_5m)
    grid = list(itertools.product(
        [1.0, 1.2, 1.5],
        [1.5, 2.0, 2.5, 3.0],
        [18, 22],
        [4, 6],
        ["all", "no_midday", "open_only"],
    ))
    print(f"Sweeping {len(grid)} configs...")
    best_pf, best_pnl, results = 0, -999999, []
    for i, (atr, tp, adx, mx, sess) in enumerate(grid):
        r = _run_fast_v2(df_1m, df_5m, atr, tp, adx, mx, sess)
        r.update({"atr_mult": atr, "tp_mult": tp, "adx": adx, "max_trades": mx, "session": sess})
        results.append(r)
        if i % 10 == 0:
            print(f"  {i}/{len(grid)}...")
    results.sort(key=lambda x: (x["pf"], x["pnl"]), reverse=True)
    print("\nTOP 8:")
    for r in results[:8]:
        print(f"  ATR={r['atr_mult']} TP={r['tp_mult']} ADX={r['adx']} max={r['max_trades']} {r['session']:<10} "
              f"| {r['trades']} tr WR={r['wr']:.1f}% PF={r['pf']:.2f} PnL=${r['pnl']:,.0f}")
    prof = [r for r in results if r["pf"] >= 1.05 and r["pnl"] > 0]
    pick = prof[0] if prof else results[0]
    with open("data/mnq_profit_config.json", "w") as f:
        json.dump(pick, f, indent=2)
    print("\nSELECTED:", pick)

if __name__ == "__main__":
    main()
