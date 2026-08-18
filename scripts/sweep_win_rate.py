#!/usr/bin/env python3
"""Find MNQ configs with higher win rate while staying profitable."""
import io
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.backtest_mtf_scalping as mtf

CONFIGS = [
    {"name": "current_tp25", "atr": 1.2, "tp": 2.5, "adx": 18, "vol": 0.5, "max_tr": 6},
    {"name": "tp15", "atr": 1.2, "tp": 1.5, "adx": 18, "vol": 0.5, "max_tr": 6},
    {"name": "tp12", "atr": 1.2, "tp": 1.2, "adx": 18, "vol": 0.5, "max_tr": 6},
    {"name": "tp15_adx22", "atr": 1.2, "tp": 1.5, "adx": 22, "vol": 0.5, "max_tr": 6},
    {"name": "tp15_adx22_vol08", "atr": 1.2, "tp": 1.5, "adx": 22, "vol": 0.8, "max_tr": 5},
    {"name": "tp18_adx22", "atr": 1.2, "tp": 1.8, "adx": 22, "vol": 0.5, "max_tr": 6},
    {"name": "tp15_adx20", "atr": 1.2, "tp": 1.5, "adx": 20, "vol": 0.6, "max_tr": 6},
    {"name": "tp15_tight_sl", "atr": 1.0, "tp": 1.5, "adx": 22, "vol": 0.5, "max_tr": 5},
]


def main():
    df_1m, df_5m = mtf.load_data("MNQ")
    results = []
    for cfg in CONFIGS:
        mtf.TP_MULT = cfg["tp"]
        mtf.ADX_THRESHOLD = cfg["adx"]
        mtf.VOLUME_RATIO_THRESHOLD = cfg["vol"]
        mtf.MAX_TRADES_PER_DAY = cfg["max_tr"]
        bt = mtf.MultiTimeframeBacktester("MNQ")
        bt.atr_mult = cfg["atr"]
        with contextlib.redirect_stdout(io.StringIO()):
            s = bt.run(df_1m, df_5m)
        row = {
            **cfg,
            "trades": s["total_trades"],
            "wr": float(s["win_rate"]),
            "pf": float(s["profit_factor"]),
            "pnl": float(s["total_pnl"]),
            "dd": float(s["max_drawdown_pct"]),
        }
        results.append(row)
        print(
            f"{cfg['name']:<22} | {row['trades']:>3} tr "
            f"WR={row['wr']:>5.1f}% PF={row['pf']:>4.2f} "
            f"PnL=${row['pnl']:>7,.0f} DD={row['dd']:.1f}%",
            flush=True,
        )

    profitable = [r for r in results if r["pf"] >= 1.0 and r["pnl"] > 0]
    profitable.sort(key=lambda x: (x["wr"], x["pf"]), reverse=True)
    best = profitable[0] if profitable else max(results, key=lambda x: x["wr"])

    out = {
        "name": best["name"],
        "atr": best["atr"],
        "tp": best["tp"],
        "adx": best["adx"],
        "vol": best["vol"],
        "max_tr": best["max_tr"],
        "session": "all",
        "trades": best["trades"],
        "wr": best["wr"],
        "pf": best["pf"],
        "pnl": best["pnl"],
        "dd": best["dd"],
        "goal": "higher_win_rate",
    }
    os.makedirs("data", exist_ok=True)
    with open("data/mnq_profit_config.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nBEST WR (profitable): {best['name']} — WR={best['wr']:.1f}% PF={best['pf']:.2f} PnL=${best['pnl']:,.0f}")


if __name__ == "__main__":
    main()
