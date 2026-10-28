#!/usr/bin/env python3
"""Sweep for higher win rate + more trades (closer TP, looser entry gates)."""
import io, contextlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.backtest_mtf_scalping as mtf

CONFIGS = [
    {"name": "current_tp25", "tp": 2.5, "adx": 18, "vol": 0.5, "di_tol": 3.0},
    {"name": "tp15_adx18", "tp": 1.5, "adx": 18, "vol": 0.5, "di_tol": 3.0},
    {"name": "tp15_loose", "tp": 1.5, "adx": 16, "vol": 0.4, "di_tol": 3.0},
    {"name": "tp13_loose", "tp": 1.3, "adx": 16, "vol": 0.4, "di_tol": 3.0},
    {"name": "tp12_loose", "tp": 1.2, "adx": 16, "vol": 0.4, "di_tol": 3.0},
    {"name": "tp15_vol04", "tp": 1.5, "adx": 18, "vol": 0.4, "di_tol": 3.0},
    {"name": "tp14_adx17", "tp": 1.4, "adx": 17, "vol": 0.4, "di_tol": 3.0},
]


def main():
    df_1m, df_5m = mtf.load_data("MNQ")
    results = []
    baseline_trades = 329

    for cfg in CONFIGS:
        mtf.TP_MULT = cfg["tp"]
        mtf.ADX_THRESHOLD = cfg["adx"]
        mtf.VOLUME_RATIO_THRESHOLD = cfg["vol"]
        mtf.DI_TOLERANCE = cfg["di_tol"]
        bt = mtf.MultiTimeframeBacktester("MNQ")
        bt.atr_mult = 1.2
        with contextlib.redirect_stdout(io.StringIO()):
            s = bt.run(df_1m, df_5m)
        row = {**cfg, **{k: s[k] for k in ("total_trades", "win_rate", "profit_factor", "total_pnl", "max_drawdown_pct")}}
        row["trades"] = row.pop("total_trades")
        row["wr"] = row.pop("win_rate")
        row["pf"] = row.pop("profit_factor")
        row["pnl"] = row.pop("total_pnl")
        row["dd"] = row.pop("max_drawdown_pct")
        results.append(row)
        print(
            f"{cfg['name']:<18} | {row['trades']:>3} tr WR={row['wr']:>5.1f}% "
            f"PF={row['pf']:>4.2f} PnL=${row['pnl']:>7,.0f}",
            flush=True,
        )

    viable = [r for r in results if r["pf"] >= 1.0 and r["trades"] >= baseline_trades * 0.95]
    viable.sort(key=lambda x: (x["wr"], x["trades"], x["pf"]), reverse=True)
    best = viable[0] if viable else max(results, key=lambda x: x["wr"])

    out = {
        "name": best["name"], "atr": 1.2, "tp": best["tp"], "adx": best["adx"],
        "vol": best["vol"], "di_tol": best["di_tol"], "max_tr": 6, "session": "all",
        "trades": best["trades"], "wr": best["wr"], "pf": best["pf"],
        "pnl": best["pnl"], "dd": best["dd"],
    }
    os.makedirs("data", exist_ok=True)
    with open("data/mnq_profit_config.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nBEST: {best['name']} — WR={best['wr']:.1f}% trades={best['trades']} PF={best['pf']:.2f}")


if __name__ == "__main__":
    main()
