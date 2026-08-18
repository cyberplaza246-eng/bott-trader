#!/usr/bin/env python3
"""Validate key MNQ configs with official backtest engine."""
import io, contextlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.backtest_mtf_scalping as mtf

CONFIGS = [
    {"name": "current_baseline", "atr": 1.2, "tp": 1.5, "adx": 18, "max_tr": 6, "session": "all"},
    {"name": "wider_tp", "atr": 1.2, "tp": 2.5, "adx": 18, "max_tr": 6, "session": "all"},
    {"name": "tighter_adx", "atr": 1.2, "tp": 2.0, "adx": 22, "max_tr": 4, "session": "all"},
    {"name": "no_midday", "atr": 1.2, "tp": 1.5, "adx": 18, "max_tr": 6, "session": "no_midday"},
    {"name": "ny_open_only", "atr": 1.0, "tp": 2.5, "adx": 22, "max_tr": 4, "session": "open_only"},
    {"name": "best_rr", "atr": 1.0, "tp": 3.0, "adx": 22, "max_tr": 4, "session": "no_midday"},
]

def session_fn(mode):
    def fn(dt):
        if mode == "all":
            return True
        h, mi = dt.hour, dt.minute
        t = h * 60 + mi
        if mode == "open_only":
            return 14 * 60 + 30 <= t < 16 * 60
        if mode == "no_midday":
            return not (16 * 60 <= t < 19 * 60)
        return True
    return fn

def main():
    df_1m, df_5m = mtf.load_data("MNQ")
    results = []
    for cfg in CONFIGS:
        mtf.TP_MULT = cfg["tp"]
        mtf.ADX_THRESHOLD = cfg["adx"]
        mtf.MAX_TRADES_PER_DAY = cfg["max_tr"]
        mtf.is_trading_session = session_fn(cfg["session"])
        bt = mtf.MultiTimeframeBacktester("MNQ")
        bt.atr_mult = cfg["atr"]
        with contextlib.redirect_stdout(io.StringIO()):
            s = bt.run(df_1m, df_5m)
        row = {**cfg, "trades": s["total_trades"], "wr": s["win_rate"],
               "pf": s["profit_factor"], "pnl": s["total_pnl"], "dd": s["max_drawdown_pct"]}
        results.append(row)
        print(f"{cfg['name']:<16} | {row['trades']:>3} tr WR={row['wr']:>5.1f}% PF={row['pf']:>4.2f} PnL=${row['pnl']:>7,.0f} DD={row['dd']:.1f}%", flush=True)

    results.sort(key=lambda x: (x["pf"], x["pnl"]), reverse=True)
    best = results[0]
    os.makedirs("data", exist_ok=True)
    with open("data/mnq_profit_config.json", "w") as f:
        json.dump(best, f, indent=2)
    print("\nWINNER:", best["name"], best)

if __name__ == "__main__":
    main()
