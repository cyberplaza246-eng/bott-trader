#!/usr/bin/env python3
"""Fast MNQ MTF parameter sweep — indicators computed once."""
import os
import sys
import itertools
import json
from copy import deepcopy
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_mtf_scalping as mtf
from scripts.backtest_mtf_scalping import (
    Trade, INITIAL_BALANCE, MAX_TRADES_PER_DAY, DAILY_LOSS_LIMIT,
    TREND_EMA_SLOW, TP_BUFFER_ATR_MULT, RESISTANCE_LOOKBACK,
    add_indicators_1m, add_indicators_5m, get_5m_context,
    load_data, check_long_entry, check_short_entry,
)


def session_ok(dt, mode: str) -> bool:
    if mode == "all":
        return True
    h, mi = dt.hour, dt.minute
    t = h * 60 + mi
    if mode == "rth":
        return (14 * 60 + 30 <= t < 16 * 60) or (19 * 60 + 30 <= t < 21 * 60)
    if mode == "open_only":
        return 14 * 60 + 30 <= t < 16 * 60
    if mode == "no_midday":
        return not (16 * 60 <= t < 19 * 60)
    return True


def run_fast(
    df_1m, df_5m,
    atr_mult: float,
    tp_mult: float,
    adx_threshold: int,
    max_trades: int,
    session_mode: str,
) -> Dict:
    mtf.ADX_THRESHOLD = adx_threshold
    point_value = 2.0
    balance = INITIAL_BALANCE
    trades: List[Trade] = []
    warmup = max(200, TREND_EMA_SLOW)
    position: Optional[Trade] = None
    daily_trades = 0
    daily_pnl = 0.0
    stopped = False
    current_date = None

    def calc_pnl(t: Trade) -> float:
        if t.direction == "LONG":
            pts = t.exit_price - t.entry_price
        else:
            pts = t.entry_price - t.exit_price
        return pts * point_value - 2.50

    def check_exit(trade: Trade, row) -> Optional[Trade]:
        hi, lo = row["high"], row["low"]
        if trade.direction == "LONG":
            if lo <= trade.sl:
                trade.exit_price, trade.exit_reason = trade.sl, "SL"
                return None
            if hi >= trade.tp:
                trade.exit_price, trade.exit_reason = trade.tp, "TP"
                return None
        else:
            if hi >= trade.sl:
                trade.exit_price, trade.exit_reason = trade.sl, "SL"
                return None
            if lo <= trade.tp:
                trade.exit_price, trade.exit_reason = trade.tp, "TP"
                return None
        return trade

    for i in range(warmup, len(df_1m)):
        row = df_1m.iloc[i]
        dt = row["datetime"]
        td = dt.date() if hasattr(dt, "date") else None
        if td != current_date:
            current_date, daily_trades, daily_pnl, stopped = td, 0, 0.0, False

        if stopped:
            if position:
                position = check_exit(position, row)
                if position is None:
                    pass
            continue

        if position:
            position = check_exit(position, row)
            if position is None:
                pnl = calc_pnl(trades[-1]) if trades else 0
            continue

        if daily_trades >= max_trades or not session_ok(dt, session_mode):
            continue

        ctx = get_5m_context(df_5m, dt)
        atr = row["atr"]
        if pd.isna(atr) or atr <= 0:
            continue

        sl_d = atr * atr_mult
        tp_d = sl_d * tp_mult
        ep = row["close"]
        direction = None

        if check_long_entry(row, ctx):
            direction = "LONG"
            tp_rr = ep + tp_d
            tp_buf = atr * TP_BUFFER_ATR_MULT
            tp_res = ctx["resistance"] - tp_buf
            tp_f = min(tp_rr, tp_res) if tp_res > ep else tp_rr
            sl = ep - sl_d
        elif check_short_entry(row, ctx):
            direction = "SHORT"
            tp_rr = ep - tp_d
            tp_buf = atr * TP_BUFFER_ATR_MULT
            tp_sup = ctx["support"] + tp_buf
            tp_f = max(tp_rr, tp_sup) if tp_sup < ep else tp_rr
            sl = ep + sl_d
        else:
            continue

        position = Trade(
            entry_time=dt, direction=direction, entry_price=ep,
            sl=sl, tp=tp_f, initial_sl=sl,
        )
        daily_trades += 1

    if position:
        last = df_1m.iloc[-1]
        position.exit_time = last["datetime"]
        position.exit_price = last["close"]
        position.exit_reason = "END"
        position.pnl = calc_pnl(position)
        trades.append(position)
        balance += position.pnl

    # Fix: record trades on exit in loop - simplified: recount from closed
    # Re-run simplified with proper recording
    return _run_fast_v2(df_1m, df_5m, atr_mult, tp_mult, adx_threshold, max_trades, session_mode)


def _run_fast_v2(df_1m, df_5m, atr_mult, tp_mult, adx_threshold, max_trades, session_mode):
    import pandas as pd
    mtf.ADX_THRESHOLD = adx_threshold
    pv = 2.0
    balance = INITIAL_BALANCE
    closed = []
    warmup = max(200, TREND_EMA_SLOW)
    pos = None
    daily_trades = 0
    daily_pnl = 0.0
    stopped = False
    cur = None

    for i in range(warmup, len(df_1m)):
        row = df_1m.iloc[i]
        dt = row["datetime"]
        td = dt.date() if hasattr(dt, "date") else None
        if td != cur:
            cur, daily_trades, daily_pnl, stopped = td, 0, 0.0, False

        if pos:
            hi, lo = row["high"], row["low"]
            done = False
            if pos.direction == "LONG":
                if lo <= pos.sl:
                    pos.exit_price, pos.exit_reason = pos.sl, "SL"
                    done = True
                elif hi >= pos.tp:
                    pos.exit_price, pos.exit_reason = pos.tp, "TP"
                    done = True
            else:
                if hi >= pos.sl:
                    pos.exit_price, pos.exit_reason = pos.sl, "SL"
                    done = True
                elif lo <= pos.tp:
                    pos.exit_price, pos.exit_reason = pos.tp, "TP"
                    done = True
            if done:
                pts = (pos.exit_price - pos.entry_price) if pos.direction == "LONG" else (pos.entry_price - pos.exit_price)
                pos.pnl = pts * pv - 2.5
                balance += pos.pnl
                daily_pnl += pos.pnl
                closed.append(pos)
                pos = None
                if daily_pnl <= -DAILY_LOSS_LIMIT:
                    stopped = True
            continue

        if stopped or daily_trades >= max_trades or not session_ok(dt, session_mode):
            continue

        ctx = get_5m_context(df_5m, dt)
        atr = row["atr"]
        if pd.isna(atr) or atr <= 0:
            continue
        sl_d, tp_d = atr * atr_mult, atr * atr_mult * tp_mult
        ep = row["close"]

        if check_long_entry(row, ctx):
            tp_res = ctx["resistance"] - atr * TP_BUFFER_ATR_MULT
            tp_f = min(ep + tp_d, tp_res) if tp_res > ep else ep + tp_d
            pos = Trade(dt, "LONG", ep, ep - sl_d, tp_f, ep - sl_d)
            daily_trades += 1
        elif check_short_entry(row, ctx):
            tp_sup = ctx["support"] + atr * TP_BUFFER_ATR_MULT
            tp_f = max(ep - tp_d, tp_sup) if tp_sup < ep else ep - tp_d
            pos = Trade(dt, "SHORT", ep, ep + sl_d, tp_f, ep + sl_d)
            daily_trades += 1

    if pos:
        last = df_1m.iloc[-1]
        pts = (last["close"] - pos.entry_price) if pos.direction == "LONG" else (pos.entry_price - last["close"])
        pos.pnl = pts * pv - 2.5
        pos.exit_reason = "END"
        closed.append(pos)
        balance += pos.pnl

    if not closed:
        return {"trades": 0, "pf": 0, "pnl": 0, "wr": 0}
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses)) or 1
    return {
        "trades": len(closed),
        "wr": len(wins) / len(closed) * 100,
        "pf": gp / gl,
        "pnl": sum(t.pnl for t in closed),
        "atr_mult": atr_mult, "tp_mult": tp_mult,
        "adx": adx_threshold, "max_trades": max_trades,
        "session": session_mode,
    }


import pandas as pd

def main():
    print("Loading MNQ data...")
    df_1m, df_5m = load_data("MNQ")
    df_1m = add_indicators_1m(df_1m)
    df_5m = add_indicators_5m(df_5m)
    print(f"Loaded {len(df_1m)} 1m bars. Sweeping...")

    results = []
    for atr, tp, adx, mx, sess in itertools.product(
        [1.0, 1.2, 1.5],
        [1.5, 2.0, 2.5, 3.0],
        [18, 22, 25],
        [3, 4, 6],
        ["all", "rth", "open_only", "no_midday"],
    ):
        r = _run_fast_v2(df_1m, df_5m, atr, tp, adx, mx, sess)
        if r["trades"] >= 15 and r["pf"] >= 1.0:
            results.append(r)

    results.sort(key=lambda x: (x["pf"], x["pnl"]), reverse=True)
    print("\nPROFITABLE CONFIGS (PF>=1.0):")
    for r in results[:15]:
        print(
            f"  ATR={r['atr_mult']} TP={r['tp_mult']} ADX={r['adx']} max={r['max_trades']} "
            f"sess={r['session']:<10} | {r['trades']} tr WR={r['wr']:.1f}% PF={r['pf']:.2f} PnL=${r['pnl']:,.0f}"
        )

    all_results = []
    for atr, tp, adx, mx, sess in itertools.product(
        [1.0, 1.2, 1.5], [1.5, 2.0, 2.5, 3.0], [18, 22, 25], [3, 4, 6],
        ["all", "rth", "open_only", "no_midday"],
    ):
        r = _run_fast_v2(df_1m, df_5m, atr, tp, adx, mx, sess)
        if r["trades"] >= 15:
            all_results.append(r)
    all_results.sort(key=lambda x: (x["pf"], x["pnl"]), reverse=True)
    best = all_results[0] if all_results else None
    if best:
        os.makedirs("data", exist_ok=True)
        with open("data/mnq_profit_config.json", "w") as f:
            json.dump(best, f, indent=2)
        print("\nBEST OVERALL:", best)


if __name__ == "__main__":
    main()
