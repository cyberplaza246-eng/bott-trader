#!/usr/bin/env python3

"""

Hybrid scalp backtest — Pullback (A), Continuation (B), and HYBRID with 30s trigger.



Usage:

    python scripts/backtest_scalp_hybrid.py

    python scripts/backtest_scalp_hybrid.py --sl 8 --tp 15 --max-hold 60

    python scripts/backtest_scalp_hybrid.py --relaxed   # compare aggressive frequency settings

"""

from __future__ import annotations



import argparse

import json

import os

import sys

from dataclasses import dataclass

from typing import Any, Dict, List, Optional



import pandas as pd



sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import scripts.backtest_scalp_momentum as scalp_bt

from src.utils.trading_session import (
    DEFAULT_SCALP_SESSIONS,
    SessionWindow,
    is_in_session_windows_et,
    parse_session_windows,
)
from src.strategy.scalp_hybrid import (

    ADX_MIN_CONTINUATION,

    ADX_MIN_PULLBACK,

    CHASE_BODY_MULT,

    CHASE_EMA_ATR,

    MAX_SETUP_BARS,

    PULLBACK_ATR,

    ScalpHybridState,

    add_30s_body_stats,

    check_hybrid_entry,

    default_rsi_gate_cfg,

)





POINT_VALUE = {"MNQ": 2.0, "NQ": 20.0, "MES": 5.0}

COMMISSION = 0.62

WARMUP = 250

MAX_TRADES_PER_DAY = 20





@dataclass

class HybridVariant:

    name: str

    pullback_enabled: bool

    continuation_enabled: bool





@dataclass

class HybridParams:

    pullback_atr: float = PULLBACK_ATR

    setup_bars: int = MAX_SETUP_BARS

    setup_window_sec: int = 0

    trend_mode: str = "both"

    continuation_volume_strict: bool = True

    cont_volume_min_ratio: float = 0.85

    chase_body_mult: float = CHASE_BODY_MULT

    chase_ema_atr: float = CHASE_EMA_ATR

    momentum_burst_enabled: bool = False

    momentum_burst_adx: int = 25

    adx_min_pullback: int = ADX_MIN_PULLBACK

    adx_min_continuation: int = ADX_MIN_CONTINUATION

    aggressive_mode: bool = False

    rsi_gate_cfg: Optional[Dict[str, Any]] = None



    @classmethod

    def baseline(cls) -> "HybridParams":

        return cls()



    @classmethod

    def relaxed(cls) -> "HybridParams":

        return cls(

            pullback_atr=0.75,

            setup_window_sec=90,

            trend_mode="vwap",

            continuation_volume_strict=False,

            cont_volume_min_ratio=0.85,

            chase_body_mult=2.0,

            chase_ema_atr=1.25,

            momentum_burst_enabled=True,

            momentum_burst_adx=25,

        )



    @classmethod

    def ultra_fast(cls, *, rsi_gate_cfg: Optional[Dict[str, Any]] = None) -> "HybridParams":

        """Backtest winner: more entries, 30s TIME exit, SL6/TP14."""

        return cls(

            pullback_atr=0.85,

            setup_bars=2,

            setup_window_sec=60,

            trend_mode="vwap",

            continuation_volume_strict=False,

            cont_volume_min_ratio=0.85,

            chase_body_mult=2.5,

            chase_ema_atr=1.5,

            momentum_burst_enabled=True,

            momentum_burst_adx=15,

            adx_min_pullback=15,

            adx_min_continuation=18,

            aggressive_mode=True,

            rsi_gate_cfg=rsi_gate_cfg or default_rsi_gate_cfg(),

        )





def attach_context(

    trigger: pd.DataFrame,

    ctx: pd.DataFrame,

    cols: List[str],

    suffix: str,

) -> pd.DataFrame:

    right = ctx[["datetime"] + cols].rename(columns={c: f"{c}{suffix}" for c in cols})

    return pd.merge_asof(

        trigger.sort_values("datetime"),

        right.sort_values("datetime"),

        on="datetime",

        direction="backward",

    ).reset_index(drop=True)





def run_hybrid_variant(

    variant: HybridVariant,

    df_1m: pd.DataFrame,

    df_5m: pd.DataFrame,

    df_30s: pd.DataFrame,

    symbol: str,

    sl_pts: float,

    tp_pts: float,

    max_hold_sec: int,

    params: HybridParams,
    rth_windows: Optional[List[SessionWindow]] = None,

) -> Dict[str, Any]:

    trigger = attach_context(df_30s, df_1m, ["ema_20", "atr", "rsi", "rsi_prev"], "_1m")

    trigger = attach_context(trigger, df_5m, ["close", "vwap", "ema_20", "adx", "volume"], "_5m")



    dt = trigger["datetime"].to_numpy()

    o = trigger["open"].to_numpy(dtype=float)

    hi = trigger["high"].to_numpy(dtype=float)

    lo = trigger["low"].to_numpy(dtype=float)

    cl = trigger["close"].to_numpy(dtype=float)

    body = trigger["body"].to_numpy(dtype=float)

    avg_body = trigger["avg_body"].to_numpy(dtype=float)



    ema20_1m = trigger["ema_20_1m"].to_numpy(dtype=float)

    atr_1m = trigger["atr_1m"].to_numpy(dtype=float)

    rsi_1m = trigger["rsi_1m"].to_numpy(dtype=float)

    rsi_prev_1m = trigger["rsi_prev_1m"].to_numpy(dtype=float)

    close_5m = trigger["close_5m"].to_numpy(dtype=float)

    vwap_5m = trigger["vwap_5m"].to_numpy(dtype=float)

    ema20_5m = trigger["ema_20_5m"].to_numpy(dtype=float)

    adx_5m = trigger["adx_5m"].to_numpy(dtype=float)

    vol_5m = trigger["volume_5m"].to_numpy(dtype=float)



    ph = np_roll(hi)

    pl = np_roll(lo)

    vol_5m_prev = np_roll(vol_5m)



    trades: List[scalp_bt.ScalpTrade] = []

    daily: Dict[str, int] = {}

    pv = POINT_VALUE.get(symbol, 2.0)

    state = ScalpHybridState()



    i = max(WARMUP, 2)

    while i < len(trigger):

        ts = pd.Timestamp(dt[i])

        day_key = str(ts.date())

        if daily.get(day_key, 0) >= MAX_TRADES_PER_DAY:

            i += 1

            continue

        if rth_windows:
            in_win, _ = is_in_session_windows_et(ts, rth_windows)
            if not in_win:
                i += 1
                continue



        row_30s = pd.Series({

            "datetime": ts,

            "open": o[i], "close": cl[i], "high": hi[i], "low": lo[i],

            "body": body[i], "avg_body": avg_body[i],

        })

        prev_30s = pd.Series({

            "datetime": pd.Timestamp(dt[i - 1]),

            "open": o[i - 1], "close": cl[i - 1],

            "high": hi[i - 1], "low": lo[i - 1],

        })

        row_1m = pd.Series({

            "datetime": ts,

            "close": cl[i], "ema_20": ema20_1m[i], "atr": atr_1m[i],

            "rsi": rsi_1m[i], "rsi_prev": rsi_prev_1m[i],

        })

        row_5m = pd.Series({

            "close": close_5m[i], "vwap": vwap_5m[i], "ema_20": ema20_5m[i],

            "adx": adx_5m[i], "volume": vol_5m[i],

        })

        prev_5m = pd.Series({"volume": vol_5m_prev[i]})



        signal, state = check_hybrid_entry(

            symbol, row_1m, row_5m, prev_5m, row_30s, prev_30s, state,

            pullback_enabled=variant.pullback_enabled,

            continuation_enabled=variant.continuation_enabled,

            adx_min_pullback=params.adx_min_pullback,

            adx_min_continuation=params.adx_min_continuation,

            pullback_atr=params.pullback_atr,

            setup_bars=params.setup_bars,

            setup_window_sec=params.setup_window_sec,

            trigger_bar_seconds=30,

            sl_pts=sl_pts,

            tp_pts=tp_pts,

            trend_mode=params.trend_mode,

            continuation_volume_strict=params.continuation_volume_strict,

            cont_volume_min_ratio=params.cont_volume_min_ratio,

            chase_body_mult=params.chase_body_mult,

            chase_ema_atr=params.chase_ema_atr,

            momentum_burst_enabled=params.momentum_burst_enabled,

            momentum_burst_adx=params.momentum_burst_adx,

            aggressive_mode=params.aggressive_mode,

            rsi_gate_cfg=params.rsi_gate_cfg,

        )

        if not signal:

            i += 1

            continue



        direction = 1 if signal["direction"] == "long" else -1

        entry = float(signal["entry"])

        sl = float(signal["sl"])

        tp = float(signal["tp"])

        mode = signal.get("scalp_mode", variant.name)



        exit_ts, exit_px, reason, hold_sec = scalp_bt.simulate_exit(

            dt, hi, lo, cl, i + 1, direction, entry, sl, tp, ts, max_hold_sec,

        )

        pts = (exit_px - entry) if direction == 1 else (entry - exit_px)

        trades.append(

            scalp_bt.ScalpTrade(

                entry_time=ts,

                direction=signal["direction"],

                entry_price=entry,

                sl=sl,

                tp=tp,

                exit_time=exit_ts,

                exit_price=exit_px,

                pnl=pts * pv - COMMISSION * 2,

                exit_reason=reason,

                version=f"{variant.name}:{mode}",

                hold_seconds=hold_sec,

            )

        )

        daily[day_key] = daily.get(day_key, 0) + 1

        j = i + 1

        while j < len(dt) and pd.Timestamp(dt[j]) <= exit_ts:

            j += 1

        i = max(j, i + 1)



    return scalp_bt.summarize(trades, variant.name, symbol)





def np_roll(arr):

    import numpy as np

    out = np.roll(arr, 1)

    out[0] = np.nan

    return out





def print_table(rows: List[Dict[str, Any]], data_note: str, label: str) -> None:

    print(f"\n{'=' * 92}")

    print(f"  HYBRID SCALP BACKTEST — {label}")

    print(f"  {data_note}")

    print(f"{'=' * 92}")

    hdr = f"{'Mode':<14} {'Trades':>6} {'WR%':>6} {'PF':>6} {'Exp$':>8} {'PnL$':>10} {'MaxDD$':>8} {'Hold s':>7}"

    print(hdr)

    print("-" * len(hdr))

    for r in rows:

        print(

            f"{r['version']:<14} {r['trades']:>6} {r['win_rate']:>6.1f} "

            f"{r['profit_factor']:>6.2f} {r['expectancy']:>8.2f} {r['total_pnl']:>10.2f} "

            f"{r['max_drawdown']:>8.2f} {r['avg_hold_sec']:>7.1f}"

        )

    print()





def main() -> None:

    parser = argparse.ArgumentParser(description="Hybrid scalp backtest")

    parser.add_argument("--symbol", default="MNQ")

    parser.add_argument("--csv-1m", default=None)

    parser.add_argument("--csv-5m", default=None)

    parser.add_argument("--csv-30s", default=None)

    parser.add_argument("--sl", type=float, default=8.0)

    parser.add_argument("--tp", type=float, default=15.0)

    parser.add_argument("--max-hold", type=int, default=60)

    parser.add_argument("--relaxed", action="store_true", help="Also run relaxed frequency params")

    parser.add_argument("--ultra-fast", action="store_true", help="Run ultra_fast winner from scalp_fast_sweep")

    parser.add_argument("--compare-rsi", action="store_true", help="Compare ultra_fast with/without RSI gate")
    parser.add_argument(
        "--rth-windows",
        action="store_true",
        help="Only enter in live scalp windows (09:30-12:00 and 13:30-16:00 ET)",
    )
    parser.add_argument(
        "--sessions",
        default="",
        help="Override windows, e.g. morning=09:30-12:00;afternoon=13:30-16:00",
    )

    args = parser.parse_args()

    # ultra_fast winner uses SL6 / TP14 / hold 30 unless user overrides explicitly
    if args.ultra_fast or args.compare_rsi:
        if args.sl == 8.0:
            args.sl = 6.0
        if args.tp == 15.0:
            args.tp = 14.0
        if args.max_hold == 60:
            args.max_hold = 30

    df_1m_raw, df_5m_raw, df_30s_raw, data_note = scalp_bt.load_data(

        args.csv_1m, args.csv_5m, args.csv_30s,

    )

    df_1m = scalp_bt.add_1m_indicators(df_1m_raw)

    df_5m = scalp_bt.add_5m_indicators(df_5m_raw)

    df_30s = add_30s_body_stats(df_30s_raw)



    start, end = df_1m["datetime"].iloc[0], df_1m["datetime"].iloc[-1]
    rth_windows = None
    if args.sessions.strip():
        rth_windows = parse_session_windows(args.sessions)
    elif args.rth_windows:
        rth_windows = list(DEFAULT_SCALP_SESSIONS)

    print(f"\nLoaded {args.symbol}: {len(df_1m):,} x 1m  ({start} -> {end})")

    print(f"SL={args.sl}pt  TP={args.tp}pt  MAX_HOLD={args.max_hold}s")
    if rth_windows:
        sess = "; ".join(f"{w.label} {w.start}-{w.end}" for w in rth_windows)
        print(f"Entry windows (ET): {sess}")
    else:
        print("Entry windows: all hours in the CSV (pass --rth-windows to match live)")



    variant = HybridVariant("hybrid", pullback_enabled=True, continuation_enabled=True)

    param_sets = [("baseline (strict)", HybridParams.baseline())]

    if args.relaxed:

        param_sets.append(("relaxed (live .env)", HybridParams.relaxed()))

    if args.ultra_fast:

        param_sets.append(("ultra_fast (optimized)", HybridParams.ultra_fast()))

    if args.compare_rsi:

        off_cfg = default_rsi_gate_cfg()

        off_cfg["rsi_gate_enabled"] = False

        on_cfg = default_rsi_gate_cfg()

        param_sets.append(("ultra_fast no RSI gate", HybridParams.ultra_fast(rsi_gate_cfg=off_cfg)))

        param_sets.append(("ultra_fast RSI gate ON", HybridParams.ultra_fast(rsi_gate_cfg=on_cfg)))



    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for label, params in param_sets:

        print(f"  Running hybrid [{label}]...", flush=True)

        stats = run_hybrid_variant(

            variant, df_1m, df_5m, df_30s, args.symbol,

            args.sl, args.tp, args.max_hold, params,
            rth_windows=rth_windows,

        )

        stats["params"] = params.__dict__

        all_results[label] = [stats]

        print_table(all_results[label], data_note, label)



    out = os.path.join(

        os.path.dirname(os.path.dirname(__file__)),
        "data",
        f"scalp_hybrid_backtest_{args.symbol.upper()}.json",


    )

    payload = {

        "symbol": args.symbol,

        "data_note": data_note,

        "period": {"start": str(start), "end": str(end), "bars_1m": len(df_1m)},

        "params": {"sl_pts": args.sl, "tp_pts": args.tp, "max_hold_sec": args.max_hold},

        "results": all_results,

    }

    with open(out, "w", encoding="utf-8") as f:

        json.dump(payload, f, indent=2)

    print(f"Saved: {out}")





if __name__ == "__main__":

    main()

