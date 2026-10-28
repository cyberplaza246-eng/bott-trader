#!/usr/bin/env python3
"""
Official locked-recipe backtest. Same rules as live check_ema15_eod_entry:
  completed 15m EMA8/21, ET RTH daily EMA20, completed 60m EMA8/21
  (every entry: daily+15m+60m same direction),
  windows 9:35/10:15/11:00/12:00/13:30/14:30,
  noon+ slots still need sep>=0.45, ATR×2 stop,
  2nd lot only if 60m+sep high confidence, flatten 15:50.
"""
from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_scalp_momentum as scalp_bt
import scripts.find_real_1m_edge as edge
from src.strategy.mnq_15m_ema_eod import (
    FLAT_MINUTE,
    MAX_TRADES_DAY,
    SEP_MIN,
    atr15_on_1m,
    daily_trend_on_1m,
    high_confidence,
    sep15_on_1m,
    sl_pts_from_atr,
    trend_15m_on_1m,
    trend_60m_on_1m,
    window_index,
    window_requires_quality,
)

ET = pytz.timezone("US/Eastern")
PT, COMM = 2.0, 1.24


def load_1m(csv: str) -> pd.DataFrame:
    raw, _, _, _ = scalp_bt.load_data(csv, None, None, allow_synthetic_30s=True)
    df = scalp_bt.add_1m_indicators(raw)
    return df


def simulate_official(df: pd.DataFrame) -> List:
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    work = df.copy()
    work["datetime"] = dt
    t15 = trend_15m_on_1m(work).to_numpy(dtype=int)
    td = daily_trend_on_1m(work).to_numpy(dtype=int)
    t60 = trend_60m_on_1m(work).to_numpy(dtype=int)
    sep = sep15_on_1m(work).to_numpy(dtype=float)
    atr = atr15_on_1m(work).to_numpy(dtype=float)
    hi = work["high"].to_numpy(dtype=float)
    lo = work["low"].to_numpy(dtype=float)
    cl = work["close"].to_numpy(dtype=float)
    et = dt.tz_convert(ET)
    mins = (et.hour * 60 + et.minute).to_numpy()
    dow = et.weekday.to_numpy()
    dates = et.date

    trades = []
    opens = []
    fired = {}
    counts = {}

    for i in range(80, len(work)):
        ts = pd.Timestamp(dt[i])
        m = int(mins[i])
        day = dates[i]
        still = []
        for p in opens:
            reason = exit_px = None
            if m >= FLAT_MINUTE:
                reason, exit_px = "EOD", float(cl[i])
            elif p["dir"] == 1:
                if lo[i] <= p["sl"]:
                    reason, exit_px = "SL", p["sl"]
                elif hi[i] >= p["tp"]:
                    reason, exit_px = "TP", p["tp"]
            else:
                if hi[i] >= p["sl"]:
                    reason, exit_px = "SL", p["sl"]
                elif lo[i] <= p["tp"]:
                    reason, exit_px = "TP", p["tp"]
            if reason:
                pts = (exit_px - p["entry"]) if p["dir"] == 1 else (p["entry"] - exit_px)
                trades.append(
                    scalp_bt.ScalpTrade(
                        entry_time=p["ts"],
                        direction="long" if p["dir"] == 1 else "short",
                        entry_price=p["entry"],
                        sl=p["sl"],
                        tp=p["tp"],
                        exit_time=ts,
                        exit_price=exit_px,
                        pnl=pts * PT - COMM,
                        exit_reason=reason,
                        version="official",
                        hold_seconds=(ts - p["ts"]).total_seconds(),
                    )
                )
            else:
                still.append(p)
        opens = still

        if dow[i] >= 5:
            continue
        win = window_index(m)
        if win is None:
            continue
        side = int(t15[i])
        if side == 0 or side != int(td[i]):
            continue
        if side != int(t60[i]):
            continue
        key = (day, win)
        conf = high_confidence(side, int(t60[i]), float(sep[i]), SEP_MIN)
        if key in fired or counts.get(day, 0) >= MAX_TRADES_DAY:
            continue
        if window_requires_quality(win) and not conf:
            continue
        if len(opens) >= 2:
            continue
        if len(opens) == 1 and (not conf or opens[0]["dir"] != side):
            continue
        use = sl_pts_from_atr(float(atr[i]))
        entry = float(cl[i])
        sl = entry - use if side == 1 else entry + use
        tp = entry + 500 if side == 1 else entry - 500
        opens.append({"ts": ts, "dir": side, "entry": entry, "sl": sl, "tp": tp})
        fired[key] = True
        counts[day] = counts.get(day, 0) + 1
    return trades


def dump(label, trades):
    s = edge.summarize(trades)
    print(
        f"{label:<16} n={s['trades']:3d}  WR={s['win_rate']:5.1f}  PF={s['profit_factor']:5.2f}  "
        f"${s['total_pnl']:8.1f}  DD=${s['max_drawdown']:7.1f}  exp=${s['expectancy']:6.1f}  "
        f"SL={s['sl_exits']} EOD/TP={s['trades']-s['sl_exits']}"
    )
    return s


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    print("OFFICIAL ema15_eod  (ET RTH daily+15m+60m, 6 windows, noon+ sep, 2-lot if sep)")
    print(f"file={csv}")
    df = load_1m(csv)
    print(f"bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")
    trades = simulate_official(df)
    ins, oos = edge.split_is_oos(trades)
    dump("ALL", trades)
    dump("IS", ins)
    dump("OOS", oos)
    days = {}
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        days[ts.tz_convert(ET).date()] = days.get(ts.tz_convert(ET).date(), 0) + 1
    print(f"fills/day (days with a trade) avg={sum(days.values())/len(days):.2f}  max={max(days.values())}")
    buckets = {}
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        buckets.setdefault(ts.tz_convert(ET).strftime("%Y-%m"), []).append(t)
    print("\nBy month:")
    for k in sorted(buckets):
        dump(k, buckets[k])
    ok = (
        edge.summarize(oos)["trades"] >= 30
        and edge.summarize(oos)["profit_factor"] >= 1.15
        and edge.summarize(oos)["total_pnl"] > 0
        and edge.summarize(ins)["profit_factor"] >= 1.05
        and edge.summarize(ins)["total_pnl"] > 0
    )
    print("\nVERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
