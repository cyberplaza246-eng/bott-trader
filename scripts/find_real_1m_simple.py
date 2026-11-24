#!/usr/bin/env python3
"""
Fewer-trade / hold-to-close search on REAL Databento MNQ 1m.
No synthetic 30s. Reuses the same simulator as find_real_1m_edge.py.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_mtf_scalping as mtf
import scripts.backtest_scalp_momentum as scalp_bt
import scripts.find_real_1m_edge as edge

ET = pytz.timezone("US/Eastern")


def et_mins(ts) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    et = t.tz_convert(ET)
    return et.hour * 60 + et.minute


def first_bar_each_day(frame: pd.DataFrame, start_min: int, end_min: int) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], utc=True)).tz_convert(ET)
    mins = idx.hour * 60 + idx.minute
    dates = idx.date
    out = np.zeros(len(frame), dtype=bool)
    last = None
    for i, (d, m) in enumerate(zip(dates, mins)):
        if d != last and start_min <= m < end_min:
            out[i] = True
            last = d
        elif d != last and m >= end_min:
            last = d
    return out


def signals_rth_long(frame: pd.DataFrame):
    long_s = first_bar_each_day(frame, 9 * 60 + 35, 10 * 60)
    return long_s, np.zeros(len(frame), dtype=bool)


def signals_rth_short(frame: pd.DataFrame):
    short_s = first_bar_each_day(frame, 9 * 60 + 35, 10 * 60)
    return np.zeros(len(frame), dtype=bool), short_s


def signals_15m_trend(frame: pd.DataFrame, long_only: bool = False):
    dt = pd.to_datetime(frame["datetime"], utc=True)
    tmp = frame.copy()
    tmp.index = dt
    b15 = tmp.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    e8 = mtf.calculate_ema(b15["close"], 8)
    e21 = mtf.calculate_ema(b15["close"], 21)
    trend = pd.Series(0, index=b15.index, dtype=int)
    trend.loc[e8 > e21] = 1
    trend.loc[e8 < e21] = -1
    aligned = trend.reindex(dt, method="ffill").fillna(0).to_numpy(dtype=int)
    idx = pd.DatetimeIndex(dt).tz_convert(ET)
    mins = idx.hour * 60 + idx.minute
    dates = idx.date
    long_s = np.zeros(len(frame), dtype=bool)
    short_s = np.zeros(len(frame), dtype=bool)
    last = None
    taken = False
    for i, (d, m, t) in enumerate(zip(dates, mins, aligned)):
        if d != last:
            last, taken = d, False
        if taken or t == 0 or m < 9 * 60 + 35 or m >= 15 * 60:
            continue
        if t > 0:
            long_s[i] = True
            taken = True
        elif t < 0 and not long_only:
            short_s[i] = True
            taken = True
    return long_s, short_s


def signals_orb(frame: pd.DataFrame, or_minutes: int):
    idx = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], utc=True)).tz_convert(ET)
    mins = idx.hour * 60 + idx.minute
    dates = idx.date
    hi = frame["high"].to_numpy(dtype=float)
    lo = frame["low"].to_numpy(dtype=float)
    c = frame["close"].to_numpy(dtype=float)
    or_end = 9 * 60 + 30 + or_minutes
    long_s = np.zeros(len(frame), dtype=bool)
    short_s = np.zeros(len(frame), dtype=bool)
    last = None
    day_hi = day_lo = np.nan
    fired = False
    for i, (d, m) in enumerate(zip(dates, mins)):
        if d != last:
            last, day_hi, day_lo, fired = d, np.nan, np.nan, False
        if m < 9 * 60 + 30 or m >= 16 * 60:
            continue
        if m < or_end:
            day_hi = hi[i] if np.isnan(day_hi) else max(day_hi, hi[i])
            day_lo = lo[i] if np.isnan(day_lo) else min(day_lo, lo[i])
            continue
        if fired or np.isnan(day_hi):
            continue
        if c[i] > day_hi:
            long_s[i] = True
            fired = True
        elif c[i] < day_lo:
            short_s[i] = True
            fired = True
    return long_s, short_s


def signals_vwap_reclaim(frame: pd.DataFrame):
    idx = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], utc=True)).tz_convert(ET)
    mins = idx.hour * 60 + idx.minute
    dates = idx.date
    c = frame["close"].to_numpy(dtype=float)
    h = frame["high"].to_numpy(dtype=float)
    l = frame["low"].to_numpy(dtype=float)
    v = frame["volume"].to_numpy(dtype=float) if "volume" in frame.columns else np.ones(len(frame))
    long_s = np.zeros(len(frame), dtype=bool)
    short_s = np.zeros(len(frame), dtype=bool)
    last = None
    pv = vv = 0.0
    below = above = fired = False
    for i, (d, m) in enumerate(zip(dates, mins)):
        if d != last:
            last, pv, vv, below, above, fired = d, 0.0, 0.0, False, False, False
        if m < 9 * 60 + 30 or m >= 16 * 60:
            continue
        typ = (h[i] + l[i] + c[i]) / 3.0
        vol = max(float(v[i]), 1.0)
        pv += typ * vol
        vv += vol
        vwap = pv / vv
        if c[i] < vwap:
            below = True
        if c[i] > vwap:
            above = True
        if fired or m < 10 * 60:
            continue
        if below and c[i] > vwap:
            long_s[i] = True
            fired = True
        elif above and c[i] < vwap:
            short_s[i] = True
            fired = True
    return long_s, short_s


def signals_open_drive(frame: pd.DataFrame, look: int):
    idx = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], utc=True)).tz_convert(ET)
    mins = idx.hour * 60 + idx.minute
    dates = idx.date
    c = frame["close"].to_numpy(dtype=float)
    long_s = np.zeros(len(frame), dtype=bool)
    short_s = np.zeros(len(frame), dtype=bool)
    last = None
    ups = downs = counted = 0
    prev = np.nan
    fired = False
    for i, (d, m) in enumerate(zip(dates, mins)):
        if d != last:
            last, ups, downs, counted, prev, fired = d, 0, 0, 0, np.nan, False
        if fired or m < 9 * 60 + 30 or m >= 16 * 60:
            continue
        if counted < look:
            if not np.isnan(prev):
                if c[i] > prev:
                    ups += 1
                elif c[i] < prev:
                    downs += 1
            prev = c[i]
            counted += 1
            if counted == look:
                if ups == look - 1:
                    long_s[i] = True
                    fired = True
                elif downs == look - 1:
                    short_s[i] = True
                    fired = True
    return long_s, short_s


def grade(oos, ins):
    if oos["trades"] < 15:
        return "THIN"
    if oos["profit_factor"] >= 1.15 and oos["total_pnl"] > 0 and ins["profit_factor"] >= 1.0:
        return "PASS"
    if oos["profit_factor"] >= 1.0 and oos["total_pnl"] > 0:
        return "WEAK"
    return "FAIL"


def score(name, frame, long_s, short_s, sl, tp):
    trades = edge.run_signals(frame, long_s, short_s, sl, tp, max_trades_day=1)
    ins, oos = edge.split_is_oos(trades)
    is_s = edge.summarize(ins)
    oos_s = edge.summarize(oos)
    g = grade(oos_s, is_s)
    hold_m = (oos_s.get("avg_hold_sec") or 0) / 60.0
    print(
        f"{name:<52} {is_s['trades']:4d} {is_s['profit_factor']:6.2f} {is_s['total_pnl']:9.1f} "
        f"{oos_s['trades']:4d} {oos_s['profit_factor']:6.2f} {oos_s['total_pnl']:9.1f} "
        f"{hold_m:6.1f}  {g}",
        flush=True,
    )
    return {"name": name, "sl": sl, "tp": tp, "is": is_s, "oos": oos_s, "grade": g}


def main() -> None:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    print("REAL 1m simple / 1-trade-day / hold-to-close — Databento MNQ")
    raw_1m, raw_5m, _, _ = scalp_bt.load_data(csv, None, None, allow_synthetic_30s=True)
    df_1m = scalp_bt.add_1m_indicators(raw_1m)
    df_5m = scalp_bt.add_5m_indicators(raw_5m)
    frame = scalp_bt.attach_context(df_1m, df_5m, ["close", "vwap", "ema_20", "adx"], "_5m")
    print(f"bars={len(frame):,}  {frame['datetime'].iloc[0]} -> {frame['datetime'].iloc[-1]}")
    print(f"{'Strategy':<52} ISn  IS PF      IS $ OOSn  OOS PF     OOS $ Hold   Grade")
    print("-" * 112)

    jobs = []
    jobs.append(("RTH long 9:35 EOD sl40/tp500", *signals_rth_long(frame), 40, 500))
    jobs.append(("RTH long 9:35 EOD sl80/tp500", *signals_rth_long(frame), 80, 500))
    jobs.append(("RTH long 9:35 EOD sl25/tp40", *signals_rth_long(frame), 25, 40))
    jobs.append(("RTH short 9:35 EOD sl80/tp500", *signals_rth_short(frame), 80, 500))
    jobs.append(("15m EMA both EOD sl30/tp500", *signals_15m_trend(frame, False), 30, 500))
    jobs.append(("15m EMA both sl30/tp60", *signals_15m_trend(frame, False), 30, 60))
    jobs.append(("15m EMA long-only EOD sl30/tp500", *signals_15m_trend(frame, True), 30, 500))
    jobs.append(("15m EMA long-only sl40/tp80", *signals_15m_trend(frame, True), 40, 80))
    jobs.append(("ORB15 1/day EOD sl25/tp500", *signals_orb(frame, 15), 25, 500))
    jobs.append(("ORB15 1/day sl30/tp60", *signals_orb(frame, 15), 30, 60))
    jobs.append(("ORB30 1/day EOD sl30/tp500", *signals_orb(frame, 30), 30, 500))
    jobs.append(("open drive 5  sl20/tp40", *signals_open_drive(frame, 5), 20, 40))
    jobs.append(("open drive 10 sl25/tp50", *signals_open_drive(frame, 10), 25, 50))
    jobs.append(("VWAP reclaim sl20/tp40", *signals_vwap_reclaim(frame), 20, 40))
    jobs.append(("VWAP reclaim EOD sl25/tp500", *signals_vwap_reclaim(frame), 25, 500))

    rows = [score(name, frame, lo, sh, sl, tp) for name, lo, sh, sl, tp in jobs]
    passed = [r for r in rows if r["grade"] == "PASS"]
    weak = [r for r in rows if r["grade"] == "WEAK"]
    print()
    if passed:
        passed.sort(key=lambda r: r["oos"]["profit_factor"], reverse=True)
        print("PASS:")
        for r in passed:
            print(f"  {r['name']}  OOS PF={r['oos']['profit_factor']}  ${r['oos']['total_pnl']}")
    elif weak:
        weak.sort(key=lambda r: r["oos"]["profit_factor"], reverse=True)
        print("No PASS. WEAK:")
        for r in weak:
            print(f"  {r['name']}  OOS PF={r['oos']['profit_factor']}  ${r['oos']['total_pnl']}")
    else:
        best = max(rows, key=lambda r: (r["oos"]["profit_factor"], r["oos"]["total_pnl"]))
        print(f"None passed. Least-bad: {best['name']} OOS PF={best['oos']['profit_factor']} ${best['oos']['total_pnl']}")

    out = os.path.join("data", "real_1m_simple_search.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
