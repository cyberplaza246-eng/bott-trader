#!/usr/bin/env python3
"""More than 1 trade/day on the locked 15m+daily recipe. Real 1m only."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.find_real_1m_edge as edge
import scripts.upgrade_ema15_eod as up


def signals_aligned(f: pd.DataFrame, *, min_sep_bars: int, start_min: int = 9 * 60 + 35):
    """Enter whenever aligned, but not closer than min_sep_bars (1m)."""
    t15 = f["t15"].to_numpy(dtype=int)
    td = f["tdaily"].to_numpy(dtype=int)
    mins = f["_et_min"].to_numpy()
    dates = f["_et_date"].to_numpy()
    long_s = np.zeros(len(f), dtype=bool)
    short_s = np.zeros(len(f), dtype=bool)
    last = None
    last_i = -10_000
    for i in range(len(f)):
        d, m, t, dayt = dates[i], int(mins[i]), int(t15[i]), int(td[i])
        if d != last:
            last, last_i = d, -10_000
        if t == 0 or t != dayt or m < start_min or m >= 15 * 60:
            continue
        if i - last_i < min_sep_bars:
            continue
        if t > 0:
            long_s[i] = True
        else:
            short_s[i] = True
        last_i = i
    return long_s, short_s


def signals_windows(f: pd.DataFrame, windows):
    """One shot per window if aligned at first bar in that window."""
    t15 = f["t15"].to_numpy(dtype=int)
    td = f["tdaily"].to_numpy(dtype=int)
    mins = f["_et_min"].to_numpy()
    dates = f["_et_date"].to_numpy()
    long_s = np.zeros(len(f), dtype=bool)
    short_s = np.zeros(len(f), dtype=bool)
    last = None
    fired = set()
    for i in range(len(f)):
        d, m, t, dayt = dates[i], int(mins[i]), int(t15[i]), int(td[i])
        if d != last:
            last, fired = d, set()
        if t == 0 or t != dayt:
            continue
        for wi, (a, b) in enumerate(windows):
            if wi in fired or not (a <= m < b):
                continue
            if t > 0:
                long_s[i] = True
            else:
                short_s[i] = True
            fired.add(wi)
            break
    return long_s, short_s


def signals_on_15m_close(f: pd.DataFrame):
    """New completed 15m bar (ET minute % 15 == 0) if aligned."""
    t15 = f["t15"].to_numpy(dtype=int)
    td = f["tdaily"].to_numpy(dtype=int)
    mins = f["_et_min"].to_numpy()
    long_s = np.zeros(len(f), dtype=bool)
    short_s = np.zeros(len(f), dtype=bool)
    for i in range(len(f)):
        m, t, dayt = int(mins[i]), int(t15[i]), int(td[i])
        if t == 0 or t != dayt or m < 9 * 60 + 45 or m >= 15 * 60:
            continue
        if m % 15 != 0:
            continue
        if t > 0:
            long_s[i] = True
        else:
            short_s[i] = True
    return long_s, short_s


def run(frame, lo, sh, max_day, name):
    # temporarily patch daily cap
    orig = up.simulate_exits

    def sim(*a, **k):
        trades = orig(*a, **k)
        return trades

    # simulate_exits hardcodes max 1. Use local copy.
    trades = simulate(frame, lo, sh, max_day)
    ins, oos = edge.split_is_oos(trades)
    i, o = edge.summarize(ins), edge.summarize(oos)
    a = edge.summarize(trades)
    days = {}
    for t in trades:
        days.setdefault(str(pd.Timestamp(t.entry_time).date()), 0)
        days[str(pd.Timestamp(t.entry_time).date())] += 1
    per = float(np.mean(list(days.values()))) if days else 0
    print(
        f"{name:<48} {i['trades']:4d} {i['profit_factor']:5.2f} {i['total_pnl']:8.0f} "
        f"{o['trades']:4d} {o['profit_factor']:5.2f} {o['total_pnl']:8.0f} "
        f"{a['total_pnl']:8.0f} {a['max_drawdown']:7.0f} {per:4.2f}/d",
        flush=True,
    )
    return o, i, a


def simulate(frame, long_s, short_s, max_day):
    dt = frame["datetime"].to_numpy()
    hi = frame["high"].to_numpy(dtype=float)
    lo = frame["low"].to_numpy(dtype=float)
    cl = frame["close"].to_numpy(dtype=float)
    atr15 = frame["atr15"].to_numpy(dtype=float)
    rth = edge.rth_mask(dt)
    trades = []
    daily = {}
    i = edge.WARMUP
    n = len(frame)
    while i < n - 1:
        if not rth[i]:
            i += 1
            continue
        ts = pd.Timestamp(dt[i])
        day = str(ts.date())
        if daily.get(day, 0) >= max_day:
            i += 1
            continue
        direction = 1 if long_s[i] else (-1 if short_s[i] else 0)
        if direction == 0:
            i += 1
            continue
        entry = float(cl[i])
        use_sl = 40.0
        if not np.isnan(atr15[i]) and atr15[i] > 0:
            use_sl = float(np.clip(atr15[i] * 2.0, 20, 60))
        sl = entry - use_sl if direction == 1 else entry + use_sl
        tp = entry + 500 if direction == 1 else entry - 500
        entry_ts = ts
        exit_ts, exit_px, reason = pd.Timestamp(dt[-1]), float(cl[-1]), "END"
        for j in range(i + 1, n):
            tss = pd.Timestamp(dt[j])
            if edge.flatten_soon(tss):
                exit_ts, exit_px, reason = tss, float(cl[j]), "EOD"
                break
            if direction == 1:
                if lo[j] <= sl:
                    exit_ts, exit_px, reason = tss, sl, "SL"
                    break
                if hi[j] >= tp:
                    exit_ts, exit_px, reason = tss, tp, "TP"
                    break
            else:
                if hi[j] >= sl:
                    exit_ts, exit_px, reason = tss, sl, "SL"
                    break
                if lo[j] <= tp:
                    exit_ts, exit_px, reason = tss, tp, "TP"
                    break
        pts = (exit_px - entry) if direction == 1 else (entry - exit_px)
        trades.append(
            __import__("scripts.backtest_scalp_momentum", fromlist=["ScalpTrade"]).ScalpTrade(
                entry_time=ts,
                direction="long" if direction == 1 else "short",
                entry_price=entry,
                sl=sl,
                tp=tp,
                exit_time=exit_ts,
                exit_price=exit_px,
                pnl=pts * up.PT - up.COMM,
                exit_reason=reason,
                version="1m",
                hold_seconds=(exit_ts - entry_ts).total_seconds(),
            )
        )
        daily[day] = daily.get(day, 0) + 1
        j = i + 1
        while j < n and pd.Timestamp(dt[j]) <= exit_ts:
            j += 1
        i = max(j, i + 1)
    return trades


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    frame = up.add_features(up.load_frame(csv))
    print("MULTI-ENTRY on locked 15m+daily+ATR2  (real 1m)")
    print(f"{'Rule':<48} ISn IS PF    IS$ OOSn OOS PF   OOS$    ALL$      DD  t/day")
    print("-" * 120)

    lo1, sh1 = up.base_signals(frame)
    td = frame["tdaily"].to_numpy()
    run(frame, lo1 & (td == 1), sh1 & (td == -1), 1, "BASE 1/day first 9:35")

    for sep, cap in ((15, 2), (15, 3), (30, 2), (30, 3), (45, 3), (60, 2), (60, 3), (60, 4), (90, 3)):
        lo, sh = signals_aligned(frame, min_sep_bars=sep)
        run(frame, lo, sh, cap, f"aligned every {sep}m cap{cap}")

    lo, sh = signals_windows(frame, [(9 * 60 + 35, 10 * 60), (13 * 60 + 30, 14 * 60)])
    run(frame, lo, sh, 2, "windows 9:35 + 13:30 cap2")
    lo, sh = signals_windows(frame, [(9 * 60 + 35, 10 * 60), (11 * 60, 11 * 60 + 30), (13 * 60 + 30, 14 * 60)])
    run(frame, lo, sh, 3, "windows 9:35 + 11:00 + 13:30 cap3")

    lo, sh = signals_on_15m_close(frame)
    run(frame, lo, sh, 2, "each 15m close cap2")
    run(frame, lo, sh, 3, "each 15m close cap3")
    run(frame, lo, sh, 4, "each 15m close cap4")


if __name__ == "__main__":
    main()
