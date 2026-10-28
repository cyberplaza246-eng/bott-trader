#!/usr/bin/env python3
"""Overnight + more windows + 2-lot high-confidence on real Databento 1m."""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_scalp_momentum as scalp_bt
import scripts.find_real_1m_edge as edge
import scripts.upgrade_ema15_eod as up

ET = pytz.timezone("US/Eastern")
PT, COMM = up.PT, up.COMM

RTH = [(9 * 60 + 35, 10 * 60), (11 * 60, 11 * 60 + 30), (13 * 60 + 30, 14 * 60)]
RTH_MORE = [
    (9 * 60 + 35, 10 * 60),
    (10 * 60 + 30, 11 * 60),
    (11 * 60 + 30, 12 * 60),
    (13 * 60 + 30, 14 * 60),
    (14 * 60 + 30, 15 * 60),
]
ON = [
    (18 * 60, 18 * 60 + 30),
    (20 * 60, 20 * 60 + 30),
    (22 * 60, 22 * 60 + 30),
    (2 * 60, 2 * 60 + 30),
    (6 * 60, 6 * 60 + 30),
]


def et_parts(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    et = t.tz_convert(ET)
    return et.hour * 60 + et.minute, et.weekday(), et.date()


def flatten_reason(mins: int, overnight: bool) -> Optional[str]:
    if overnight:
        if 9 * 60 + 20 <= mins < 16 * 60:
            return "ON_FLAT"
        return None
    if mins >= 15 * 60 + 50:
        return "EOD"
    return None


def in_window(mins, windows) -> Optional[int]:
    for i, (a, b) in enumerate(windows):
        if a <= mins < b:
            return i
    return None


def session_key(mins, day, overnight: bool) -> str:
    """Globex session: 18:00+ belongs to next calendar day's 'night' bucket."""
    if overnight and mins >= 18 * 60:
        return f"{day}|ON"
    if overnight and mins < 9 * 60 + 20:
        return f"{day}|ON"
    return f"{day}|RTH"


def simulate(
    frame: pd.DataFrame,
    windows,
    *,
    overnight: bool,
    max_day: int,
    max_open: int,
    conf_2lot: bool,
) -> List:
    dt = frame["datetime"].to_numpy()
    hi = frame["high"].to_numpy(dtype=float)
    lo = frame["low"].to_numpy(dtype=float)
    cl = frame["close"].to_numpy(dtype=float)
    t15 = frame["t15"].to_numpy(dtype=int)
    td = frame["tdaily"].to_numpy(dtype=int)
    t60 = frame["t60"].to_numpy(dtype=int)
    sep = frame["sep15"].to_numpy(dtype=float)
    atr15 = frame["atr15"].to_numpy(dtype=float)
    n = len(frame)
    trades = []
    open_pos = []  # dicts
    daily = {}
    fired = {}

    def sl_pts(i):
        a = atr15[i]
        if np.isnan(a) or a <= 0:
            return 40.0
        return float(np.clip(a * 2.0, 20, 60))

    for i in range(edge.WARMUP, n):
        ts = pd.Timestamp(dt[i])
        mins, dow, day = et_parts(ts)
        if dow >= 5:
            continue
        # manage opens
        still = []
        for p in open_pos:
            fr = flatten_reason(mins, p["overnight"])
            exit_px = reason = None
            if fr:
                exit_px, reason = float(cl[i]), fr
            elif p["dir"] == 1:
                if lo[i] <= p["sl"]:
                    exit_px, reason = p["sl"], "SL"
                elif hi[i] >= p["tp"]:
                    exit_px, reason = p["tp"], "TP"
            else:
                if hi[i] >= p["sl"]:
                    exit_px, reason = p["sl"], "SL"
                elif lo[i] <= p["tp"]:
                    exit_px, reason = p["tp"], "TP"
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
                        version="1m",
                        hold_seconds=(ts - p["ts"]).total_seconds(),
                    )
                )
            else:
                still.append(p)
        open_pos = still

        side = int(t15[i])
        daily_ok = side != 0 and side == int(td[i])
        if not daily_ok:
            continue
        win = in_window(mins, windows)
        if win is None:
            continue
        sk = session_key(mins, day, overnight and win is not None and windows is ON or (overnight and mins < 9 * 60 + 30 or mins >= 18 * 60))
        # simpler session key
        is_on = mins >= 18 * 60 or mins < 9 * 60 + 20
        if overnight and not is_on and windows is ON:
            continue
        if not overnight and is_on:
            continue
        sk = f"{day}|{'ON' if is_on else 'RTH'}"
        if daily.get(sk, 0) >= max_day:
            continue
        fk = (sk, win)
        if fk in fired:
            continue
        high = (int(t60[i]) == side) and (not np.isnan(sep[i]) and sep[i] >= 0.45)
        if conf_2lot:
            cap_open = 2 if high else 1
        else:
            cap_open = max_open
        cap_open = min(cap_open, max_open)
        if len(open_pos) >= cap_open:
            continue
        # second lot only same direction
        if open_pos and open_pos[0]["dir"] != side:
            continue
        use = sl_pts(i)
        entry = float(cl[i])
        sl = entry - use if side == 1 else entry + use
        tp = entry + 500 if side == 1 else entry - 500
        open_pos.append({
            "ts": ts, "dir": side, "entry": entry, "sl": sl, "tp": tp,
            "overnight": is_on,
        })
        fired[fk] = True
        daily[sk] = daily.get(sk, 0) + 1

    return trades


def dump(name, trades):
    ins, oos = edge.split_is_oos(trades)
    i, o, a = edge.summarize(ins), edge.summarize(oos), edge.summarize(trades)
    print(
        f"{name:<46} {i['trades']:4d} {i['profit_factor']:5.2f} {i['total_pnl']:8.0f} "
        f"{o['trades']:4d} {o['profit_factor']:5.2f} {o['total_pnl']:8.0f} "
        f"{a['total_pnl']:8.0f} {a['max_drawdown']:7.0f}",
        flush=True,
    )
    return i, o, a


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    frame = up.add_features(up.load_frame(csv))
    print("MORE TRADES / OVERNIGHT / 2-LOT  (real 1m, completed bars)")
    print(f"{'Rule':<46} ISn IS PF    IS$ OOSn OOS PF   OOS$    ALL$      DD")
    print("-" * 118)
    dump("RTH 3win 1 open (current)", simulate(frame, RTH, overnight=False, max_day=3, max_open=1, conf_2lot=False))
    dump("RTH 5win 1 open", simulate(frame, RTH_MORE, overnight=False, max_day=5, max_open=1, conf_2lot=False))
    dump("RTH 5win 2 open always", simulate(frame, RTH_MORE, overnight=False, max_day=5, max_open=2, conf_2lot=False))
    dump("RTH 3win 2lot if 60m+sep", simulate(frame, RTH, overnight=False, max_day=3, max_open=2, conf_2lot=True))
    dump("RTH 5win 2lot if 60m+sep", simulate(frame, RTH_MORE, overnight=False, max_day=5, max_open=2, conf_2lot=True))
    dump("ON 5win flatten 9:20 1 open", simulate(frame, ON, overnight=True, max_day=5, max_open=1, conf_2lot=False))
    dump("ON 5win 2lot if 60m+sep", simulate(frame, ON, overnight=True, max_day=5, max_open=2, conf_2lot=True))
    # combine: run both and concat
    rth = simulate(frame, RTH, overnight=False, max_day=3, max_open=2, conf_2lot=True)
    on = simulate(frame, ON, overnight=True, max_day=5, max_open=1, conf_2lot=False)
    dump("COMBINE RTH3+2lot + ON5", rth + on)
    rth5 = simulate(frame, RTH_MORE, overnight=False, max_day=5, max_open=2, conf_2lot=True)
    dump("COMBINE RTH5+2lot + ON5", rth5 + on)


if __name__ == "__main__":
    main()
