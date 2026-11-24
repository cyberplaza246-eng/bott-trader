#!/usr/bin/env python3
"""Overnight MNQ: VWAP + EMA/SMA 10/20 cross, enter a few bars after the cross."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backtest_overnight_edge import (
    WARMUP,
    Spec,
    _close_pos,
    _manage,
    _open,
    load_frame,
    pack_row,
    prepare,
)
from scripts.backtest_overnight_fill_count import _print

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FADE = dict(sl_mult=1.5, sl_min=12, sl_max=40, tp_r=1.0, tp_pts=0.0)


def _ma(cl: np.ndarray, sess: np.ndarray, span: int, kind: str) -> np.ndarray:
    n = len(cl)
    out = np.full(n, np.nan)
    last = None
    avg = None
    k = 2.0 / (span + 1.0)
    count = 0
    seed = 0.0
    buf: list = []
    for i in range(n):
        s = sess[i]
        if not s:
            last = None
            continue
        if s != last:
            last = s
            avg = None
            count = 0
            seed = 0.0
            buf = []
        x = float(cl[i])
        if kind == "sma":
            buf.append(x)
            if len(buf) > span:
                buf.pop(0)
            if len(buf) == span:
                out[i] = float(sum(buf) / span)
            continue
        if avg is None:
            count += 1
            seed += x
            if count >= span:
                avg = seed / count
                out[i] = avg
        else:
            avg = x * k + avg * (1.0 - k)
            out[i] = avg
    return out


def simulate_cross(
    frame: dict,
    *,
    kind: str,
    delay: int,
    vwap_mode: str,
    max_per_night: int,
) -> list:
    dt, hi, lo, cl = frame["dt"], frame["hi"], frame["lo"], frame["cl"]
    atr, vwap, sess = frame["atr"], frame["vwap"], frame["sess"]
    can_enter, flatten = frame["can_enter"], frame["flatten"]
    n = len(dt)
    fast = _ma(cl, sess, 10, kind)
    slow = _ma(cl, sess, 20, kind)
    spec = Spec(
        f"{kind}10/20 d{delay} {vwap_mode} {max_per_night}/n",
        "cross",
        max_per_night=max_per_night,
        note="",
        **FADE,
    )
    trades = []
    pos = None
    filled = 0
    cur = ""
    pending_side = 0
    pending_i = -1
    for i in range(WARMUP, n):
        s = sess[i]
        ts = dt[i]
        if s != cur:
            if pos is not None:
                trades.append(_close_pos(pos, ts, float(cl[i]), "FLAT", pos["spec"]))
                pos = None
            cur = s
            filled = 0
            pending_side = 0
            pending_i = -1
        if pos is not None:
            closed = _manage(pos, float(hi[i]), float(lo[i]), float(cl[i]), ts, bool(flatten[i]))
            if closed:
                trades.append(closed)
                pos = None
            else:
                continue
        if not s or not can_enter[i] or filled >= max_per_night:
            continue
        f0, s0 = fast[i], slow[i]
        f1, s1 = fast[i - 1], slow[i - 1]
        if s and sess[i - 1] == s and not np.isnan(f0) and not np.isnan(s0) and not np.isnan(f1) and not np.isnan(s1):
            if f1 <= s1 and f0 > s0:
                pending_side, pending_i = 1, i
            elif f1 >= s1 and f0 < s0:
                pending_side, pending_i = -1, i
        if pending_side == 0 or i < pending_i + delay:
            continue
        if np.isnan(f0) or np.isnan(s0):
            pending_side = 0
            continue
        still = (pending_side == 1 and f0 > s0) or (pending_side == -1 and f0 < s0)
        if not still:
            pending_side = 0
            pending_i = -1
            continue
        close = float(cl[i])
        vw = float(vwap[i]) if not np.isnan(vwap[i]) else np.nan
        ok = True
        if vwap_mode == "close":
            ok = not np.isnan(vw) and ((pending_side == 1 and close > vw) or (pending_side == -1 and close < vw))
        elif vwap_mode == "ema_vwap":
            ok = (
                not np.isnan(vw)
                and ((pending_side == 1 and f0 > vw and s0 > vw and close > vw)
                     or (pending_side == -1 and f0 < vw and s0 < vw and close < vw))
            )
        if not ok:
            continue
        a = float(atr[i]) if not np.isnan(atr[i]) else np.nan
        cue = "cross_long" if pending_side == 1 else "cross_short"
        pos = _open(spec, ts, s, pending_side, close, a, cue)
        filled += 1
        pending_side = 0
        pending_i = -1
    return spec, trades


def main() -> int:
    csv_1m = os.path.join(ROOT, "data", "MNQ_1m.csv")
    df, note = load_frame(csv_1m)
    print(note)
    print("Preparing...")
    frame = prepare(df)
    print("=== VWAP + 10/20 cross, wait N 1m candles, overnight 1R ===")
    for kind in ("ema", "sma"):
        for delay in (2, 3, 4, 5):
            for vwap_mode in ("none", "close", "ema_vwap"):
                for mx in (1, 6):
                    spec, trades = simulate_cross(
                        frame, kind=kind, delay=delay, vwap_mode=vwap_mode, max_per_night=mx,
                    )
                    row = pack_row(spec, trades, frame)
                    _print(spec.name, row, trades)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
