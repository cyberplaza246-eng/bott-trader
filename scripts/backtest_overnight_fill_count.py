#!/usr/bin/env python3
"""How many overnight fills still pay? Live-accurate ONH update while in a trade."""
from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace

from scripts.backtest_overnight_edge import (
    OOS_START,
    SETTLE_BARS,
    Spec,
    TOUCH_TICK,
    WARMUP,
    WARMUP_BARS,
    _close_pos,
    _clip_sl,
    _open,
    load_frame,
    pack_row,
    prepare,
)
from src.strategy.overnight_research import ROLL_BARS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def simulate_fills(
    frame: Dict[str, Any],
    spec: Spec,
    *,
    update_range_while_open: bool,
    block_same_cue: bool,
    settle_bars: int = SETTLE_BARS,
    warmup_bars: int = WARMUP_BARS,
) -> List[Any]:
    dt = frame["dt"]
    hi = frame["hi"]
    lo = frame["lo"]
    cl = frame["cl"]
    atr = frame["atr"]
    mins = frame["mins"]
    sess = frame["sess"]
    can_enter = frame["can_enter"]
    flatten = frame["flatten"]
    n = len(dt)
    trades: List[Any] = []
    pos = None
    filled = 0
    used: set = set()
    cur = ""
    bars_in = 0
    onh = onl = np.nan
    onh_i = onl_i = -1

    def bump_range(i: int) -> None:
        nonlocal onh, onl, onh_i, onl_i
        h, l = float(hi[i]), float(lo[i])
        if np.isnan(onh) or h > onh:
            onh, onh_i = h, bars_in
        if np.isnan(onl) or l < onl:
            onl, onl_i = l, bars_in

    for i in range(WARMUP, n):
        s = sess[i]
        ts = dt[i]
        if s != cur:
            if pos is not None:
                trades.append(_close_pos(pos, ts, float(cl[i]), "FLAT", pos["spec"]))
                pos = None
            cur = s
            filled = 0
            used = set()
            bars_in = 0
            onh = onl = np.nan
            onh_i = onl_i = -1

        if pos is not None:
            closed = None
            from scripts.backtest_overnight_edge import _manage
            closed = _manage(pos, float(hi[i]), float(lo[i]), float(cl[i]), ts, bool(flatten[i]))
            if update_range_while_open and s:
                bump_range(i)
                bars_in += 1
            if closed:
                trades.append(closed)
                pos = None
                if not block_same_cue:
                    used.clear()
            else:
                if s and not update_range_while_open:
                    bars_in += 1
                continue

        if not s or not can_enter[i] or filled >= spec.max_per_night:
            if s:
                bump_range(i)
                bars_in += 1
            continue

        side = 0
        cue = spec.kind
        a = float(atr[i]) if not np.isnan(atr[i]) else np.nan
        if bars_in >= warmup_bars and not np.isnan(onh) and not np.isnan(onl):
            age_hi = bars_in - 1 - onh_i
            age_lo = bars_in - 1 - onl_i
            close = float(cl[i])
            if age_hi >= settle_bars and close > onh + TOUCH_TICK and (not block_same_cue or "break_onh" not in used):
                side, cue = 1, "break_onh"
            elif age_lo >= settle_bars and close < onl - TOUCH_TICK and (not block_same_cue or "break_onl" not in used):
                side, cue = -1, "break_onl"

        if side != 0:
            pos = _open(spec, ts, s, side, float(cl[i]), a, cue)
            filled += 1
            used.add(cue)

        bars_in += 1
        bump_range(i)

    return trades


def simulate_roll(
    frame: Dict[str, Any],
    spec: Spec,
    *,
    roll_bars: int = ROLL_BARS,
    warmup_bars: int = WARMUP_BARS,
) -> List[Any]:
    """Last N completed 1m high/low (live break_roll20). Entry on 1m close through the level."""
    dt = frame["dt"]
    hi = frame["hi"]
    lo = frame["lo"]
    cl = frame["cl"]
    atr = frame["atr"]
    sess = frame["sess"]
    can_enter = frame["can_enter"]
    flatten = frame["flatten"]
    n = len(dt)
    trades: List[Any] = []
    pos = None
    filled = 0
    cur = ""
    bars_in = 0
    sess_i0 = WARMUP

    for i in range(WARMUP, n):
        s = sess[i]
        ts = dt[i]
        if s != cur:
            if pos is not None:
                trades.append(_close_pos(pos, ts, float(cl[i]), "FLAT", pos["spec"]))
                pos = None
            cur = s
            filled = 0
            bars_in = 0
            sess_i0 = i

        if pos is not None:
            closed = None
            from scripts.backtest_overnight_edge import _manage
            closed = _manage(pos, float(hi[i]), float(lo[i]), float(cl[i]), ts, bool(flatten[i]))
            if closed:
                trades.append(closed)
                pos = None
            else:
                if s:
                    bars_in += 1
                continue

        if not s or not can_enter[i] or filled >= spec.max_per_night:
            if s:
                bars_in += 1
            continue

        side = 0
        cue = spec.kind
        a = float(atr[i]) if not np.isnan(atr[i]) else np.nan
        if bars_in >= warmup_bars:
            win0 = max(sess_i0, i - roll_bars)
            if i > win0:
                rh = float(np.max(hi[win0:i]))
                rl = float(np.min(lo[win0:i]))
                close = float(cl[i])
                if close > rh + TOUCH_TICK:
                    side, cue = 1, "break_onh"
                elif close < rl - TOUCH_TICK:
                    side, cue = -1, "break_onl"

        if side != 0:
            pos = _open(spec, ts, s, side, float(cl[i]), a, cue)
            filled += 1

        bars_in += 1

    return trades


def _scale_qty(trades: List[Any], qty: int) -> List[Any]:
    if qty == 1:
        return trades
    return [replace(t, pnl=round(float(t.pnl) * qty, 4)) for t in trades]


def _print(label: str, row: Dict[str, Any], trades: List[Any]) -> None:
    nights = Counter(t.session for t in trades)
    per = sorted(nights.values()) if nights else [0]
    fills = [nights[s] for s in nights] or [0]
    six_plus = sum(1 for v in fills if v >= 6)
    is_s, oos_s = row["is"], row["oos"]
    print(
        f"{label:42s}  grade={row['grade']:8s}  "
        f"IS ${is_s['total_pnl']:>8.0f} n={is_s['trades']:<4} PF={is_s['profit_factor']:<5}  "
        f"OOS ${oos_s['total_pnl']:>8.0f} n={oos_s['trades']:<4} PF={oos_s['profit_factor']:<5}  "
        f"avg/night={sum(fills)/max(len(fills),1):.2f}  nights>=6={six_plus}  max/night={max(per)}"
    )


def main() -> int:
    csv_1m = os.path.join(ROOT, "data", "MNQ_1m.csv")
    df, note = load_frame(csv_1m)
    print(note)
    print("Preparing frame...")
    frame = prepare(df)
    fade = dict(sl_mult=1.5, sl_min=12, sl_max=40, tp_r=1.0, tp_pts=0.0)
    print("PASS = OOS PF>=1.20, OOS n>=30, IS n>=40, IS PF>=1.0, OOS$ >0, DD cap")
    print("Session ONH/ONL settle=0, live range while open, 1 MNQ")
    session_cases = [
        ("session 1/night", 1, True, False),
        ("session 2/night re-settle", 2, True, False),
        ("session 3/night re-settle", 3, True, False),
        ("session 4/night re-settle", 4, True, False),
        ("session 5/night re-settle", 5, True, False),
        ("session 6/night re-settle", 6, True, False),
        ("session 6/night block same cue", 6, True, True),
    ]
    passed: List[tuple] = []
    for label, n, live, block in session_cases:
        spec = Spec(f"sess_{n}_{int(live)}_{int(block)}", "break", max_per_night=n, note=label, **fade)
        trades = simulate_fills(frame, spec, update_range_while_open=live, block_same_cue=block)
        row = pack_row(spec, trades, frame)
        _print(label, row, trades)
        if row["grade"] == "PASS":
            passed.append((label, spec, trades, row, 1))

    print()
    print(f"Roll last {ROLL_BARS} completed 1m high/low (live recipe), 1 MNQ")
    for n in (4, 6, 8, 10, 12, 20):
        label = f"roll{ROLL_BARS} {n}/night"
        spec = Spec(f"roll{ROLL_BARS}_{n}", "break", max_per_night=n, note=label, **fade)
        trades = simulate_roll(frame, spec, roll_bars=ROLL_BARS)
        row = pack_row(spec, trades, frame)
        _print(label, row, trades)
        if row["grade"] == "PASS":
            passed.append((label, spec, trades, row, 1))

    print()
    print("2 MNQ qty (same fills; $ and DD * 2, absolute DD cap $4000)")
    scaled = list(passed)
    for label, spec, trades, row, _qty in list(passed):
        t2 = _scale_qty(trades, 2)
        row2 = pack_row(spec, t2, frame)
        lab2 = f"{label} x2 MNQ"
        _print(lab2, row2, t2)
        if row2["grade"] == "PASS":
            scaled.append((lab2, spec, t2, row2, 2))

    print()
    print("PASSED:")
    shown = scaled if scaled else passed
    if not shown:
        print("  none")
    for label, spec, trades, row, qty in shown:
        o = row["oos"]
        print(
            f"  {label}  qty={qty}  OOS ${o['total_pnl']:.0f}  PF={o['profit_factor']}  "
            f"n={o['trades']}  $/night={o['avg_per_night']:.1f}  DD=${o['max_drawdown']:.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
