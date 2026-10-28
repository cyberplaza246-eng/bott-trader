#!/usr/bin/env python3
"""Backtest live overnight paper recipe (roll20 break) with TP / MACD variants.

  python scripts/backtest_overnight_live_recipe.py
  python scripts/backtest_overnight_live_recipe.py --csv-1m data/MNQ_1m.csv

Uses real 1m MNQ. IS < 2026-06-01, OOS >= 2026-06-01. Flatten 09:25 ET.
Does not write live config.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backtest_overnight_edge import (
    OOS_START,
    Spec,
    TOUCH_TICK,
    WARMUP,
    WARMUP_BARS,
    _close_pos,
    _manage,
    _open,
    load_frame,
    pack_row,
    prepare,
)
from src.strategy.overnight_research import (
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    ROLL_BARS,
    TP_R,
    _as_utc_frame,
    _ohlc_resample,
    entry_confidence_pct,
    overnight_macd_ok,
    overnight_min_confidence,
    overnight_stack_gate,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "overnight_live_recipe_backtest.json")


def _macd_15m_on_1m(df: pd.DataFrame) -> np.ndarray:
    """Completed 15m MACD hist sign on 1m index (+1/-1/0)."""
    work = df.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], utc=True)
    tmp = _as_utc_frame(work)
    b15 = _ohlc_resample(tmp, "15min")
    close = b15["close"]
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    hist = (macd - signal).shift(1)
    trend = pd.Series(0, index=b15.index, dtype=int)
    trend.loc[hist > 0] = 1
    trend.loc[hist < 0] = -1
    aligned = trend.reindex(tmp.index, method="ffill").fillna(0).astype(int)
    return aligned.to_numpy(dtype=int)


def _conf_ok(
    side: int,
    close: float,
    level: float,
    t15: int,
    t60: int,
    macd_t: int,
    *,
    min_conf: int,
) -> Tuple[bool, int]:
    side_s = "long" if side == 1 else "short"
    gate = overnight_stack_gate(side_s, close, close, level, t15, t60, macd_t)
    conf = entry_confidence_pct(side_s, gate)
    return int(conf["pct"]) >= int(min_conf), int(conf["pct"])


def simulate_live(
    frame: Dict[str, Any],
    spec: Spec,
    *,
    macd: Optional[np.ndarray] = None,
    require_macd: bool = False,
    min_confidence: int = 0,
    roll_bars: int = ROLL_BARS,
    max_per_night: int = 12,
) -> List[Any]:
    dt = frame["dt"]
    hi = frame["hi"]
    lo = frame["lo"]
    cl = frame["cl"]
    atr = frame["atr"]
    t15a = frame["t15"]
    t60a = frame["t60"]
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
            closed = _manage(pos, float(hi[i]), float(lo[i]), float(cl[i]), ts, bool(flatten[i]))
            if closed:
                trades.append(closed)
                pos = None
            else:
                if s:
                    bars_in += 1
                continue

        if not s or not can_enter[i] or filled >= max_per_night:
            if s:
                bars_in += 1
            continue

        side = 0
        cue = ""
        level = 0.0
        a = float(atr[i]) if not np.isnan(atr[i]) else np.nan
        if bars_in >= WARMUP_BARS:
            win0 = max(sess_i0, i - roll_bars)
            if i > win0:
                rh = float(np.max(hi[win0:i]))
                rl = float(np.min(lo[win0:i]))
                close = float(cl[i])
                if close > rh + TOUCH_TICK:
                    side, cue, level = 1, "break_onh", rh
                elif close < rl - TOUCH_TICK:
                    side, cue, level = -1, "break_onl", rl

        if side != 0:
            macd_t = int(macd[i]) if macd is not None else 0
            t15 = int(t15a[i])
            t60 = int(t60a[i])
            if require_macd and not overnight_macd_ok(
                "long" if side == 1 else "short", macd_t,
            ):
                bars_in += 1
                continue
            if min_confidence > 0:
                ok, _ = _conf_ok(
                    side, float(cl[i]), level, t15, t60, macd_t,
                    min_conf=min_confidence,
                )
                if not ok:
                    bars_in += 1
                    continue
            pos = _open(spec, ts, s, side, float(cl[i]), a, cue)
            filled += 1

        bars_in += 1

    return trades


def _line(label: str, row: Dict[str, Any]) -> str:
    is_s, oos = row["is"], row["oos"]
    return (
        f"{label:48s}  grade={row['grade']:8s}  "
        f"IS ${is_s['total_pnl']:>7.0f} n={is_s['trades']:<4} "
        f"PF={is_s['profit_factor']:<5} WR={is_s['win_rate']:<5}  "
        f"OOS ${oos['total_pnl']:>7.0f} n={oos['trades']:<4} "
        f"PF={oos['profit_factor']:<5} WR={oos['win_rate']:<5}  "
        f"TP={oos['tp_exits']} SL={oos['sl_exits']} FL={oos['flatten_exits']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Overnight live-recipe backtest")
    parser.add_argument("--csv-1m", default=os.path.join(ROOT, "data", "MNQ_1m.csv"))
    parser.add_argument("--max-per-night", type=int, default=12)
    args = parser.parse_args()

    df, note = load_frame(args.csv_1m)
    print(note)
    print("Preparing frame + 15m MACD...")
    frame = prepare(df)
    macd = _macd_15m_on_1m(df)
    min_conf = overnight_min_confidence()
    print(
        f"Live recipe: roll{ROLL_BARS} break · ATR×1.5 stop 12–40 · "
        f"default TP={TP_R}R · min_conf={min_conf} · max/night={args.max_per_night}"
    )
    print(f"IS < {OOS_START.date()} · OOS >= {OOS_START.date()} · flatten 09:25 ET\n")

    cases: List[Tuple[str, float, bool, int]] = [
        ("CURRENT 0.65R MACD+conf65", 0.65, True, min_conf),
        ("0.65R MACD require only", 0.65, True, 0),
        ("0.65R conf65 no MACD", 0.65, False, min_conf),
        ("0.65R bare (no filters)", 0.65, False, 0),
        ("0.8R MACD+conf65", 0.8, True, min_conf),
        ("0.8R MACD require only", 0.8, True, 0),
        ("0.8R bare (no filters)", 0.8, False, 0),
        ("1.0R bare (old paper)", 1.0, False, 0),
    ]

    rows_out: List[Dict[str, Any]] = []
    for label, tp_r, req_macd, conf_min in cases:
        stop = dict(sl_mult=1.5, sl_min=12.0, sl_max=40.0, tp_r=tp_r, tp_pts=0.0)
        spec = Spec(
            name=label.replace(" ", "_"),
            kind="break_roll20",
            max_per_night=args.max_per_night,
            note=label,
            **stop,
        )
        trades = simulate_live(
            frame,
            spec,
            macd=macd if (req_macd or conf_min > 0) else None,
            require_macd=req_macd,
            min_confidence=conf_min,
            max_per_night=args.max_per_night,
        )
        row = pack_row(spec, trades, frame)
        print(_line(label, row))
        rows_out.append({
            "label": label,
            "tp_r": tp_r,
            "require_macd": req_macd,
            "min_confidence": conf_min,
            "grade": row["grade"],
            "is": row["is"],
            "oos": row["oos"],
            "aug": row.get("aug"),
        })

    payload = {
        "csv": args.csv_1m,
        "note": note,
        "tp_default": TP_R,
        "roll_bars": ROLL_BARS,
        "max_per_night": args.max_per_night,
        "cases": rows_out,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
