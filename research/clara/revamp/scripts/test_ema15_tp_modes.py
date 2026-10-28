#!/usr/bin/env python3
"""
Research only: realistic TP vs official hold-to-15:50.

Same entries / ATR stop / 15:50 flatten as scripts/backtest_ema15_official.py.
Does not change live defaults or ENTRY_WINDOWS.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_official as official
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
PT, COMM = official.PT, official.COMM


def completed_15m_hl(df: pd.DataFrame):
    """Last *finished* 15m high/low on the 1m index (no current bucket)."""
    work = df.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], utc=True)
    tmp = work.set_index("datetime")
    b15 = tmp.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    hi = b15["high"].shift(1).reindex(tmp.index, method="ffill")
    lo = b15["low"].shift(1).reindex(tmp.index, method="ffill")
    return hi.to_numpy(dtype=float), lo.to_numpy(dtype=float)


def tp_from_mode(mode: str, side: int, entry: float, sl_pts: float, prior_hi: float, prior_lo: float) -> float:
    if mode == "hold":
        pts = 500.0
    elif mode == "2r":
        pts = 2.0 * sl_pts
    elif mode == "80":
        pts = 80.0
    elif mode == "swing":
        if side == 1:
            raw = prior_hi - entry if not np.isnan(prior_hi) else sl_pts
        else:
            raw = entry - prior_lo if not np.isnan(prior_lo) else sl_pts
        pts = max(float(raw), sl_pts)  # min 1R; if swing is behind, use 1R
    else:
        raise ValueError(mode)
    return entry + pts if side == 1 else entry - pts


def collect_official_entries(df: pd.DataFrame) -> list:
    """Exact official fills (same occupancy / windows). TP is assigned later."""
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    work = df.copy()
    work["datetime"] = dt
    t15 = trend_15m_on_1m(work).to_numpy(dtype=int)
    td = daily_trend_on_1m(work).to_numpy(dtype=int)
    t60 = trend_60m_on_1m(work).to_numpy(dtype=int)
    sep = sep15_on_1m(work).to_numpy(dtype=float)
    atr = atr15_on_1m(work).to_numpy(dtype=float)
    prior_hi, prior_lo = completed_15m_hl(work)
    hi = work["high"].to_numpy(dtype=float)
    lo = work["low"].to_numpy(dtype=float)
    cl = work["close"].to_numpy(dtype=float)
    et = dt.tz_convert(ET)
    mins = (et.hour * 60 + et.minute).to_numpy()
    dow = et.weekday.to_numpy()
    dates = et.date

    entries = []
    opens = []
    fired = {}
    counts = {}

    for i in range(80, len(work)):
        ts = pd.Timestamp(dt[i])
        m = int(mins[i])
        day = dates[i]
        still = []
        for p in opens:
            reason = None
            if m >= FLAT_MINUTE:
                reason = "EOD"
            elif p["dir"] == 1:
                if lo[i] <= p["sl"] or hi[i] >= p["hold_tp"]:
                    reason = "X"
            else:
                if hi[i] >= p["sl"] or lo[i] <= p["hold_tp"]:
                    reason = "X"
            if not reason:
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
        hold_tp = entry + 500 if side == 1 else entry - 500
        rec = {
            "i": i,
            "ts": ts,
            "dir": side,
            "entry": entry,
            "sl": sl,
            "sl_pts": use,
            "prior_hi": float(prior_hi[i]),
            "prior_lo": float(prior_lo[i]),
            "hold_tp": hold_tp,
        }
        entries.append(rec)
        opens.append(rec)
        fired[key] = True
        counts[day] = counts.get(day, 0) + 1
    return entries


def simulate_same_entries(df: pd.DataFrame, entries: list, mode: str) -> List:
    """Replay official fills; only the take-profit changes."""
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    et = dt.tz_convert(ET)
    mins = (et.hour * 60 + et.minute).to_numpy()

    pending = []
    for e in entries:
        pending.append({
            **e,
            "tp": tp_from_mode(mode, e["dir"], e["entry"], e["sl_pts"], e["prior_hi"], e["prior_lo"]),
        })
    by_i = {}
    for p in pending:
        by_i.setdefault(p["i"], []).append(p)

    trades = []
    opens = []
    for i in range(80, len(df)):
        ts = pd.Timestamp(dt[i])
        m = int(mins[i])
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
                        version=mode,
                        hold_seconds=(ts - p["ts"]).total_seconds(),
                    )
                )
            else:
                still.append(p)
        opens = still
        for p in by_i.get(i, []):
            opens.append(p)
    return trades


def verdict(ins, oos) -> bool:
    si, so = edge.summarize(ins), edge.summarize(oos)
    return (
        so["trades"] >= 30
        and so["profit_factor"] >= 1.15
        and so["total_pnl"] > 0
        and si["profit_factor"] >= 1.05
        and si["total_pnl"] > 0
    )


def row(label: str, trades) -> dict:
    ins, oos = edge.split_is_oos(trades)
    si, so = edge.summarize(ins), edge.summarize(oos)
    official.dump(f"{label} ALL", trades)
    official.dump(f"{label} IS", ins)
    official.dump(f"{label} OOS", oos)
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print(f"  exits: {reasons}  PASS={verdict(ins, oos)}")
    return {
        "label": label,
        "is_pf": si["profit_factor"],
        "is_pnl": si["total_pnl"],
        "is_n": si["trades"],
        "oos_pf": so["profit_factor"],
        "oos_pnl": so["total_pnl"],
        "oos_n": so["trades"],
        "pass": verdict(ins, oos),
        "reasons": reasons,
    }


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    print("TP-mode research vs official hold-to-15:50")
    print(f"file={csv}")
    df = official.load_1m(csv)
    print(f"bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")
    modes = [
        ("Current (hold to 3:50)", "hold"),
        ("2R TP", "2r"),
        ("15m swing TP", "swing"),
        ("80pt TP", "80"),
    ]
    print("Collecting official entries (same occupancy as hold-to-15:50)...")
    entries = collect_official_entries(df)
    print(f"official fills={len(entries)}")
    rows = []
    for label, mode in modes:
        print(f"\n=== {label} ===")
        rows.append(row(label, simulate_same_entries(df, entries, mode)))

    print("\nTABLE")
    print(f"{'variant':<24} {'IS PF':>7} {'IS $':>10} {'IS n':>6} {'OOS PF':>7} {'OOS $':>10} {'OOS n':>6}  lock?")
    for r in rows:
        lock = "NO — research only"
        if r["label"].startswith("Current"):
            lock = "YES (current)"
        elif r["pass"] and r["oos_pnl"] > rows[0]["oos_pnl"]:
            lock = "would lock"
        print(
            f"{r['label']:<24} {r['is_pf']:7.2f} {r['is_pnl']:10.1f} {r['is_n']:6d} "
            f"{r['oos_pf']:7.2f} {r['oos_pnl']:10.1f} {r['oos_n']:6d}  {lock}"
        )
    best = max(rows, key=lambda x: x["oos_pnl"])
    print(f"\nOOS $ winner: {best['label']}  ${best['oos_pnl']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
