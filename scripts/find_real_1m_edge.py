#!/usr/bin/env python3
"""
Find a 1-minute MNQ strategy that works on REAL Databento bars (no synthetic 30s).

In-sample: before 2026-06-01
Out-of-sample: 2026-06-01 onward (same window that broke hybrid 30s)

Usage:
    python scripts/find_real_1m_edge.py
    python scripts/find_real_1m_edge.py --csv-1m data/MNQ_1m.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_mtf_scalping as mtf
import scripts.backtest_scalp_momentum as scalp_bt
from src.ai.mnq_context import compute_vwap

ET = pytz.timezone("US/Eastern")
POINT_VALUE = 2.0
COMMISSION = 1.24  # round trip
MAX_TRADES_DAY = 8
WARMUP = 80
OOS_START = pd.Timestamp("2026-06-01", tz="UTC")
MIN_OOS_TRADES = 40
PF_PASS = 1.20


@dataclass
class Spec:
    name: str
    kind: str
    sl_pts: float
    tp_pts: float
    adx_min: float
    extra: float = 0.0


def rth_mask(dt: np.ndarray) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(dt, utc=True)).tz_convert(ET)
    mins = idx.hour * 60 + idx.minute
    return (idx.weekday < 5) & (mins >= 9 * 60 + 30) & (mins < 16 * 60)


def flatten_soon(ts: pd.Timestamp) -> bool:
    et = pd.Timestamp(ts).tz_convert(ET) if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts, tz="UTC").tz_convert(ET)
    return (et.hour * 60 + et.minute) >= (15 * 60 + 50)


def summarize(trades: List[scalp_bt.ScalpTrade]) -> Dict[str, Any]:
    return scalp_bt.summarize(trades, "x", "MNQ")


def split_is_oos(trades: List[scalp_bt.ScalpTrade]) -> Tuple[List, List]:
    ins, oos = [], []
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        (oos if ts >= OOS_START else ins).append(t)
    return ins, oos


def simulate(
    dt, hi, lo, cl, start, direction, entry, sl, tp,
) -> Tuple[pd.Timestamp, float, str, float]:
    entry_ts = pd.Timestamp(dt[start - 1])
    for j in range(start, len(dt)):
        ts = pd.Timestamp(dt[j])
        if flatten_soon(ts):
            return ts, float(cl[j]), "EOD", (ts - entry_ts).total_seconds()
        if direction == 1:
            if lo[j] <= sl:
                return ts, sl, "SL", (ts - entry_ts).total_seconds()
            if hi[j] >= tp:
                return ts, tp, "TP", (ts - entry_ts).total_seconds()
        else:
            if hi[j] >= sl:
                return ts, sl, "SL", (ts - entry_ts).total_seconds()
            if lo[j] <= tp:
                return ts, tp, "TP", (ts - entry_ts).total_seconds()
    ts = pd.Timestamp(dt[-1])
    return ts, float(cl[-1]), "END", (ts - entry_ts).total_seconds()


def run_signals(
    frame: pd.DataFrame,
    long_sig: np.ndarray,
    short_sig: np.ndarray,
    sl_pts: float,
    tp_pts: float,
    max_trades_day: int = MAX_TRADES_DAY,
) -> List[scalp_bt.ScalpTrade]:
    dt = frame["datetime"].to_numpy()
    hi = frame["high"].to_numpy(dtype=float)
    lo = frame["low"].to_numpy(dtype=float)
    cl = frame["close"].to_numpy(dtype=float)
    rth = rth_mask(dt)
    trades: List[scalp_bt.ScalpTrade] = []
    daily: Dict[str, int] = {}
    i = WARMUP
    n = len(frame)
    while i < n - 1:
        if not rth[i]:
            i += 1
            continue
        ts = pd.Timestamp(dt[i])
        day = str(ts.date())
        if daily.get(day, 0) >= max_trades_day:
            i += 1
            continue
        direction = 0
        if long_sig[i]:
            direction = 1
        elif short_sig[i]:
            direction = -1
        if direction == 0:
            i += 1
            continue
        entry = float(cl[i])
        sl = entry - sl_pts if direction == 1 else entry + sl_pts
        tp = entry + tp_pts if direction == 1 else entry - tp_pts
        exit_ts, exit_px, reason, hold = simulate(
            dt, hi, lo, cl, i + 1, direction, entry, sl, tp,
        )
        pts = (exit_px - entry) if direction == 1 else (entry - exit_px)
        trades.append(
            scalp_bt.ScalpTrade(
                entry_time=ts,
                direction="long" if direction == 1 else "short",
                entry_price=entry,
                sl=sl,
                tp=tp,
                exit_time=exit_ts,
                exit_price=exit_px,
                pnl=pts * POINT_VALUE - COMMISSION,
                exit_reason=reason,
                version="1m",
                hold_seconds=hold,
            )
        )
        daily[day] = daily.get(day, 0) + 1
        j = i + 1
        while j < n and pd.Timestamp(dt[j]) <= exit_ts:
            j += 1
        i = max(j, i + 1)
    return trades


def signals_pullback(f: pd.DataFrame, spec: Spec) -> Tuple[np.ndarray, np.ndarray]:
    adx = f["adx_5m"].to_numpy(dtype=float)
    vwap = f["vwap_5m"].to_numpy(dtype=float)
    c5 = f["close_5m"].to_numpy(dtype=float)
    c = f["close"].to_numpy(dtype=float)
    o = f["open"].to_numpy(dtype=float)
    ema = f["ema_20"].to_numpy(dtype=float)
    atr = f["atr"].to_numpy(dtype=float)
    zone = atr * spec.extra
    trend_up = (adx >= spec.adx_min) & (c5 > vwap)
    trend_dn = (adx >= spec.adx_min) & (c5 < vwap)
    in_long = (c <= ema + zone) & (c >= ema - zone * 1.2)
    in_short = (c >= ema - zone) & (c <= ema + zone * 1.2)
    long_s = trend_up & in_long & (c > o) & (c > ema)
    short_s = trend_dn & in_short & (c < o) & (c < ema)
    return long_s, short_s


def signals_ema_cross(f: pd.DataFrame, spec: Spec) -> Tuple[np.ndarray, np.ndarray]:
    adx = f["adx_5m"].to_numpy(dtype=float)
    vwap = f["vwap_5m"].to_numpy(dtype=float)
    c5 = f["close_5m"].to_numpy(dtype=float)
    e9 = f["ema_9"].to_numpy(dtype=float)
    e21 = f["ema_21"].to_numpy(dtype=float)
    e9p = np.roll(e9, 1)
    e21p = np.roll(e21, 1)
    e9p[0] = np.nan
    cross_up = (e9p <= e21p) & (e9 > e21)
    cross_dn = (e9p >= e21p) & (e9 < e21)
    long_s = (adx >= spec.adx_min) & (c5 > vwap) & cross_up
    short_s = (adx >= spec.adx_min) & (c5 < vwap) & cross_dn
    return long_s, short_s


def signals_breakout(f: pd.DataFrame, spec: Spec) -> Tuple[np.ndarray, np.ndarray]:
    n = int(spec.extra)
    adx = f["adx_5m"].to_numpy(dtype=float)
    vwap = f["vwap_5m"].to_numpy(dtype=float)
    c5 = f["close_5m"].to_numpy(dtype=float)
    c = f["close"].to_numpy(dtype=float)
    o = f["open"].to_numpy(dtype=float)
    hi = f["high"].to_numpy(dtype=float)
    lo = f["low"].to_numpy(dtype=float)
    prior_hi = pd.Series(hi).shift(1).rolling(n).max().to_numpy()
    prior_lo = pd.Series(lo).shift(1).rolling(n).min().to_numpy()
    long_s = (adx >= spec.adx_min) & (c5 > vwap) & (c > o) & (c > prior_hi)
    short_s = (adx >= spec.adx_min) & (c5 < vwap) & (c < o) & (c < prior_lo)
    return long_s, short_s


def signals_rsi_fade(f: pd.DataFrame, spec: Spec) -> Tuple[np.ndarray, np.ndarray]:
    """Fade only in quiet tape (low ADX)."""
    adx = f["adx_5m"].to_numpy(dtype=float)
    rsi = f["rsi"].to_numpy(dtype=float)
    c = f["close"].to_numpy(dtype=float)
    o = f["open"].to_numpy(dtype=float)
    vwap = f["vwap_5m"].to_numpy(dtype=float)
    long_s = (adx <= spec.adx_min) & (rsi <= spec.extra) & (c > o) & (c < vwap)
    short_s = (adx <= spec.adx_min) & (rsi >= (100 - spec.extra)) & (c < o) & (c > vwap)
    return long_s, short_s


def signals_orb(f: pd.DataFrame, spec: Spec) -> Tuple[np.ndarray, np.ndarray]:
    """Break the 09:30-10:00 ET opening range after 10:00, one shot style via daily cap."""
    idx = pd.DatetimeIndex(pd.to_datetime(f["datetime"], utc=True)).tz_convert(ET)
    mins = idx.hour * 60 + idx.minute
    dates = idx.date
    hi = f["high"].to_numpy(dtype=float)
    lo = f["low"].to_numpy(dtype=float)
    c = f["close"].to_numpy(dtype=float)
    o = f["open"].to_numpy(dtype=float)
    orb_hi = np.full(len(f), np.nan)
    orb_lo = np.full(len(f), np.nan)
    last_d = None
    day_hi = day_lo = np.nan
    for i, (d, m) in enumerate(zip(dates, mins)):
        if d != last_d:
            last_d = d
            day_hi = day_lo = np.nan
        if (9 * 60 + 30) <= m < (10 * 60):
            day_hi = hi[i] if np.isnan(day_hi) else max(day_hi, hi[i])
            day_lo = lo[i] if np.isnan(day_lo) else min(day_lo, lo[i])
        orb_hi[i] = day_hi
        orb_lo[i] = day_lo
    after = mins >= 10 * 60
    long_s = after & ~np.isnan(orb_hi) & (c > o) & (c > orb_hi)
    short_s = after & ~np.isnan(orb_lo) & (c < o) & (c < orb_lo)
    return long_s, short_s


HANDLERS: Dict[str, Callable] = {
    "pullback": signals_pullback,
    "ema_cross": signals_ema_cross,
    "breakout": signals_breakout,
    "rsi_fade": signals_rsi_fade,
    "orb": signals_orb,
}


def build_grid() -> List[Spec]:
    specs: List[Spec] = []
    for sl, tp, adx, pb in (
        (8, 12, 18, 0.6),
        (10, 15, 18, 0.75),
        (12, 18, 20, 0.75),
        (15, 22, 20, 1.0),
        (10, 20, 22, 0.75),
        (8, 16, 25, 0.5),
    ):
        specs.append(Spec(f"pullback sl{sl}/tp{tp} adx{adx} pb{pb}", "pullback", sl, tp, adx, pb))
    for sl, tp, adx in (
        (10, 15, 18),
        (12, 18, 20),
        (15, 22, 22),
        (10, 20, 25),
    ):
        specs.append(Spec(f"ema_cross sl{sl}/tp{tp} adx{adx}", "ema_cross", sl, tp, adx))
    for sl, tp, adx, n in (
        (10, 15, 18, 10),
        (12, 18, 20, 15),
        (15, 22, 20, 20),
        (12, 24, 25, 20),
    ):
        specs.append(Spec(f"breakout{n} sl{sl}/tp{tp} adx{adx}", "breakout", sl, tp, adx, float(n)))
    for sl, tp, adx_max, rsi in (
        (10, 10, 18, 28),
        (12, 12, 20, 30),
        (15, 15, 22, 25),
    ):
        specs.append(Spec(f"rsi_fade sl{sl} adx<={adx_max} rsi{rsi}", "rsi_fade", sl, tp, adx_max, rsi))
    for sl, tp in ((12, 18), (15, 22), (20, 30)):
        specs.append(Spec(f"orb sl{sl}/tp{tp}", "orb", sl, tp, 0.0))
    return specs


def grade(oos: Dict[str, Any], ins: Dict[str, Any]) -> str:
    if oos["trades"] < MIN_OOS_TRADES:
        return "THIN"
    if oos["profit_factor"] >= PF_PASS and oos["total_pnl"] > 0 and ins["profit_factor"] >= 1.05:
        return "PASS"
    if oos["profit_factor"] >= 1.0 and oos["total_pnl"] > 0:
        return "WEAK"
    return "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-1m", default=os.path.join("data", "MNQ_1m.csv"))
    args = parser.parse_args()

    print("REAL 1m search — Databento MNQ, no synthetic 30s")
    print(f"File: {args.csv_1m}")
    raw_1m, raw_5m, _, _ = scalp_bt.load_data(
        args.csv_1m, None, None, allow_synthetic_30s=True,
    )
    # load_data still builds unused 30s if allowed; we ignore it
    df_1m = scalp_bt.add_1m_indicators(raw_1m)
    df_1m["ema_9"] = mtf.calculate_ema(df_1m["close"], 9)
    df_1m["ema_21"] = mtf.calculate_ema(df_1m["close"], 21)
    df_5m = scalp_bt.add_5m_indicators(raw_5m)
    frame = scalp_bt.attach_context(df_1m, df_5m, ["close", "vwap", "ema_20", "adx"], "_5m")
    print(
        f"1m bars: {len(frame):,}  {frame['datetime'].iloc[0]} -> {frame['datetime'].iloc[-1]}"
    )
    print(f"IS < {OOS_START.date()}  |  OOS >= {OOS_START.date()}")
    print(
        f"{'Strategy':<42} {'IS n':>5} {'IS PF':>6} {'IS $':>9} "
        f"{'OOS n':>5} {'OOS PF':>7} {'OOS $':>9} {'Hold m':>6}  Grade"
    )
    print("-" * 110)

    rows = []
    for spec in build_grid():
        fn = HANDLERS[spec.kind]
        long_s, short_s = fn(frame, spec)
        trades = run_signals(frame, long_s, short_s, spec.sl_pts, spec.tp_pts)
        ins, oos = split_is_oos(trades)
        is_s = summarize(ins)
        oos_s = summarize(oos)
        g = grade(oos_s, is_s)
        hold_m = (oos_s["avg_hold_sec"] or 0) / 60.0
        print(
            f"{spec.name:<42} {is_s['trades']:>5} {is_s['profit_factor']:>6.2f} {is_s['total_pnl']:>9.1f} "
            f"{oos_s['trades']:>5} {oos_s['profit_factor']:>7.2f} {oos_s['total_pnl']:>9.1f} "
            f"{hold_m:>6.1f}  {g}",
            flush=True,
        )
        rows.append({
            "name": spec.name,
            "kind": spec.kind,
            "sl": spec.sl_pts,
            "tp": spec.tp_pts,
            "adx": spec.adx_min,
            "extra": spec.extra,
            "is": is_s,
            "oos": oos_s,
            "grade": g,
        })

    winners = [r for r in rows if r["grade"] == "PASS"]
    weak = [r for r in rows if r["grade"] == "WEAK"]
    winners.sort(key=lambda r: r["oos"]["profit_factor"], reverse=True)
    print()
    if winners:
        best = winners[0]
        print(f"BEST PASS: {best['name']}  OOS PF={best['oos']['profit_factor']}  ${best['oos']['total_pnl']}")
    elif weak:
        weak.sort(key=lambda r: r["oos"]["profit_factor"], reverse=True)
        best = weak[0]
        print(f"No PASS. Best WEAK: {best['name']}  OOS PF={best['oos']['profit_factor']}")
    else:
        best = max(rows, key=lambda r: (r["oos"]["profit_factor"], r["oos"]["total_pnl"]))
        print("No profitable OOS strategy in this grid.")
        print(f"Least-bad: {best['name']}  OOS PF={best['oos']['profit_factor']}  ${best['oos']['total_pnl']}")

    out = os.path.join("data", "real_1m_edge_search.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"oos_start": str(OOS_START), "rows": rows, "best": best}, f, indent=2, default=str)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
