#!/usr/bin/env python3
"""
Research only: same HTF signals, different 1-minute fills. 1R exit.

  python scripts/backtest_1m_entries.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_official as official
import scripts.backtest_exit_methods as exits
import scripts.backtest_overnight_edge as ov
import scripts.test_ema15_tp_modes as tp_modes
from src.strategy.mnq_15m_ema_eod import ENTRY_WINDOWS, quiet_1m_ok, window_index

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "entry_1m_backtest.json")

METHODS = [
    ("now", "First 1m close (current)"),
    ("next_with", "Next 1m must close with the trade"),
    ("pullback_go", "1m pullback, then next with-trend close"),
    ("ema20_reclaim", "1m touches EMA20 and closes back with trend"),
    ("no_chase", "Skip stretched 1m; wait for a small with-trend bar"),
    ("retest", "Come back to the signal price / break level, then hold"),
    ("live_quiet", "LIVE day path: quiet 1m body <=4"),
]


def _ema(close: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(close).ewm(span=span, adjust=False).mean().to_numpy()


def _window_end(mins0: int) -> int:
    win = window_index(mins0)
    if win is None:
        return mins0 + 10
    return ENTRY_WINDOWS[win][1]


def try_fill(
    method: str,
    side: int,
    sig_i: int,
    hi, lo, op, cl, ema20, mins,
    *,
    book: str,
    level: Optional[float],
    max_i: int,
) -> Optional[Tuple[int, float]]:
    """Return (bar_index, entry_close) or None. side +1 long / -1 short."""
    n = len(cl)
    end_min = 9 * 60 + 20 if book == "overnight" else _window_end(int(mins[sig_i]))
    last = min(n - 1, max_i, sig_i + (20 if book == "overnight" else 25))

    def in_time(i: int) -> bool:
        m = int(mins[i])
        if book == "overnight":
            return not ((9 * 60 + 20) <= m < 16 * 60)
        return m < end_min

    def with_bar(i: int) -> bool:
        return (cl[i] > op[i]) if side == 1 else (cl[i] < op[i])

    if method == "now":
        return sig_i, float(cl[sig_i])

    if method == "next_with":
        j = sig_i + 1
        if j > last or not in_time(j):
            return None
        if with_bar(j):
            return j, float(cl[j])
        return None

    pulled = False
    sig_px = float(cl[sig_i])
    stretch = abs(float(cl[sig_i]) - float(op[sig_i]))
    for i in range(sig_i, last + 1):
        if not in_time(i):
            break
        if method == "pullback_go":
            if side == 1 and lo[i] <= sig_px - 0.5:
                pulled = True
            elif side == -1 and hi[i] >= sig_px + 0.5:
                pulled = True
            if pulled and i > sig_i and with_bar(i):
                return i, float(cl[i])
        elif method == "ema20_reclaim":
            e = float(ema20[i])
            if np.isnan(e):
                continue
            if side == 1 and lo[i] <= e and cl[i] > e and with_bar(i):
                return i, float(cl[i])
            if side == -1 and hi[i] >= e and cl[i] < e and with_bar(i):
                return i, float(cl[i])
        elif method == "no_chase":
            body = abs(float(cl[i]) - float(op[i]))
            if i == sig_i and stretch <= 4.0 and with_bar(i):
                return i, float(cl[i])
            if i > sig_i and body <= 4.0 and with_bar(i):
                return i, float(cl[i])
        elif method == "live_quiet":
            if quiet_1m_ok(side, op[i], cl[i]):
                return i, float(cl[i])
        elif method == "retest":
            lvl = float(level) if level is not None and not np.isnan(level) else sig_px
            if side == 1 and lo[i] <= lvl + 1.0 and cl[i] > lvl and with_bar(i) and i > sig_i:
                return i, float(cl[i])
            if side == -1 and hi[i] >= lvl - 1.0 and cl[i] < lvl and with_bar(i) and i > sig_i:
                return i, float(cl[i])
    return None


def refine_book(signals: List[Dict], arrays: Dict[str, Any], method: str, book: str) -> List[Dict]:
    hi, lo, op, cl = arrays["hi"], arrays["lo"], arrays["op"], arrays["cl"]
    ema20, mins = arrays["ema20"], arrays["mins"]
    out = []
    used = set()
    for sig in signals:
        i0 = int(sig["i"])
        if i0 in used:
            continue
        lvl = sig.get("level")
        hit = try_fill(
            method, int(sig["dir"]), i0, hi, lo, op, cl, ema20, mins,
            book=book, level=lvl, max_i=len(cl) - 1,
        )
        if hit is None:
            continue
        j, px = hit
        if j in used:
            continue
        sl_pts = float(sig["sl_pts"])
        sl = px - sl_pts if sig["dir"] == 1 else px + sl_pts
        rec = dict(sig)
        rec["i"] = j
        rec["entry"] = px
        rec["sl"] = sl
        rec["ts"] = arrays["dt"][j]
        out.append(rec)
        used.add(j)
    return out


def arrays_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    cl = df["close"].to_numpy(dtype=float)
    return {
        "dt": dt,
        "hi": df["high"].to_numpy(dtype=float),
        "lo": df["low"].to_numpy(dtype=float),
        "op": df["open"].to_numpy(dtype=float),
        "cl": cl,
        "ema20": _ema(cl, 20),
        "mins": (dt.tz_convert(exits.ET).hour * 60 + dt.tz_convert(exits.ET).minute).to_numpy(),
    }


def run_book(name: str, df: pd.DataFrame, signals: List[Dict], book: str, comm: float) -> List[Dict]:
    arr = arrays_from_df(df)
    print(f"\n=== {name}: {len(signals)} signals ===")
    rows = []
    for method, label in METHODS:
        fills = refine_book(signals, arr, method, book)
        trades = exits.replay(df, fills, "live_1r", book=book, comm=comm)
        row = exits.pack(label, method, trades)
        row["fills"] = len(fills)
        row["signals"] = len(signals)
        print(
            f"  {label:<44} fills={len(fills):<4} "
            f"IS {row['is_pf']:.2f} ${row['is_pnl']:>8.0f}  "
            f"OOS {row['oos_pf']:.2f} ${row['oos_pnl']:>8.0f} n={row['oos_n']}"
        )
        rows.append(row)
    return rows


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "MNQ_1m.csv")
    print("1-minute entry research (1R exits, HTF signals unchanged)")
    df_day = official.load_1m(csv)
    day_sig = tp_modes.collect_official_entries(df_day)
    day_rows = run_book("DAY ema15", df_day, day_sig, "day", official.COMM)

    df_ov, note = ov.load_frame(csv)
    frame = ov.prepare(df_ov)
    ov_sig = exits.collect_overnight_breaks(frame, df_ov)
    ov_rows = run_book("OVERNIGHT break", df_ov, ov_sig, "overnight", exits.OV_COMM)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv": csv,
        "data_note": note,
        "live_note": "Research only until a method beats current 1m close on test AND train.",
        "day": day_rows,
        "overnight": ov_rows,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT}")
    for book, rows in (("DAY", day_rows), ("OVERNIGHT", ov_rows)):
        live = next(r for r in rows if r["id"] == "now")
        best = max(rows, key=lambda r: (r["oos_pnl"], r["is_pnl"]))
        print(f"{book} OOS $ winner: {best['label']} ${best['oos_pnl']:.1f} vs current ${live['oos_pnl']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
