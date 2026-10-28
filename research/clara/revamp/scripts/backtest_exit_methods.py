#!/usr/bin/env python3
"""
Research only: same entries, different exits.

Day = official ema15_eod fills. Overnight = break_settled_onh_onl fills.
Does not change live defaults, Lucid orders, or mnq_profit_config.

  python scripts/backtest_exit_methods.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_official as official
import scripts.backtest_overnight_edge as ov
import scripts.backtest_scalp_momentum as scalp_bt
import scripts.find_real_1m_edge as edge
import scripts.test_ema15_tp_modes as tp_modes
from src.strategy.mnq_15m_ema_eod import FLAT_MINUTE, sl_pts_from_atr
from src.strategy.overnight_research import (
    ATR_MULT,
    COMMISSION as OV_COMM,
    SETTLE_BARS,
    SL_MAX as OV_SL_MAX,
    SL_MIN as OV_SL_MIN,
    TOUCH_TICK,
    WARMUP_BARS,
)

ET = pytz.timezone("US/Eastern")
PT, COMM = official.PT, official.COMM
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "exit_methods_backtest.json")

# id, label, how it exits
MODES: List[Tuple[str, str]] = [
    ("live_1r", "Current live: take 1R"),
    ("hold", "Hold to flatten (no TP)"),
    ("2r", "Take 2R"),
    ("resist_15m", "TP at last finished 15m high/low (min 1R)"),
    ("resist_rth", "TP at prior RTH high/low if in front, else 1R"),
    ("trail_after_1r", "After 1R, trail 0.5R; flatten if still open"),
    ("be_after_1r", "After 1R, stop to break-even; flatten if still open"),
    ("run_past_resist", "15m target; if price pushes 2pts past, trail 0.5R"),
]


def prior_rth_levels(dt: pd.DatetimeIndex, hi: np.ndarray, lo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    et = dt.tz_convert(ET)
    mins = (et.hour * 60 + et.minute).to_numpy()
    dates = np.array(et.date)
    rth = (et.weekday.to_numpy() < 5) & (mins >= 9 * 60 + 30) & (mins < 16 * 60)
    day_hi: Dict[Any, float] = {}
    day_lo: Dict[Any, float] = {}
    for i in range(len(dt)):
        if not rth[i]:
            continue
        d = dates[i]
        h, l = float(hi[i]), float(lo[i])
        day_hi[d] = h if d not in day_hi else max(day_hi[d], h)
        day_lo[d] = l if d not in day_lo else min(day_lo[d], l)
    ordered = sorted(day_hi)
    prev: Dict[Any, Tuple[float, float]] = {}
    last = None
    for d in ordered:
        if last is not None:
            prev[d] = (day_hi[last], day_lo[last])
        last = d
    out_hi = np.full(len(dt), np.nan)
    out_lo = np.full(len(dt), np.nan)
    last_hi = last_lo = np.nan
    last_date = None
    for i in range(len(dt)):
        d = dates[i]
        if d != last_date:
            if mins[i] >= 16 * 60 and d in day_hi:
                last_hi, last_lo = day_hi[d], day_lo[d]
            elif d in prev:
                last_hi, last_lo = prev[d]
            last_date = d
        out_hi[i] = last_hi
        out_lo[i] = last_lo
    return out_hi, out_lo


def _tp_pts(mode: str, side: int, sl_pts: float, entry: float, prior_hi: float, prior_lo: float, rth_hi: float, rth_lo: float) -> float:
    if mode in ("hold", "trail_after_1r", "be_after_1r"):
        return 500.0
    if mode == "2r":
        return 2.0 * sl_pts
    if mode in ("resist_15m", "run_past_resist"):
        raw = (prior_hi - entry) if side == 1 else (entry - prior_lo)
        if np.isnan(raw):
            raw = sl_pts
        return max(float(raw), sl_pts)
    if mode == "resist_rth":
        raw = (rth_hi - entry) if side == 1 else (entry - rth_lo)
        if np.isnan(raw) or raw < sl_pts:
            return sl_pts
        return float(raw)
    return sl_pts  # live_1r


def _close(p: Dict, ts, exit_px: float, reason: str, mode: str, comm: float) -> scalp_bt.ScalpTrade:
    pts = (exit_px - p["entry"]) if p["dir"] == 1 else (p["entry"] - exit_px)
    return scalp_bt.ScalpTrade(
        entry_time=p["ts"],
        direction="long" if p["dir"] == 1 else "short",
        entry_price=p["entry"],
        sl=p["sl0"],
        tp=p["tp"],
        exit_time=ts,
        exit_price=exit_px,
        pnl=pts * PT - comm,
        exit_reason=reason,
        version=mode,
        hold_seconds=(ts - p["ts"]).total_seconds(),
    )


def manage_bar(p: Dict, mode: str, high: float, low: float, close: float, flatten: bool) -> Optional[Tuple[str, float]]:
    """Return (reason, exit) or None. Mutates trail/BE state on p."""
    sl_pts = float(p["sl_pts"])
    trail = 0.5 * sl_pts
    one_r = sl_pts
    if p["dir"] == 1:
        favorable = high - p["entry"]
        if mode == "trail_after_1r" or (mode == "run_past_resist" and p.get("trail_on")):
            if favorable >= one_r:
                p["armed"] = True
            if p.get("armed"):
                p["extreme"] = max(float(p.get("extreme") or p["entry"]), high)
                p["sl"] = max(p["sl"], p["extreme"] - trail)
        elif mode == "be_after_1r" and favorable >= one_r:
            p["sl"] = max(p["sl"], p["entry"])
        elif mode == "run_past_resist" and high >= p["tp"] + 2.0:
            p["trail_on"] = True
            p["armed"] = True
            p["extreme"] = max(float(p.get("extreme") or p["entry"]), high)
            p["sl"] = max(p["sl"], p["extreme"] - trail)
        if flatten:
            return "FLAT", close
        if low <= p["sl"]:
            return "SL", p["sl"]
        if mode in ("live_1r", "2r", "resist_15m", "resist_rth") and high >= p["tp"]:
            return "TP", p["tp"]
        if mode == "run_past_resist" and not p.get("trail_on") and high >= p["tp"]:
            return "TP", p["tp"]
        return None
    favorable = p["entry"] - low
    if mode == "trail_after_1r" or (mode == "run_past_resist" and p.get("trail_on")):
        if favorable >= one_r:
            p["armed"] = True
        if p.get("armed"):
            ext = p.get("extreme")
            p["extreme"] = min(float(ext if ext is not None else p["entry"]), low)
            p["sl"] = min(p["sl"], p["extreme"] + trail)
    elif mode == "be_after_1r" and favorable >= one_r:
        p["sl"] = min(p["sl"], p["entry"])
    elif mode == "run_past_resist" and low <= p["tp"] - 2.0:
        p["trail_on"] = True
        p["armed"] = True
        ext = p.get("extreme")
        p["extreme"] = min(float(ext if ext is not None else p["entry"]), low)
        p["sl"] = min(p["sl"], p["extreme"] + trail)
    if flatten:
        return "FLAT", close
    if high >= p["sl"]:
        return "SL", p["sl"]
    if mode in ("live_1r", "2r", "resist_15m", "resist_rth") and low <= p["tp"]:
        return "TP", p["tp"]
    if mode == "run_past_resist" and not p.get("trail_on") and low <= p["tp"]:
        return "TP", p["tp"]
    return None


def should_flatten(mins: int, book: str) -> bool:
    if book == "overnight":
        return (9 * 60 + 25) <= mins < (16 * 60)
    return mins >= FLAT_MINUTE


def replay(df: pd.DataFrame, entries: List[Dict], mode: str, *, book: str, comm: float) -> List:
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    et = dt.tz_convert(ET)
    mins = (et.hour * 60 + et.minute).to_numpy()
    pending = []
    for e in entries:
        tp_pts = _tp_pts(
            mode, e["dir"], e["sl_pts"], e["entry"],
            e.get("prior_hi", np.nan), e.get("prior_lo", np.nan),
            e.get("rth_hi", np.nan), e.get("rth_lo", np.nan),
        )
        pending.append({
            **e,
            "sl0": e["sl"],
            "sl": e["sl"],
            "tp": e["entry"] + tp_pts if e["dir"] == 1 else e["entry"] - tp_pts,
            "armed": False,
            "trail_on": False,
            "extreme": None,
        })
    by_i: Dict[int, List] = {}
    for p in pending:
        by_i.setdefault(p["i"], []).append(p)
    trades = []
    opens: List[Dict] = []
    start = min((e["i"] for e in entries), default=80)
    for i in range(start, len(df)):
        ts = pd.Timestamp(dt[i])
        still = []
        for p in opens:
            hit = manage_bar(p, mode, float(hi[i]), float(lo[i]), float(cl[i]), should_flatten(int(mins[i]), book))
            if hit:
                trades.append(_close(p, ts, hit[1], hit[0], mode, comm))
            else:
                still.append(p)
        opens = still
        opens.extend(by_i.get(i, []))
    for p in opens:
        trades.append(_close(p, pd.Timestamp(dt[-1]), float(cl[-1]), "FLAT", mode, comm))
    return trades


def pack(label: str, mode: str, trades: List) -> Dict[str, Any]:
    ins, oos = edge.split_is_oos(trades)
    si, so = edge.summarize(ins), edge.summarize(oos)
    reasons: Dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    return {
        "id": mode,
        "label": label,
        "is_n": int(si.get("trades") or 0),
        "is_pf": round(float(si.get("profit_factor") or 0), 2),
        "is_pnl": round(float(si.get("total_pnl") or 0), 1),
        "is_wr": round(float(si.get("win_rate") or 0), 1),
        "oos_n": int(so.get("trades") or 0),
        "oos_pf": round(float(so.get("profit_factor") or 0), 2),
        "oos_pnl": round(float(so.get("total_pnl") or 0), 1),
        "oos_wr": round(float(so.get("win_rate") or 0), 1),
        "exits": reasons,
    }


def enrich_day_entries(df: pd.DataFrame, entries: List[Dict]) -> List[Dict]:
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    rth_hi, rth_lo = prior_rth_levels(dt, df["high"].to_numpy(float), df["low"].to_numpy(float))
    for e in entries:
        i = e["i"]
        e["rth_hi"] = float(rth_hi[i]) if not np.isnan(rth_hi[i]) else np.nan
        e["rth_lo"] = float(rth_lo[i]) if not np.isnan(rth_lo[i]) else np.nan
    return entries


def collect_overnight_breaks(frame: Dict[str, Any], df: pd.DataFrame) -> List[Dict]:
    dt = frame["dt"]
    hi, lo, cl = frame["hi"], frame["lo"], frame["cl"]
    atr, mins = frame["atr"], frame["mins"]
    sess, can_enter = frame["sess"], frame["can_enter"]
    prior_hi, prior_lo = tp_modes.completed_15m_hl(df)
    rth_hi, rth_lo = prior_rth_levels(dt, hi, lo)
    entries: List[Dict] = []
    cur = ""
    filled = 0
    bars_in = 0
    onh = onl = np.nan
    onh_i = onl_i = -1
    for i in range(ov.WARMUP, len(dt)):
        s = sess[i]
        if s != cur:
            cur = s
            filled = 0
            bars_in = 0
            onh = onl = np.nan
            onh_i = onl_i = -1
        if not s:
            continue
        if filled == 0 and can_enter[i] and bars_in >= WARMUP_BARS and not np.isnan(onh) and not np.isnan(onl):
            age_hi = bars_in - 1 - onh_i
            age_lo = bars_in - 1 - onl_i
            close = float(cl[i])
            side = 0
            if age_hi >= SETTLE_BARS and close > onh + TOUCH_TICK:
                side = 1
            elif age_lo >= SETTLE_BARS and close < onl - TOUCH_TICK:
                side = -1
            if side != 0:
                a = float(atr[i]) if not np.isnan(atr[i]) else np.nan
                sl_pts = sl_pts_from_atr(a, mult=ATR_MULT, sl_min=OV_SL_MIN, sl_max=OV_SL_MAX, fallback=24.0)
                entry = close
                entries.append({
                    "i": i,
                    "ts": pd.Timestamp(dt[i]),
                    "dir": side,
                    "entry": entry,
                    "sl": entry - sl_pts if side == 1 else entry + sl_pts,
                    "sl_pts": sl_pts,
                    "prior_hi": float(prior_hi[i]),
                    "prior_lo": float(prior_lo[i]),
                    "rth_hi": float(rth_hi[i]) if not np.isnan(rth_hi[i]) else np.nan,
                    "rth_lo": float(rth_lo[i]) if not np.isnan(rth_lo[i]) else np.nan,
                    "level": onh if side == 1 else onl,
                })
                filled = 1
        h, l = float(hi[i]), float(lo[i])
        if np.isnan(onh) or h > onh:
            onh, onh_i = h, bars_in
        if np.isnan(onl) or l < onl:
            onl, onl_i = l, bars_in
        bars_in += 1
    return entries


def run_book(label: str, df: pd.DataFrame, entries: List[Dict], book: str, comm: float) -> List[Dict]:
    print(f"\n=== {label}: {len(entries)} fills ===")
    rows = []
    for mode, name in MODES:
        trades = replay(df, entries, mode, book=book, comm=comm)
        row = pack(name, mode, trades)
        print(
            f"  {name:<48} IS {row['is_pf']:.2f} ${row['is_pnl']:>8.0f} n={row['is_n']:<4}  "
            f"OOS {row['oos_pf']:.2f} ${row['oos_pnl']:>8.0f} n={row['oos_n']}"
        )
        rows.append(row)
    return rows


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "MNQ_1m.csv")
    print("Exit-method research (same entries, exits only)")
    print(f"file={csv}")
    df_day = official.load_1m(csv)
    print(f"day bars={len(df_day):,}  {df_day['datetime'].iloc[0]} -> {df_day['datetime'].iloc[-1]}")
    day_entries = enrich_day_entries(df_day, tp_modes.collect_official_entries(df_day))
    day_rows = run_book("DAY ema15_eod", df_day, day_entries, "day", COMM)

    df_ov, note = ov.load_frame(csv)
    print(f"overnight {note} bars={len(df_ov):,}")
    frame = ov.prepare(df_ov)
    ov_entries = collect_overnight_breaks(frame, df_ov)
    ov_rows = run_book("OVERNIGHT break_settled_onh_onl", df_ov, ov_entries, "overnight", OV_COMM)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv": csv,
        "data_note": note,
        "range": [str(df_day["datetime"].iloc[0]), str(df_day["datetime"].iloc[-1])],
        "oos_start": "2026-06-01",
        "live_note": "Research only. Does not change the live 1R bot.",
        "day": day_rows,
        "overnight": ov_rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT}")
    for book, rows in (("DAY", day_rows), ("OVERNIGHT", ov_rows)):
        best = max(rows, key=lambda r: r["oos_pnl"])
        live = next(r for r in rows if r["id"] == "live_1r")
        print(f"{book} OOS $ winner: {best['label']}  ${best['oos_pnl']:.1f}  vs live 1R ${live['oos_pnl']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
