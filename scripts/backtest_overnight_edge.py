#!/usr/bin/env python3
"""
Overnight Globex MNQ search on REAL 1m bars.

Not the locked RTH ema15_eod recipe. Same 15m windows on Globex already
failed IS (PF ~0.97). This file only backtests overnight hypotheses.

  python scripts/backtest_overnight_edge.py
  python scripts/backtest_overnight_edge.py --csv-1m data/MNQ_1m.csv

IS: entry < 2026-06-01 UTC. OOS: entry >= 2026-06-01 (includes August 2026
when present). Flatten 09:25 ET. Skip RTH 09:30-16:00. 1 MNQ, fees in.
No synthetic 30s. Does not start paper/live and does not write live config.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_scalp_momentum as scalp_bt
from src.strategy.mnq_15m_ema_eod import (
    atr15_on_1m,
    daily_trend_on_1m,
    sl_pts_from_atr,
    trend_15m_on_1m,
    trend_60m_on_1m,
)
from src.strategy.overnight_research import (
    ATR_MULT as FADE_ATR_MULT,
    COMMISSION,
    FLATTEN_MINUTE,
    GLOBEX_OPEN_MINUTE,
    NO_ENTRY_MINUTE,
    POINT_VALUE,
    SETTLE_BARS,
    SL_MAX as FADE_SL_MAX,
    SL_MIN as FADE_SL_MIN,
    TOUCH_TICK,
    TP_R as FADE_TP_R,
    WARMUP_BARS,
)

ET = pytz.timezone("US/Eastern")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OOS_START = pd.Timestamp("2026-06-01", tz="UTC")
AUG_START = pd.Timestamp("2026-08-01", tz="UTC")
WARMUP = 80
PF_PASS = 1.20
MIN_OOS_TRADES = 30
MIN_IS_TRADES = 40
DD_CAP_USD = 4000.0
TP_CAP = 500.0
QTY = 1

# ET windows. Asia ~ Tokyo morning, London cash open, US premarket, Globex reopen.
WIN_ASIA = ((20 * 60, 21 * 60),)
WIN_LONDON = ((3 * 60, 4 * 60),)
WIN_PRE = ((8 * 60, 9 * 60),)
WIN_GLOBEX = ((18 * 60, 19 * 60),)
WIN_EMA15_ON = (
    (18 * 60, 18 * 60 + 30),
    (20 * 60, 20 * 60 + 30),
    (22 * 60, 22 * 60 + 30),
    (2 * 60, 2 * 60 + 30),
    (6 * 60, 6 * 60 + 30),
)
WIN_FIRST_THREE = WIN_ASIA + WIN_LONDON + WIN_PRE


@dataclass
class Spec:
    name: str
    kind: str
    windows: Tuple[Tuple[int, int], ...] = ()
    max_per_night: int = 1
    require_daily: bool = False
    require_60m: bool = False
    use_atr_stop: bool = True
    sl_mult: float = 2.0
    sl_min: float = 20.0
    sl_max: float = 60.0
    tp_r: float = 0.0
    tp_pts: float = TP_CAP
    side: int = 1  # BH only: +1 long / -1 short
    note: str = ""


@dataclass
class OnTrade:
    entry_time: str
    exit_time: str
    direction: str
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    pnl: float
    pts: float
    exit_reason: str
    spec: str
    session: str
    hold_seconds: float
    cue: str = ""


def in_window(mins: int, windows: Sequence[Tuple[int, int]]) -> Optional[int]:
    for i, (a, b) in enumerate(windows):
        if a <= mins < b:
            return i
    return None


RTH_OPEN_MINUTE = 9 * 60 + 30


def globex_session_id(et_date: date, mins: int, dow: int) -> str:
    """Morning-date key for a Globex overnight, else empty.

    Sun-Thu 18:00 starts the next weekday morning. Mon-Fri through the
    09:25 flatten bar (before 09:30 RTH) continues that night.
    Friday 18:00 and Saturday are closed.
    """
    if mins >= GLOBEX_OPEN_MINUTE and dow in (6, 0, 1, 2, 3):
        return str(et_date + timedelta(days=1))
    if mins < RTH_OPEN_MINUTE and dow in (0, 1, 2, 3, 4):
        return str(et_date)
    return ""


def allow_entry(mins: int, dow: int, session: str) -> bool:
    if not session:
        return False
    if dow < 5 and NO_ENTRY_MINUTE <= mins < GLOBEX_OPEN_MINUTE:
        return False
    return True


def should_flatten(mins: int, dow: int) -> bool:
    if dow >= 5:
        return False
    return FLATTEN_MINUTE <= mins < GLOBEX_OPEN_MINUTE


def build_session_arrays(dt: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    et = dt.tz_convert(ET)
    mins = (et.hour * 60 + et.minute).to_numpy(dtype=int)
    dow = et.weekday.to_numpy(dtype=int)
    dates = et.date
    n = len(dt)
    sess = np.empty(n, dtype=object)
    can_enter = np.zeros(n, dtype=bool)
    flatten = np.zeros(n, dtype=bool)
    for i in range(n):
        s = globex_session_id(dates[i], int(mins[i]), int(dow[i]))
        sess[i] = s
        can_enter[i] = allow_entry(int(mins[i]), int(dow[i]), s)
        flatten[i] = should_flatten(int(mins[i]), int(dow[i]))
    return mins, dow, sess, can_enter, flatten


def overnight_vwap(hi, lo, cl, vol, sess) -> np.ndarray:
    n = len(cl)
    out = np.full(n, np.nan)
    last = None
    pv = vv = 0.0
    for i in range(n):
        s = sess[i]
        if not s:
            continue
        if s != last:
            last, pv, vv = s, 0.0, 0.0
        typ = (float(hi[i]) + float(lo[i]) + float(cl[i])) / 3.0
        v = max(float(vol[i]), 1.0)
        pv += typ * v
        vv += v
        out[i] = pv / vv
    return out


def _pnl(side: int, entry: float, exit_px: float) -> Tuple[float, float]:
    pts = (exit_px - entry) * side
    usd = pts * POINT_VALUE * QTY - COMMISSION
    return round(pts, 4), round(usd, 4)


def _clip_sl(atr: float, spec: Spec) -> float:
    return float(sl_pts_from_atr(
        atr, mult=spec.sl_mult, sl_min=spec.sl_min, sl_max=spec.sl_max,
        fallback=max(spec.sl_min, 24.0),
    ))


def _summarize(trades: List[OnTrade], nights: int, version: str) -> Dict[str, Any]:
    if not trades:
        return {
            "version": version,
            "trades": 0,
            "nights": int(nights),
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "avg_per_night": 0.0,
            "avg_hold_sec": 0.0,
            "tp_exits": 0,
            "sl_exits": 0,
            "flatten_exits": 0,
        }
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    equity = peak = max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    net = float(sum(pnls))
    return {
        "version": version,
        "trades": len(trades),
        "nights": int(nights),
        "win_rate": round(100.0 * len(wins) / len(trades), 1),
        "profit_factor": round(min(pf, 999.0), 2),
        "expectancy": round(float(np.mean(pnls)), 2),
        "total_pnl": round(net, 2),
        "max_drawdown": round(max_dd, 2),
        "avg_per_night": round(net / nights, 2) if nights else 0.0,
        "avg_hold_sec": round(float(np.mean([t.hold_seconds for t in trades])), 1),
        "tp_exits": sum(1 for t in trades if t.exit_reason == "TP"),
        "sl_exits": sum(1 for t in trades if t.exit_reason == "SL"),
        "flatten_exits": sum(1 for t in trades if t.exit_reason == "FLAT"),
    }


def split_is_oos(trades: List[OnTrade]) -> Tuple[List[OnTrade], List[OnTrade], List[OnTrade]]:
    ins, oos, aug = [], [], []
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if ts >= AUG_START:
            aug.append(t)
        if ts >= OOS_START:
            oos.append(t)
        else:
            ins.append(t)
    return ins, oos, aug


def nights_in_split(sess: np.ndarray, dt: pd.DatetimeIndex, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> int:
    seen = set()
    for i, s in enumerate(sess):
        if not s:
            continue
        ts = pd.Timestamp(dt[i])
        if start is not None and ts < start:
            continue
        if end is not None and ts >= end:
            continue
        seen.add(s)
    return len(seen)


def grade(is_s: Dict[str, Any], oos_s: Dict[str, Any]) -> str:
    """PASS only with OOS PF, trade count, and DD that is not catastrophic vs net $."""
    if oos_s["trades"] < MIN_OOS_TRADES:
        return "THIN"
    if oos_s["profit_factor"] < PF_PASS or oos_s["total_pnl"] <= 0:
        return "FAIL"
    dd = float(oos_s["max_drawdown"])
    net = float(oos_s["total_pnl"])
    if dd > DD_CAP_USD or dd > max(2.0 * net, 1500.0):
        return "FAIL_DD"
    if is_s["trades"] < MIN_IS_TRADES:
        return "THIN_IS"
    if is_s["profit_factor"] < 1.0:
        return "FAIL_IS"
    return "PASS"


def monthly_pnl(trades: List[OnTrade]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        key = ts.tz_convert(ET).strftime("%Y-%m")
        out[key] = round(out.get(key, 0.0) + t.pnl, 2)
    return dict(sorted(out.items()))


def _close_pos(pos, ts, px, reason, spec_name) -> OnTrade:
    pts, usd = _pnl(pos["dir"], pos["entry"], px)
    hold = (pd.Timestamp(ts) - pd.Timestamp(pos["ts"])).total_seconds()
    return OnTrade(
        entry_time=str(pos["ts"]),
        exit_time=str(ts),
        direction="long" if pos["dir"] == 1 else "short",
        entry_price=round(pos["entry"], 2),
        exit_price=round(float(px), 2),
        sl=round(pos["sl"], 2),
        tp=round(pos["tp"], 2),
        pnl=usd,
        pts=pts,
        exit_reason=reason,
        spec=spec_name,
        session=pos["session"],
        hold_seconds=hold,
        cue=pos.get("cue") or spec_name,
    )


def _manage(pos, hi, lo, cl, ts, do_flat) -> Optional[OnTrade]:
    if do_flat:
        return _close_pos(pos, ts, cl, "FLAT", pos["spec"])
    if pos["dir"] == 1:
        hit_sl = lo <= pos["sl"]
        hit_tp = pos["tp"] and hi >= pos["tp"]
        if hit_sl:
            return _close_pos(pos, ts, pos["sl"], "SL", pos["spec"])
        if hit_tp:
            return _close_pos(pos, ts, pos["tp"], "TP", pos["spec"])
    else:
        hit_sl = hi >= pos["sl"]
        hit_tp = pos["tp"] and lo <= pos["tp"]
        if hit_sl:
            return _close_pos(pos, ts, pos["sl"], "SL", pos["spec"])
        if hit_tp:
            return _close_pos(pos, ts, pos["tp"], "TP", pos["spec"])
    return None


def _open(spec: Spec, ts, session, side: int, entry: float, atr: float, cue: str) -> Dict[str, Any]:
    sl_pts = _clip_sl(atr, spec) if spec.use_atr_stop else 1e9
    if spec.tp_r > 0 and spec.use_atr_stop:
        tp_pts = sl_pts * spec.tp_r
    else:
        tp_pts = spec.tp_pts
    if side == 1:
        sl = entry - sl_pts
        tp = entry + tp_pts
    else:
        sl = entry + sl_pts
        tp = entry - tp_pts
    return {
        "ts": ts,
        "dir": side,
        "entry": float(entry),
        "sl": float(sl),
        "tp": float(tp),
        "session": session,
        "spec": spec.name,
        "cue": cue,
    }


def simulate(frame: Dict[str, Any], spec: Spec) -> List[OnTrade]:
    dt = frame["dt"]
    hi = frame["hi"]
    lo = frame["lo"]
    cl = frame["cl"]
    t15 = frame["t15"]
    t60 = frame["t60"]
    td = frame["td"]
    atr = frame["atr"]
    vwap = frame["vwap"]
    mins = frame["mins"]
    sess = frame["sess"]
    can_enter = frame["can_enter"]
    flatten = frame["flatten"]
    n = len(dt)
    trades: List[OnTrade] = []
    pos = None
    filled = 0
    fired_win: set = set()
    cur = ""
    bars_in = 0
    onh = onl = np.nan
    onh_i = onl_i = -1
    or_hi = or_lo = np.nan
    or_done = False

    def reset_night(s: str) -> None:
        nonlocal filled, fired_win, bars_in, onh, onl, onh_i, onl_i, or_hi, or_lo, or_done
        filled = 0
        fired_win = set()
        bars_in = 0
        onh = onl = np.nan
        onh_i = onl_i = -1
        or_hi = or_lo = np.nan
        or_done = False

    for i in range(WARMUP, n):
        s = sess[i]
        ts = dt[i]
        if s != cur:
            if pos is not None:
                # Night ended (or 09:25 bar missing) — flatten at this close.
                trades.append(_close_pos(pos, ts, float(cl[i]), "FLAT", pos["spec"]))
                pos = None
            cur = s
            reset_night(s)

        if pos is not None:
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
                h, l = float(hi[i]), float(lo[i])
                if np.isnan(onh) or h > onh:
                    onh, onh_i = h, bars_in
                if np.isnan(onl) or l < onl:
                    onl, onl_i = l, bars_in
                bars_in += 1
            continue

        a = float(atr[i]) if not np.isnan(atr[i]) else np.nan
        side = 0
        cue = spec.kind
        win = in_window(int(mins[i]), spec.windows) if spec.windows else None

        if spec.kind == "bh":
            if bars_in == 0:
                side = spec.side
                cue = "bh_long" if side == 1 else "bh_short"
        elif spec.kind == "window_ema":
            if win is not None and win not in fired_win:
                d15 = int(t15[i])
                if d15 != 0:
                    ok = True
                    if spec.require_daily and int(td[i]) != d15:
                        ok = False
                    if spec.require_60m and int(t60[i]) != d15:
                        ok = False
                    if ok:
                        side = d15
                        cue = f"win{win}"
                        fired_win.add(win)
        elif spec.kind in ("fade", "break", "paper"):
            if bars_in >= WARMUP_BARS and not np.isnan(onh) and not np.isnan(onl):
                age_hi = bars_in - 1 - onh_i
                age_lo = bars_in - 1 - onl_i
                onh_ok = age_hi >= SETTLE_BARS
                onl_ok = age_lo >= SETTLE_BARS
                close = float(cl[i])
                high = float(hi[i])
                low = float(lo[i])
                if spec.kind in ("break", "paper") and onh_ok and close > onh + TOUCH_TICK:
                    side, cue = 1, "break_onh"
                elif spec.kind in ("break", "paper") and onl_ok and close < onl - TOUCH_TICK:
                    side, cue = -1, "break_onl"
                elif spec.kind in ("fade", "paper") and onh_ok and high >= onh - TOUCH_TICK and close <= onh:
                    side, cue = -1, "fade_onh"
                elif spec.kind in ("fade", "paper") and onl_ok and low <= onl + TOUCH_TICK and close >= onl:
                    side, cue = 1, "fade_onl"
        elif spec.kind == "vwap_fade":
            if bars_in >= WARMUP_BARS and not np.isnan(vwap[i]) and not np.isnan(a) and a > 0:
                dist = float(cl[i]) - float(vwap[i])
                if dist >= 1.5 * a:
                    side, cue = -1, "vwap_fade_short"
                elif dist <= -1.5 * a:
                    side, cue = 1, "vwap_fade_long"
        elif spec.kind == "orb":
            # First window is the range build; trade starts at window end.
            if spec.windows:
                a0, b0 = spec.windows[0]
                m = int(mins[i])
                if a0 <= m < b0:
                    h, l = float(hi[i]), float(lo[i])
                    or_hi = h if np.isnan(or_hi) else max(or_hi, h)
                    or_lo = l if np.isnan(or_lo) else min(or_lo, l)
                elif m >= b0 and not np.isnan(or_hi) and not or_done:
                    width = or_hi - or_lo
                    if 4.0 <= width <= 40.0:
                        close = float(cl[i])
                        if close > or_hi + TOUCH_TICK:
                            side, cue, or_done = 1, "orb_long", True
                        elif close < or_lo - TOUCH_TICK:
                            side, cue, or_done = -1, "orb_short", True

        if side != 0:
            pos = _open(spec, ts, s, side, float(cl[i]), a, cue)
            filled += 1

        bars_in += 1
        h, l = float(hi[i]), float(lo[i])
        if np.isnan(onh) or h > onh:
            onh, onh_i = h, bars_in - 1
        if np.isnan(onl) or l < onl:
            onl, onl_i = l, bars_in - 1

    return trades


def specs() -> List[Spec]:
    fade_stop = dict(sl_mult=FADE_ATR_MULT, sl_min=FADE_SL_MIN, sl_max=FADE_SL_MAX, tp_r=FADE_TP_R, tp_pts=0.0)
    ema_stop = dict(sl_mult=2.0, sl_min=20.0, sl_max=60.0, tp_r=0.0, tp_pts=TP_CAP)
    return [
        Spec("no_trade", "none", note="Do nothing overnight."),
        Spec("bh_long", "bh", use_atr_stop=False, tp_pts=1e9, side=1, note="Long first Globex bar, flatten 09:25, no stop."),
        Spec("bh_short", "bh", use_atr_stop=False, tp_pts=1e9, side=-1, note="Short first Globex bar, flatten 09:25, no stop."),
        Spec("bh_long_atr", "bh", side=1, note="Overnight long with ATR×2 20-60 stop.", **ema_stop),
        Spec(
            "ema_asia_20et", "window_ema", windows=WIN_ASIA, require_daily=True, require_60m=True,
            note="20:00-21:00 ET, daily+15m+60m aligned, 1/night.", **ema_stop,
        ),
        Spec(
            "ema_london_03et", "window_ema", windows=WIN_LONDON, require_daily=True, require_60m=True,
            note="03:00-04:00 ET, daily+15m+60m aligned, 1/night.", **ema_stop,
        ),
        Spec(
            "ema_premarket_08et", "window_ema", windows=WIN_PRE, require_daily=True, require_60m=True,
            note="08:00-09:00 ET, daily+15m+60m aligned, 1/night.", **ema_stop,
        ),
        Spec(
            "ema_globex_18et", "window_ema", windows=WIN_GLOBEX, require_daily=True, require_60m=True,
            note="18:00-19:00 ET, daily+15m+60m aligned, 1/night.", **ema_stop,
        ),
        Spec(
            "ema_first_asia_london_pre", "window_ema", windows=WIN_FIRST_THREE, require_daily=True, require_60m=True,
            max_per_night=1, note="First aligned Asia/London/premarket fill only.", **ema_stop,
        ),
        Spec(
            "ema15_on_5win_replica", "window_ema", windows=WIN_EMA15_ON, require_daily=True, require_60m=True,
            max_per_night=5, note="Prior failed Globex ema15 windows (confirm).", **ema_stop,
        ),
        Spec("fade_settled_onh_onl", "fade", note="Paper fade of settled overnight high/low.", **fade_stop),
        Spec("break_settled_onh_onl", "break", note="Paper break-and-hold of settled ONH/ONL.", **fade_stop),
        Spec("paper_cues_combined", "paper", note="Break then fade, 1 fill/night, flatten 09:25.", **fade_stop),
        Spec("vwap_fade_1p5atr", "vwap_fade", note="Fade 1.5×ATR from overnight VWAP after 60m warmup.", **fade_stop),
        Spec(
            "london_orb_0315", "orb", windows=((3 * 60, 3 * 60 + 15),),
            note="03:00-03:15 ET range, break, ATR stop, flatten 09:25.", **ema_stop,
        ),
        Spec(
            "premarket_orb_0815", "orb", windows=((8 * 60, 8 * 60 + 15),),
            note="08:00-08:15 ET range, break, ATR stop, flatten 09:25.", **ema_stop,
        ),
    ]


def _read_1m(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, parse_dates=["datetime"])
    df = scalp_bt.normalize_dt(raw)
    keep = [c for c in ("datetime", "open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep]


def load_frame(csv_1m: str, extra: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, str]:
    """Load real 1m MNQ and stitch later Databento slices so August is not a hole."""
    parts = [_read_1m(csv_1m)]
    used = [csv_1m]
    data_dir = os.path.dirname(os.path.abspath(csv_1m)) or os.path.join(ROOT, "data")
    extras = list(extra) if extra is not None else [
        os.path.join(data_dir, "MNQ_1m_real_window.csv"),
        os.path.join(data_dir, "MNQ_1m_august_databento.csv"),
        os.path.join(data_dir, "MNQ_1m_august.csv"),
    ]
    for path in extras:
        if os.path.isfile(path) and os.path.abspath(path) != os.path.abspath(csv_1m):
            parts.append(_read_1m(path))
            used.append(path)
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    note = "real 1m merged: " + ", ".join(os.path.basename(p) for p in used)
    return df, note


def prepare(df: pd.DataFrame) -> Dict[str, Any]:
    work = df.copy()
    dt = pd.DatetimeIndex(pd.to_datetime(work["datetime"], utc=True))
    work["datetime"] = dt
    t15 = trend_15m_on_1m(work).to_numpy(dtype=int)
    t60 = trend_60m_on_1m(work).to_numpy(dtype=int)
    td = daily_trend_on_1m(work).to_numpy(dtype=int)
    atr = atr15_on_1m(work).to_numpy(dtype=float)
    mins, dow, sess, can_enter, flatten = build_session_arrays(dt)
    hi = work["high"].to_numpy(dtype=float)
    lo = work["low"].to_numpy(dtype=float)
    cl = work["close"].to_numpy(dtype=float)
    vol = work["volume"].to_numpy(dtype=float) if "volume" in work.columns else np.ones(len(work))
    vwap = overnight_vwap(hi, lo, cl, vol, sess)
    return {
        "dt": dt,
        "hi": hi,
        "lo": lo,
        "cl": cl,
        "vol": vol,
        "t15": t15,
        "t60": t60,
        "td": td,
        "atr": atr,
        "vwap": vwap,
        "mins": mins,
        "dow": dow,
        "sess": sess,
        "can_enter": can_enter,
        "flatten": flatten,
    }


def pack_row(spec: Spec, trades: List[OnTrade], frame: Dict[str, Any]) -> Dict[str, Any]:
    dt = frame["dt"]
    sess = frame["sess"]
    ins, oos, aug = split_is_oos(trades)
    n_all = nights_in_split(sess, dt, None, None)
    n_is = nights_in_split(sess, dt, None, OOS_START)
    n_oos = nights_in_split(sess, dt, OOS_START, None)
    n_aug = nights_in_split(sess, dt, AUG_START, None)
    is_s = _summarize(ins, n_is, spec.name)
    oos_s = _summarize(oos, n_oos, spec.name)
    all_s = _summarize(trades, n_all, spec.name)
    aug_s = _summarize(aug, n_aug, spec.name)
    g = "BASELINE" if spec.kind in ("none", "bh") else grade(is_s, oos_s)
    if spec.name == "no_trade":
        g = "BASELINE"
    return {
        "name": spec.name,
        "kind": spec.kind,
        "note": spec.note,
        "grade": g,
        "is": is_s,
        "oos": oos_s,
        "all": all_s,
        "august": aug_s,
        "oos_monthly": monthly_pnl(oos),
    }


def recipe_from(row: Dict[str, Any], spec: Spec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "windows_et": (
            [f"{a//60:02d}:{a%60:02d}-{b//60:02d}:{b%60:02d}" for a, b in spec.windows]
            if spec.windows
            else ["Globex 18:00-09:25, first settled ONH/ONL break"]
        ),
        "filters": {
            "daily_rth_ema20": spec.require_daily,
            "ema15_8_21": spec.kind == "window_ema",
            "ema60_8_21": spec.require_60m,
        },
        "max_trades_per_night": spec.max_per_night,
        "stop": "none" if not spec.use_atr_stop else f"clip(15m ATR14×{spec.sl_mult}, {spec.sl_min}-{spec.sl_max})",
        "tp": f"{spec.tp_r}R" if spec.tp_r else f"{spec.tp_pts} pt cap",
        "flatten_et": "09:25",
        "qty": 1,
        "oos": row["oos"],
        "is": row["is"],
        "grade": row["grade"],
        "live": False,
        "paper_only": True,
    }


def print_table(rows: List[Dict[str, Any]]) -> None:
    hdr = (
        f"{'Rule':<28} {'ISn':>4} {'IS PF':>6} {'IS$':>8} "
        f"{'OOSn':>5} {'OOS PF':>7} {'OOS$':>8} {'OOS DD':>7} "
        f"{'$/n':>7} {'Aug n':>5} {'Aug PF':>6} {'grade':<10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        i, o, a = r["is"], r["oos"], r["august"]
        print(
            f"{r['name']:<28} {i['trades']:4d} {i['profit_factor']:6.2f} {i['total_pnl']:8.0f} "
            f"{o['trades']:5d} {o['profit_factor']:7.2f} {o['total_pnl']:8.0f} {o['max_drawdown']:7.0f} "
            f"{o['avg_per_night']:7.1f} {a['trades']:5d} {a['profit_factor']:6.2f} {r['grade']:<10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight Globex MNQ edge search (real 1m)")
    parser.add_argument("--csv-1m", default=os.path.join(ROOT, "data", "MNQ_1m.csv"))
    parser.add_argument("--out", default=os.path.join(ROOT, "data", "overnight_edge_search.json"))
    args = parser.parse_args()

    df, note = load_frame(args.csv_1m)
    frame = prepare(df)
    start, end = frame["dt"][0], frame["dt"][-1]
    print("OVERNIGHT GLOBEX SEARCH — real 1m MNQ")
    print(f"  {args.csv_1m}")
    print(f"  {start} -> {end}  bars={len(df):,}  {note}")
    print(f"  IS < {OOS_START.date()} UTC   OOS >= {OOS_START.date()} (August reported separately)")
    print(f"  Flatten 09:25 ET  |  1 MNQ  |  fee ${COMMISSION:.2f} RT  |  ${POINT_VALUE:.0f}/pt")
    print(f"  PASS = OOS PF>={PF_PASS}, OOS trades>={MIN_OOS_TRADES}, IS trades>={MIN_IS_TRADES},")
    print(f"         OOS net>0, DD <= min($4000, max(2×OOS$, $1500))")
    print(f"  NOT live. Day bot stays locked ema15 RTH.")
    print()

    rows = []
    spec_map = {s.name: s for s in specs()}
    for spec in specs():
        trades = [] if spec.kind == "none" else simulate(frame, spec)
        rows.append(pack_row(spec, trades, frame))
    print_table(rows)

    passed = [r for r in rows if r["grade"] == "PASS"]
    passed.sort(key=lambda r: (r["oos"]["profit_factor"], r["oos"]["total_pnl"]), reverse=True)
    best = passed[0] if passed else None
    print()
    if best:
        print(f"BEST PASS: {best['name']}  OOS PF {best['oos']['profit_factor']}  "
              f"${best['oos']['total_pnl']:.0f}  DD ${best['oos']['max_drawdown']:.0f}")
        print("  Still NOT for live money. Paper-only if you want more samples.")
        recipe = recipe_from(best, spec_map[best["name"]])
    else:
        print("NO overnight candidate passed. Do not paper for edge. Do not go live.")
        recipe = None

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv": args.csv_1m,
        "data_note": note,
        "bars": int(len(df)),
        "range": [str(start), str(end)],
        "oos_start": str(OOS_START),
        "point_value": POINT_VALUE,
        "commission": COMMISSION,
        "flatten_et": "09:25",
        "qty": QTY,
        "pass_rule": {
            "oos_pf": PF_PASS,
            "min_oos_trades": MIN_OOS_TRADES,
            "min_is_trades": MIN_IS_TRADES,
            "dd_cap_usd": DD_CAP_USD,
        },
        "live_note": (
            "NOT for live money. Locked day recipe stays ema15_eod RTH only "
            "(windows 9:35/10:15/11:00/12:00/13:30/14:30, flatten 15:50). "
            "Do not set OVERNIGHT_TRADING on live."
        ),
        "candidates": rows,
        "best": best,
        "recipe": recipe,
        "verdict": "PASS" if best else "NO_EDGE",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
