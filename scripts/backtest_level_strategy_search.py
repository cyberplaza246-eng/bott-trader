#!/usr/bin/env python3
"""
MNQ level-mode strategy search — how to use PDH/PDL/ONH/ONL/RTH levels.

Variants:
  L0  Baseline prior: daily+15m+60m + quiet 1m + 15m ATR×2 stop (no levels)
  L1  Level mode as live: 15m strict (structure+EMA+VWAP), 5m, level interact,
      1m confirm, 1m ATR stop, TP next-level-or-1R
  L2  Soft VWAP: 15m = structure+EMA only (VWAP advisory), else same as L1
  L3  L2 but morning windows only (9:35, 10:15, 11:00)
  L4a L2 + TP next major level when practical (live hybrid)
  L4b L2 + TP 1R only
  L5a L2 + interact only at PDH/PDL
  L5b L2 + interact at all majors (PDH/PDL/ONH/ONL/RTH) — same as L2
  L6  Responsive 15m+5m no levels + 1m ATR×1.5 clip 6–25

Shared: 6 RTH windows (L3 morning-only), conf≥75, max 20/day, window re-entry,
flatten 15:50, max 2 lots, $2/pt + $1.24 RT. IS/OOS split Jun 2026.

Rank by OOS expectancy, then PF, then DD (lower better).
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_ema15_15_5_1_responsive as responsive
import scripts.backtest_ema15_gapped_vs_full as gapped
import scripts.backtest_ema15_official as official
import scripts.backtest_ema15_timeframe_stacks as stacks
import scripts.backtest_scalp_momentum as scalp_bt
import scripts.find_real_1m_edge as edge
from src.strategy.mnq_15m_ema_eod import (
    DEFAULT_MAX_OPEN_LOTS,
    ENTRY_MINUTE,
    ENTRY_WINDOWS,
    FLAT_MINUTE,
    MAX_TRADES_DAY,
    TP_R,
    _as_utc_frame,
    _atr,
    _ohlc_resample,
    atr15_on_1m,
    ema_series,
    level_mode_confidence_pct,
    quiet_1m_ok,
    sl_pts_from_atr,
    window_index,
    window_label,
)
from src.strategy.session_levels import (
    EMA_FAST_CHART,
    EMA_SLOW_CHART,
    RTH_END_MINUTE,
    RTH_OPEN_MINUTE,
    TOUCH_PTS_DEFAULT,
    level_interaction,
    major_level_map,
    next_target_level,
    one_min_level_confirm,
    session_vwap_series,
    structure_side_from_ohlc,
)

ET = pytz.timezone("US/Eastern")
WARMUP = 80
PT, COMM = official.PT, official.COMM
MIN_CONF = 75
MORNING_WINDOWS = frozenset({0, 1, 2})  # 9:35, 10:15, 11:00
MORNING_END = 12 * 60
WINDOW_NAMES = tuple(window_label(i) for i in range(len(ENTRY_WINDOWS)))

# Level-mode stop band (matches check_ema15_level_entry defaults)
SL_ATR_MULT = 1.5
SL_MIN = 8.0
SL_MAX = 30.0

LEVEL_KEYS_ALL = ("pdh", "pdl", "onh", "onl", "rth_open")
LEVEL_KEYS_PD = ("pdh", "pdl")


@dataclass
class TradeRec:
    trade: scalp_bt.ScalpTrade
    window: str
    sl_pts: float
    variant: str
    level_name: str = ""
    tp_mode: str = ""


def _et_index(tmp: pd.DataFrame) -> pd.DatetimeIndex:
    return tmp.index.tz_convert(ET)


def _rth_mask(et_idx: pd.DatetimeIndex) -> np.ndarray:
    mins = et_idx.hour * 60 + et_idx.minute
    wd = et_idx.weekday
    return (wd < 5) & (mins >= RTH_OPEN_MINUTE) & (mins < RTH_END_MINUTE)


def _prior_rth_day(et_day: date) -> Optional[date]:
    d = et_day - timedelta(days=1)
    for _ in range(8):
        if d.weekday() < 5:
            return d
        d -= timedelta(days=1)
    return None


def _globex_start_for_rth_day(et_day: date) -> datetime:
    start_day = et_day - timedelta(days=1)
    while start_day.weekday() == 5:
        start_day -= timedelta(days=1)
    return ET.localize(datetime(start_day.year, start_day.month, start_day.day, 18, 0))


def precompute_day_levels(tmp: pd.DataFrame) -> Dict[date, Dict[str, Any]]:
    """PDH/PDL/ONH/ONL/RTH Open per ET RTH day (static overnight + prior day)."""
    et = _et_index(tmp)
    day_arr = pd.Series(et.date, index=tmp.index)
    rth = _rth_mask(et)
    out: Dict[date, Dict[str, Any]] = {}
    unique_days = sorted(set(day_arr[rth].tolist()))
    for et_day in unique_days:
        prior = _prior_rth_day(et_day)
        pdh = pdl = None
        if prior is not None:
            prior_mask = rth & (day_arr == prior)
            prior_bars = tmp.loc[prior_mask]
            if not prior_bars.empty:
                pdh = float(prior_bars["high"].max())
                pdl = float(prior_bars["low"].min())
        onh = onl = None
        globex_start = _globex_start_for_rth_day(et_day).astimezone(pytz.UTC)
        rth_open_dt = ET.localize(
            datetime(et_day.year, et_day.month, et_day.day, 9, 30)
        ).astimezone(pytz.UTC)
        on_mask = (tmp.index >= globex_start) & (tmp.index < rth_open_dt)
        on_bars = tmp.loc[on_mask]
        if not on_bars.empty:
            onh = float(on_bars["high"].max())
            onl = float(on_bars["low"].min())
        today_mask = rth & (day_arr == et_day)
        today = tmp.loc[today_mask]
        rth_open = float(today.iloc[0]["open"]) if not today.empty else None
        # OR / day extremes evolve — store open-only static; update day H/L in loop via arrays
        out[et_day] = {
            "pdh": pdh,
            "pdl": pdl,
            "onh": onh,
            "onl": onl,
            "rth_open": rth_open,
            "orh": None,
            "orl": None,
            "day_high": None,
            "day_low": None,
            "prev_week_close": None,
        }
    return out


def precompute_bias_arrays(
    tmp: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return aligned 1m arrays:
      side15_strict, side15_soft, side5, struct_label_code,
      ema_side_15, vwap_side
    Labels: 0=mixed, 1=HH_HL, -1=LH_LL encoded in struct_side separately.
    """
    n = len(tmp)
    side15_strict = np.zeros(n, dtype=np.int8)
    side15_soft = np.zeros(n, dtype=np.int8)
    side5 = np.zeros(n, dtype=np.int8)
    struct_side = np.zeros(n, dtype=np.int8)
    ema_side_15 = np.zeros(n, dtype=np.int8)
    vwap_side = np.zeros(n, dtype=np.int8)

    b15 = _ohlc_resample(tmp, "15min")
    b5 = _ohlc_resample(tmp, "5min")
    if len(b15) < 6 or len(b5) < 4:
        return side15_strict, side15_soft, side5, struct_side, ema_side_15, vwap_side

    # Completed bars only → shift by 1 on TF then ffill
    e10_15 = ema_series(b15["close"], EMA_FAST_CHART)
    e20_15 = ema_series(b15["close"], EMA_SLOW_CHART)
    ema15 = pd.Series(0, index=b15.index, dtype=int)
    ema15.loc[e10_15 > e20_15] = 1
    ema15.loc[e10_15 < e20_15] = -1

    struct15 = pd.Series(0, index=b15.index, dtype=int)
    for i in range(5, len(b15)):
        s, _ = structure_side_from_ohlc(b15.iloc[i - 5 : i + 1], lookback=6)
        struct15.iloc[i] = s

    e10_5 = ema_series(b5["close"], EMA_FAST_CHART)
    e20_5 = ema_series(b5["close"], EMA_SLOW_CHART)
    ema5 = pd.Series(0, index=b5.index, dtype=int)
    ema5.loc[e10_5 > e20_5] = 1
    ema5.loc[e10_5 < e20_5] = -1
    struct5 = pd.Series(0, index=b5.index, dtype=int)
    for i in range(3, len(b5)):
        s, _ = structure_side_from_ohlc(b5.iloc[i - 3 : i + 1], lookback=4)
        struct5.iloc[i] = s
    # Soft structure disagree → flat 5m
    side5_tf = ema5.copy()
    disagree = (struct5 != 0) & (ema5 != 0) & (struct5 != ema5)
    side5_tf.loc[disagree] = 0

    # Completed-only: shift then ffill to 1m
    struct_ff = struct15.shift(1).reindex(tmp.index, method="ffill").fillna(0).astype(int)
    ema15_ff = ema15.shift(1).reindex(tmp.index, method="ffill").fillna(0).astype(int)
    side5_ff = side5_tf.shift(1).reindex(tmp.index, method="ffill").fillna(0).astype(int)

    vwap_s = session_vwap_series(tmp)
    cl = tmp["close"].astype(float)
    vwap_side_s = pd.Series(0, index=tmp.index, dtype=int)
    valid = vwap_s.notna()
    vwap_side_s.loc[valid & (cl > vwap_s)] = 1
    vwap_side_s.loc[valid & (cl < vwap_s)] = -1

    ss = struct_ff.to_numpy(dtype=int)
    es = ema15_ff.to_numpy(dtype=int)
    vs = vwap_side_s.to_numpy(dtype=int)
    s5 = side5_ff.to_numpy(dtype=int)

    soft = np.zeros(n, dtype=np.int8)
    strict = np.zeros(n, dtype=np.int8)
    # soft: structure + EMA only
    soft[(ss > 0) & (es >= 0)] = 1
    soft[(ss < 0) & (es <= 0)] = -1
    soft[ss == 0] = 0
    # strict: structure + EMA + VWAP
    strict[(ss > 0) & (es >= 0) & (vs >= 0)] = 1
    strict[(ss < 0) & (es <= 0) & (vs <= 0)] = -1
    strict[ss == 0] = 0
    # conflicts already zeroed by conditions

    return (
        strict.astype(np.int8),
        soft.astype(np.int8),
        s5.astype(np.int8),
        ss.astype(np.int8),
        es.astype(np.int8),
        vs.astype(np.int8),
    )


def filter_levels(levels: Dict[str, Any], keys: Tuple[str, ...]) -> Dict[str, Any]:
    out = dict(levels)
    majors = {k: out.get(k) for k in LEVEL_KEYS_ALL}
    for k in LEVEL_KEYS_ALL:
        if k not in keys:
            out[k] = None
    # keep optional target levels
    return out


def atr1_on_1m(df_1m: pd.DataFrame) -> pd.Series:
    return responsive.atr1_on_1m(df_1m)


def group_metrics(trades: List[scalp_bt.ScalpTrade]) -> Dict[str, Any]:
    return stacks.group_metrics(trades)


def session_bucket(entry_ts: pd.Timestamp) -> str:
    return stacks.session_bucket(entry_ts)


def full_report(variant_id: str, label: str, recs: List[TradeRec]) -> Dict[str, Any]:
    trades = [r.trade for r in recs]
    ins, oos = edge.split_is_oos(trades)
    overall = group_metrics(trades)
    is_m = group_metrics(ins)
    oos_m = group_metrics(oos)
    by_side = {
        lab: group_metrics([r.trade for r in recs if r.trade.direction == lab])
        for lab in ("long", "short")
    }
    by_session = {
        lab: group_metrics([r.trade for r in recs if session_bucket(r.trade.entry_time) == lab])
        for lab in ("morning", "afternoon")
    }
    by_window = {
        wn: group_metrics([r.trade for r in recs if r.window == wn]) for wn in WINDOW_NAMES
    }
    level_counts: Dict[str, int] = {}
    tp_counts: Dict[str, int] = {}
    for r in recs:
        if r.level_name:
            level_counts[r.level_name] = level_counts.get(r.level_name, 0) + 1
        if r.tp_mode:
            tp_counts[r.tp_mode] = tp_counts.get(r.tp_mode, 0) + 1
    avg_sl = round(float(np.mean([r.sl_pts for r in recs])), 1) if recs else 0.0
    return {
        "id": variant_id,
        "label": label,
        "avg_stop_pts": avg_sl,
        "overall": overall,
        "is": is_m,
        "oos": oos_m,
        "by_side": by_side,
        "by_session": by_session,
        "by_window": by_window,
        "level_counts": level_counts,
        "tp_mode_counts": tp_counts,
        "n_trades": overall["trades"],
    }


def rank_reports(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        reports,
        key=lambda r: (
            r["oos"]["expectancy"],
            r["oos"]["profit_factor"],
            -r["oos"]["max_drawdown"],
        ),
        reverse=True,
    )


@dataclass
class LevelCtx:
    work: pd.DataFrame
    dt: pd.DatetimeIndex
    day_levels: Dict[date, Dict[str, Any]]
    side15_strict: np.ndarray
    side15_soft: np.ndarray
    side5_arr: np.ndarray
    struct_arr: np.ndarray
    ema15_arr: np.ndarray
    vwap_arr: np.ndarray
    atr1: np.ndarray
    hi: np.ndarray
    lo: np.ndarray
    cl: np.ndarray
    mins: np.ndarray
    dow: np.ndarray
    dates: Any
    day_hi: np.ndarray
    day_lo: np.ndarray


def build_level_ctx(df: pd.DataFrame) -> LevelCtx:
    """One-time precompute shared by all level variants."""
    dt = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))
    work = df.copy()
    work["datetime"] = dt
    tmp = _as_utc_frame(work)
    print("  precomputing day levels + 15m/5m bias...")
    t0 = time.time()
    day_levels = precompute_day_levels(tmp)
    (
        side15_strict,
        side15_soft,
        side5_arr,
        struct_arr,
        ema15_arr,
        vwap_arr,
    ) = precompute_bias_arrays(tmp)
    atr1 = atr1_on_1m(work).to_numpy(dtype=float)
    hi = work["high"].to_numpy(dtype=float)
    lo = work["low"].to_numpy(dtype=float)
    cl = work["close"].to_numpy(dtype=float)
    et = dt.tz_convert(ET)
    mins = (et.hour * 60 + et.minute).to_numpy()
    dow = et.weekday.to_numpy()
    dates = et.date
    day_hi = np.full(len(work), np.nan)
    day_lo = np.full(len(work), np.nan)
    cur_d = None
    cur_hi = cur_lo = np.nan
    for i in range(len(work)):
        if dow[i] >= 5:
            continue
        m = int(mins[i])
        if m < RTH_OPEN_MINUTE or m >= RTH_END_MINUTE:
            continue
        d = dates[i]
        if d != cur_d:
            cur_d = d
            cur_hi = float(hi[i])
            cur_lo = float(lo[i])
        else:
            cur_hi = max(cur_hi, float(hi[i]))
            cur_lo = min(cur_lo, float(lo[i]))
        day_hi[i] = cur_hi
        day_lo[i] = cur_lo
    print(f"  precompute done in {time.time() - t0:.1f}s")
    return LevelCtx(
        work=work,
        dt=dt,
        day_levels=day_levels,
        side15_strict=side15_strict,
        side15_soft=side15_soft,
        side5_arr=side5_arr,
        struct_arr=struct_arr,
        ema15_arr=ema15_arr,
        vwap_arr=vwap_arr,
        atr1=atr1,
        hi=hi,
        lo=lo,
        cl=cl,
        mins=mins,
        dow=dow,
        dates=dates,
        day_hi=day_hi,
        day_lo=day_lo,
    )


def simulate_level(
    ctx: LevelCtx,
    *,
    variant_id: str,
    soft_vwap: bool,
    allowed_windows: Optional[Set[int]] = None,
    level_keys: Tuple[str, ...] = LEVEL_KEYS_ALL,
    tp_mode: str = "hybrid",  # hybrid | 1r
    min_confidence: int = MIN_CONF,
) -> List[TradeRec]:
    """
    Level-mode simulation with precomputed bias + day levels.
    tp_mode hybrid = next major level when practical else 1R (live).
    tp_mode 1r = always 1R.
    """
    work = ctx.work
    dt = ctx.dt
    day_levels = ctx.day_levels
    side15 = ctx.side15_soft if soft_vwap else ctx.side15_strict
    side5_arr = ctx.side5_arr
    struct_arr = ctx.struct_arr
    ema15_arr = ctx.ema15_arr
    vwap_arr = ctx.vwap_arr
    atr1 = ctx.atr1
    hi, lo, cl = ctx.hi, ctx.lo, ctx.cl
    mins, dow, dates = ctx.mins, ctx.dow, ctx.dates
    day_hi, day_lo = ctx.day_hi, ctx.day_lo

    recs: List[TradeRec] = []
    opens: List[Dict] = []
    fired_windows: Set[int] = set()
    counts: Dict = {}

    for i in range(WARMUP, len(work)):
        ts = pd.Timestamp(dt[i])
        m = int(mins[i])
        day = dates[i]

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
                tr = gapped._close_position(p, ts, float(exit_px), reason)
                tr.version = variant_id
                recs.append(
                    TradeRec(
                        tr,
                        p["win_name"],
                        p["sl_pts"],
                        variant_id,
                        level_name=p.get("level_name", ""),
                        tp_mode=p.get("tp_mode", ""),
                    )
                )
                if p.get("win") is not None:
                    fired_windows.discard(int(p["win"]))
            else:
                still.append(p)
        opens = still

        if dow[i] >= 5:
            continue
        if m < ENTRY_MINUTE or m >= FLAT_MINUTE:
            continue
        if counts.get(day, 0) >= MAX_TRADES_DAY:
            continue

        win = window_index(m)
        if win is None or win in fired_windows:
            continue
        if allowed_windows is not None and win not in allowed_windows:
            continue

        side = int(side15[i])
        if side == 0:
            continue
        if int(side5_arr[i]) != side:
            continue

        base_lv = dict(day_levels.get(day) or {})
        if not base_lv:
            continue
        # evolving day extremes for targets
        if not np.isnan(day_hi[i]):
            base_lv["day_high"] = float(day_hi[i])
            base_lv["day_low"] = float(day_lo[i])
        levels = filter_levels(base_lv, level_keys)
        if not major_level_map(levels):
            continue

        # Recent 1m slice for interaction / confirm (indexed frame)
        lookback = 14
        lo_i = max(0, i - lookback)
        slice_df = work.iloc[lo_i : i + 1]
        interaction = level_interaction(slice_df, side, levels, touch_pts=TOUCH_PTS_DEFAULT)
        if not interaction:
            continue
        if not one_min_level_confirm(slice_df, side, interaction):
            continue

        bias = {
            "structure_15m": (
                "HH_HL" if struct_arr[i] > 0 else ("LH_LL" if struct_arr[i] < 0 else "mixed")
            ),
            "ema_side_15m": int(ema15_arr[i]),
            "vwap_side_15m": int(vwap_arr[i]),
            "side_5m": int(side5_arr[i]),
        }
        conf = level_mode_confidence_pct(side, bias, interaction, one_min_ok=True)
        if conf["pct"] < min_confidence:
            continue

        if len(opens) >= DEFAULT_MAX_OPEN_LOTS:
            continue
        if len(opens) == 1 and opens[0]["dir"] != side:
            continue

        entry = float(cl[i])
        lvl = float(interaction["level"])
        a1 = float(atr1[i]) if not np.isnan(atr1[i]) and atr1[i] > 0 else 10.0
        atr_sl = sl_pts_from_atr(a1, mult=SL_ATR_MULT, sl_min=SL_MIN, sl_max=SL_MAX)
        struct_sl = abs(entry - lvl) + 2.0
        use_sl = float(max(atr_sl, min(struct_sl, SL_MAX)))
        use_sl = float(np.clip(use_sl, SL_MIN, SL_MAX))
        sl = entry - use_sl if side > 0 else entry + use_sl

        used_tp_mode = "1R"
        tp_level_name = ""
        if tp_mode == "hybrid":
            tgt = next_target_level(side, entry, levels, stop_dist=use_sl)
            if tgt is not None:
                tname, tpx = tgt
                tp_dist = abs(tpx - entry)
                if tp_dist >= use_sl * 0.5:
                    tp = float(tpx)
                    used_tp_mode = "next_level"
                    tp_level_name = tname
                else:
                    tp_dist = use_sl * TP_R
                    tp = entry + tp_dist if side > 0 else entry - tp_dist
            else:
                tp_dist = use_sl * TP_R
                tp = entry + tp_dist if side > 0 else entry - tp_dist
        else:
            tp_dist = use_sl * TP_R
            tp = entry + tp_dist if side > 0 else entry - tp_dist

        opens.append({
            "ts": ts,
            "dir": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "win": win,
            "win_name": window_label(win),
            "sl_pts": use_sl,
            "variant": variant_id,
            "level_name": str(interaction.get("level_name") or ""),
            "tp_mode": used_tp_mode if not tp_level_name else f"{used_tp_mode}:{tp_level_name}",
        })
        fired_windows.add(win)
        counts[day] = counts.get(day, 0) + 1

    return recs


def stacks_to_recs(variant: str, stack_recs: List[stacks.TradeRec]) -> List[TradeRec]:
    return [
        TradeRec(r.trade, r.window, r.sl_pts, variant, level_name="", tp_mode="1R")
        for r in stack_recs
    ]


def responsive_to_recs(variant: str, resp_recs: List[responsive.TradeRec]) -> List[TradeRec]:
    return [
        TradeRec(r.trade, r.window, r.sl_pts, variant, level_name="", tp_mode="1R")
        for r in resp_recs
    ]


def print_row(r: Dict[str, Any]) -> None:
    o, oo = r["overall"], r["oos"]
    print(
        f"  {r['id']:<5} n={o['trades']:4d} WR={o['win_rate']:5.1f}% "
        f"PF={o['profit_factor']:5.2f} Exp=${o['expectancy']:7.2f} Net=${o['total_pnl']:8.1f} "
        f"DD=${o['max_drawdown']:7.1f} | "
        f"OOS n={oo['trades']:3d} Exp=${oo['expectancy']:7.2f} "
        f"PF={oo['profit_factor']:5.2f} DD=${oo['max_drawdown']:7.1f}"
    )


def recommend(ranked: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not ranked:
        return {"winner": None, "text": "No results.", "live": {}}
    w = ranked[0]
    # Prefer actionable live knobs from winner family
    live: Dict[str, Any] = {
        "winner_id": w["id"],
        "EMA15_LEVEL_MODE": "0",
        "notes": [],
    }
    wid = w["id"]
    if wid == "L0":
        live["notes"].append("Prefer classic ema15_eod stack (daily+15m+60m), not level mode.")
        live["SCALP_MODE"] = "ema15_eod"
        live["EMA15_LEVEL_MODE"] = "0"
    elif wid == "L6":
        live["notes"].append(
            "Prefer responsive 15m+5m quiet 1m with 1m ATR x1.5 clip 6-25; levels hurt OOS."
        )
        live["EMA15_LEVEL_MODE"] = "0"
        live["suggested_stop"] = "1m ATR x1.5 clip 6-25"
        level_only = [r for r in ranked if r["id"] not in ("L0", "L6")]
        if level_only:
            best_lvl = level_only[0]
            live["if_levels_anyway"] = {
                "id": best_lvl["id"],
                "soft_vwap": True,
                "tp_mode": "1R" if best_lvl["id"] == "L4b" else "hybrid",
                "level_keys": (
                    ["pdh", "pdl"] if best_lvl["id"] == "L5a" else list(LEVEL_KEYS_ALL)
                ),
                "morning_only": best_lvl["id"] == "L3",
                "note": (
                    f"{best_lvl['id']} best among level variants on OOS Exp "
                    f"${best_lvl['oos']['expectancy']:.2f} but still worse than L6; "
                    "do not flip live default to level mode."
                ),
            }
    else:
        live["EMA15_LEVEL_MODE"] = "1"
        live["SCALP_MODE"] = "level_15_5_1"
        if wid == "L1":
            live["soft_vwap"] = False
            live["notes"].append("Keep strict 15m structure+EMA+VWAP (as live).")
        else:
            live["soft_vwap"] = True
            live["notes"].append(
                "Use soft VWAP: structure+EMA for 15m direction; VWAP advisory for confidence only."
            )
            live["suggested_env"] = "EMA15_LEVEL_SOFT_VWAP=1 (if wired) or relax live bias"
        if wid == "L3":
            live["morning_only"] = True
            live["allowed_windows"] = ["9:35", "10:15", "11:00"]
            live["notes"].append("Restrict entries to morning windows only.")
        else:
            live["morning_only"] = False
        if wid == "L5a":
            live["level_keys"] = ["pdh", "pdl"]
            live["notes"].append("Interact only at PDH/PDL (drop ONH/ONL/RTH Open).")
        else:
            live["level_keys"] = list(LEVEL_KEYS_ALL)
        if wid == "L4b":
            live["tp_mode"] = "1R"
            live["notes"].append("Force 1R targets (skip next-level TP).")
        else:
            live["tp_mode"] = "hybrid"
            live["notes"].append("TP = next major level when practical, else 1R.")

    caveats = [
        "OOS = Jun 2026+ only; small n for level variants means ranking is noisy.",
        "Level interaction on historical 1m is path-dependent; live fills/slippage not modeled.",
        f"Costs fixed at ${PT}/pt + ${COMM} RT; no partial fills.",
        "Window re-entry after close matches live fired_windows discard.",
        "Soft VWAP is a backtest knob - live still uses strict bias unless code/env updated.",
        "L2 / L4a / L5b are identical rule sets (listed separately for the search table).",
    ]
    if w["oos"]["trades"] < 20:
        caveats.append(
            f"Winner OOS n={w['oos']['trades']} is thin - treat as exploratory, not production proof."
        )
    if w["oos"]["expectancy"] <= 0:
        caveats.append(
            "Best OOS expectancy is still <=0 - prefer paper / no auto-size-up until positive."
        )

    text = (
        f"Winner {w['id']} ({w['label']}): OOS Exp ${w['oos']['expectancy']:.2f}/tr, "
        f"PF {w['oos']['profit_factor']:.2f}, DD ${w['oos']['max_drawdown']:.0f}, "
        f"n={w['oos']['trades']}. "
        + " ".join(live.get("notes") or [])
    )
    if live.get("if_levels_anyway"):
        text += " " + live["if_levels_anyway"]["note"]
    return {"winner": w["id"], "text": text, "live": live, "caveats": caveats}


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "MNQ_1m.csv")
    out_json = os.path.join("data", "level_strategy_search.json")

    print("MNQ LEVEL STRATEGY SEARCH")
    print(f"Data: {csv}")
    print(
        f"Shared: windows={len(ENTRY_WINDOWS)}, conf>={MIN_CONF}, max {MAX_TRADES_DAY}/day, "
        f"re-entry, max {DEFAULT_MAX_OPEN_LOTS} lots, flatten 15:50, "
        f"${PT}/pt + ${COMM} RT, IS/OOS Jun 2026"
    )
    print(
        "Level stop: max(1m ATR×1.5, invalidation+2) clip 8–30; "
        "TP hybrid = next level else 1R"
    )

    t0 = time.time()
    df = official.load_1m(csv)
    print(f"bars={len(df):,}  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    reports: List[Dict[str, Any]] = []

    # L0 — baseline stacks A
    print("\n--- L0 baseline daily+15m+60m ---")
    recs0 = stacks_to_recs("L0", stacks.simulate_variant(df, variant="A"))
    r0 = full_report(
        "L0",
        "Baseline: daily+15m+60m + quiet 1m + 15m ATR×2 (no levels)",
        recs0,
    )
    reports.append(r0)
    print_row(r0)

    # L6 — responsive B + 1m atr1.5 6-25
    print("\n--- L6 responsive 15m+5m 1mATR×1.5 ---")
    stop_cfg = {
        "id": "1m_atr1.5_6_25",
        "label": "1m ATR×1.5 clip 6-25",
        "kind": "atr",
        "tf": "1m",
        "mult": 1.5,
        "min": 6.0,
        "max": 25.0,
    }
    recs6 = responsive_to_recs(
        "L6", responsive.simulate(df, variant="B", stop_cfg=stop_cfg)
    )
    r6 = full_report(
        "L6",
        "Responsive 15m+5m no levels + 1m ATR×1.5 clip 6–25",
        recs6,
    )
    reports.append(r6)
    print_row(r6)

    level_specs = [
        (
            "L1",
            "Level live: strict VWAP 15m + all levels + hybrid TP",
            dict(soft_vwap=False, level_keys=LEVEL_KEYS_ALL, tp_mode="hybrid"),
        ),
        (
            "L2",
            "Soft VWAP 15m + all levels + hybrid TP",
            dict(soft_vwap=True, level_keys=LEVEL_KEYS_ALL, tp_mode="hybrid"),
        ),
        (
            "L3",
            "L2 morning windows only (9:35/10:15/11:00)",
            dict(
                soft_vwap=True,
                level_keys=LEVEL_KEYS_ALL,
                tp_mode="hybrid",
                allowed_windows=MORNING_WINDOWS,
            ),
        ),
        (
            "L4a",
            "L2 + TP next major level (hybrid)",
            dict(soft_vwap=True, level_keys=LEVEL_KEYS_ALL, tp_mode="hybrid"),
        ),
        (
            "L4b",
            "L2 + TP 1R only",
            dict(soft_vwap=True, level_keys=LEVEL_KEYS_ALL, tp_mode="1r"),
        ),
        (
            "L5a",
            "L2 + PDH/PDL levels only",
            dict(soft_vwap=True, level_keys=LEVEL_KEYS_PD, tp_mode="hybrid"),
        ),
        (
            "L5b",
            "L2 + all majors (alias of L2)",
            dict(soft_vwap=True, level_keys=LEVEL_KEYS_ALL, tp_mode="hybrid"),
        ),
    ]

    print("\n--- Level variants L1-L5 ---")
    ctx = build_level_ctx(df)
    for vid, label, kwargs in level_specs:
        print(f"  simulating {vid}...")
        t1 = time.time()
        recs = simulate_level(ctx, variant_id=vid, **kwargs)
        rep = full_report(vid, label, recs)
        reports.append(rep)
        print_row(rep)
        print(f"    ({time.time() - t1:.1f}s, levels={rep.get('level_counts')})")

    # Deduplicate identical L2/L4a/L5b for ranking note but keep all rows
    ranked = rank_reports(reports)
    rec = recommend(ranked)

    print("\n" + "=" * 100)
    print("RANKED BY OOS EXPECTANCY -> PF -> DD")
    print("=" * 100)
    print(
        f"{'#':<3} {'ID':<5} {'OOS_n':>5} {'OOS_Exp':>9} {'OOS_PF':>7} {'OOS_DD':>9} "
        f"{'All_n':>5} {'All_Exp':>9}  label"
    )
    for i, r in enumerate(ranked, 1):
        print(
            f"{i:<3} {r['id']:<5} {r['oos']['trades']:5d} "
            f"${r['oos']['expectancy']:8.2f} {r['oos']['profit_factor']:7.2f} "
            f"${r['oos']['max_drawdown']:8.1f} "
            f"{r['overall']['trades']:5d} ${r['overall']['expectancy']:8.2f}  "
            f"{r['label'][:48]}"
        )

    print("\nRECOMMENDATION")
    print(f"  {rec['text']}")
    print("  Live knobs:", json.dumps(rec.get("live"), indent=2))
    print("  Caveats:")
    for c in rec.get("caveats") or []:
        print(f"    - {c}")

    assumptions = {
        "period": f"{df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}",
        "is_oos_split": "before Jun 2026 / Jun 2026+",
        "costs": f"${PT}/pt + ${COMM} RT",
        "max_trades_day": MAX_TRADES_DAY,
        "max_open_lots": DEFAULT_MAX_OPEN_LOTS,
        "windows": [window_label(i) for i in range(len(ENTRY_WINDOWS))],
        "flatten": "15:50 ET",
        "min_confidence": MIN_CONF,
        "window_reentry": "fired window cleared on position close (same as live/gapped)",
        "level_stop": (
            f"max(1m ATR×{SL_ATR_MULT}, |entry-level|+2) clipped [{SL_MIN},{SL_MAX}]"
        ),
        "soft_vwap": "structure+EMA set 15m side; VWAP still counts toward confidence %",
        "strict_vwap": "structure+EMA+VWAP must agree or no 15m side",
        "tp_hybrid": "next_target_level if dist>=0.5R and not too far, else 1R",
        "no_slippage": True,
        "l0_source": "backtest_ema15_timeframe_stacks variant A",
        "l6_source": "backtest_ema15_15_5_1_responsive variant B stop 1m_atr1.5_6_25",
        "note_l2_l4a_l5b": "L2, L4a, L5b share identical rules (listed for table clarity)",
    }

    payload = {
        "csv": csv,
        "generated_at": datetime.now(ET).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "assumptions": assumptions,
        "variants": reports,
        "ranked_ids": [r["id"] for r in ranked],
        "winner": ranked[0] if ranked else None,
        "recommendation": rec,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nJSON saved: {out_json}  ({time.time() - t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
