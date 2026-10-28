"""
Session reference levels for the day desk chart and level-mode entries.

Computed from 1m OHLC (ET):
  Must: PDH, PDL, ONH, ONL, RTH Open, VWAP, EMA10, EMA20, Volume
  Optional: ORH/ORL (first 15m), Day H/L, previous week close
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

from src.strategy.mnq_15m_ema_eod import (
    ET,
    _as_utc_frame,
    _et_minutes,
    _ohlc_resample,
    ema_series,
)

RTH_OPEN_MINUTE = 9 * 60 + 30
RTH_END_MINUTE = 16 * 60
GLOBEX_OPEN_MINUTE = 18 * 60
OR_MINUTES = 15  # opening range = first 15 RTH minutes
TOUCH_PTS_DEFAULT = 8.0
EMA_FAST_CHART = 10
EMA_SLOW_CHART = 20


def _round(v: Optional[float], n: int = 2) -> Optional[float]:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return None


def _et_index(tmp: pd.DataFrame) -> pd.DatetimeIndex:
    return tmp.index.tz_convert(ET)


def _rth_mask(et_idx: pd.DatetimeIndex) -> pd.Series:
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
    """Overnight window starts at prior 18:00 ET (Sun for Monday RTH)."""
    start_day = et_day - timedelta(days=1)
    while start_day.weekday() == 5:  # Saturday → Sunday open
        start_day -= timedelta(days=1)
    if start_day.weekday() == 6:  # Sunday
        pass
    return ET.localize(datetime(start_day.year, start_day.month, start_day.day, 18, 0))


def session_vwap_series(tmp: pd.DataFrame) -> pd.Series:
    """RTH-session VWAP reset each ET calendar day at 09:30."""
    et = _et_index(tmp)
    typical = (tmp["high"] + tmp["low"] + tmp["close"]) / 3.0
    vol = tmp["volume"].astype(float).fillna(0.0) if "volume" in tmp.columns else pd.Series(0.0, index=tmp.index)
    rth = _rth_mask(et)
    day_key = pd.Series(et.date, index=tmp.index)
    day_key = day_key.where(rth, other=None)
    tp_vol = typical * vol
    # groupby date only on RTH bars; non-RTH → NaN
    cum_pv = tp_vol.groupby(day_key).cumsum()
    cum_v = vol.groupby(day_key).cumsum().replace(0, np.nan)
    out = cum_pv / cum_v
    out = out.where(rth)
    return out


def compute_session_levels(
    df_1m: Optional[pd.DataFrame],
    *,
    now: Optional[datetime] = None,
    or_minutes: int = OR_MINUTES,
) -> Dict[str, Any]:
    """Return desk/chart levels + last EMA/VWAP values from 1m bars."""
    empty: Dict[str, Any] = {
        "pdh": None,
        "pdl": None,
        "onh": None,
        "onl": None,
        "rth_open": None,
        "vwap": None,
        "ema10": None,
        "ema20": None,
        "orh": None,
        "orl": None,
        "day_high": None,
        "day_low": None,
        "prev_week_close": None,
        "last_price": None,
        "volume": None,
        "asof_et": None,
        "labels": {},
    }
    if df_1m is None or getattr(df_1m, "empty", True):
        return empty
    tmp = _as_utc_frame(df_1m)
    if tmp.empty:
        return empty
    et = _et_index(tmp)
    if now is None:
        now_ts = tmp.index[-1]
    else:
        now_ts = pd.Timestamp(now)
        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC")
        else:
            now_ts = now_ts.tz_convert("UTC")
    now_et = now_ts.tz_convert(ET)
    et_day = now_et.date()
    prior = _prior_rth_day(et_day)

    # Previous RTH day H/L
    pdh = pdl = None
    if prior is not None:
        prior_mask = _rth_mask(et) & (pd.Series(et.date, index=tmp.index) == prior)
        prior_bars = tmp.loc[prior_mask]
        if not prior_bars.empty:
            pdh = float(prior_bars["high"].max())
            pdl = float(prior_bars["low"].min())

    # Overnight H/L: prior Globex 18:00 → today RTH open
    onh = onl = None
    globex_start = _globex_start_for_rth_day(et_day).astimezone(pytz.UTC)
    rth_open_dt = ET.localize(datetime(et_day.year, et_day.month, et_day.day, 9, 30)).astimezone(pytz.UTC)
    on_mask = (tmp.index >= globex_start) & (tmp.index < rth_open_dt)
    on_bars = tmp.loc[on_mask]
    if not on_bars.empty:
        onh = float(on_bars["high"].max())
        onl = float(on_bars["low"].min())

    # Today's RTH
    today_mask = _rth_mask(et) & (pd.Series(et.date, index=tmp.index) == et_day)
    today = tmp.loc[today_mask]
    rth_open = day_high = day_low = orh = orl = None
    vol_last = None
    if not today.empty:
        rth_open = float(today.iloc[0]["open"])
        day_high = float(today["high"].max())
        day_low = float(today["low"].min())
        or_end = rth_open_dt + timedelta(minutes=int(or_minutes))
        or_bars = today.loc[today.index < or_end]
        if not or_bars.empty:
            orh = float(or_bars["high"].max())
            orl = float(or_bars["low"].min())
        if "volume" in today.columns:
            vol_last = float(today["volume"].iloc[-1])

    # Previous week close = last RTH close on/before prior Friday
    prev_week_close = None
    week_start = et_day - timedelta(days=et_day.weekday())  # Monday of this week
    prev_week_end = week_start - timedelta(days=1)  # Sunday; walk back to Friday RTH
    for i in range(7):
        d = prev_week_end - timedelta(days=i)
        if d.weekday() >= 5:
            continue
        m = _rth_mask(et) & (pd.Series(et.date, index=tmp.index) == d)
        bars = tmp.loc[m]
        if not bars.empty:
            prev_week_close = float(bars["close"].iloc[-1])
            break

    vwap_s = session_vwap_series(tmp)
    ema10_s = ema_series(tmp["close"], EMA_FAST_CHART)
    ema20_s = ema_series(tmp["close"], EMA_SLOW_CHART)
    last = float(tmp["close"].iloc[-1])
    vwap = float(vwap_s.dropna().iloc[-1]) if vwap_s.dropna().size else None
    ema10 = float(ema10_s.iloc[-1]) if len(ema10_s) else None
    ema20 = float(ema20_s.iloc[-1]) if len(ema20_s) else None

    labels = {
        "pdh": "PDH",
        "pdl": "PDL",
        "onh": "ONH",
        "onl": "ONL",
        "rth_open": "RTH Open",
        "vwap": "VWAP",
        "ema10": "10 EMA",
        "ema20": "20 EMA",
        "orh": "ORH",
        "orl": "ORL",
        "day_high": "Day High",
        "day_low": "Day Low",
        "prev_week_close": "Prev Week Close",
    }
    return {
        "pdh": _round(pdh),
        "pdl": _round(pdl),
        "onh": _round(onh),
        "onl": _round(onl),
        "rth_open": _round(rth_open),
        "vwap": _round(vwap),
        "ema10": _round(ema10),
        "ema20": _round(ema20),
        "orh": _round(orh),
        "orl": _round(orl),
        "day_high": _round(day_high),
        "day_low": _round(day_low),
        "prev_week_close": _round(prev_week_close),
        "last_price": _round(last),
        "volume": _round(vol_last, 0),
        "asof_et": now_et.strftime("%Y-%m-%d %H:%M ET"),
        "labels": labels,
    }


def chart_series_payload(
    df_1m: Optional[pd.DataFrame],
    *,
    now: Optional[datetime] = None,
    lookback_bars: int = 390,
) -> Dict[str, Any]:
    """Bars + overlay series for the desk Lightweight Chart."""
    levels = compute_session_levels(df_1m, now=now)
    if df_1m is None or getattr(df_1m, "empty", True):
        return {"bars": [], "vwap": [], "ema10": [], "ema20": [], "levels": levels}
    tmp = _as_utc_frame(df_1m).tail(int(lookback_bars))
    et = _et_index(tmp)
    vwap_s = session_vwap_series(tmp)
    ema10_s = ema_series(tmp["close"], EMA_FAST_CHART)
    ema20_s = ema_series(tmp["close"], EMA_SLOW_CHART)
    bars: List[Dict[str, Any]] = []
    vwap_pts: List[Dict[str, Any]] = []
    e10: List[Dict[str, Any]] = []
    e20: List[Dict[str, Any]] = []
    for i, (ts, row) in enumerate(tmp.iterrows()):
        # Lightweight Charts wants unix seconds (UTC) for intraday
        t = int(pd.Timestamp(ts).tz_convert("UTC").timestamp())
        bars.append({
            "time": t,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]) if "volume" in row.index and not pd.isna(row["volume"]) else 0.0,
        })
        e10.append({"time": t, "value": float(ema10_s.iloc[i])})
        e20.append({"time": t, "value": float(ema20_s.iloc[i])})
        vv = vwap_s.iloc[i]
        if pd.notna(vv):
            vwap_pts.append({"time": t, "value": float(vv)})
    return {
        "bars": bars,
        "vwap": vwap_pts,
        "ema10": e10,
        "ema20": e20,
        "levels": levels,
        "et_hint": et[-1].strftime("%Y-%m-%d %H:%M ET") if len(et) else None,
    }


def major_level_map(levels: Dict[str, Any]) -> Dict[str, float]:
    """Named major levels used for entry interaction."""
    out: Dict[str, float] = {}
    for key in ("pdh", "pdl", "onh", "onl", "rth_open"):
        v = levels.get(key)
        if v is not None:
            try:
                out[key] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def nearest_major_level(
    price: float,
    levels: Dict[str, Any],
    *,
    touch_pts: float = TOUCH_PTS_DEFAULT,
) -> Optional[Tuple[str, float, float]]:
    """Return (name, level, distance) if price is within touch_pts of a major level."""
    best = None
    for name, lvl in major_level_map(levels).items():
        dist = abs(float(price) - lvl)
        if dist <= touch_pts and (best is None or dist < best[2]):
            best = (name, lvl, dist)
    return best


def next_target_level(
    side: int,
    entry: float,
    levels: Dict[str, Any],
    *,
    stop_dist: float,
) -> Optional[Tuple[str, float]]:
    """Next meaningful major level in trade direction; None if none practical."""
    majors = major_level_map(levels)
    # Prefer optional day extremes / OR as secondary targets too
    for key in ("orh", "orl", "day_high", "day_low", "prev_week_close"):
        v = levels.get(key)
        if v is not None:
            try:
                majors[key] = float(v)
            except (TypeError, ValueError):
                pass
    candidates: List[Tuple[str, float]] = []
    for name, lvl in majors.items():
        if side > 0 and lvl > entry + max(1.0, 0.35 * stop_dist):
            candidates.append((name, lvl))
        elif side < 0 and lvl < entry - max(1.0, 0.35 * stop_dist):
            candidates.append((name, lvl))
    if not candidates:
        return None
    if side > 0:
        candidates.sort(key=lambda x: x[1])
    else:
        candidates.sort(key=lambda x: -x[1])
    name, lvl = candidates[0]
    # Cap target stretch vs stop (avoid silly far targets)
    if abs(lvl - entry) > max(stop_dist * 3.0, 60.0):
        return None
    return name, lvl


def structure_side_from_ohlc(ohlc: pd.DataFrame, *, lookback: int = 6) -> Tuple[int, str]:
    """
    HH/HL → +1, LH/LL → -1, else mixed 0.
    Uses completed bars only (caller should pass shifted/completed frame).
    """
    if ohlc is None or len(ohlc) < max(4, lookback):
        return 0, "mixed"
    tail = ohlc.tail(lookback)
    highs = tail["high"].astype(float)
    lows = tail["low"].astype(float)
    # Simple swing: mid-window max/min vs earlier half
    mid = len(tail) // 2
    early_h, late_h = float(highs.iloc[:mid].max()), float(highs.iloc[mid:].max())
    early_l, late_l = float(lows.iloc[:mid].min()), float(lows.iloc[mid:].min())
    hh = late_h > early_h
    hl = late_l > early_l
    lh = late_h < early_h
    ll = late_l < early_l
    if hh and hl:
        return 1, "HH_HL"
    if lh and ll:
        return -1, "LH_LL"
    return 0, "mixed"


def bias_15m_5m(
    df_1m: pd.DataFrame,
    *,
    completed_only: bool = True,
) -> Dict[str, Any]:
    """15m structure + VWAP + EMA10/20; 5m EMA10/20 confirmation."""
    empty = {
        "side_15m": 0,
        "structure_15m": "mixed",
        "ema_side_15m": 0,
        "vwap_side_15m": 0,
        "side_5m": 0,
        "agree": False,
        "bias_15m": "mixed",
        "bias_5m": "flat",
    }
    if df_1m is None or getattr(df_1m, "empty", True) or len(df_1m) < 80:
        return empty
    tmp = _as_utc_frame(df_1m)
    b15 = _ohlc_resample(tmp, "15min")
    b5 = _ohlc_resample(tmp, "5min")
    if completed_only:
        b15 = b15.iloc[:-1] if len(b15) > 1 else b15.iloc[0:0]
        b5 = b5.iloc[:-1] if len(b5) > 1 else b5.iloc[0:0]
    if len(b15) < 6 or len(b5) < 4:
        return empty

    struct_side, struct_label = structure_side_from_ohlc(b15)
    e10_15 = ema_series(b15["close"], EMA_FAST_CHART)
    e20_15 = ema_series(b15["close"], EMA_SLOW_CHART)
    ema_side = 1 if float(e10_15.iloc[-1]) > float(e20_15.iloc[-1]) else (
        -1 if float(e10_15.iloc[-1]) < float(e20_15.iloc[-1]) else 0
    )
    vwap_s = session_vwap_series(tmp)
    last_close = float(tmp["close"].iloc[-1])
    vwap_val = float(vwap_s.dropna().iloc[-1]) if vwap_s.dropna().size else last_close
    vwap_side = 1 if last_close > vwap_val else (-1 if last_close < vwap_val else 0)

    # Combined 15m: structure + EMA10/20 + VWAP. Mixed structure → no trade.
    if struct_side == 0:
        side_15 = 0
    elif struct_side > 0 and ema_side >= 0 and vwap_side >= 0:
        side_15 = 1
    elif struct_side < 0 and ema_side <= 0 and vwap_side <= 0:
        side_15 = -1
    else:
        side_15 = 0  # structure vs EMA/VWAP conflict = mixed

    e10_5 = ema_series(b5["close"], EMA_FAST_CHART)
    e20_5 = ema_series(b5["close"], EMA_SLOW_CHART)
    side_5 = 1 if float(e10_5.iloc[-1]) > float(e20_5.iloc[-1]) else (
        -1 if float(e10_5.iloc[-1]) < float(e20_5.iloc[-1]) else 0
    )
    # Soft structure confirm on 5m
    s5, _ = structure_side_from_ohlc(b5, lookback=4)
    if s5 != 0 and side_5 != 0 and s5 != side_5:
        side_5 = 0  # 5m mixed / disagree internally

    agree = side_15 != 0 and side_5 == side_15
    return {
        "side_15m": int(side_15),
        "structure_15m": struct_label,
        "ema_side_15m": int(ema_side),
        "vwap_side_15m": int(vwap_side),
        "ema10_15m": _round(float(e10_15.iloc[-1])),
        "ema20_15m": _round(float(e20_15.iloc[-1])),
        "side_5m": int(side_5),
        "agree": bool(agree),
        "bias_15m": "long" if side_15 > 0 else ("short" if side_15 < 0 else "mixed"),
        "bias_5m": "long" if side_5 > 0 else ("short" if side_5 < 0 else "flat"),
        "vwap": _round(vwap_val),
        "last": _round(last_close),
    }


def level_interaction(
    df_1m: pd.DataFrame,
    side: int,
    levels: Dict[str, Any],
    *,
    touch_pts: float = TOUCH_PTS_DEFAULT,
    lookback: int = 12,
) -> Optional[Dict[str, Any]]:
    """
    Detect break+retest or rejection at a major level in `side` direction.
    Returns interaction dict or None.
    """
    if side == 0 or df_1m is None or len(df_1m) < 5:
        return None
    tmp = _as_utc_frame(df_1m)
    recent = tmp.tail(min(len(tmp), lookback + 2))
    majors = major_level_map(levels)
    if not majors:
        return None
    last = recent.iloc[-1]
    px = float(last["close"])
    hi = float(recent["high"].max())
    lo = float(recent["low"].min())

    for name, lvl in majors.items():
        near_now = abs(px - lvl) <= touch_pts
        touched = (recent["low"].astype(float).min() - touch_pts) <= lvl <= (recent["high"].astype(float).max() + touch_pts)
        if not touched:
            continue
        # Break: price traded through level in direction
        broke_up = hi > lvl + 0.25 and any(
            float(recent.iloc[i]["close"]) > lvl and float(recent.iloc[i - 1]["close"]) <= lvl
            for i in range(1, len(recent))
        )
        broke_dn = lo < lvl - 0.25 and any(
            float(recent.iloc[i]["close"]) < lvl and float(recent.iloc[i - 1]["close"]) >= lvl
            for i in range(1, len(recent))
        )
        # Retest after break: came back to level and held
        retest_long = broke_up and float(recent["low"].iloc[-3:].min()) <= lvl + touch_pts and px >= lvl - 0.5
        retest_short = broke_dn and float(recent["high"].iloc[-3:].max()) >= lvl - touch_pts and px <= lvl + 0.5
        # Rejection: touch and close back with side (support/resistance)
        reject_long = near_now and float(last["low"]) <= lvl + touch_pts and px > lvl and side > 0
        reject_short = near_now and float(last["high"]) >= lvl - touch_pts and px < lvl and side < 0

        if side > 0 and (retest_long or reject_long or (broke_up and near_now and px >= lvl)):
            kind = "retest" if retest_long else ("rejection" if reject_long else "break")
            return {"level_name": name, "level": _round(lvl), "kind": kind, "side": "long"}
        if side < 0 and (retest_short or reject_short or (broke_dn and near_now and px <= lvl)):
            kind = "retest" if retest_short else ("rejection" if reject_short else "break")
            return {"level_name": name, "level": _round(lvl), "kind": kind, "side": "short"}
    return None


def one_min_level_confirm(
    df_1m: pd.DataFrame,
    side: int,
    interaction: Dict[str, Any],
    *,
    max_body: float = 4.0,
) -> bool:
    """1m break/retest/hold confirmation — not a mid-range EMA cross."""
    if side == 0 or not interaction or df_1m is None or len(df_1m) < 3:
        return False
    tmp = _as_utc_frame(df_1m)
    row = tmp.iloc[-1]
    prev = tmp.iloc[-2]
    o, c = float(row["open"]), float(row["close"])
    body = abs(c - o)
    if body > max_body:
        return False
    lvl = float(interaction["level"])
    kind = str(interaction.get("kind") or "")
    if side > 0:
        if c < o:  # need with-trend close
            return False
        if kind == "break":
            return c > lvl and float(prev["close"]) <= lvl + 0.5
        # retest / rejection / hold: close holds above level
        return c >= lvl - 0.25 and float(row["low"]) <= lvl + 8.0
    if c > o:
        return False
    if kind == "break":
        return c < lvl and float(prev["close"]) >= lvl - 0.5
    return c <= lvl + 0.25 and float(row["high"]) >= lvl - 8.0


def desk_levels_block(
    df_1m: Optional[pd.DataFrame],
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """bot_status / desk payload: levels + 15m/5m bias."""
    levels = compute_session_levels(df_1m, now=now)
    bias = bias_15m_5m(df_1m, completed_only=True) if df_1m is not None else {
        "bias_15m": "mixed", "bias_5m": "flat", "agree": False, "structure_15m": "mixed",
    }
    return {
        "chart_levels": levels,
        "bias_15m": bias.get("bias_15m"),
        "bias_5m": bias.get("bias_5m"),
        "structure_15m": bias.get("structure_15m"),
        "bias_agree": bool(bias.get("agree")),
        "bias": bias,
    }
