"""
15-minute EMA trend, up to six MNQ RTH slots, hold to 15:50 ET.

Uses only *completed* bars (no same-day daily close, no unfinished 15m bucket).

Every entry needs daily permission + completed 15m EMA8/21 + completed 60m EMA8/21
in the same direction. Afternoon windows and a 2nd lot still need sep/ATR >= 0.45.

  SL = clip(completed 15m ATR14 * 2.0, 20, 60)
    ~1.8 fills/RTH day on the prior recipe (6/day is not available on this edge).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

ET = pytz.timezone("US/Eastern")

EMA_FAST = 8
EMA_SLOW = 21
DAILY_EMA = 20
ENTRY_MINUTE = 9 * 60 + 35
FLAT_MINUTE = 15 * 60 + 50
SESSION_END = 16 * 60
# Six RTH slots. Every entry needs daily+15m+60m; noon+ and 2nd lot still need sep.
ENTRY_WINDOWS = (
    (9 * 60 + 35, 10 * 60),
    (10 * 60 + 15, 10 * 60 + 40),
    (11 * 60, 11 * 60 + 30),
    (12 * 60, 12 * 60 + 30),
    (13 * 60 + 30, 14 * 60),
    (14 * 60 + 30, 15 * 60),
)
MAX_TRADES_DAY = 6
QUALITY_AFTER_MINUTE = 12 * 60  # 12:00 / 13:30 / 14:30 require high_confidence
SEP_MIN = 0.45  # |EMA8-EMA21| / ATR15 for a 2nd concurrent lot or afternoon window
SL_ATR_MULT = 2.0
SL_MIN = 20.0
SL_MAX = 60.0
TP_PTS = 500.0
ADX_MIN = 0.0
FLATTEN_ET = "15:50"
LOCKED_RULES_PARAGRAPH = (
    "Locked ema15_eod: RTH only, entries only in 9:35 / 10:15 / 11:00 / 12:00 / 13:30 / 14:30 ET. "
    "Enter only when daily + 15m + 60m agree: completed 15m EMA8 vs EMA21, prior RTH close vs "
    "daily EMA20, and completed 60m EMA8 vs EMA21, all the same direction. "
    "From 12:00 ET, also need 15m sep/ATR >= 0.45. Second lot needs that sep gate too. "
    "Stop = clip(completed 15m ATR14 × 2, 20–60 pts). TP is a wide 500-pt cap, not a target. "
    "Flatten 15:50 ET. Do not trail. Overnight is off. Gemini is a second opinion only — "
    "it must not place orders, change stops, change windows, or rewrite mnq_profit_config."
)


def _as_utc_frame(df_1m: pd.DataFrame) -> pd.DataFrame:
    tmp = df_1m.copy()
    if "datetime" in tmp.columns:
        tmp = tmp.copy()
        tmp["datetime"] = parse_mnq_1m_datetime(tmp["datetime"])
        tmp = tmp.dropna(subset=["datetime"]).set_index("datetime")
    elif not isinstance(tmp.index, pd.DatetimeIndex):
        raise ValueError("df_1m needs datetime column or DatetimeIndex")
    elif tmp.index.tz is None:
        tmp.index = tmp.index.tz_localize("UTC")
    else:
        tmp.index = tmp.index.tz_convert("UTC")
    return tmp


def _et_minutes(ts) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    et = t.tz_convert(ET)
    return int(et.hour * 60 + et.minute)


def _et_date(ts) -> date:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(ET).date()


def ema_series(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def _ohlc_resample(tmp: pd.DataFrame, rule: str) -> pd.DataFrame:
    return tmp.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()


def _atr(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        ohlc["high"] - ohlc["low"],
        (ohlc["high"] - ohlc["close"].shift()).abs(),
        (ohlc["low"] - ohlc["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def trend_15m_on_1m(df_1m: pd.DataFrame, *, completed_only: bool = True) -> pd.Series:
    """15m EMA8 vs EMA21 on 1m index. completed_only uses the last finished 15m bar."""
    tmp = _as_utc_frame(df_1m)
    b15 = _ohlc_resample(tmp, "15min")
    fast = ema_series(b15["close"], EMA_FAST)
    slow = ema_series(b15["close"], EMA_SLOW)
    trend = pd.Series(0, index=b15.index, dtype=int)
    trend.loc[fast > slow] = 1
    trend.loc[fast < slow] = -1
    if completed_only:
        trend = trend.shift(1)
    return trend.reindex(tmp.index, method="ffill").fillna(0).astype(int)


def daily_trend_on_1m(df_1m: pd.DataFrame) -> pd.Series:
    """Prior *RTH session* close vs EMA20 (ET calendar, no same-day leak)."""
    tmp = _as_utc_frame(df_1m)
    et = tmp.index.tz_convert(ET)
    mins = et.hour * 60 + et.minute
    rth = (et.weekday < 5) & (mins >= 9 * 60 + 30) & (mins < 16 * 60)
    if not rth.any():
        return pd.Series(0, index=tmp.index, dtype=int)
    sess = tmp.loc[rth].copy()
    sess.index = et[rth]
    closes = sess.groupby(sess.index.date)["close"].last()
    e20 = ema_series(closes, DAILY_EMA)
    trend = pd.Series(0, index=closes.index, dtype=int)
    trend.loc[closes > e20] = 1
    trend.loc[closes < e20] = -1
    prior = trend.shift(1)
    lookup = {d: int(v) if pd.notna(v) else 0 for d, v in prior.items()}
    values = [lookup.get(d, 0) for d in et.date]
    return pd.Series(values, index=tmp.index, dtype=int)


def trend_60m_on_1m(df_1m: pd.DataFrame, *, completed_only: bool = True) -> pd.Series:
    tmp = _as_utc_frame(df_1m)
    b60 = _ohlc_resample(tmp, "60min")
    fast = ema_series(b60["close"], EMA_FAST)
    slow = ema_series(b60["close"], EMA_SLOW)
    trend = pd.Series(0, index=b60.index, dtype=int)
    trend.loc[fast > slow] = 1
    trend.loc[fast < slow] = -1
    if completed_only:
        trend = trend.shift(1)
    return trend.reindex(tmp.index, method="ffill").fillna(0).astype(int)


def sep15_on_1m(df_1m: pd.DataFrame) -> pd.Series:
    tmp = _as_utc_frame(df_1m)
    b15 = _ohlc_resample(tmp, "15min")
    fast = ema_series(b15["close"], EMA_FAST)
    slow = ema_series(b15["close"], EMA_SLOW)
    atr = _atr(b15, 14)
    sep = (fast - slow).abs() / atr.replace(0, np.nan)
    return sep.shift(1).reindex(tmp.index, method="ffill")


def high_confidence(side: int, trend60: int, sep: float, sep_min: float = SEP_MIN) -> bool:
    if side == 0 or trend60 != side:
        return False
    if sep is None or (isinstance(sep, float) and (np.isnan(sep) or sep < sep_min)):
        return False
    return True


def atr15_on_1m(df_1m: pd.DataFrame) -> pd.Series:
    tmp = _as_utc_frame(df_1m)
    b15 = _ohlc_resample(tmp, "15min")
    atr = _atr(b15, 14).shift(1)
    return atr.reindex(tmp.index, method="ffill")


def parse_mnq_1m_datetime(values) -> pd.Series:
    """Parse mixed 1m timestamps. Coerce 2-digit years (26-05-27 → 2026-05-27). Never raise."""
    def _one(val) -> pd.Timestamp:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return pd.NaT
        if isinstance(val, pd.Timestamp):
            t = val
            if t.tzinfo is None:
                return t.tz_localize("UTC")
            return t.tz_convert("UTC")
        text = str(val).strip()
        if not text or text.lower() in ("nan", "nat", "none"):
            return pd.NaT
        if re.match(r"^\d{2}-\d{2}-\d{2}[ T]", text):
            text = "20" + text
        try:
            t = pd.to_datetime(text, utc=True, format="mixed", errors="coerce")
        except (TypeError, ValueError, OSError):
            try:
                t = pd.to_datetime(text, utc=True, errors="coerce")
            except Exception:
                return pd.NaT
        if pd.isna(t):
            return pd.NaT
        return t

    try:
        s = pd.Series(values, dtype="object").astype(str).str.strip()
        s = s.str.replace(r"^(\d{2})-(\d{2})-(\d{2})([ T])", r"20\1-\2-\3\4", regex=True)
        try:
            parsed = pd.to_datetime(s, utc=True, format="mixed", errors="coerce")
        except (TypeError, ValueError, OSError):
            parsed = None
        if parsed is None or not isinstance(parsed, pd.Series):
            parsed = pd.Series([_one(v) for v in values], dtype="datetime64[ns, UTC]")
        return parsed
    except Exception:
        try:
            seq = list(values)
        except TypeError:
            seq = [values]
        return pd.Series([_one(v) for v in seq], dtype="datetime64[ns, UTC]")


def load_1m_seed_csv(path: str) -> Optional[pd.DataFrame]:
    """Local Databento (or other) 1m history so daily EMA20 is real on live start."""
    if not path or not os.path.isfile(path):
        return None
    try:
        raw = pd.read_csv(path)
    except Exception:
        return None
    if "datetime" not in raw.columns:
        return None
    raw = raw.copy()
    try:
        raw["datetime"] = parse_mnq_1m_datetime(raw["datetime"])
    except Exception:
        return None
    raw = raw.dropna(subset=["datetime"])
    if raw.empty:
        return None
    return raw.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)


def merge_1m_history(seed: Optional[pd.DataFrame], live: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if live is None or live.empty:
        return seed
    if seed is None or seed.empty:
        return live
    a = seed.copy()
    b = live.copy()
    a["datetime"] = parse_mnq_1m_datetime(a["datetime"])
    b["datetime"] = parse_mnq_1m_datetime(b["datetime"])
    a = a.dropna(subset=["datetime"])
    b = b.dropna(subset=["datetime"])
    out = pd.concat([a, b], ignore_index=True)
    return out.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)


def overlay_ticker_last_on_1m(
    df: Optional[pd.DataFrame],
    last: Optional[float],
    now: Optional[datetime] = None,
) -> Optional[pd.DataFrame]:
    """Stamp Rithmic ticker last onto the current 1m clock.

    Same minute: update OHLC. Later minute (including hours-stale CSV): append
    ONE live minute bar. Does not fill the gap with invented bars.
    """
    if df is None or df.empty or last is None:
        return df
    last_px = float(last)
    if last_px <= 0:
        return df
    now_ts = pd.Timestamp(now or datetime.now(pytz.UTC))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    minute = now_ts.floor("min")
    out = df.copy()
    out["datetime"] = parse_mnq_1m_datetime(out["datetime"])
    out = out.dropna(subset=["datetime"]).sort_values("datetime")
    if out.empty:
        return df
    row = out.iloc[-1]
    ts = pd.Timestamp(row["datetime"])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    ts_min = ts.floor("min")
    if ts_min == minute:
        idx = out.index[-1]
        out.loc[idx, "close"] = last_px
        out.loc[idx, "high"] = max(float(row["high"]), last_px)
        out.loc[idx, "low"] = min(float(row["low"]), last_px)
        return out.reset_index(drop=True)
    if minute > ts_min:
        new = {col: row[col] for col in out.columns}
        prev_close = float(row["close"])
        new["datetime"] = minute
        new["open"] = prev_close
        new["high"] = max(prev_close, last_px)
        new["low"] = min(prev_close, last_px)
        new["close"] = last_px
        if "volume" in out.columns:
            new["volume"] = 0
        out = pd.concat([out, pd.DataFrame([new])], ignore_index=True)
    return out.reset_index(drop=True)


def _side_word(side: int) -> str:
    if int(side) > 0:
        return "long"
    if int(side) < 0:
        return "short"
    return "flat"


def _et_hhmm(ts) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(ET).strftime("%H:%M")


def sl_pts_from_atr(
    atr15: float,
    *,
    mult: float = SL_ATR_MULT,
    sl_min: float = SL_MIN,
    sl_max: float = SL_MAX,
    fallback: float = 40.0,
) -> float:
    if atr15 is None or (isinstance(atr15, float) and (np.isnan(atr15) or atr15 <= 0)):
        return fallback
    return float(np.clip(float(atr15) * mult, sl_min, sl_max))


def window_index(et_minutes: int) -> Optional[int]:
    for i, (start, end) in enumerate(ENTRY_WINDOWS):
        if start <= et_minutes < end:
            return i
    return None


def window_requires_quality(win: Optional[int]) -> bool:
    """Afternoon extra slots need 60m+sep even when flat."""
    if win is None or win < 0 or win >= len(ENTRY_WINDOWS):
        return False
    return ENTRY_WINDOWS[win][0] >= QUALITY_AFTER_MINUTE


def window_label(win: Optional[int]) -> str:
    """Human window name from ENTRY_WINDOWS start (does not change the windows)."""
    if win is None or win < 0 or win >= len(ENTRY_WINDOWS):
        return "none"
    start = ENTRY_WINDOWS[win][0]
    return f"{start // 60:02d}:{start % 60:02d}"


def next_window_info(et_minutes: int) -> Dict[str, Any]:
    """Current locked window, or the next ENTRY_WINDOWS slot (does not invent windows)."""
    mins = int(et_minutes)
    win = window_index(mins)
    if win is not None:
        _start, end = ENTRY_WINDOWS[win]
        name = window_label(win)
        return {
            "in_window": True,
            "window_index": win,
            "window_name": name,
            "window_end": f"{end // 60:02d}:{end % 60:02d}",
            "next_window_name": name,
            "next_window_start": name,
            "minutes_until_next": 0,
            "next_is_tomorrow": False,
        }
    for i, (start, _end) in enumerate(ENTRY_WINDOWS):
        if mins < start:
            name = window_label(i)
            return {
                "in_window": False,
                "window_index": None,
                "window_name": "none",
                "window_end": None,
                "next_window_name": name,
                "next_window_start": name,
                "minutes_until_next": int(start - mins),
                "next_is_tomorrow": False,
            }
    first = ENTRY_WINDOWS[0][0]
    name = window_label(0)
    until = (24 * 60 - mins) + first
    return {
        "in_window": False,
        "window_index": None,
        "window_name": "none",
        "window_end": None,
        "next_window_name": name,
        "next_window_start": name,
        "minutes_until_next": int(until),
        "next_is_tomorrow": True,
    }


def minutes_into_window(et_minutes: int, win: Optional[int] = None) -> Optional[int]:
    if win is None:
        win = window_index(et_minutes)
    if win is None:
        return None
    return int(et_minutes - ENTRY_WINDOWS[win][0])


def _session_label(ts) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    et = t.tz_convert(ET)
    mins = int(et.hour * 60 + et.minute)
    if int(et.weekday()) >= 5:
        return "weekend"
    if 9 * 60 + 30 <= mins < 16 * 60:
        return "RTH"
    return "closed"


def _tf_state(df_1m: pd.DataFrame, rule: str) -> Dict[str, Any]:
    """Last completed-bar EMA8/21 (+ ATR on 15m) aligned to the latest 1m stamp."""
    tmp = _as_utc_frame(df_1m)
    bars = _ohlc_resample(tmp, rule)
    empty = {"ema_fast": None, "ema_slow": None, "atr": None, "trend": 0, "sep": None}
    if bars.empty or tmp.empty:
        return empty
    fast = ema_series(bars["close"], EMA_FAST)
    slow = ema_series(bars["close"], EMA_SLOW)
    atr = _atr(bars, 14)
    fast_s = fast.shift(1).reindex(tmp.index, method="ffill")
    slow_s = slow.shift(1).reindex(tmp.index, method="ffill")
    atr_s = atr.shift(1).reindex(tmp.index, method="ffill")
    f = float(fast_s.iloc[-1]) if pd.notna(fast_s.iloc[-1]) else None
    s = float(slow_s.iloc[-1]) if pd.notna(slow_s.iloc[-1]) else None
    a = float(atr_s.iloc[-1]) if pd.notna(atr_s.iloc[-1]) else None
    trend = 0
    if f is not None and s is not None:
        if f > s:
            trend = 1
        elif f < s:
            trend = -1
    sep = None
    if f is not None and s is not None and a is not None and a > 0:
        sep = abs(f - s) / a
    return {
        "ema_fast": None if f is None else round(f, 2),
        "ema_slow": None if s is None else round(s, 2),
        "atr": None if a is None else round(a, 2),
        "trend": trend,
        "sep": None if sep is None else round(float(sep), 3),
    }


def daily_permission_snapshot(df_1m: pd.DataFrame) -> Dict[str, Any]:
    """Prior RTH close vs EMA20 (same completed-session rule as daily_trend_on_1m)."""
    out = {
        "daily_permission": 0,
        "prior_rth_close": None,
        "daily_ema20": None,
    }
    if df_1m is None or df_1m.empty:
        return out
    tmp = _as_utc_frame(df_1m)
    et = tmp.index.tz_convert(ET)
    mins = et.hour * 60 + et.minute
    rth = (et.weekday < 5) & (mins >= 9 * 60 + 30) & (mins < 16 * 60)
    if not rth.any():
        out["daily_permission"] = int(daily_trend_on_1m(df_1m).iloc[-1])
        return out
    sess = tmp.loc[rth].copy()
    sess.index = et[rth]
    closes = sess.groupby(sess.index.date)["close"].last()
    e20 = ema_series(closes, DAILY_EMA)
    day = et[-1].date()
    prior_dates = [d for d in closes.index if d < day]
    if not prior_dates:
        out["daily_permission"] = int(daily_trend_on_1m(df_1m).iloc[-1])
        return out
    prior_d = prior_dates[-1]
    prior_close = float(closes.loc[prior_d])
    ema20 = float(e20.loc[prior_d]) if pd.notna(e20.loc[prior_d]) else None
    perm = 0
    if ema20 is not None:
        if prior_close > ema20:
            perm = 1
        elif prior_close < ema20:
            perm = -1
    out["daily_permission"] = perm
    out["prior_rth_close"] = round(prior_close, 2)
    out["daily_ema20"] = None if ema20 is None else round(ema20, 2)
    return out


def capture_market_snapshot(
    df_1m: pd.DataFrame,
    *,
    now: Optional[datetime] = None,
    side: int = 0,
    atr_stop_pts: Optional[float] = None,
    window: Optional[int] = None,
    sep_min: float = SEP_MIN,
) -> Dict[str, Any]:
    """Market facts at a point in time for the trade journal (logging only)."""
    empty = {
        "ts_et": None,
        "session": None,
        "window_name": "none",
        "window": window,
        "minutes_into_window": None,
        "daily_permission": 0,
        "prior_rth_close": None,
        "daily_ema20": None,
        "ema15_fast": None,
        "ema15_slow": None,
        "trend_15m": 0,
        "atr15": None,
        "ema60_fast": None,
        "ema60_slow": None,
        "trend_60m": 0,
        "sep15": None,
        "price_1m": None,
        "price_vs_ema15_fast": None,
        "price_vs_ema15_slow": None,
        "atr_stop_pts": None if atr_stop_pts is None else round(float(atr_stop_pts), 2),
        "daily_agree": False,
        "tf15_agree": False,
        "tf60_agree": False,
        "high_confidence": False,
    }
    if df_1m is None or df_1m.empty:
        return empty
    row = df_1m.iloc[-1]
    ts = now or row.get("datetime")
    if ts is None:
        ts = datetime.now(pytz.UTC)
    mins = _et_minutes(ts)
    win = window if window is not None else window_index(mins)
    t15 = _tf_state(df_1m, "15min")
    t60 = _tf_state(df_1m, "60min")
    daily = daily_permission_snapshot(df_1m)
    price = float(row["close"])
    ema_f = t15.get("ema_fast")
    ema_s = t15.get("ema_slow")
    sep = t15.get("sep")
    if sep is None:
        try:
            sep_v = float(sep15_on_1m(df_1m).iloc[-1])
            sep = None if np.isnan(sep_v) else round(sep_v, 3)
        except Exception:
            sep = None
    trend15 = int(t15.get("trend") or 0)
    trend60 = int(t60.get("trend") or 0)
    perm = int(daily.get("daily_permission") or 0)
    stop = atr_stop_pts
    if stop is None and t15.get("atr") is not None:
        stop = sl_pts_from_atr(float(t15["atr"]))
    confident = high_confidence(side, trend60, float(sep) if sep is not None else float("nan"), sep_min)
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    ts_et = t.tz_convert(ET).isoformat()
    return {
        "ts_et": ts_et,
        "session": _session_label(ts),
        "window_name": window_label(win),
        "window": win,
        "minutes_into_window": minutes_into_window(mins, win),
        "daily_permission": perm,
        "prior_rth_close": daily.get("prior_rth_close"),
        "daily_ema20": daily.get("daily_ema20"),
        "ema15_fast": ema_f,
        "ema15_slow": ema_s,
        "trend_15m": trend15,
        "atr15": t15.get("atr"),
        "ema60_fast": t60.get("ema_fast"),
        "ema60_slow": t60.get("ema_slow"),
        "trend_60m": trend60,
        "sep15": sep,
        "price_1m": round(price, 2),
        "price_vs_ema15_fast": None if ema_f is None else round(price - float(ema_f), 2),
        "price_vs_ema15_slow": None if ema_s is None else round(price - float(ema_s), 2),
        "atr_stop_pts": None if stop is None else round(float(stop), 2),
        "daily_agree": bool(side and perm == side),
        "tf15_agree": bool(side and trend15 == side),
        "tf60_agree": bool(side and trend60 == side),
        "high_confidence": bool(confident),
    }


@dataclass
class Ema15EodState:
    taken_date: Optional[date] = None
    decided_date: Optional[date] = None
    day: Optional[date] = None
    trades_today: int = 0
    fired_windows: tuple = ()


def _reset_day(state: Ema15EodState, day: date) -> Ema15EodState:
    if state.day == day:
        return state
    state.day = day
    state.trades_today = 0
    state.fired_windows = ()
    state.taken_date = None
    state.decided_date = None
    return state


@dataclass
class Ema15EodConfig:
    require_daily: bool = True
    require_60m: bool = True
    sl_atr_mult: float = SL_ATR_MULT
    sl_min: float = SL_MIN
    sl_max: float = SL_MAX
    tp_pts: float = TP_PTS
    adx_min: float = ADX_MIN
    completed_15m: bool = True


def check_ema15_eod_entry(
    symbol: str,
    df_1m: pd.DataFrame,
    state: Ema15EodState,
    *,
    now: Optional[datetime] = None,
    sl_pts: Optional[float] = None,
    tp_pts: float = TP_PTS,
    require_daily: bool = True,
    require_60m: bool = True,
    adx_5m: Optional[float] = None,
    adx_min: float = ADX_MIN,
    sl_atr_mult: float = SL_ATR_MULT,
    sl_min: float = SL_MIN,
    sl_max: float = SL_MAX,
    completed_15m: bool = True,
    max_trades_day: int = MAX_TRADES_DAY,
    open_count: int = 0,
    open_direction: Optional[str] = None,
    sep_min: float = SEP_MIN,
) -> Tuple[Optional[Dict[str, Any]], Ema15EodState]:
    if df_1m is None or df_1m.empty or len(df_1m) < 80:
        return None, state
    row = df_1m.iloc[-1]
    ts = now or row.get("datetime")
    if ts is None:
        ts = datetime.now(pytz.UTC)
    mins = _et_minutes(ts)
    day = _et_date(ts)
    state = _reset_day(state, day)
    if mins < ENTRY_MINUTE or mins >= FLAT_MINUTE:
        return None, state
    if state.trades_today >= max_trades_day:
        return None, state
    win = window_index(mins)
    if win is None or win in state.fired_windows:
        return None, state

    side = int(trend_15m_on_1m(df_1m, completed_only=completed_15m).iloc[-1])
    if side == 0:
        return None, state
    if require_daily:
        daily = int(daily_trend_on_1m(df_1m).iloc[-1])
        if daily != side:
            return None, state
    trend60 = int(trend_60m_on_1m(df_1m).iloc[-1])
    if require_60m and trend60 != side:
        return None, state
    if adx_min > 0:
        adx_val = adx_5m
        if adx_val is None and "adx" in row.index:
            adx_val = row.get("adx")
        if adx_val is None or pd.isna(adx_val) or float(adx_val) < adx_min:
            return None, state

    atr15 = float(atr15_on_1m(df_1m).iloc[-1])
    use_sl = sl_pts if sl_pts is not None else sl_pts_from_atr(
        atr15, mult=sl_atr_mult, sl_min=sl_min, sl_max=sl_max,
    )
    entry = float(row["close"])
    direction = "long" if side > 0 else "short"
    sep = float(sep15_on_1m(df_1m).iloc[-1])
    confident = high_confidence(side, trend60, sep, sep_min)
    if window_requires_quality(win) and not confident:
        return None, state
    if open_count >= 2:
        return None, state
    if open_count >= 1:
        if not confident:
            return None, state
        if open_direction and open_direction != direction:
            return None, state
    sl = entry - use_sl if side > 0 else entry + use_sl
    tp = entry + tp_pts if side > 0 else entry - tp_pts
    state.taken_date = day
    state.decided_date = day
    state.trades_today += 1
    state.fired_windows = tuple(sorted(set(state.fired_windows) | {win}))
    return {
        "symbol": symbol,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "atr": atr15 if not np.isnan(atr15) else use_sl,
        "scalp_mode": "ema15_eod",
        "scalp_hybrid": False,
        "structure_capped": False,
        "entry_meta": {
            "entry_mode": "ema15_eod",
            "trend_15m": side,
            "daily_trend": int(daily_trend_on_1m(df_1m).iloc[-1]) if require_daily else 0,
            "sl_pts": round(use_sl, 2),
            "atr15": None if np.isnan(atr15) else round(atr15, 2),
            "et_date": str(day),
            "trend_60m": trend60,
            "require_60m": require_60m,
            "sep15": None if np.isnan(sep) else round(sep, 3),
            "high_confidence": confident,
            "add_on": open_count >= 1,
            "window": win,
            "window_name": window_label(win),
        },
    }, state


def explain_ema15_skip(
    df_1m: Optional[pd.DataFrame],
    state: Ema15EodState,
    *,
    now: Optional[datetime] = None,
    require_daily: bool = True,
    require_60m: bool = True,
    adx_5m: Optional[float] = None,
    adx_min: float = ADX_MIN,
    completed_15m: bool = True,
    max_trades_day: int = MAX_TRADES_DAY,
    open_count: int = 0,
    open_direction: Optional[str] = None,
    sep_min: float = SEP_MIN,
) -> str:
    """Human reason the locked recipe will not enter on this bar. Does not mutate state."""
    n = 0 if df_1m is None or df_1m.empty else len(df_1m)
    if df_1m is None or df_1m.empty or n < 80:
        return (
            f"Need more 1m history (have {n}, need 80) so daily EMA20 is real — "
            "check data/MNQ_1m.csv / Databento"
        )
    row = df_1m.iloc[-1]
    ts = now or row.get("datetime")
    if ts is None:
        ts = datetime.now(pytz.UTC)
    mins = _et_minutes(ts)
    clock = _et_hhmm(ts)
    day = _et_date(ts)
    fired = state.fired_windows if state.day == day else ()
    trades_today = state.trades_today if state.day == day else 0
    info = next_window_info(mins)
    if mins >= FLAT_MINUTE:
        return f"After flatten 15:50 ET — no more entries today (now {clock} ET)"
    if mins < ENTRY_MINUTE:
        wait = int(info.get("minutes_until_next") or 0)
        return f"Not in an entry window yet — 9:35 starts in {wait} min (now {clock} ET)"
    if trades_today >= max_trades_day:
        return f"Daily trade cap reached ({trades_today}/{max_trades_day})"
    win = window_index(mins)
    if win is None:
        nxt = info.get("next_window_name") or "next window"
        wait = int(info.get("minutes_until_next") or 0)
        tomorrow = " tomorrow" if info.get("next_is_tomorrow") else ""
        return (
            f"Not in a window yet — {nxt}{tomorrow} starts in {wait} min "
            f"(now {clock} ET)"
        )
    name = window_label(win)
    if win in fired:
        return f"{name} window already used today"
    side = int(trend_15m_on_1m(df_1m, completed_only=completed_15m).iloc[-1])
    if side == 0:
        return f"{name}: 15m EMA8/21 has no side yet (flat / not enough completed 15m bars)"
    if require_daily:
        daily = int(daily_trend_on_1m(df_1m).iloc[-1])
        if daily != side:
            return (
                f"{name} skipped — daily EMA20 is {_side_word(daily)}, "
                f"15m is {_side_word(side)} (they must agree)"
            )
    trend60 = int(trend_60m_on_1m(df_1m).iloc[-1])
    if require_60m and trend60 != side:
        return (
            f"{name} skipped — 60m EMA is {_side_word(trend60)}, "
            f"15m is {_side_word(side)} (daily+15m+60m must agree)"
        )
    if adx_min > 0:
        adx_val = adx_5m
        if adx_val is None and "adx" in row.index:
            adx_val = row.get("adx")
        if adx_val is None or pd.isna(adx_val) or float(adx_val) < adx_min:
            shown = "n/a" if adx_val is None or pd.isna(adx_val) else f"{float(adx_val):.0f}"
            return f"{name} skipped — ADX {shown} is below {adx_min:.0f}"
    sep = float(sep15_on_1m(df_1m).iloc[-1])
    confident = high_confidence(side, trend60, sep, sep_min)
    if window_requires_quality(win) and not confident:
        return (
            f"{name} skipped — noon+ quality filter "
            f"(need 60m agree + 15m sep/ATR ≥ {sep_min:.2f})"
        )
    if open_count >= 2:
        return f"{name} skipped — already at 2-lot cap"
    direction = "long" if side > 0 else "short"
    if open_count >= 1:
        if not confident:
            return f"{name} skipped — add-on needs 15m sep/ATR ≥ {sep_min:.2f}"
        if open_direction and open_direction != direction:
            return (
                f"{name} skipped — already {open_direction}, "
                f"will not flip to {direction}"
            )
    return f"{name} ready — {direction} (daily+15m+60m agree)"
