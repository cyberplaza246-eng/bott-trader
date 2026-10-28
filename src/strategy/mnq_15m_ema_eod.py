"""
15-minute EMA trend, up to six MNQ RTH slots, hold to 15:50 ET.

Uses only *completed* bars (no same-day daily close, no unfinished 15m bucket).

Default live recipe (L6 / responsive B): completed 15m EMA8/21 + completed 5m EMA8/21
agree (no daily, no 60m). Afternoon windows and a 2nd lot still need sep/ATR >= 0.45.
EMA15_MIN_CONFIDENCE (default 75) gates entries via a stack confidence %; sep quality
is usually required to clear 75.

  SL = clip(completed 1m ATR14 * 1.5, 6, 25) when EMA15_SL_TF=1m
  Legacy stack: set EMA15_REQUIRE_DAILY/60M=true, EMA15_REQUIRE_5M=false,
  EMA15_SL_TF=15m for daily+15m+60m with 15m ATR×2 clip 20–60.
    Daily cap defaults to 20 fills; same RTH window can re-fire after a close.
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
# Six RTH slots. L6: 15m+5m; noon+ and 2nd lot still need sep.
ENTRY_WINDOWS = (
    (9 * 60 + 35, 10 * 60),
    (10 * 60 + 15, 10 * 60 + 40),
    (11 * 60, 11 * 60 + 30),
    (12 * 60, 12 * 60 + 30),
    (13 * 60 + 30, 14 * 60),
    (14 * 60 + 30, 15 * 60),
)
MAX_TRADES_DAY = 20
DEFAULT_MAX_OPEN_LOTS = 2  # concurrent MNQ lots (MNQ_MAX_POSITIONS); not the daily cap
QUALITY_AFTER_MINUTE = 12 * 60  # 12:00 / 13:30 / 14:30 require high_confidence
SEP_MIN = 0.45  # |EMA8-EMA21| / ATR15 for a 2nd concurrent lot or afternoon window
SL_ATR_MULT = 1.5
SL_MIN = 6.0
SL_MAX = 25.0
TP_PTS = 500.0  # unused when tp_r > 0
TP_R = 1.0
ADX_MIN = 0.0
ONE_MIN_MAX_BODY = 4.0  # skip a stretched 1m; wait for a quiet with-trend close
FLATTEN_ET = "15:50"
LOCKED_RULES_PARAGRAPH = (
    "Locked ema15_eod L6: RTH only, entries only in 9:35 / 10:15 / 11:00 / 12:00 / 13:30 / 14:30 ET. "
    "Enter when completed 15m EMA8 vs EMA21 and completed 5m EMA8 vs EMA21 agree "
    "(no daily / 60m gate). From 12:00 ET, also need 15m sep/ATR >= 0.45. "
    "Second lot needs that sep gate too. "
    "Fill only a quiet with-trend 1m (body ≤ 4 pts). Do not chase a stretched 1m. "
    "Stop = clip(completed 1m ATR14 × 1.5, 6–25 pts). Target = 1R (same distance as the stop). "
    "Chart levels (PDH/PDL/…) are map-only — not an entry gate. "
    "Flatten 15:50 ET only if SL/TP have not filled. Do not trail. Overnight is off. "
    "Gemini is a second opinion only — it must not place orders, change stops, change "
    "windows, or rewrite mnq_profit_config."
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


def trend_5m_on_1m(df_1m: pd.DataFrame, *, completed_only: bool = True) -> pd.Series:
    """5m EMA8 vs EMA21 on 1m index (L6 / responsive 15m+5m stack)."""
    tmp = _as_utc_frame(df_1m)
    b5 = _ohlc_resample(tmp, "5min")
    fast = ema_series(b5["close"], EMA_FAST)
    slow = ema_series(b5["close"], EMA_SLOW)
    trend = pd.Series(0, index=b5.index, dtype=int)
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


def quiet_1m_ok(side: int, open_px: float, close_px: float, max_body: float = ONE_MIN_MAX_BODY) -> bool:
    """True when the 1m is small and closes with the trade (doji counts)."""
    try:
        o = float(open_px)
        c = float(close_px)
    except (TypeError, ValueError):
        return False
    if abs(c - o) > float(max_body):
        return False
    if side > 0:
        return c >= o
    if side < 0:
        return c <= o
    return False


def high_confidence(side: int, confirm_side: int, sep: float, sep_min: float = SEP_MIN) -> bool:
    """Afternoon / 2nd-lot gate: confirm TF agrees + 15m sep/ATR."""
    if side == 0 or confirm_side != side:
        return False
    if sep is None or (isinstance(sep, float) and (np.isnan(sep) or sep < sep_min)):
        return False
    return True


def ema15_min_confidence_from_env(default: int = 75) -> int:
    """Min entry-confidence % for ema15_eod (EMA15_MIN_CONFIDENCE, default 75)."""
    raw = os.getenv("EMA15_MIN_CONFIDENCE", str(default)).strip()
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return default
    return max(0, min(100, n))


def _env_bool(key: str, default: bool = False) -> bool:
    raw = str(os.getenv(key, "true" if default else "false") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def ema15_require_5m_from_env(default: bool = True) -> bool:
    """True when EMA15_REQUIRE_5M is on (L6 15m+5m stack)."""
    return _env_bool("EMA15_REQUIRE_5M", default)


def ema15_require_quiet_from_env(default: bool = True) -> bool:
    """PASS clean-forward sets EMA15_REQUIRE_QUIET=false (no quiet-1m filter)."""
    return _env_bool("EMA15_REQUIRE_QUIET", default)


def ema15_side_mode_from_env(default: str = "both") -> str:
    """long | short | both. Clean-forward PASS uses long."""
    raw = str(os.getenv("EMA15_SIDE", default) or default).strip().lower()
    if raw in ("long", "long_only", "l"):
        return "long"
    if raw in ("short", "short_only", "s"):
        return "short"
    return "both"


def ema15_allowed_windows_from_env() -> Optional[frozenset]:
    """
    Restrict ENTRY_WINDOWS indices.
    EMA15_WINDOWS=0,2 or w2_open → clean-forward PASS (09:35 + 11:00).
    Empty / unset → all six slots.
    """
    raw = str(os.getenv("EMA15_WINDOWS", "") or "").strip().lower()
    if not raw:
        return None
    aliases = {
        "w2_open": frozenset({0, 2}),
        "w2": frozenset({0, 2}),
        "w6": frozenset({0, 1, 2, 3, 4, 5}),
        "w6_all": frozenset({0, 1, 2, 3, 4, 5}),
        "all": frozenset({0, 1, 2, 3, 4, 5}),
    }
    if raw in aliases:
        return aliases[raw]
    idxs = set()
    for part in re.split(r"[,\s]+", raw):
        if not part:
            continue
        try:
            idxs.add(int(part))
        except ValueError:
            continue
    return frozenset(idxs) if idxs else None


def ema15_fixed_sl_from_env() -> Optional[float]:
    """EMA15_SL_FIXED=25 → hard stop pts (clean-forward PASS)."""
    raw = str(os.getenv("EMA15_SL_FIXED", "") or "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v > 0 else None


def ema15_continuous_from_env(default: bool = False) -> bool:
    """Hunt any minute 09:35–15:50 ET (not only slotted windows)."""
    return _env_bool("EMA15_CONTINUOUS", default)


def ema15_entry_cooldown_mins_from_env(default: int = 15) -> int:
    """Minutes after an entry before another continuous-mode signal."""
    raw = str(os.getenv("EMA15_ENTRY_COOLDOWN_MIN", "") or "").strip()
    if not raw:
        return max(0, int(default))
    try:
        return max(0, int(raw))
    except ValueError:
        return max(0, int(default))


def ema15_sl_tf_from_env(default: str = "1m") -> str:
    """ATR timeframe for stop: '1m' (L6) or '15m' (legacy)."""
    raw = str(os.getenv("EMA15_SL_TF", default) or default).strip().lower()
    if raw in ("1m", "1", "1min", "1minute"):
        return "1m"
    if raw in ("5m", "5", "5min"):
        return "5m"
    return "15m"


def ema15_entry_confidence_pct(
    side: int,
    confirm_side: int,
    sep: float,
    *,
    daily_ok: bool = True,
    sep_min: float = SEP_MIN,
    confirm_name: str = "60m",
) -> Dict[str, Any]:
    """Heuristic entry confidence for day bot — comparable to overnight win %.

    50 base (15m side), +10 daily agree, +10 confirm TF agree (60m or 5m),
    +15 sep/ATR quality. Clamped 40–95. With EMA15_MIN_CONFIDENCE=75, sep
    quality is usually required.
    """
    label = confirm_name or "60m"
    breakdown: list = ["base 15m 50"]
    score = 50
    if daily_ok:
        score += 10
        breakdown.append("daily +10")
    else:
        breakdown.append("daily missing 0")
    if side != 0 and confirm_side == side:
        score += 10
        breakdown.append(f"{label} +10")
    elif confirm_side == 0:
        score += 3
        breakdown.append(f"{label} flat +3")
    else:
        score -= 6
        breakdown.append(f"{label} opposed -6")
    sep_ok = False
    if sep is not None and not (isinstance(sep, float) and np.isnan(sep)):
        sep_ok = float(sep) >= float(sep_min)
    if sep_ok:
        score += 15
        breakdown.append(f"sep/ATR>={sep_min:.2f} +15")
    else:
        breakdown.append(f"sep/ATR soft 0")
    pct = int(max(40, min(95, round(score))))
    parts = []
    if daily_ok:
        parts.append("daily+15m")
    else:
        parts.append("15m")
    if side != 0 and confirm_side == side:
        parts.append(label)
    if sep_ok:
        parts.append("sep quality")
    return {
        "pct": pct,
        "breakdown": breakdown,
        "summary": ", ".join(parts) if parts else "partial stack",
        "high_confidence": bool(sep_ok and side != 0 and confirm_side == side),
    }


def atr15_on_1m(df_1m: pd.DataFrame) -> pd.Series:
    tmp = _as_utc_frame(df_1m)
    b15 = _ohlc_resample(tmp, "15min")
    atr = _atr(b15, 14).shift(1)
    return atr.reindex(tmp.index, method="ffill")


def atr1_on_1m(df_1m: pd.DataFrame) -> pd.Series:
    """Completed 1m ATR14 (shifted) on the 1m index — L6 stop source."""
    tmp = _as_utc_frame(df_1m)
    atr = _atr(tmp, 14).shift(1)
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
        gap_min = int((minute - ts_min).total_seconds() // 60)
        new["datetime"] = minute
        # Hours-stale CSV: bridging prev_close→last creates a fake 200+ pt body and
        # blocks quiet_1m_ok forever. Stamp a flat live minute instead.
        if gap_min > 1:
            new["open"] = last_px
            new["high"] = last_px
            new["low"] = last_px
        else:
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


def max_trades_day_from_env(default: int = MAX_TRADES_DAY) -> int:
    """Daily entry cap from EMA15_MAX_TRADES_DAY or MAX_TRADES_PER_DAY (default 20)."""
    for key in ("EMA15_MAX_TRADES_DAY", "MAX_TRADES_PER_DAY"):
        raw = os.getenv(key, "").strip()
        if not raw:
            continue
        try:
            return max(1, int(raw))
        except ValueError:
            continue
    return default


def max_open_lots_from_env(default: int = DEFAULT_MAX_OPEN_LOTS) -> int:
    """Concurrent open lots from MNQ_MAX_POSITIONS or EMA15_MAX_OPEN_LOTS (default 2)."""
    for key in ("MNQ_MAX_POSITIONS", "EMA15_MAX_OPEN_LOTS"):
        raw = os.getenv(key, "").strip()
        if not raw:
            continue
        try:
            return max(1, int(raw))
        except ValueError:
            continue
    return default


def allow_ema15_entry_confidence(confidence_pct, min_pct: Optional[int] = None) -> bool:
    """True when day-bot confidence meets EMA15_MIN_CONFIDENCE (default 75)."""
    need = ema15_min_confidence_from_env() if min_pct is None else int(min_pct)
    if need <= 0:
        return True
    try:
        pct = int(confidence_pct)
    except (TypeError, ValueError):
        return False
    return pct >= need


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
    last_entry_minute: Optional[int] = None  # ET minutes; continuous-mode cooldown anchor


def _reset_day(state: Ema15EodState, day: date) -> Ema15EodState:
    if state.day == day:
        return state
    state.day = day
    state.trades_today = 0
    state.fired_windows = ()
    state.last_entry_minute = None
    state.taken_date = None
    state.decided_date = None
    return state


def rollback_ema15_pending_entry(
    state: Ema15EodState, window: Optional[int],
) -> Ema15EodState:
    """Undo signal reservation when bracket risk or order placement fails."""
    state.trades_today = max(0, state.trades_today - 1)
    if window is not None:
        state.fired_windows = tuple(w for w in state.fired_windows if w != window)
    state.taken_date = None
    return state


def release_ema15_window(
    state: Ema15EodState, window: Optional[int],
) -> Ema15EodState:
    """Allow re-entry in the same RTH window after a fill closes."""
    if window is not None:
        state.fired_windows = tuple(w for w in state.fired_windows if w != window)
    return state


@dataclass
class Ema15EodConfig:
    require_daily: bool = False
    require_60m: bool = False
    require_5m: bool = True
    sl_atr_mult: float = SL_ATR_MULT
    sl_min: float = SL_MIN
    sl_max: float = SL_MAX
    sl_tf: str = "1m"
    tp_pts: float = TP_PTS
    tp_r: float = TP_R
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
    tp_r: float = TP_R,
    require_daily: bool = False,
    require_60m: bool = False,
    require_5m: bool = True,
    adx_5m: Optional[float] = None,
    adx_min: float = ADX_MIN,
    sl_atr_mult: float = SL_ATR_MULT,
    sl_min: float = SL_MIN,
    sl_max: float = SL_MAX,
    sl_tf: str = "1m",
    completed_15m: bool = True,
    max_trades_day: Optional[int] = None,
    max_open_lots: Optional[int] = None,
    open_count: int = 0,
    open_direction: Optional[str] = None,
    sep_min: float = SEP_MIN,
    min_confidence: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], Ema15EodState]:
    if max_trades_day is None:
        max_trades_day = max_trades_day_from_env()
    if max_open_lots is None:
        max_open_lots = max_open_lots_from_env()
    if min_confidence is None:
        min_confidence = ema15_min_confidence_from_env()
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

    continuous = ema15_continuous_from_env()
    win = window_index(mins)
    if continuous:
        # Any RTH minute; space entries with cooldown (1m quiet times the fill).
        cool = ema15_entry_cooldown_mins_from_env(15)
        if (
            cool > 0
            and state.last_entry_minute is not None
            and mins < int(state.last_entry_minute) + cool
        ):
            return None, state
    else:
        if win is None or win in state.fired_windows:
            return None, state
        allowed = ema15_allowed_windows_from_env()
        if allowed is not None and win not in allowed:
            return None, state

    side = int(trend_15m_on_1m(df_1m, completed_only=completed_15m).iloc[-1])
    if side == 0:
        return None, state
    side_mode = ema15_side_mode_from_env()
    if side_mode == "long" and side != 1:
        return None, state
    if side_mode == "short" and side != -1:
        return None, state
    daily_ok = True
    if require_daily:
        daily = int(daily_trend_on_1m(df_1m).iloc[-1])
        if daily != side:
            return None, state
    else:
        daily_ok = False
    trend60 = int(trend_60m_on_1m(df_1m).iloc[-1])
    if require_60m and trend60 != side:
        return None, state
    trend5 = int(trend_5m_on_1m(df_1m).iloc[-1])
    if require_5m and trend5 != side:
        return None, state
    if adx_min > 0:
        adx_val = adx_5m
        if adx_val is None and "adx" in row.index:
            adx_val = row.get("adx")
        if adx_val is None or pd.isna(adx_val) or float(adx_val) < adx_min:
            return None, state

    atr15 = float(atr15_on_1m(df_1m).iloc[-1])
    atr1 = float(atr1_on_1m(df_1m).iloc[-1])
    # Continuous: quiet 1m times the fill. Slotted PASS can set EMA15_REQUIRE_QUIET=false.
    if ema15_require_quiet_from_env(default=True) and not quiet_1m_ok(
        side, row.get("open"), row.get("close")
    ):
        return None, state
    fixed_sl = ema15_fixed_sl_from_env()
    if sl_pts is None and fixed_sl is not None:
        sl_pts = fixed_sl
    tf = (sl_tf or "1m").strip().lower()
    if sl_pts is not None:
        use_sl = float(sl_pts)
        atr_used = atr1 if tf == "1m" else atr15
    elif tf == "1m":
        atr_used = atr1
        use_sl = sl_pts_from_atr(atr1, mult=sl_atr_mult, sl_min=sl_min, sl_max=sl_max)
    else:
        atr_used = atr15
        use_sl = sl_pts_from_atr(atr15, mult=sl_atr_mult, sl_min=sl_min, sl_max=sl_max)
    entry = float(row["close"])
    direction = "long" if side > 0 else "short"
    sep = float(sep15_on_1m(df_1m).iloc[-1])
    if require_5m:
        confirm_side, confirm_name = trend5, "5m"
    else:
        confirm_side, confirm_name = trend60, "60m"
    confident = high_confidence(side, confirm_side, sep, sep_min)
    conf = ema15_entry_confidence_pct(
        side, confirm_side, sep, daily_ok=daily_ok, sep_min=sep_min,
        confirm_name=confirm_name,
    )
    need_quality = (
        (win is not None and window_requires_quality(win))
        or (continuous and mins >= QUALITY_AFTER_MINUTE)
    )
    if need_quality and not confident:
        return None, state
    if not allow_ema15_entry_confidence(conf["pct"], min_confidence):
        return None, state
    if open_count >= max_open_lots:
        return None, state
    if open_count >= 1:
        if not confident:
            return None, state
        if open_direction and open_direction != direction:
            return None, state
    sl = entry - use_sl if side > 0 else entry + use_sl
    tp_dist = use_sl * tp_r if tp_r and tp_r > 0 else tp_pts
    tp = entry + tp_dist if side > 0 else entry - tp_dist
    state.taken_date = day
    state.decided_date = day
    state.trades_today += 1
    state.last_entry_minute = int(mins)
    if win is not None:
        state.fired_windows = tuple(sorted(set(state.fired_windows) | {win}))
    atr_nan = atr_used if not (isinstance(atr_used, float) and np.isnan(atr_used)) else use_sl
    return {
        "symbol": symbol,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "atr": atr_nan,
        "scalp_mode": "ema15_eod",
        "scalp_hybrid": False,
        "structure_capped": False,
        "entry_meta": {
            "entry_mode": "ema15_eod",
            "stack": "15m+5m" if require_5m else ("daily+15m+60m" if require_daily else "15m+60m"),
            "trend_15m": side,
            "daily_trend": int(daily_trend_on_1m(df_1m).iloc[-1]) if require_daily else 0,
            "sl_pts": round(use_sl, 2),
            "sl_tf": tf,
            "tp_pts": round(tp_dist, 2),
            "tp_r": tp_r,
            "atr15": None if np.isnan(atr15) else round(atr15, 2),
            "atr1": None if (isinstance(atr1, float) and np.isnan(atr1)) else round(atr1, 2),
            "et_date": str(day),
            "trend_60m": trend60,
            "trend_5m": trend5,
            "require_60m": require_60m,
            "require_5m": require_5m,
            "sep15": None if np.isnan(sep) else round(sep, 3),
            "high_confidence": confident,
            "entry_confidence_pct": conf["pct"],
            "entry_confidence": conf,
            "add_on": open_count >= 1,
            "window": win,
            "window_name": window_label(win),
            "entry_1m": "quiet_close",
            "body_pts": round(abs(float(row["close"]) - float(row["open"])), 2),
        },
    }, state


def explain_ema15_skip(
    df_1m: Optional[pd.DataFrame],
    state: Ema15EodState,
    *,
    now: Optional[datetime] = None,
    require_daily: bool = False,
    require_60m: bool = False,
    require_5m: bool = True,
    adx_5m: Optional[float] = None,
    adx_min: float = ADX_MIN,
    completed_15m: bool = True,
    max_trades_day: Optional[int] = None,
    max_open_lots: Optional[int] = None,
    open_count: int = 0,
    open_direction: Optional[str] = None,
    sep_min: float = SEP_MIN,
    min_confidence: Optional[int] = None,
) -> str:
    """Human reason the locked recipe will not enter on this bar. Does not mutate state."""
    if max_trades_day is None:
        max_trades_day = max_trades_day_from_env()
    if max_open_lots is None:
        max_open_lots = max_open_lots_from_env()
    if min_confidence is None:
        min_confidence = ema15_min_confidence_from_env()
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
    continuous = ema15_continuous_from_env()
    win = window_index(mins)
    if continuous:
        cool = ema15_entry_cooldown_mins_from_env(15)
        last_m = state.last_entry_minute if state.day == day else None
        if cool > 0 and last_m is not None and mins < int(last_m) + cool:
            left = int(last_m) + cool - mins
            return (
                f"Continuous hunt cooldown — {left} min left "
                f"(last entry ~{int(last_m) // 60:02d}:{int(last_m) % 60:02d} ET, now {clock})"
            )
        name = "continuous"
    else:
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
            return f"{name} window active — wait for open trade to close before re-entry"
    side = int(trend_15m_on_1m(df_1m, completed_only=completed_15m).iloc[-1])
    if side == 0:
        return f"{name}: 15m EMA8/21 has no side yet (flat / not enough completed 15m bars)"
    daily_ok = True
    if require_daily:
        daily = int(daily_trend_on_1m(df_1m).iloc[-1])
        if daily != side:
            return (
                f"{name} skipped — daily EMA20 is {_side_word(daily)}, "
                f"15m is {_side_word(side)} (they must agree)"
            )
    else:
        daily_ok = False
    trend60 = int(trend_60m_on_1m(df_1m).iloc[-1])
    if require_60m and trend60 != side:
        return (
            f"{name} skipped — 60m EMA is {_side_word(trend60)}, "
            f"15m is {_side_word(side)} (daily+15m+60m must agree)"
        )
    trend5 = int(trend_5m_on_1m(df_1m).iloc[-1])
    if require_5m and trend5 != side:
        return (
            f"{name} skipped — 5m EMA is {_side_word(trend5)}, "
            f"15m is {_side_word(side)} (15m+5m must agree)"
        )
    if adx_min > 0:
        adx_val = adx_5m
        if adx_val is None and "adx" in row.index:
            adx_val = row.get("adx")
        if adx_val is None or pd.isna(adx_val) or float(adx_val) < adx_min:
            shown = "n/a" if adx_val is None or pd.isna(adx_val) else f"{float(adx_val):.0f}"
            return f"{name} skipped — ADX {shown} is below {adx_min:.0f}"
    sep = float(sep15_on_1m(df_1m).iloc[-1])
    if require_5m:
        confirm_side, confirm_name = trend5, "5m"
        stack_label = "15m+5m"
    else:
        confirm_side, confirm_name = trend60, "60m"
        stack_label = "daily+15m+60m" if require_daily else "15m+60m"
    confident = high_confidence(side, confirm_side, sep, sep_min)
    conf = ema15_entry_confidence_pct(
        side, confirm_side, sep, daily_ok=daily_ok, sep_min=sep_min,
        confirm_name=confirm_name,
    )
    if window_requires_quality(win) if win is not None else (
        continuous and mins >= QUALITY_AFTER_MINUTE
    ):
        if not confident:
            return (
                f"{name} skipped — noon+ quality filter "
                f"(need {confirm_name} agree + 15m sep/ATR ≥ {sep_min:.2f})"
            )
    if not allow_ema15_entry_confidence(conf["pct"], min_confidence):
        return (
            f"{name} skipped — confidence {conf['pct']}% below "
            f"EMA15_MIN_CONFIDENCE={min_confidence}% "
            f"({conf.get('summary') or 'stack'})"
        )
    if open_count >= max_open_lots:
        return f"{name} skipped — already at {max_open_lots}-lot cap"
    direction = "long" if side > 0 else "short"
    if open_count >= 1:
        if not confident:
            return f"{name} skipped — add-on needs 15m sep/ATR ≥ {sep_min:.2f}"
        if open_direction and open_direction != direction:
            return (
                f"{name} skipped — already {open_direction}, "
                f"will not flip to {direction}"
            )
    require_quiet = ema15_require_quiet_from_env(True)
    if require_quiet and not quiet_1m_ok(side, row.get("open"), row.get("close")):
        body = abs(float(row.get("close") or 0) - float(row.get("open") or 0))
        return (
            f"{name} waiting — 1m body {body:.1f} pts (need ≤ {ONE_MIN_MAX_BODY:.0f} "
            f"and close with the {direction})"
        )
    return (
        f"{name} ready — {direction} conf {conf['pct']}% "
        f"({stack_label} agree, quiet 1m)"
    )


def ema15_level_mode_from_env(environ: Optional[Dict[str, str]] = None) -> bool:
    """True when EMA15_LEVEL_MODE=1 or SCALP_MODE=level_15_5_1."""
    env = environ if environ is not None else os.environ
    raw = str(env.get("EMA15_LEVEL_MODE", "") or "").strip().lower()
    if raw in ("1", "true", "yes", "on", "level"):
        return True
    scalp = str(env.get("SCALP_MODE", "") or env.get("STRATEGY_MODE", "") or "").strip().lower()
    return scalp in ("level_15_5_1", "level_break_retest", "ema15_level")


def level_mode_confidence_pct(
    side: int,
    bias: Dict[str, Any],
    interaction: Optional[Dict[str, Any]],
    *,
    one_min_ok: bool,
) -> Dict[str, Any]:
    """Confidence for level_15_5_1 — needs ≥75 by default to clear EMA15_MIN_CONFIDENCE."""
    pct = 40
    bits: list = []
    if side != 0 and bias.get("structure_15m") in ("HH_HL", "LH_LL"):
        pct += 15
        bits.append(str(bias.get("structure_15m")))
    if int(bias.get("ema_side_15m") or 0) == side and side != 0:
        pct += 10
        bits.append("EMA10/20")
    if int(bias.get("vwap_side_15m") or 0) == side and side != 0:
        pct += 10
        bits.append("VWAP")
    if int(bias.get("side_5m") or 0) == side and side != 0:
        pct += 15
        bits.append("5m")
    if interaction:
        pct += 10
        bits.append(f"{interaction.get('kind')}@{interaction.get('level_name')}")
    if one_min_ok:
        pct += 5
        bits.append("1m")
    pct = int(max(40, min(95, pct)))
    return {"pct": pct, "summary": "+".join(bits) if bits else "thin", "bits": bits}


def check_ema15_level_entry(
    symbol: str,
    df_1m: pd.DataFrame,
    state: Ema15EodState,
    *,
    now: Optional[datetime] = None,
    sl_pts: Optional[float] = None,
    tp_pts: float = TP_PTS,
    tp_r: float = TP_R,
    max_trades_day: Optional[int] = None,
    max_open_lots: Optional[int] = None,
    open_count: int = 0,
    open_direction: Optional[str] = None,
    min_confidence: Optional[int] = None,
    touch_pts: float = 8.0,
    sl_atr_mult: float = 1.5,
    sl_min: float = 8.0,
    sl_max: float = 30.0,
) -> Tuple[Optional[Dict[str, Any]], Ema15EodState]:
    """
    Level break/retest mode (within locked 6 windows).

    15m HH/HL or LH/LL + VWAP + EMA10/20 → 5m agree → major level interaction
    → 1m confirm. Mixed 15m = no trade. Target next level when practical else 1R.
    Uses responsive-style 1m ATR stop band (8–30) as risk cap.
    """
    from src.strategy.session_levels import (
        bias_15m_5m,
        compute_session_levels,
        level_interaction,
        next_target_level,
        one_min_level_confirm,
    )

    if max_trades_day is None:
        max_trades_day = max_trades_day_from_env()
    if max_open_lots is None:
        max_open_lots = max_open_lots_from_env()
    if min_confidence is None:
        min_confidence = ema15_min_confidence_from_env()
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

    bias = bias_15m_5m(df_1m, completed_only=True)
    side = int(bias.get("side_15m") or 0)
    if side == 0:
        return None, state
    if int(bias.get("side_5m") or 0) != side:
        return None, state

    levels = compute_session_levels(df_1m, now=ts)
    interaction = level_interaction(df_1m, side, levels, touch_pts=touch_pts)
    if not interaction:
        return None, state
    one_ok = one_min_level_confirm(df_1m, side, interaction)
    if not one_ok:
        return None, state

    conf = level_mode_confidence_pct(side, bias, interaction, one_min_ok=one_ok)
    if not allow_ema15_entry_confidence(conf["pct"], min_confidence):
        return None, state
    if open_count >= max_open_lots:
        return None, state
    direction = "long" if side > 0 else "short"
    if open_count >= 1 and open_direction and open_direction != direction:
        return None, state

    # Stop: beyond invalidation (interaction level) vs 1m ATR band — use wider of buffer,
    # clipped to responsive band.
    entry = float(row["close"])
    lvl = float(interaction["level"])
    tmp = _as_utc_frame(df_1m)
    atr1 = float(_atr(tmp.tail(60), 14).iloc[-1]) if len(tmp) >= 20 else 10.0
    if np.isnan(atr1) or atr1 <= 0:
        atr1 = 10.0
    atr_sl = sl_pts_from_atr(atr1, mult=sl_atr_mult, sl_min=sl_min, sl_max=sl_max)
    struct_sl = abs(entry - lvl) + 2.0
    use_sl = float(sl_pts) if sl_pts is not None else float(max(atr_sl, min(struct_sl, sl_max)))
    use_sl = float(np.clip(use_sl, sl_min, sl_max))
    sl = entry - use_sl if side > 0 else entry + use_sl

    tgt = next_target_level(side, entry, levels, stop_dist=use_sl)
    structure_capped = False
    if tgt is not None:
        tname, tpx = tgt
        tp_dist = abs(tpx - entry)
        if tp_dist >= use_sl * 0.5:
            tp = float(tpx)
            structure_capped = True
            tp_meta = {"tp_mode": "next_level", "tp_level": tname, "tp_pts": round(tp_dist, 2)}
        else:
            tp_dist = use_sl * tp_r if tp_r and tp_r > 0 else tp_pts
            tp = entry + tp_dist if side > 0 else entry - tp_dist
            tp_meta = {"tp_mode": "1R", "tp_pts": round(abs(tp - entry), 2)}
    else:
        tp_dist = use_sl * tp_r if tp_r and tp_r > 0 else tp_pts
        tp = entry + tp_dist if side > 0 else entry - tp_dist
        tp_meta = {"tp_mode": "1R", "tp_pts": round(abs(tp - entry), 2)}

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
        "size": 1,
        "structure_capped": structure_capped,
        "entry_meta": {
            "entry_mode": "level_15_5_1",
            "window": win,
            "window_name": window_label(win),
            "sl_pts": round(use_sl, 2),
            "tp_pts": tp_meta.get("tp_pts"),
            "tp_mode": tp_meta.get("tp_mode"),
            "tp_level": tp_meta.get("tp_level"),
            "level_name": interaction.get("level_name"),
            "level": interaction.get("level"),
            "level_kind": interaction.get("kind"),
            "structure_15m": bias.get("structure_15m"),
            "bias_15m": bias.get("bias_15m"),
            "bias_5m": bias.get("bias_5m"),
            "entry_confidence_pct": conf["pct"],
            "entry_confidence": conf,
            "flatten_et": FLATTEN_ET,
            "chart_levels": {
                k: levels.get(k)
                for k in ("pdh", "pdl", "onh", "onl", "rth_open", "vwap", "ema10", "ema20")
            },
        },
    }, state


def explain_ema15_level_skip(
    df_1m: Optional[pd.DataFrame],
    state: Ema15EodState,
    *,
    now: Optional[datetime] = None,
    max_trades_day: Optional[int] = None,
    max_open_lots: Optional[int] = None,
    open_count: int = 0,
    open_direction: Optional[str] = None,
    min_confidence: Optional[int] = None,
    touch_pts: float = 8.0,
) -> str:
    """Console skip reasons for level_15_5_1 mode."""
    from src.strategy.session_levels import (
        bias_15m_5m,
        compute_session_levels,
        level_interaction,
        one_min_level_confirm,
    )

    if max_trades_day is None:
        max_trades_day = max_trades_day_from_env()
    if max_open_lots is None:
        max_open_lots = max_open_lots_from_env()
    if min_confidence is None:
        min_confidence = ema15_min_confidence_from_env()
    n = 0 if df_1m is None or df_1m.empty else len(df_1m)
    if df_1m is None or df_1m.empty or n < 80:
        return (
            f"Need more 1m history (have {n}, need 80) for levels — "
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
        return f"{name} window active — wait for open trade to close before re-entry"

    bias = bias_15m_5m(df_1m, completed_only=True)
    side = int(bias.get("side_15m") or 0)
    if side == 0:
        return (
            f"{name}: 15m mixed "
            f"(structure {bias.get('structure_15m')}, "
            f"EMA {bias.get('ema_side_15m')}, VWAP {bias.get('vwap_side_15m')})"
        )
    if int(bias.get("side_5m") or 0) != side:
        return (
            f"{name}: 5m disagree "
            f"(15m {_side_word(side)}, 5m {_side_word(int(bias.get('side_5m') or 0))})"
        )
    levels = compute_session_levels(df_1m, now=ts)
    interaction = level_interaction(df_1m, side, levels, touch_pts=touch_pts)
    if not interaction:
        return (
            f"{name}: waiting for level "
            f"(PDH/PDL/ONH/ONL/RTH Open within {touch_pts:.0f} pts — "
            f"15m {_side_word(side)})"
        )
    one_ok = one_min_level_confirm(df_1m, side, interaction)
    if not one_ok:
        return (
            f"{name}: waiting 1m confirm at {interaction.get('level_name')} "
            f"{interaction.get('level')} ({interaction.get('kind')})"
        )
    conf = level_mode_confidence_pct(side, bias, interaction, one_min_ok=one_ok)
    if not allow_ema15_entry_confidence(conf["pct"], min_confidence):
        return (
            f"{name} skipped — confidence {conf['pct']}% below "
            f"EMA15_MIN_CONFIDENCE={min_confidence}% "
            f"({conf.get('summary') or 'stack'})"
        )
    if open_count >= max_open_lots:
        return f"{name} skipped — already at {max_open_lots}-lot cap"
    direction = "long" if side > 0 else "short"
    if open_count >= 1 and open_direction and open_direction != direction:
        return (
            f"{name} skipped — already {open_direction}, "
            f"will not flip to {direction}"
        )
    return (
        f"{name} ready — {direction} conf {conf['pct']}% "
        f"({interaction.get('kind')} {interaction.get('level_name')} "
        f"@ {interaction.get('level')}, 15m+5m agree)"
    )
