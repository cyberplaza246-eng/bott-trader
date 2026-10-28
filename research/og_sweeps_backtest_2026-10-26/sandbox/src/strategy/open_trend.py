"""OpenTrend MNQ paper strategy: 5-minute EMA crossover with ATR brackets."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
import pytz

ET = pytz.timezone("America/New_York")
BAR_MINUTES = 5
EMA_FAST = 9
EMA_SLOW = 21
ATR_PERIOD = 14
SL_ATR = 1.5
TP_ATR = 2.0
RTH_OPEN_MINUTE = 9 * 60 + 30
FLATTEN_MINUTE = 15 * 60 + 55
SESSION_NAME = "open_trend"


def _as_et(now: Optional[datetime] = None) -> datetime:
    value = now or datetime.now(ET)
    if value.tzinfo is None:
        return ET.localize(value)
    return value.astimezone(ET)


def is_rth_et(now: Optional[datetime] = None) -> bool:
    et = _as_et(now)
    minute = et.hour * 60 + et.minute
    return et.weekday() < 5 and RTH_OPEN_MINUTE <= minute < 16 * 60


def entries_allowed_et(now: Optional[datetime] = None) -> bool:
    et = _as_et(now)
    minute = et.hour * 60 + et.minute
    return et.weekday() < 5 and RTH_OPEN_MINUTE <= minute < FLATTEN_MINUTE


def should_flatten_rth(now: Optional[datetime] = None) -> bool:
    et = _as_et(now)
    return et.weekday() < 5 and et.hour * 60 + et.minute >= FLATTEN_MINUTE


def prepare_5m_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute OHLCV data into left-labeled 5-minute bars."""
    if df is None or df.empty or "datetime" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    out = out.dropna(subset=["datetime"]).sort_values("datetime")
    for col in ("open", "high", "low", "close"):
        if col not in out.columns:
            return pd.DataFrame()
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["open", "high", "low", "close"])
    if out.empty:
        return pd.DataFrame()
    out = out.set_index("datetime")
    bars = out.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return bars.dropna(subset=["open", "high", "low", "close"]).reset_index()


def add_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """Return bars with EMA(9), EMA(21), and Wilder-style ATR(14)."""
    if bars is None or bars.empty:
        return pd.DataFrame()
    out = bars.copy()
    out["ema9"] = out["close"].ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean()
    out["ema21"] = out["close"].ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean()
    previous_close = out["close"].shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - previous_close).abs(),
            (out["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14"] = true_range.ewm(alpha=1 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()
    return out


def crossed_above(previous_fast: float, previous_slow: float, fast: float, slow: float) -> bool:
    return previous_fast <= previous_slow and fast > slow


def crossed_below(previous_fast: float, previous_slow: float, fast: float, slow: float) -> bool:
    return previous_fast >= previous_slow and fast < slow


def closed_bars(bars: pd.DataFrame, now: Optional[datetime] = None) -> pd.DataFrame:
    """Exclude the still-forming 5-minute bar when its end is in the future."""
    if bars is None or bars.empty or "datetime" not in bars.columns:
        return pd.DataFrame()
    out = bars.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(now or datetime.now(pytz.UTC))
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    cutoff = cutoff.tz_convert("UTC")
    return out[out["datetime"] + pd.Timedelta(minutes=BAR_MINUTES) <= cutoff].reset_index(drop=True)


def signal_on_closed_bar(
    bars: pd.DataFrame, now: Optional[datetime] = None
) -> Optional[dict]:
    """Return one crossover signal from the newest completed bar, if any."""
    done = closed_bars(add_indicators(bars), now=now)
    if len(done) < 2:
        return None
    previous = done.iloc[-2]
    current = done.iloc[-1]
    values = [previous["ema9"], previous["ema21"], current["ema9"], current["ema21"], current["atr14"]]
    if any(pd.isna(value) for value in values) or float(current["atr14"]) <= 0:
        return None
    direction: Optional[str] = None
    if crossed_above(float(previous["ema9"]), float(previous["ema21"]), float(current["ema9"]), float(current["ema21"])) and float(current["close"]) > float(current["ema21"]):
        direction = "long"
    elif crossed_below(float(previous["ema9"]), float(previous["ema21"]), float(current["ema9"]), float(current["ema21"])) and float(current["close"]) < float(current["ema21"]):
        direction = "short"
    if direction is None:
        return None
    entry = float(current["close"])
    atr = float(current["atr14"])
    sl_distance = SL_ATR * atr
    tp_distance = TP_ATR * atr
    if direction == "long":
        sl, tp = entry - sl_distance, entry + tp_distance
    else:
        sl, tp = entry + sl_distance, entry - tp_distance
    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "atr": atr,
        "sl_distance": sl_distance,
        "tp_distance": tp_distance,
        "signal_bar": str(current["datetime"]),
        "strategy": SESSION_NAME,
        "size": 1,
    }


def check_bar_exit(pos: dict, bar_high: float, bar_low: float, bar_close: float, *, flat_eod: bool = False):
    """Return (reason, price) using conservative stop-first intrabar ordering."""
    if flat_eod:
        return "EOD", float(bar_close)
    direction = str(pos["direction"])
    if direction == "long":
        if bar_low <= float(pos["sl"]):
            return "SL", float(pos["sl"])
        if bar_high >= float(pos["tp"]):
            return "TP", float(pos["tp"])
    else:
        if bar_high >= float(pos["sl"]):
            return "SL", float(pos["sl"])
        if bar_low <= float(pos["tp"]):
            return "TP", float(pos["tp"])
    return None
