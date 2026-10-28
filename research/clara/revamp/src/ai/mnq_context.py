"""
MNQ trading context — session, structure, VWAP, volatility, multi-TF.

Feeds both rule-based filters and the LLM advisor.
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import pytz
    ET = pytz.timezone("US/Eastern")
except ImportError:
    ET = None


def _to_et(dt: datetime) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if ET is None:
        return dt
    return dt.astimezone(ET)


def classify_session(dt: datetime) -> Dict[str, Any]:
    """US/Eastern session bucket for MNQ."""
    et = _to_et(dt)
    hm = et.hour * 60 + et.minute
    wd = et.weekday()

    if wd >= 5:
        return {"session": "weekend", "quality": 0, "allow_scalp": False, "et_time": et.strftime("%H:%M"), "weekday": wd}

    # Times in ET minutes
    if hm < 4 * 60:
        label, quality = "overnight", 30
    elif hm < 9 * 60 + 30:
        label, quality = "pre_market", 55
    elif hm < 11 * 60:
        label, quality = "ny_open", 95
    elif hm < 14 * 60:
        label, quality = "midday_chop", 35
    elif hm < 15 * 60:
        label, quality = "afternoon", 65
    elif hm < 16 * 60:
        label, quality = "power_hour", 80
    else:
        label, quality = "after_hours", 40

    allow = label in ("ny_open", "power_hour", "pre_market", "afternoon")
    return {
        "session": label,
        "quality": quality,
        "allow_scalp": allow,
        "weekday": wd,
        "et_time": et.strftime("%H:%M"),
    }


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    if "datetime" not in df.columns:
        cum_pv = (typical * df["volume"]).cumsum()
        cum_v = df["volume"].cumsum().replace(0, np.nan)
        return cum_pv / cum_v
    dates = pd.to_datetime(df["datetime"]).dt.date
    tp_vol = typical * df["volume"]
    cum_pv = tp_vol.groupby(dates).cumsum()
    cum_v = df["volume"].groupby(dates).cumsum().replace(0, np.nan)
    return cum_pv / cum_v


def classify_market_structure(adx: float, volume_ratio: float, atr_ratio: float) -> str:
    if pd.isna(adx):
        adx = 0
    if adx >= 25 and volume_ratio >= 0.8:
        return "trending"
    if adx >= 18:
        return "transition"
    if atr_ratio > 2.5:
        return "volatile_chop"
    if adx < 15:
        return "chop"
    return "ranging"


def compute_setup_score(
    direction: str,
    ctx: Dict[str, Any],
    min_score: float = 60,
) -> Dict[str, Any]:
    """0–100 setup score from user-advice factors."""
    score = 50.0
    reasons_plus = []
    reasons_minus = []

    session_q = ctx.get("session_quality", 50)
    score += (session_q - 50) * 0.25
    if session_q >= 80:
        reasons_plus.append("high-quality session window")
    elif session_q <= 40:
        reasons_minus.append("low-probability session")

    structure = ctx.get("market_structure", "ranging")
    if structure == "trending":
        score += 12
        reasons_plus.append("trending structure")
    elif structure in ("chop", "volatile_chop"):
        score -= 20
        reasons_minus.append(f"{structure} environment")

    if ctx.get("mtf_aligned"):
        score += 10
        reasons_plus.append("multi-timeframe aligned")
    else:
        score -= 15
        reasons_minus.append("MTF conflict")

    price = ctx.get("price")
    vwap = ctx.get("vwap")
    if price and vwap and not pd.isna(vwap):
        above = price > vwap
        if (direction in ("long", "LONG", "BUY") and above) or (
            direction in ("short", "SHORT", "SELL") and not above
        ):
            score += 8
            reasons_plus.append("VWAP alignment")
        else:
            score -= 8
            reasons_minus.append("against VWAP")

    if ctx.get("volume_ratio", 0) >= 1.0:
        score += 5
        reasons_plus.append("healthy volume")
    elif ctx.get("volume_ratio", 1) < 0.5:
        score -= 10
        reasons_minus.append("weak volume")

    if ctx.get("near_resistance") and direction in ("long", "LONG", "BUY"):
        score -= 8
        reasons_minus.append("near resistance")
    if ctx.get("near_support") and direction in ("short", "SHORT", "SELL"):
        score -= 8
        reasons_minus.append("near support")

    if ctx.get("news_blocked"):
        score -= 30
        reasons_minus.append(f"news window: {ctx.get('news_event')}")

    if ctx.get("atr_ratio", 1) > 2.5:
        score -= 12
        reasons_minus.append("volatility spike")

    score = max(0, min(100, score))
    if score >= 95:
        size_pct = 100
    elif score >= 80:
        size_pct = 100
    elif score >= 60:
        size_pct = 50
    else:
        size_pct = 0

    return {
        "setup_score": round(score, 1),
        "position_size_pct": size_pct,
        "allow": score >= min_score,
        "reasons_plus": reasons_plus,
        "reasons_minus": reasons_minus,
    }


def build_mnq_context(
    dt: datetime,
    row_1m: pd.Series,
    ctx_5m: Dict[str, Any],
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_15m: Optional[pd.DataFrame] = None,
    news_blocked: bool = False,
    news_event: Optional[str] = None,
) -> Dict[str, Any]:
    """Rich context packet for LLM + rule filters."""
    sess = classify_session(dt)
    price = float(row_1m["close"])
    atr = float(row_1m.get("atr", 0) or 0)

    # ATR vs median
    atr_med = float(df_1m["atr"].tail(200).median()) if "atr" in df_1m.columns else atr
    atr_ratio = (atr / atr_med) if atr_med > 0 else 1.0

    vwap_val = None
    if "vwap" in row_1m.index and not pd.isna(row_1m.get("vwap")):
        vwap_val = float(row_1m["vwap"])
    elif "vwap" in df_1m.columns:
        idx = df_1m.index.get_loc(row_1m.name) if row_1m.name in df_1m.index else -1
        if idx >= 0:
            vwap_val = float(df_1m.iloc[idx]["vwap"])

    # Previous day high/low (ET date)
    et = _to_et(dt)
    et_date = et.date()
    if "datetime" in df_5m.columns:
        dts = pd.to_datetime(df_5m["datetime"])
        if dts.dt.tz is None:
            dts = dts.dt.tz_localize("UTC")
        et_dates = dts.dt.tz_convert("US/Eastern").dt.date
        prior = df_5m[et_dates < et_date]
        prev_day = prior.tail(78) if len(prior) else pd.DataFrame()
        prev_high = float(prev_day["high"].max()) if len(prev_day) else None
        prev_low = float(prev_day["low"].min()) if len(prev_day) else None
    else:
        prev_high = prev_low = None

    # Opening range (first 30 min RTH = 14:30–15:00 UTC approx)
    or_high = or_low = None
    if "datetime" in df_5m.columns:
        dts = pd.to_datetime(df_5m["datetime"])
        day_mask = dts.dt.date == (dt.date() if dt.tzinfo else dt.replace(tzinfo=timezone.utc).date())
        day_bars = df_5m[day_mask].head(6)
        if len(day_bars):
            or_high = float(day_bars["high"].max())
            or_low = float(day_bars["low"].min())

    trend_1m = "bullish" if row_1m.get("ema_9", 0) > row_1m.get("ema_21", 0) else "bearish"
    trend_5m = ctx_5m.get("trend")
    trend_15m = None
    if df_15m is not None and len(df_15m) > 50 and "datetime" in df_15m.columns:
        mask = df_15m["datetime"] <= dt
        if mask.any():
            r15 = df_15m[mask].iloc[-1]
            if "ema_50" in r15 and "ema_200" in r15:
                trend_15m = "bullish" if r15["ema_50"] > r15["ema_200"] else "bearish"

    vol_ratio = float(row_1m.get("volume_ratio", 1) or 1)
    structure = classify_market_structure(ctx_5m.get("adx", 0), vol_ratio, atr_ratio)

    resistance = ctx_5m.get("resistance", price)
    support = ctx_5m.get("support", price)
    near_res = abs(price - resistance) < atr * 0.5 if atr else False
    near_sup = abs(price - support) < atr * 0.5 if atr else False

    mtf_aligned = trend_5m == trend_1m and (trend_15m is None or trend_15m == trend_1m)

    return {
        "timestamp_utc": dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
        "session": sess["session"],
        "session_quality": sess["quality"],
        "session_allow_scalp": sess["allow_scalp"],
        "et_time": sess["et_time"],
        "price": price,
        "vwap": vwap_val,
        "ema_20_1m": float(row_1m.get("ema_21", 0)),
        "ema_50_5m": float(ctx_5m.get("ema_50", 0)) if "ema_50" in ctx_5m else None,
        "rsi_1m": float(row_1m.get("rsi", 50)),
        "atr_1m": atr,
        "atr_ratio_vs_median": round(atr_ratio, 2),
        "volume_ratio_1m": vol_ratio,
        "trend_1m": trend_1m,
        "trend_5m": trend_5m,
        "trend_15m": trend_15m,
        "mtf_aligned": mtf_aligned,
        "market_structure": structure,
        "adx_5m": float(ctx_5m.get("adx", 0)),
        "prev_day_high": prev_high,
        "prev_day_low": prev_low,
        "opening_range_high": or_high,
        "opening_range_low": or_low,
        "resistance": resistance,
        "support": support,
        "near_resistance": near_res,
        "near_support": near_sup,
        "news_blocked": news_blocked,
        "news_event": news_event,
    }
