"""
Entry-quality gates — optional filters to avoid weak 1M entries in MTF scalping.

Loaded from data/mnq_profit_config.json → entry_quality block.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.utils.trading_session import SESSION_EXTENDED, SESSION_RTH, is_rth_session_et

DEFAULT_ENTRY_QUALITY: Dict[str, Any] = {
    "enabled": False,
    # Shorts: price must be at/below EMA9 (no rally tolerance above)
    "short_below_ema9": False,
    "long_above_ema9": False,
    # MACD histogram sign + slope (negative & falling for shorts)
    "short_macd_negative": False,
    "long_macd_positive": False,
    # Block if last N 1M bars were all green (short) or all red (long)
    "short_block_green_bars": 0,
    "long_block_red_bars": 0,
    # Block if last 1M candle is bearish (close < open)
    "long_block_bearish_bar": False,
    # 5M DI margin for trend shorts/longs (not counter-trend)
    "short_di_margin": 0.0,
    "long_di_margin": 0.0,
    # Stricter ADX outside RTH when session_mode=extended
    "extended_adx_min": 0,
    # RSI momentum: short needs RSI falling from elevated zone
    "short_rsi_falling_max": 0,
    "long_rsi_rising_min": 0,
}


def parse_entry_quality(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge entry_quality block from profit config with defaults."""
    out = dict(DEFAULT_ENTRY_QUALITY)
    if not cfg:
        return out
    block = cfg.get("entry_quality")
    if isinstance(block, dict):
        out.update(block)
    # Flat legacy keys (optional)
    for key in DEFAULT_ENTRY_QUALITY:
        if key in cfg and key not in ("enabled",):
            out[key] = cfg[key]
    return out


def entry_quality_enabled(eq: Dict[str, Any]) -> bool:
    if eq.get("enabled"):
        return True
    return any(
        eq.get(k)
        for k in (
            "short_below_ema9",
            "long_above_ema9",
            "short_macd_negative",
            "long_macd_positive",
            "short_block_green_bars",
            "long_block_red_bars",
            "long_block_bearish_bar",
            "short_di_margin",
            "long_di_margin",
            "extended_adx_min",
            "short_rsi_falling_max",
            "long_rsi_rising_min",
        )
    )


def _last_n_bars(df_1m: Optional[pd.DataFrame], n: int) -> Optional[pd.DataFrame]:
    if df_1m is None or n <= 0 or len(df_1m) < n:
        return None
    return df_1m.tail(n)


def _all_green(df: pd.DataFrame) -> bool:
    return bool((df["close"] > df["open"]).all())


def _all_red(df: pd.DataFrame) -> bool:
    return bool((df["close"] < df["open"]).all())


def extended_session_adx_min(
    eq: Dict[str, Any],
    ctx_5m: Dict[str, Any],
    timestamp,
    session_mode: str = SESSION_RTH,
) -> Tuple[bool, str]:
    """Require higher ADX during extended (non-RTH) hours."""
    min_adx = float(eq.get("extended_adx_min") or 0)
    if min_adx <= 0:
        return True, ""
    if session_mode != SESSION_EXTENDED:
        return True, ""
    if timestamp is None:
        return True, ""
    if is_rth_session_et(timestamp):
        return True, ""
    adx = ctx_5m.get("adx", 0)
    if adx >= min_adx:
        return True, ""
    return False, (
        f"Overnight session needs stronger trend (ADX {adx:.0f}, want {min_adx:.0f}+) "
        f"— skipping chop"
    )


def check_short_entry_quality(
    row_1m: pd.Series,
    ctx_5m: Dict[str, Any],
    eq: Dict[str, Any],
    *,
    is_counter_trend: bool = False,
    df_1m: Optional[pd.DataFrame] = None,
    timestamp=None,
    session_mode: str = SESSION_RTH,
) -> Tuple[bool, str]:
    """Return (ok, skip_reason). Empty skip_reason when ok."""
    if not entry_quality_enabled(eq):
        return True, ""

    ok, msg = extended_session_adx_min(eq, ctx_5m, timestamp, session_mode)
    if not ok:
        return False, msg

    price = row_1m["close"]
    ema_9 = row_1m["ema_9"]
    macd_hist = row_1m["macd_hist"]
    macd_hist_prev = row_1m["macd_hist_prev"]
    rsi = row_1m.get("rsi")
    rsi_prev = row_1m.get("rsi_prev")

    if not is_counter_trend:
        margin = float(eq.get("short_di_margin") or 0)
        if margin > 0:
            di_gap = ctx_5m.get("di_minus", 0) - ctx_5m.get("di_plus", 0)
            if di_gap < margin:
                return False, (
                    f"Sellers not dominant enough on 5M (DI- lead {di_gap:.0f} pts, "
                    f"want {margin:.0f}+)"
                )

    if eq.get("short_below_ema9") and not pd.isna(ema_9) and price > ema_9:
        return False, "Price still above EMA9 — wait for pullback, not a 1M rally"

    if eq.get("short_macd_negative"):
        if pd.isna(macd_hist) or pd.isna(macd_hist_prev):
            return False, "MACD not ready yet"
        if macd_hist >= 0:
            return False, "MACD still positive — momentum not bearish enough to short"
        if macd_hist >= macd_hist_prev:
            return False, "MACD histogram not falling — wait for downward momentum"

    n_green = int(eq.get("short_block_green_bars") or 0)
    if n_green > 0:
        recent = _last_n_bars(df_1m, n_green)
        if recent is not None and _all_green(recent):
            return False, (
                f"Last {n_green} minute bars were all green — price still bouncing up"
            )

    rsi_max = float(eq.get("short_rsi_falling_max") or 0)
    if rsi_max > 0 and not pd.isna(rsi) and not pd.isna(rsi_prev):
        if rsi > rsi_max or rsi >= rsi_prev:
            return False, (
                f"RSI not rolling over yet ({rsi:.0f}) — wait for momentum to fade"
            )

    return True, ""


def check_long_entry_quality(
    row_1m: pd.Series,
    ctx_5m: Dict[str, Any],
    eq: Dict[str, Any],
    *,
    df_1m: Optional[pd.DataFrame] = None,
    timestamp=None,
    session_mode: str = SESSION_RTH,
    is_counter_trend: bool = False,
) -> Tuple[bool, str]:
    """Return (ok, skip_reason). Empty skip_reason when ok."""
    if not entry_quality_enabled(eq):
        return True, ""

    ok, msg = extended_session_adx_min(eq, ctx_5m, timestamp, session_mode)
    if not ok:
        return False, msg

    price = row_1m["close"]
    ema_9 = row_1m["ema_9"]
    macd_hist = row_1m["macd_hist"]
    macd_hist_prev = row_1m["macd_hist_prev"]
    rsi = row_1m.get("rsi")
    rsi_prev = row_1m.get("rsi_prev")

    margin = float(eq.get("long_di_margin") or 0)
    if margin > 0 and not is_counter_trend:
        di_gap = ctx_5m.get("di_plus", 0) - ctx_5m.get("di_minus", 0)
        if di_gap < margin:
            return False, (
                f"Buyers not dominant enough on 5M (DI+ lead {di_gap:.0f} pts, "
                f"want {margin:.0f}+)"
            )

    if eq.get("long_above_ema9") and not pd.isna(ema_9) and price < ema_9:
        return False, "Price still below EMA9 — wait for bounce, not a 1M dip"

    if eq.get("long_macd_positive"):
        if pd.isna(macd_hist) or pd.isna(macd_hist_prev):
            return False, "MACD not ready yet"
        if macd_hist <= 0:
            return False, "MACD still negative — momentum not bullish enough to buy"
        if macd_hist <= macd_hist_prev:
            return False, "MACD histogram not rising — wait for upward momentum"

    n_red = int(eq.get("long_block_red_bars") or 0)
    if n_red > 0:
        recent = _last_n_bars(df_1m, n_red)
        if recent is not None and _all_red(recent):
            return False, (
                f"Last {n_red} minute bars were all red — price still selling off"
            )

    if eq.get("long_block_bearish_bar"):
        candle_open = row_1m.get("open")
        if not pd.isna(candle_open) and price < candle_open:
            return False, "Last 1M candle bearish — wait for buyers to step in"

    rsi_min = float(eq.get("long_rsi_rising_min") or 0)
    if rsi_min > 0 and not pd.isna(rsi) and not pd.isna(rsi_prev):
        if rsi < rsi_min or rsi <= rsi_prev:
            return False, (
                f"RSI not turning up yet ({rsi:.0f}) — wait for momentum to build"
            )

    return True, ""
