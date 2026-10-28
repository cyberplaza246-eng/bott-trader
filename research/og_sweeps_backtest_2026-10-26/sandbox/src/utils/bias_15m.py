"""Shared 15M higher-timeframe bias rules (live + backtest parity)."""
from __future__ import annotations

from typing import Any, Dict, Optional

VALID_15M_BIAS_MODES = ("ema_cross", "price_ema50", "hybrid")
DEFAULT_15M_BIAS_MODE = "ema_cross"


def normalize_15m_bias_mode(mode: Optional[str]) -> str:
    m = (mode or DEFAULT_15M_BIAS_MODE).strip().lower()
    if m in VALID_15M_BIAS_MODES:
        return m
    return DEFAULT_15M_BIAS_MODE


def resolve_15m_bias_mode(cfg: Optional[Dict[str, Any]] = None) -> str:
    import os

    if cfg and cfg.get("15m_bias_mode"):
        return normalize_15m_bias_mode(str(cfg["15m_bias_mode"]))
    env = os.getenv("15M_BIAS_MODE", "").strip()
    if env:
        return normalize_15m_bias_mode(env)
    return DEFAULT_15M_BIAS_MODE


def resolve_15m_bias_buffer_pts(cfg: Optional[Dict[str, Any]] = None) -> float:
    import os

    if cfg and "15m_bias_buffer_pts" in cfg:
        return float(cfg["15m_bias_buffer_pts"])
    raw = os.getenv("15M_BIAS_BUFFER_PTS", "").strip()
    if raw:
        return float(raw)
    return 0.0


def compute_15m_trend(
    close: float,
    ema50: float,
    ema200: float,
    mode: str = DEFAULT_15M_BIAS_MODE,
    buffer_pts: float = 0.0,
) -> str:
    """Return bullish / bearish / neutral from last 15M bar."""
    mode = normalize_15m_bias_mode(mode)

    if buffer_pts > 0 and mode in ("price_ema50", "hybrid"):
        if abs(close - ema50) <= buffer_pts:
            return "neutral"

    if mode == "ema_cross":
        if ema50 > ema200:
            return "bullish"
        if ema50 < ema200:
            return "bearish"
        return "neutral"

    if mode == "price_ema50":
        if close > ema50:
            return "bullish"
        if close < ema50:
            return "bearish"
        return "neutral"

    # hybrid: bullish if price above EMA50 OR structure bullish; bearish only when both agree down
    if close > ema50 or ema50 > ema200:
        return "bullish"
    if close < ema50 and ema50 < ema200:
        return "bearish"
    return "neutral"


def trend_rule_label(
    trend: str,
    close: float,
    ema50: float,
    ema200: float,
    mode: str = DEFAULT_15M_BIAS_MODE,
) -> str:
    mode = normalize_15m_bias_mode(mode)
    gap = ema50 - ema200
    if mode == "price_ema50":
        diff = close - ema50
        if trend == "bullish":
            return f"close {close:.2f} > EMA50 {ema50:.2f} (+{diff:.1f})"
        if trend == "bearish":
            return f"close {close:.2f} < EMA50 {ema50:.2f} ({diff:.1f})"
        return f"close {close:.2f} ≈ EMA50 {ema50:.2f}"
    if mode == "hybrid":
        return (
            f"close {close:.2f} vs EMA50 {ema50:.2f}; "
            f"EMA50 {ema50:.2f} vs EMA200 {ema200:.2f} ({gap:+.1f})"
        )
    if trend == "bullish":
        return f"EMA50 {ema50:.2f} > EMA200 {ema200:.2f} (+{gap:.1f})"
    if trend == "bearish":
        return f"EMA50 {ema50:.2f} < EMA200 {ema200:.2f} ({gap:.1f})"
    return f"EMA50 {ema50:.2f} ≈ EMA200 {ema200:.2f}"
