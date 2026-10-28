"""
Live Version B scalp momentum — 5M VWAP+EMA20+ADX, 1M pullback, trigger bar.

Mirrors scripts/backtest_scalp_momentum.py variant B (no no-chase filter).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.strategy.entry_diagnostics import gate_line


ADX_MIN = 20
EMA_TREND = 20
PULLBACK_ATR = 0.5
MAX_SETUP_BARS = 3


@dataclass
class ScalpSymbolState:
    phase: int = 0
    trend_dir: int = 0
    setup_until_ts: Optional[pd.Timestamp] = None


def scalp_5m_trend(row_5m: pd.Series, adx_min: int = ADX_MIN) -> int:
    """Return 1 long, -1 short, 0 none."""
    adx = float(row_5m.get("adx", 0))
    if pd.isna(adx) or adx < adx_min:
        return 0
    close = float(row_5m["close"])
    vwap = float(row_5m.get("vwap", float("nan")))
    ema20 = float(row_5m.get("ema_20", float("nan")))
    if pd.isna(vwap) or pd.isna(ema20):
        return 0
    if close > vwap and close > ema20:
        return 1
    if close < vwap and close < ema20:
        return -1
    return 0


def _in_pullback_zone(
    close: float,
    ema20: float,
    atr: float,
    direction: int,
    pb_limit: float = PULLBACK_ATR,
) -> bool:
    if pd.isna(atr) or atr <= 0 or pd.isna(ema20):
        return False
    zone = atr * pb_limit
    dist = abs(close - ema20)
    if dist > zone:
        return False
    if direction == 1:
        return close >= ema20 - zone * 0.2
    if direction == -1:
        return close <= ema20 + zone * 0.2
    return False


def _trigger_fired(o: float, c: float, prev_high: float, prev_low: float, direction: int) -> bool:
    if direction == 1:
        return c > o and c > prev_high
    if direction == -1:
        return c < o and c < prev_low
    return False


def evaluate_scalp_gates(
    row_1m: pd.Series,
    row_5m: pd.Series,
    prev_1m: Optional[pd.Series],
    state: ScalpSymbolState,
    *,
    row_30s: Optional[pd.Series] = None,
    prev_30s: Optional[pd.Series] = None,
    adx_min: int = ADX_MIN,
    pullback_atr: float = PULLBACK_ATR,
    setup_bars: int = MAX_SETUP_BARS,
    sl_pts: float = 8.0,
    tp_pts: float = 15.0,
) -> Tuple[List[str], ScalpSymbolState]:
    """Return diagnostic lines and updated state (dry-run friendly copy)."""
    st = ScalpSymbolState(
        phase=state.phase,
        trend_dir=state.trend_dir,
        setup_until_ts=state.setup_until_ts,
    )
    lines: List[str] = []
    direction = scalp_5m_trend(row_5m, adx_min=adx_min)
    adx = float(row_5m.get("adx", 0))
    close_5m = float(row_5m["close"])
    vwap = float(row_5m.get("vwap", float("nan")))
    ema20_5m = float(row_5m.get("ema_20", float("nan")))

    if direction == 1:
        lines.append(gate_line(True, "long", f"5M scalp trend UP (close>{vwap:.2f} VWAP, >EMA20 {ema20_5m:.2f}, ADX {adx:.0f})"))
        lines.append(gate_line(False, "short", f"5M trend bullish (counter-trend disabled in scalp_b)"))
    elif direction == -1:
        lines.append(gate_line(False, "long", f"5M trend bearish (counter-trend disabled in scalp_b)"))
        lines.append(gate_line(True, "short", f"5M scalp trend DOWN (close<{vwap:.2f} VWAP, <EMA20 {ema20_5m:.2f}, ADX {adx:.0f})"))
    else:
        reason = f"ADX {adx:.0f} < {adx_min}" if adx < adx_min else "price between VWAP/EMA20"
        lines.append(gate_line(False, "long", f"no 5M scalp trend ({reason})"))
        lines.append(gate_line(False, "short", f"no 5M scalp trend ({reason})"))
        st.phase = 0
        st.trend_dir = 0
        return lines, st

    for side, d in (("long", 1), ("short", -1)):
        if direction != d:
            continue
        close = float(row_1m["close"])
        ema20_1m = float(row_1m.get("ema_20", float("nan")))
        atr = float(row_1m.get("atr", 0))
        ts = pd.Timestamp(row_1m.get("datetime")) if row_1m.get("datetime") is not None else None

        in_pb = _in_pullback_zone(close, ema20_1m, atr, d, pb_limit=pullback_atr)
        if st.phase == 0:
            if in_pb:
                st.phase = 1
                st.trend_dir = d
                if ts is not None:
                    st.setup_until_ts = ts + pd.Timedelta(minutes=setup_bars)
                lines.append(gate_line(True, side, f"1M pullback to EMA20 (dist {abs(close - ema20_1m):.2f}) — setup armed"))
            else:
                lines.append(gate_line(
                    False, side,
                    f"1M pullback zone (dist {abs(close - ema20_1m):.2f}, need ≤{atr * pullback_atr:.2f} ATR×{pullback_atr})",
                ))
            lines.append(gate_line(False, side, "30s/1M trigger not fired yet"))
            continue

        if st.phase == 1 and st.trend_dir == d:
            if ts is not None and st.setup_until_ts is not None and ts > st.setup_until_ts:
                lines.append(gate_line(False, side, "setup window expired — waiting for new pullback"))
                st.phase = 0
                continue
            if direction != st.trend_dir:
                lines.append(gate_line(False, side, "5M trend changed — setup reset"))
                st.phase = 0
                continue
            trig_row = row_30s if row_30s is not None else row_1m
            prev_trig = prev_30s if row_30s is not None else prev_1m
            prev_h = float(prev_trig["high"]) if prev_trig is not None else float("nan")
            prev_l = float(prev_trig["low"]) if prev_trig is not None else float("nan")
            o = float(trig_row["open"])
            c = float(trig_row["close"])
            fired = _trigger_fired(o, c, prev_h, prev_l, d)
            tf = "30s" if row_30s is not None else "1M"
            if fired:
                lines.append(gate_line(True, side, f"{tf} trigger fired (SL {sl_pts}pt TP {tp_pts}pt)"))
                lines.append(gate_line(True, side, "ALL GATES PASSED — scalp_b entry ready"))
            else:
                need = "green close > prev high" if d == 1 else "red close < prev low"
                lines.append(gate_line(False, side, f"waiting for {tf} trigger ({need})"))
        else:
            lines.append(gate_line(False, side, "setup not armed"))

    lines.append(gate_line(True, "long" if direction == 1 else "short", "no MACD/RSI/15M/flow gates in scalp_b"))
    return lines, st


def check_scalp_entry(
    symbol: str,
    row_1m: pd.Series,
    row_5m: pd.Series,
    prev_1m: Optional[pd.Series],
    state: ScalpSymbolState,
    *,
    row_30s: Optional[pd.Series] = None,
    prev_30s: Optional[pd.Series] = None,
    adx_min: int = ADX_MIN,
    pullback_atr: float = PULLBACK_ATR,
    setup_bars: int = MAX_SETUP_BARS,
    sl_pts: float = 8.0,
    tp_pts: float = 15.0,
) -> Tuple[Optional[Dict[str, Any]], ScalpSymbolState]:
    """Evaluate scalp_b entry; returns signal dict or None and updated state."""
    direction = scalp_5m_trend(row_5m, adx_min=adx_min)
    ts = pd.Timestamp(row_1m.get("datetime")) if row_1m.get("datetime") is not None else None
    close = float(row_1m["close"])
    ema20_1m = float(row_1m.get("ema_20", float("nan")))
    atr = float(row_1m.get("atr", 0))

    if direction == 0:
        state.phase = 0
        state.trend_dir = 0
        return None, state

    if state.phase == 0:
        if _in_pullback_zone(close, ema20_1m, atr, direction, pb_limit=pullback_atr):
            state.phase = 1
            state.trend_dir = direction
            if ts is not None:
                state.setup_until_ts = ts + pd.Timedelta(minutes=setup_bars)
        return None, state

    if state.phase != 1 or state.trend_dir != direction:
        state.phase = 0
        state.trend_dir = 0
        return None, state

    if ts is not None and state.setup_until_ts is not None and ts > state.setup_until_ts:
        state.phase = 0
        state.trend_dir = 0
        return None, state

    if direction != state.trend_dir:
        state.phase = 0
        state.trend_dir = 0
        return None, state

    trig_row = row_30s if row_30s is not None else row_1m
    prev_trig = prev_30s if row_30s is not None else prev_1m
    if prev_trig is None:
        return None, state

    o = float(trig_row["open"])
    c = float(trig_row["close"])
    prev_h = float(prev_trig["high"])
    prev_l = float(prev_trig["low"])
    if not _trigger_fired(o, c, prev_h, prev_l, direction):
        return None, state

    entry_price = c
    if direction == 1:
        sl = entry_price - sl_pts
        tp = entry_price + tp_pts
        dir_str = "long"
    else:
        sl = entry_price + sl_pts
        tp = entry_price - tp_pts
        dir_str = "short"

    state.phase = 0
    state.trend_dir = 0
    state.setup_until_ts = None

    return {
        "symbol": symbol,
        "direction": dir_str,
        "entry": entry_price,
        "sl": sl,
        "tp": tp,
        "atr": atr if not pd.isna(atr) else sl_pts,
        "scalp_b": True,
        "structure_capped": False,
    }, state
