"""
Hybrid scalp — Pullback (A) + Continuation (B) with 30s trigger and chase protection.

State: WAIT → TREND → PULLBACK|CONTINUATION setup → 30s trigger → ENTER
Optional momentum burst: strong ADX + 30s break without full setup arming.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.strategy.entry_diagnostics import gate_line


PHASE_WAIT = 0
PHASE_SETUP = 1

ADX_MIN_PULLBACK = 20
ADX_MIN_CONTINUATION = 25
EMA_TREND = 20
PULLBACK_ATR = 0.5
MAX_SETUP_BARS = 3
CHASE_BODY_MULT = 1.5
CHASE_EMA_ATR = 1.0
CHASE_EMA_ATR_FLOW = 2.0
CHASE_BODY_MULT_FLOW = 2.0
CHASE_FLOW_RELAX = True
CHASE_SKIP_ON_FLOW = False
CHASE_FLOW_BUY_PCT_LONG = 0.60
CHASE_FLOW_BUY_PCT_SHORT = 0.40
CHASE_FLOW_DELTA_MIN = 0.0
CANDLE_AVG_PERIOD = 20
TREND_MODE_BOTH = "both"
TREND_MODE_VWAP = "vwap"


@dataclass
class ScalpHybridState:
    long_phase: int = 0
    long_mode: str = ""
    long_setup_until: Optional[pd.Timestamp] = None
    short_phase: int = 0
    short_mode: str = ""
    short_setup_until: Optional[pd.Timestamp] = None


def _bar_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    dt = row.get("datetime")
    if dt is None or pd.isna(dt):
        return None
    return pd.Timestamp(dt)


def _trend_pullback(
    row_5m: pd.Series,
    adx_min: int = ADX_MIN_PULLBACK,
    trend_mode: str = TREND_MODE_BOTH,
) -> int:
    """5M trend for pullback scalper: 1 long, -1 short, 0 none."""
    adx = float(row_5m.get("adx", 0))
    if pd.isna(adx) or adx < adx_min:
        return 0
    close = float(row_5m["close"])
    vwap = float(row_5m.get("vwap", float("nan")))
    ema20 = float(row_5m.get("ema_20", float("nan")))
    if pd.isna(vwap):
        return 0
    if trend_mode == TREND_MODE_VWAP:
        if close > vwap:
            return 1
        if close < vwap:
            return -1
        return 0
    if pd.isna(ema20):
        return 0
    if close > vwap and close > ema20:
        return 1
    if close < vwap and close < ema20:
        return -1
    return 0


def _continuation_ok(
    row_5m: pd.Series,
    prev_5m: Optional[pd.Series],
    direction: int,
    adx_min: int = ADX_MIN_CONTINUATION,
    trend_mode: str = TREND_MODE_BOTH,
    volume_strict: bool = True,
    volume_min_ratio: float = 0.85,
) -> Tuple[bool, str]:
    adx = float(row_5m.get("adx", 0))
    if pd.isna(adx) or adx < adx_min:
        return False, f"ADX {adx:.0f} < {adx_min}"
    close = float(row_5m["close"])
    vwap = float(row_5m.get("vwap", float("nan")))
    ema20 = float(row_5m.get("ema_20", float("nan")))
    if pd.isna(vwap):
        return False, "missing VWAP"
    if trend_mode == TREND_MODE_VWAP:
        if direction == 1 and not (close > vwap):
            return False, "5M not above VWAP"
        if direction == -1 and not (close < vwap):
            return False, "5M not below VWAP"
    else:
        if pd.isna(ema20):
            return False, "missing EMA20"
        if direction == 1:
            if not (close > vwap and close > ema20):
                return False, "5M not above VWAP/EMA20"
        elif direction == -1:
            if not (close < vwap and close < ema20):
                return False, "5M not below VWAP/EMA20"
        else:
            return False, "invalid direction"
    vol = float(row_5m.get("volume", 0))
    if prev_5m is None:
        return False, "no prior 5M bar for volume compare"
    prev_vol = float(prev_5m.get("volume", 0))
    if volume_strict:
        if vol <= prev_vol:
            return False, f"volume not increasing ({vol:.0f} ≤ {prev_vol:.0f})"
    elif prev_vol > 0 and vol < prev_vol * volume_min_ratio:
        return False, (
            f"volume declining sharply ({vol:.0f} < {volume_min_ratio:.0%}× prev {prev_vol:.0f})"
        )
    vol_note = "vol rising" if volume_strict else "vol OK"
    return True, f"ADX {adx:.0f}, {vol_note}"


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


TRIGGER_MICRO_BREAK_PTS = 0.25
FLOW_TRIGGER_DELTA_MIN = 30.0
FLOW_TRIGGER_BUY_PCT_LONG = 0.55
FLOW_TRIGGER_BUY_PCT_SHORT = 0.45
FLOW_STRONG_DELTA_MIN = 100.0
FLOW_STRONG_BUY_PCT_LONG = 0.60
FLOW_STRONG_BUY_PCT_SHORT = 0.40
TRIGGER_RELAX_FLOW = False
TRIGGER_RELAX_BUY_PCT_LONG = 0.60
TRIGGER_RELAX_BUY_PCT_SHORT = 0.40
TRIGGER_RELAX_DELTA_MIN = 0.0
RSI_GATE_ENABLED = True
RSI_MIN_SHORT = 40.0
RSI_MAX_LONG = 65.0
RSI_RELAX_ADX = 25.0
RSI_RELAX_MIN_SHORT = 35.0
RSI_RELAX_MAX_LONG = 70.0
RSI_OVERBOUGHT = 65.0
RSI_OVERSOLD = 35.0


def _flow_confirms_trigger(
    flow_snap: Optional[Dict],
    direction: int,
    *,
    delta_min: float = FLOW_TRIGGER_DELTA_MIN,
    buy_pct_long_min: float = FLOW_TRIGGER_BUY_PCT_LONG,
    buy_pct_short_max: float = FLOW_TRIGGER_BUY_PCT_SHORT,
) -> bool:
    if not flow_snap:
        return False
    delta = float(flow_snap.get("delta", 0) or 0)
    buy_pct = float(flow_snap.get("buy_pct", 0.5) or 0.5)
    if direction == 1:
        return delta > delta_min and buy_pct > buy_pct_long_min
    if direction == -1:
        return delta < -delta_min and buy_pct < buy_pct_short_max
    return False


def _flow_very_strong(
    flow_snap: Optional[Dict],
    direction: int,
    *,
    delta_min: float = FLOW_STRONG_DELTA_MIN,
    buy_pct_long_min: float = FLOW_STRONG_BUY_PCT_LONG,
    buy_pct_short_max: float = FLOW_STRONG_BUY_PCT_SHORT,
) -> bool:
    if not flow_snap:
        return False
    delta = float(flow_snap.get("delta", 0) or 0)
    buy_pct = float(flow_snap.get("buy_pct", 0.5) or 0.5)
    if direction == 1:
        return delta > delta_min and buy_pct > buy_pct_long_min
    if direction == -1:
        return delta < -delta_min and buy_pct < buy_pct_short_max
    return False


def default_rsi_gate_cfg() -> Dict[str, Any]:
    """RSI confirmation gate (overridden via env in live runner)."""
    return {
        "rsi_gate_enabled": RSI_GATE_ENABLED,
        "rsi_min_short": RSI_MIN_SHORT,
        "rsi_max_long": RSI_MAX_LONG,
        "rsi_relax_adx": RSI_RELAX_ADX,
        "rsi_relax_min_short": RSI_RELAX_MIN_SHORT,
        "rsi_relax_max_long": RSI_RELAX_MAX_LONG,
        "rsi_overbought": RSI_OVERBOUGHT,
        "rsi_oversold": RSI_OVERSOLD,
    }


def _rsi_gate_cfg_from(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    cfg = default_rsi_gate_cfg()
    explicit = kwargs.get("rsi_gate_cfg")
    if isinstance(explicit, dict):
        cfg.update(explicit)
    for key in cfg:
        if key in kwargs:
            cfg[key] = kwargs[key]
    return cfg


def _rsi_trend_relax_allowed(
    direction: int,
    row_5m: pd.Series,
    flow_snap: Optional[Dict],
    *,
    adx_min_pullback: int,
    trend_mode: str,
    relax_adx: float,
) -> bool:
    adx = float(row_5m.get("adx", 0))
    if pd.isna(adx) or adx < relax_adx:
        return False
    if _trend_pullback(row_5m, adx_min=adx_min_pullback, trend_mode=trend_mode) != direction:
        return False
    return _flow_confirms_trigger(flow_snap, direction) if flow_snap else False


def rsi_entry_block_reason(
    direction: int,
    row_1m: pd.Series,
    row_5m: pd.Series,
    row_30s: Optional[pd.Series],
    prev_30s: Optional[pd.Series],
    flow_snap: Optional[Dict],
    *,
    adx_min_pullback: int = ADX_MIN_PULLBACK,
    trend_mode: str = TREND_MODE_BOTH,
    rsi_gate_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return block reason when RSI gate rejects entry; None if allowed."""
    cfg = rsi_gate_cfg or default_rsi_gate_cfg()
    if not cfg.get("rsi_gate_enabled", RSI_GATE_ENABLED):
        return None
    rsi = float(row_1m.get("rsi", float("nan")))
    if pd.isna(rsi):
        return None
    rsi_prev = float(row_1m.get("rsi_prev", float("nan")))
    close_1m = float(row_1m["close"])
    ema20_1m = float(row_1m.get("ema_20", float("nan")))
    min_short = float(cfg.get("rsi_min_short", RSI_MIN_SHORT))
    max_long = float(cfg.get("rsi_max_long", RSI_MAX_LONG))
    relax_adx = float(cfg.get("rsi_relax_adx", RSI_RELAX_ADX))
    relax_min_short = float(cfg.get("rsi_relax_min_short", RSI_RELAX_MIN_SHORT))
    relax_max_long = float(cfg.get("rsi_relax_max_long", RSI_RELAX_MAX_LONG))
    overbought = float(cfg.get("rsi_overbought", RSI_OVERBOUGHT))
    oversold = float(cfg.get("rsi_oversold", RSI_OVERSOLD))
    can_relax = _rsi_trend_relax_allowed(
        direction, row_5m, flow_snap,
        adx_min_pullback=adx_min_pullback,
        trend_mode=trend_mode,
        relax_adx=relax_adx,
    )

    if direction == -1:
        floor = relax_min_short if can_relax else min_short
        if not pd.isna(ema20_1m) and close_1m <= ema20_1m and rsi < min_short:
            return (
                f"block: SHORT RSI oversold at EMA20 ({rsi:.0f} < {min_short:.0f}, "
                f"price ≤ EMA20 {ema20_1m:.2f})"
            )
        if row_30s is not None and prev_30s is not None:
            c30 = float(row_30s["close"])
            o30 = float(row_30s["open"])
            l30 = float(row_30s["low"])
            prev_l = float(prev_30s["low"])
            if c30 < o30 and l30 <= prev_l and rsi < floor:
                return f"block: SHORT extended down at low (RSI {rsi:.0f} < {floor:.0f})"
        falling_from_ob = (
            not pd.isna(rsi_prev) and rsi_prev >= overbought and rsi < rsi_prev
        )
        if rsi >= floor or falling_from_ob:
            return None
        return f"block: SHORT RSI oversold ({rsi:.0f} < {floor:.0f})"

    if direction == 1:
        ceiling = relax_max_long if can_relax else max_long
        if not pd.isna(ema20_1m) and close_1m >= ema20_1m and rsi > max_long:
            return (
                f"block: LONG RSI overbought at EMA20 ({rsi:.0f} > {max_long:.0f}, "
                f"price ≥ EMA20 {ema20_1m:.2f})"
            )
        rising_from_os = (
            not pd.isna(rsi_prev) and rsi_prev <= oversold and rsi > rsi_prev
        )
        if rsi <= max_long or rising_from_os:
            return None
        if can_relax and rsi <= ceiling:
            return None
        return f"block: LONG RSI overbought ({rsi:.0f} > {ceiling:.0f})"

    return None


def default_trigger_flow_cfg() -> Dict[str, Any]:
    """Flow-aware 30s trigger relax (overridden via env in live runner)."""
    return {
        "trigger_relax_flow": TRIGGER_RELAX_FLOW,
        "trigger_relax_buy_pct_long": TRIGGER_RELAX_BUY_PCT_LONG,
        "trigger_relax_buy_pct_short": TRIGGER_RELAX_BUY_PCT_SHORT,
        "trigger_relax_delta_min": TRIGGER_RELAX_DELTA_MIN,
    }


def _trigger_flow_cfg_from(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    cfg = default_trigger_flow_cfg()
    explicit = kwargs.get("trigger_flow_cfg")
    if isinstance(explicit, dict):
        cfg.update(explicit)
    for key in cfg:
        if key in kwargs:
            cfg[key] = kwargs[key]
    return cfg


def _flow_confirms_trigger_relax(
    flow_snap: Optional[Dict],
    direction: int,
    *,
    delta_min: float = TRIGGER_RELAX_DELTA_MIN,
    buy_pct_long_min: float = TRIGGER_RELAX_BUY_PCT_LONG,
    buy_pct_short_max: float = TRIGGER_RELAX_BUY_PCT_SHORT,
) -> bool:
    """Strong tape: skip green/red 30s bar requirement when trend + flow align."""
    if not flow_snap or direction not in (1, -1):
        return False
    delta = float(flow_snap.get("delta", 0) or 0)
    buy_pct = float(flow_snap.get("buy_pct", 0.5) or 0.5)
    if direction == 1:
        return delta > delta_min and buy_pct >= buy_pct_long_min
    return delta < -delta_min and buy_pct <= buy_pct_short_max


def _try_flow_trigger_relax(
    o: float,
    h: float,
    l: float,
    c: float,
    prev_high: float,
    prev_low: float,
    prev_close: float,
    direction: int,
    flow_snap: Optional[Dict],
    trigger_flow_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[bool, str]]:
    """Alternate 30s trigger paths when flow dominates but bar color fails."""
    cfg = trigger_flow_cfg or default_trigger_flow_cfg()
    if not cfg.get("trigger_relax_flow", TRIGGER_RELAX_FLOW):
        return None
    if not _flow_confirms_trigger_relax(
        flow_snap,
        direction,
        delta_min=float(cfg.get("trigger_relax_delta_min", TRIGGER_RELAX_DELTA_MIN)),
        buy_pct_long_min=float(cfg.get("trigger_relax_buy_pct_long", TRIGGER_RELAX_BUY_PCT_LONG)),
        buy_pct_short_max=float(cfg.get("trigger_relax_buy_pct_short", TRIGGER_RELAX_BUY_PCT_SHORT)),
    ):
        return None
    bar_mid = (h + l) / 2.0
    if direction == 1:
        if c > prev_high:
            return True, "flow relax: close > prev high (no green req)"
        if c > prev_close:
            return True, "flow relax: close > prev close"
        if c >= bar_mid:
            return True, "flow relax: close in upper half"
        if c > prev_low:
            return True, "flow relax: close > prev low"
        return None
    if c < prev_low:
        return True, "flow relax: close < prev low (no red req)"
    if c < prev_close:
        return True, "flow relax: close < prev close"
    if c <= bar_mid:
        return True, "flow relax: close in lower half"
    if c < prev_high:
        return True, "flow relax: close < prev high"
    return None


def _trigger_eval(
    o: float,
    c: float,
    prev_high: float,
    prev_low: float,
    prev_close: float,
    direction: int,
    *,
    h: Optional[float] = None,
    l: Optional[float] = None,
    aggressive: bool = False,
    setup_mode: str = "",
    flow_snap: Optional[Dict] = None,
    micro_break_pts: float = TRIGGER_MICRO_BREAK_PTS,
    flow_trigger_delta_min: float = FLOW_TRIGGER_DELTA_MIN,
    flow_trigger_buy_pct_long: float = FLOW_TRIGGER_BUY_PCT_LONG,
    flow_trigger_buy_pct_short: float = FLOW_TRIGGER_BUY_PCT_SHORT,
    flow_strong_delta_min: float = FLOW_STRONG_DELTA_MIN,
    flow_strong_buy_pct_long: float = FLOW_STRONG_BUY_PCT_LONG,
    flow_strong_buy_pct_short: float = FLOW_STRONG_BUY_PCT_SHORT,
    trigger_flow_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Evaluate 30s entry trigger; returns (fired, reason)."""
    hi = h if h is not None else max(o, c)
    lo = l if l is not None else min(o, c)
    if direction == 1:
        is_green = c > o
        if is_green and c > prev_high:
            if flow_snap is None or _flow_confirms_trigger(
                flow_snap, 1,
                delta_min=flow_trigger_delta_min,
                buy_pct_long_min=flow_trigger_buy_pct_long,
                buy_pct_short_max=flow_trigger_buy_pct_short,
            ):
                suffix = " + flow confirm" if flow_snap is not None else ""
                return True, f"bullish close > prev high{suffix}"
        if aggressive and setup_mode in ("continuation", "burst"):
            if c > prev_high + micro_break_pts:
                return True, f"micro-break +{micro_break_pts:.2f}pt above prev high"
            if _flow_very_strong(
                flow_snap, 1,
                delta_min=flow_strong_delta_min,
                buy_pct_long_min=flow_strong_buy_pct_long,
                buy_pct_short_max=flow_strong_buy_pct_short,
            ) and c > prev_close:
                return True, "strong flow burst (close > prev close)"
            if is_green and c > prev_close and _flow_confirms_trigger(
                flow_snap, 1,
                delta_min=flow_trigger_delta_min,
                buy_pct_long_min=flow_trigger_buy_pct_long,
                buy_pct_short_max=flow_trigger_buy_pct_short,
            ):
                return True, "green close > prev close + flow confirm"
        relax = _try_flow_trigger_relax(
            o, hi, lo, c, prev_high, prev_low, prev_close, 1, flow_snap, trigger_flow_cfg,
        )
        if relax is not None:
            return relax
        if not is_green:
            return False, "not bullish (close <= open)"
        if c <= prev_high:
            return False, f"close {c:.2f} <= prev high {prev_high:.2f}"
        return False, "trigger not met"

    if direction == -1:
        is_red = c < o
        if is_red and c < prev_low:
            if flow_snap is None or _flow_confirms_trigger(
                flow_snap, -1,
                delta_min=flow_trigger_delta_min,
                buy_pct_long_min=flow_trigger_buy_pct_long,
                buy_pct_short_max=flow_trigger_buy_pct_short,
            ):
                suffix = " + flow confirm" if flow_snap is not None else ""
                return True, f"bearish close < prev low{suffix}"
        if aggressive and setup_mode in ("continuation", "burst"):
            if c < prev_low - micro_break_pts:
                return True, f"micro-break -{micro_break_pts:.2f}pt below prev low"
            if _flow_very_strong(
                flow_snap, -1,
                delta_min=flow_strong_delta_min,
                buy_pct_long_min=flow_strong_buy_pct_long,
                buy_pct_short_max=flow_strong_buy_pct_short,
            ) and c < prev_close:
                return True, "strong flow burst (close < prev close)"
            if is_red and c < prev_close and _flow_confirms_trigger(
                flow_snap, -1,
                delta_min=flow_trigger_delta_min,
                buy_pct_long_min=flow_trigger_buy_pct_long,
                buy_pct_short_max=flow_trigger_buy_pct_short,
            ):
                return True, "red close < prev close + flow confirm"
        relax = _try_flow_trigger_relax(
            o, hi, lo, c, prev_high, prev_low, prev_close, -1, flow_snap, trigger_flow_cfg,
        )
        if relax is not None:
            return relax
        if not is_red:
            return False, "not bearish (close >= open)"
        if c >= prev_low:
            return False, f"close {c:.2f} >= prev low {prev_low:.2f}"
        return False, "trigger not met"

    return False, "invalid direction"


def _trigger_fired(
    o: float,
    c: float,
    prev_high: float,
    prev_low: float,
    direction: int,
    *,
    prev_close: Optional[float] = None,
    h: Optional[float] = None,
    l: Optional[float] = None,
    aggressive: bool = False,
    setup_mode: str = "",
    flow_snap: Optional[Dict] = None,
    **trigger_kwargs,
) -> bool:
    if prev_close is None:
        prev_close = prev_high if direction == 1 else prev_low
    fired, _ = _trigger_eval(
        o, c, prev_high, prev_low, prev_close, direction,
        h=h, l=l,
        aggressive=aggressive, setup_mode=setup_mode, flow_snap=flow_snap,
        **trigger_kwargs,
    )
    return fired


def format_30s_trigger_log(
    row_30s: pd.Series,
    prev_30s: pd.Series,
    direction: int,
    *,
    aggressive: bool = False,
    setup_mode: str = "",
    flow_snap: Optional[Dict] = None,
    **trigger_kwargs,
) -> str:
    """One-line 30s bar trigger evaluation for live scan logs."""
    o = float(row_30s["open"])
    h = float(row_30s["high"])
    l = float(row_30s["low"])
    c = float(row_30s["close"])
    prev_h = float(prev_30s["high"])
    prev_l = float(prev_30s["low"])
    prev_c = float(prev_30s["close"])
    fired, reason = _trigger_eval(
        o, c, prev_h, prev_l, prev_c, direction,
        h=h, l=l,
        aggressive=aggressive, setup_mode=setup_mode, flow_snap=flow_snap,
        **trigger_kwargs,
    )
    side = "LONG" if direction == 1 else "SHORT"
    return (
        f"30s eval {side}: O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f} | "
        f"trigger={'yes' if fired else 'no'} reason={reason}"
    )


def default_chase_flow_cfg() -> Dict[str, Any]:
    """Flow-aware chase relax settings (overridden via env in live runner)."""
    return {
        "chase_flow_relax": CHASE_FLOW_RELAX,
        "chase_skip_on_flow": CHASE_SKIP_ON_FLOW,
        "chase_ema_atr_flow": CHASE_EMA_ATR_FLOW,
        "chase_body_mult_flow": CHASE_BODY_MULT_FLOW,
        "chase_flow_buy_pct_long": CHASE_FLOW_BUY_PCT_LONG,
        "chase_flow_buy_pct_short": CHASE_FLOW_BUY_PCT_SHORT,
        "chase_flow_delta_min": CHASE_FLOW_DELTA_MIN,
    }


def _chase_flow_cfg_from(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    cfg = default_chase_flow_cfg()
    explicit = kwargs.get("chase_flow_cfg")
    if isinstance(explicit, dict):
        cfg.update(explicit)
    for key in cfg:
        if key in kwargs:
            cfg[key] = kwargs[key]
    return cfg


def _flow_confirms_chase_relax(
    flow_snap: Optional[Dict],
    direction: int,
    *,
    delta_min: float = CHASE_FLOW_DELTA_MIN,
    buy_pct_long_min: float = CHASE_FLOW_BUY_PCT_LONG,
    buy_pct_short_max: float = CHASE_FLOW_BUY_PCT_SHORT,
) -> bool:
    """Momentum continuation: positive delta + dominant buy/sell %."""
    if not flow_snap or direction not in (1, -1):
        return False
    delta = float(flow_snap.get("delta", 0) or 0)
    buy_pct = float(flow_snap.get("buy_pct", 0.5) or 0.5)
    if direction == 1:
        return delta > delta_min and buy_pct >= buy_pct_long_min
    return delta < -delta_min and buy_pct <= buy_pct_short_max


def _effective_chase_limits(
    body_mult: float,
    ema_atr: float,
    direction: Optional[int],
    flow_snap: Optional[Dict],
    cfg: Dict[str, Any],
) -> Tuple[bool, float, float]:
    """Return (skip_chase, body_mult, ema_atr). skip_chase=True allows extended entries."""
    if direction not in (1, -1):
        return False, body_mult, ema_atr
    if not _flow_confirms_chase_relax(
        flow_snap,
        direction,
        delta_min=float(cfg.get("chase_flow_delta_min", CHASE_FLOW_DELTA_MIN)),
        buy_pct_long_min=float(cfg.get("chase_flow_buy_pct_long", CHASE_FLOW_BUY_PCT_LONG)),
        buy_pct_short_max=float(cfg.get("chase_flow_buy_pct_short", CHASE_FLOW_BUY_PCT_SHORT)),
    ):
        return False, body_mult, ema_atr
    if cfg.get("chase_skip_on_flow", CHASE_SKIP_ON_FLOW):
        return True, body_mult, ema_atr
    if cfg.get("chase_flow_relax", CHASE_FLOW_RELAX):
        return (
            False,
            max(body_mult, float(cfg.get("chase_body_mult_flow", CHASE_BODY_MULT_FLOW))),
            max(ema_atr, float(cfg.get("chase_ema_atr_flow", CHASE_EMA_ATR_FLOW))),
        )
    return False, body_mult, ema_atr


def _chase_blocked(
    row_30s: pd.Series,
    row_1m: pd.Series,
    *,
    body_mult: float = CHASE_BODY_MULT,
    ema_atr: float = CHASE_EMA_ATR,
    direction: Optional[int] = None,
    flow_snap: Optional[Dict] = None,
    chase_flow_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    cfg = chase_flow_cfg or default_chase_flow_cfg()
    skip, body_mult, ema_atr = _effective_chase_limits(
        body_mult, ema_atr, direction, flow_snap, cfg,
    )
    if skip:
        return False, ""
    o = float(row_30s["open"])
    c = float(row_30s["close"])
    body = abs(c - o)
    avg_body = float(row_30s.get("avg_body", float("nan")))
    if not pd.isna(avg_body) and avg_body > 0 and body > body_mult * avg_body:
        return True, f"30s body {body:.2f} > {body_mult}x avg {avg_body:.2f}"
    ema20_1m = float(row_1m.get("ema_20", float("nan")))
    atr_1m = float(row_1m.get("atr", 0))
    if not pd.isna(ema20_1m) and not pd.isna(atr_1m) and atr_1m > 0:
        dist = abs(c - ema20_1m)
        if dist > ema_atr * atr_1m:
            return True, f"dist from 1M EMA20 {dist:.2f} > {ema_atr}x ATR ({atr_1m * ema_atr:.2f})"
    return False, ""


def _chase_check(
    row_30s: pd.Series,
    row_1m: pd.Series,
    direction: int,
    flow_snap: Optional[Dict],
    *,
    body_mult: float,
    ema_atr: float,
    chase_flow_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    return _chase_blocked(
        row_30s,
        row_1m,
        body_mult=body_mult,
        ema_atr=ema_atr,
        direction=direction,
        flow_snap=flow_snap,
        chase_flow_cfg=chase_flow_cfg,
    )


def _setup_window_sec(setup_bars: int, setup_window_sec: int, trigger_bar_seconds: int) -> int:
    if setup_window_sec > 0:
        return setup_window_sec
    return max(trigger_bar_seconds, setup_bars * trigger_bar_seconds)


def _side_label(direction: int) -> str:
    return "long" if direction == 1 else "short"


def _get_side_state(state: ScalpHybridState, direction: int) -> Tuple[int, str, Optional[pd.Timestamp]]:
    if direction == 1:
        return state.long_phase, state.long_mode, state.long_setup_until
    return state.short_phase, state.short_mode, state.short_setup_until


def _set_side_state(
    state: ScalpHybridState,
    direction: int,
    phase: int,
    mode: str,
    setup_until: Optional[pd.Timestamp],
) -> None:
    if direction == 1:
        state.long_phase = phase
        state.long_mode = mode
        state.long_setup_until = setup_until
    else:
        state.short_phase = phase
        state.short_mode = mode
        state.short_setup_until = setup_until


def _copy_state(state: ScalpHybridState) -> ScalpHybridState:
    return ScalpHybridState(
        long_phase=state.long_phase,
        long_mode=state.long_mode,
        long_setup_until=state.long_setup_until,
        short_phase=state.short_phase,
        short_mode=state.short_mode,
        short_setup_until=state.short_setup_until,
    )


def _evaluate_side_gates(
    side: str,
    direction: int,
    row_1m: pd.Series,
    row_5m: pd.Series,
    prev_5m: Optional[pd.Series],
    row_30s: Optional[pd.Series],
    prev_30s: Optional[pd.Series],
    state: ScalpHybridState,
    *,
    pullback_enabled: bool,
    continuation_enabled: bool,
    adx_min_pullback: int,
    adx_min_continuation: int,
    pullback_atr: float,
    setup_bars: int,
    setup_window_sec: int,
    trigger_bar_seconds: int,
    sl_pts: float,
    tp_pts: float,
    trend_mode: str,
    continuation_volume_strict: bool,
    cont_volume_min_ratio: float,
    chase_body_mult: float,
    chase_ema_atr: float,
    momentum_burst_enabled: bool,
    momentum_burst_adx: int,
    aggressive_mode: bool = False,
    flow_snap: Optional[Dict] = None,
    micro_break_pts: float = TRIGGER_MICRO_BREAK_PTS,
    flow_trigger_delta_min: float = FLOW_TRIGGER_DELTA_MIN,
    flow_trigger_buy_pct_long: float = FLOW_TRIGGER_BUY_PCT_LONG,
    flow_trigger_buy_pct_short: float = FLOW_TRIGGER_BUY_PCT_SHORT,
    flow_strong_delta_min: float = FLOW_STRONG_DELTA_MIN,
    flow_strong_buy_pct_long: float = FLOW_STRONG_BUY_PCT_LONG,
    flow_strong_buy_pct_short: float = FLOW_STRONG_BUY_PCT_SHORT,
    chase_flow_cfg: Optional[Dict[str, Any]] = None,
    trigger_flow_cfg: Optional[Dict[str, Any]] = None,
    rsi_gate_cfg: Optional[Dict[str, Any]] = None,
) -> List[str]:
    lines: List[str] = []
    phase, mode, setup_until = _get_side_state(state, direction)
    ts_30s = _bar_ts(row_30s) if row_30s is not None else _bar_ts(row_1m)
    window = _setup_window_sec(setup_bars, setup_window_sec, trigger_bar_seconds)

    pb_trend = _trend_pullback(row_5m, adx_min=adx_min_pullback, trend_mode=trend_mode)
    cont_ok, cont_reason = _continuation_ok(
        row_5m, prev_5m, direction, adx_min=adx_min_continuation, trend_mode=trend_mode,
        volume_strict=continuation_volume_strict, volume_min_ratio=cont_volume_min_ratio,
    )

    close_1m = float(row_1m["close"])
    ema20_1m = float(row_1m.get("ema_20", float("nan")))
    atr_1m = float(row_1m.get("atr", 0))
    in_pb = _in_pullback_zone(close_1m, ema20_1m, atr_1m, direction, pb_limit=pullback_atr)

    trend_ok = pb_trend == direction
    if not trend_ok and not (continuation_enabled and cont_ok):
        adx = float(row_5m.get("adx", 0))
        trend_label = "VWAP" if trend_mode == TREND_MODE_VWAP else "VWAP+EMA20"
        lines.append(gate_line(
            False, side,
            f"no 5M {side} trend ({trend_label}, ADX≥{adx_min_pullback}: dir={pb_trend}, ADX={adx:.0f})",
        ))
        _set_side_state(state, direction, PHASE_WAIT, "", None)
        if momentum_burst_enabled:
            adx = float(row_5m.get("adx", 0))
            if row_30s is not None and prev_30s is not None and not pd.isna(adx) and adx >= momentum_burst_adx:
                o = float(row_30s["open"])
                h = float(row_30s["high"])
                l = float(row_30s["low"])
                c = float(row_30s["close"])
                prev_h = float(prev_30s["high"])
                prev_l = float(prev_30s["low"])
                prev_c = float(prev_30s["close"])
                fired = _trigger_fired(
                    o, c, prev_h, prev_l, direction,
                    prev_close=prev_c,
                    h=h, l=l,
                    aggressive=aggressive_mode,
                    setup_mode="burst",
                    flow_snap=flow_snap,
                    micro_break_pts=micro_break_pts,
                    flow_trigger_delta_min=flow_trigger_delta_min,
                    flow_trigger_buy_pct_long=flow_trigger_buy_pct_long,
                    flow_trigger_buy_pct_short=flow_trigger_buy_pct_short,
                    flow_strong_delta_min=flow_strong_delta_min,
                    flow_strong_buy_pct_long=flow_strong_buy_pct_long,
                    flow_strong_buy_pct_short=flow_strong_buy_pct_short,
                    trigger_flow_cfg=trigger_flow_cfg,
                )
                burst_trend = _trend_pullback(row_5m, adx_min=momentum_burst_adx, trend_mode=trend_mode)
                if burst_trend == direction and fired:
                    chased, chase_reason = _chase_check(
                        row_30s, row_1m, direction, flow_snap,
                        body_mult=chase_body_mult, ema_atr=chase_ema_atr,
                        chase_flow_cfg=chase_flow_cfg,
                    )
                    if not chased:
                        rsi_block = rsi_entry_block_reason(
                            direction, row_1m, row_5m, row_30s, prev_30s, flow_snap,
                            adx_min_pullback=adx_min_pullback,
                            trend_mode=trend_mode,
                            rsi_gate_cfg=rsi_gate_cfg,
                        )
                        if rsi_block:
                            lines.append(gate_line(False, side, f"[BURST] {rsi_block}"))
                        else:
                            lines.append(gate_line(
                                True, side,
                                f"[BURST] ADX {adx:.0f} + 30s break — micro-entry ready",
                            ))
                    else:
                        lines.append(gate_line(False, side, f"[BURST] chase block — {chase_reason}"))
                else:
                    lines.append(gate_line(
                        False, side,
                        f"[BURST] waiting (ADX {adx:.0f}, trend={burst_trend}, trigger={fired})",
                    ))
        return lines

    if trend_ok:
        vwap = float(row_5m.get("vwap", 0))
        ema20_5m = float(row_5m.get("ema_20", 0))
        adx = float(row_5m.get("adx", 0))
        trend_label = "VWAP" if trend_mode == TREND_MODE_VWAP else "VWAP+EMA20"
        lines.append(gate_line(
            True, side,
            f"5M trend {side.upper()} ({trend_label}: VWAP {vwap:.2f}, ADX {adx:.0f})",
        ))
    elif continuation_enabled and cont_ok:
        lines.append(gate_line(True, side, f"5M continuation {side.upper()} ({cont_reason})"))

    if phase == PHASE_WAIT:
        armed = False
        if pullback_enabled and trend_ok and in_pb:
            if ts_30s is not None:
                _set_side_state(
                    state, direction, PHASE_SETUP, "pullback",
                    ts_30s + pd.Timedelta(seconds=window),
                )
            lines.append(gate_line(
                True, side,
                f"[PULLBACK] 1M {'pullback' if direction == 1 else 'rally'} to EMA20 "
                f"(dist {abs(close_1m - ema20_1m):.2f}) — setup armed ({window}s window)",
            ))
            armed = True
        elif pullback_enabled and trend_ok:
            lines.append(gate_line(
                False, side,
                f"[PULLBACK] waiting 1M zone (dist {abs(close_1m - ema20_1m):.2f}, "
                f"need ≤{atr_1m * pullback_atr:.2f})",
            ))
        if continuation_enabled and cont_ok and not in_pb:
            if ts_30s is not None:
                _set_side_state(
                    state, direction, PHASE_SETUP, "continuation",
                    ts_30s + pd.Timedelta(seconds=window),
                )
            lines.append(gate_line(
                True, side,
                "[CONTINUATION] 5M momentum OK — no pullback required, setup armed",
            ))
            armed = True
        elif continuation_enabled and not cont_ok:
            lines.append(gate_line(False, side, f"[CONTINUATION] {cont_reason}"))
        if not armed:
            lines.append(gate_line(False, side, "30s trigger not fired yet"))
        return lines

    phase, mode, setup_until = _get_side_state(state, direction)
    if ts_30s is not None and setup_until is not None and ts_30s > setup_until:
        lines.append(gate_line(False, side, f"[{mode.upper()}] setup window expired"))
        _set_side_state(state, direction, PHASE_WAIT, "", None)
        return lines

    if mode == "pullback":
        if not trend_ok or not in_pb:
            lines.append(gate_line(False, side, "[PULLBACK] left pullback zone — reset"))
            _set_side_state(state, direction, PHASE_WAIT, "", None)
            return lines
        lines.append(gate_line(True, side, "[PULLBACK] in 1M EMA20 zone"))
    elif mode == "continuation":
        if not cont_ok:
            lines.append(gate_line(False, side, f"[CONTINUATION] conditions lost ({cont_reason})"))
            _set_side_state(state, direction, PHASE_WAIT, "", None)
            return lines
        lines.append(gate_line(True, side, f"[CONTINUATION] momentum intact ({cont_reason})"))

    if row_30s is None or prev_30s is None:
        lines.append(gate_line(False, side, "30s bars unavailable"))
        return lines

    o = float(row_30s["open"])
    h = float(row_30s["high"])
    l = float(row_30s["low"])
    c = float(row_30s["close"])
    prev_h = float(prev_30s["high"])
    prev_l = float(prev_30s["low"])
    prev_c = float(prev_30s["close"])
    fired, trigger_reason = _trigger_eval(
        o, c, prev_h, prev_l, prev_c, direction,
        h=h, l=l,
        aggressive=aggressive_mode,
        setup_mode=mode,
        flow_snap=flow_snap,
        micro_break_pts=micro_break_pts,
        flow_trigger_delta_min=flow_trigger_delta_min,
        flow_trigger_buy_pct_long=flow_trigger_buy_pct_long,
        flow_trigger_buy_pct_short=flow_trigger_buy_pct_short,
        flow_strong_delta_min=flow_strong_delta_min,
        flow_strong_buy_pct_long=flow_strong_buy_pct_long,
        flow_strong_buy_pct_short=flow_strong_buy_pct_short,
        trigger_flow_cfg=trigger_flow_cfg,
    )
    if not fired:
        lines.append(gate_line(False, side, f"[{mode.upper()}] waiting 30s trigger ({trigger_reason})"))
        return lines

    chased, chase_reason = _chase_check(
        row_30s, row_1m, direction, flow_snap,
        body_mult=chase_body_mult, ema_atr=chase_ema_atr,
        chase_flow_cfg=chase_flow_cfg,
    )
    if chased:
        lines.append(gate_line(False, side, f"[{mode.upper()}] chase block — {chase_reason}"))
        return lines

    rsi_block = rsi_entry_block_reason(
        direction, row_1m, row_5m, row_30s, prev_30s, flow_snap,
        adx_min_pullback=adx_min_pullback,
        trend_mode=trend_mode,
        rsi_gate_cfg=rsi_gate_cfg,
    )
    if rsi_block:
        lines.append(gate_line(False, side, f"[{mode.upper()}] {rsi_block}"))
        return lines

    lines.append(gate_line(True, side, f"[{mode.upper()}] 30s trigger fired (SL {sl_pts}pt TP {tp_pts}pt)"))
    lines.append(gate_line(True, side, f"[{mode.upper()}] ALL GATES PASSED — hybrid entry ready"))
    return lines


def evaluate_hybrid_gates(
    row_1m: pd.Series,
    row_5m: pd.Series,
    prev_5m: Optional[pd.Series],
    row_30s: Optional[pd.Series],
    prev_30s: Optional[pd.Series],
    state: ScalpHybridState,
    *,
    pullback_enabled: bool = True,
    continuation_enabled: bool = True,
    adx_min_pullback: int = ADX_MIN_PULLBACK,
    adx_min_continuation: int = ADX_MIN_CONTINUATION,
    pullback_atr: float = PULLBACK_ATR,
    setup_bars: int = MAX_SETUP_BARS,
    setup_window_sec: int = 0,
    trigger_bar_seconds: int = 30,
    sl_pts: float = 8.0,
    tp_pts: float = 15.0,
    trend_mode: str = TREND_MODE_BOTH,
    continuation_volume_strict: bool = True,
    cont_volume_min_ratio: float = 0.85,
    chase_body_mult: float = CHASE_BODY_MULT,
    chase_ema_atr: float = CHASE_EMA_ATR,
    momentum_burst_enabled: bool = False,
    momentum_burst_adx: int = 25,
    aggressive_mode: bool = False,
    flow_snap: Optional[Dict] = None,
    micro_break_pts: float = TRIGGER_MICRO_BREAK_PTS,
    flow_trigger_delta_min: float = FLOW_TRIGGER_DELTA_MIN,
    flow_trigger_buy_pct_long: float = FLOW_TRIGGER_BUY_PCT_LONG,
    flow_trigger_buy_pct_short: float = FLOW_TRIGGER_BUY_PCT_SHORT,
    flow_strong_delta_min: float = FLOW_STRONG_DELTA_MIN,
    flow_strong_buy_pct_long: float = FLOW_STRONG_BUY_PCT_LONG,
    flow_strong_buy_pct_short: float = FLOW_STRONG_BUY_PCT_SHORT,
    chase_flow_cfg: Optional[Dict[str, Any]] = None,
    trigger_flow_cfg: Optional[Dict[str, Any]] = None,
    rsi_gate_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], ScalpHybridState]:
    st = _copy_state(state)
    lines: List[str] = []
    mode_label = "HYBRID" if (pullback_enabled and continuation_enabled) else (
        "PULLBACK" if pullback_enabled else "CONTINUATION"
    )
    burst_note = " + BURST" if momentum_burst_enabled else ""
    lines.append(f"   --- Scalp {mode_label}{burst_note} (trend={trend_mode}) ---")
    for direction in (1, -1):
        side = _side_label(direction)
        lines.extend(_evaluate_side_gates(
            side, direction, row_1m, row_5m, prev_5m, row_30s, prev_30s, st,
            pullback_enabled=pullback_enabled,
            continuation_enabled=continuation_enabled,
            adx_min_pullback=adx_min_pullback,
            adx_min_continuation=adx_min_continuation,
            pullback_atr=pullback_atr,
            setup_bars=setup_bars,
            setup_window_sec=setup_window_sec,
            trigger_bar_seconds=trigger_bar_seconds,
            sl_pts=sl_pts,
            tp_pts=tp_pts,
            trend_mode=trend_mode,
            continuation_volume_strict=continuation_volume_strict,
            cont_volume_min_ratio=cont_volume_min_ratio,
            chase_body_mult=chase_body_mult,
            chase_ema_atr=chase_ema_atr,
            momentum_burst_enabled=momentum_burst_enabled,
            momentum_burst_adx=momentum_burst_adx,
            aggressive_mode=aggressive_mode,
            flow_snap=flow_snap,
            micro_break_pts=micro_break_pts,
            flow_trigger_delta_min=flow_trigger_delta_min,
            flow_trigger_buy_pct_long=flow_trigger_buy_pct_long,
            flow_trigger_buy_pct_short=flow_trigger_buy_pct_short,
            flow_strong_delta_min=flow_strong_delta_min,
            flow_strong_buy_pct_long=flow_strong_buy_pct_long,
            flow_strong_buy_pct_short=flow_strong_buy_pct_short,
            chase_flow_cfg=chase_flow_cfg,
            trigger_flow_cfg=trigger_flow_cfg,
            rsi_gate_cfg=rsi_gate_cfg,
        ))
    return lines, st


def _flow_confirms_burst(
    flow_snap: Optional[Dict],
    direction: int,
    *,
    delta_min: float = 50,
    buy_pct_long_min: float = 0.55,
    buy_pct_short_max: float = 0.45,
) -> bool:
    """Strong tick flow confirms burst direction (aggressive / chop sessions)."""
    if not flow_snap:
        return False
    delta = float(flow_snap.get("delta", 0) or 0)
    buy_pct = float(flow_snap.get("buy_pct", 0.5) or 0.5)
    if direction == 1:
        return delta > delta_min and buy_pct > buy_pct_long_min
    if direction == -1:
        return delta < -delta_min and buy_pct < buy_pct_short_max
    return False


def _try_momentum_burst(
    direction: int,
    row_5m: pd.Series,
    row_30s: pd.Series,
    prev_30s: pd.Series,
    row_1m: pd.Series,
    *,
    adx_min: int,
    trend_mode: str,
    chase_body_mult: float,
    chase_ema_atr: float,
    adx_min_pullback: int = ADX_MIN_PULLBACK,
    flow_snap: Optional[Dict] = None,
    aggressive: bool = False,
    flow_burst_delta_min: float = 50,
    micro_break_pts: float = TRIGGER_MICRO_BREAK_PTS,
    flow_trigger_delta_min: float = FLOW_TRIGGER_DELTA_MIN,
    flow_trigger_buy_pct_long: float = FLOW_TRIGGER_BUY_PCT_LONG,
    flow_trigger_buy_pct_short: float = FLOW_TRIGGER_BUY_PCT_SHORT,
    flow_strong_delta_min: float = FLOW_STRONG_DELTA_MIN,
    flow_strong_buy_pct_long: float = FLOW_STRONG_BUY_PCT_LONG,
    flow_strong_buy_pct_short: float = FLOW_STRONG_BUY_PCT_SHORT,
    chase_flow_cfg: Optional[Dict[str, Any]] = None,
    trigger_flow_cfg: Optional[Dict[str, Any]] = None,
    rsi_gate_cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    trend_dir = _trend_pullback(row_5m, adx_min=adx_min_pullback, trend_mode=trend_mode)
    adx = float(row_5m.get("adx", 0))

    if aggressive and trend_dir == 0 and flow_snap is not None:
        if not _flow_confirms_burst(flow_snap, direction, delta_min=flow_burst_delta_min):
            return False
    elif trend_dir != direction:
        return False
    elif pd.isna(adx) or adx < adx_min:
        return False

    o = float(row_30s["open"])
    h = float(row_30s["high"])
    l = float(row_30s["low"])
    c = float(row_30s["close"])
    prev_h = float(prev_30s["high"])
    prev_l = float(prev_30s["low"])
    prev_c = float(prev_30s["close"])
    if not _trigger_fired(
        o, c, prev_h, prev_l, direction,
        prev_close=prev_c,
        h=h, l=l,
        aggressive=aggressive,
        setup_mode="burst",
        flow_snap=flow_snap,
        micro_break_pts=micro_break_pts,
        flow_trigger_delta_min=flow_trigger_delta_min,
        flow_trigger_buy_pct_long=flow_trigger_buy_pct_long,
        flow_trigger_buy_pct_short=flow_trigger_buy_pct_short,
        flow_strong_delta_min=flow_strong_delta_min,
        flow_strong_buy_pct_long=flow_strong_buy_pct_long,
        flow_strong_buy_pct_short=flow_strong_buy_pct_short,
        trigger_flow_cfg=trigger_flow_cfg,
    ):
        return False
    chased, _ = _chase_check(
        row_30s, row_1m, direction, flow_snap,
        body_mult=chase_body_mult, ema_atr=chase_ema_atr,
        chase_flow_cfg=chase_flow_cfg,
    )
    if chased:
        return False
    if rsi_entry_block_reason(
        direction, row_1m, row_5m, row_30s, prev_30s, flow_snap,
        adx_min_pullback=adx_min_pullback,
        trend_mode=trend_mode,
        rsi_gate_cfg=rsi_gate_cfg,
    ):
        return False
    return True


def _try_aggressive_direct_trigger(
    direction: int,
    row_1m: pd.Series,
    row_5m: pd.Series,
    row_30s: pd.Series,
    prev_30s: pd.Series,
    state: ScalpHybridState,
    *,
    adx_min_pullback: int,
    trend_mode: str,
    chase_body_mult: float,
    chase_ema_atr: float,
    aggressive_mode: bool,
    flow_snap: Optional[Dict] = None,
    micro_break_pts: float = TRIGGER_MICRO_BREAK_PTS,
    flow_trigger_delta_min: float = FLOW_TRIGGER_DELTA_MIN,
    flow_trigger_buy_pct_long: float = FLOW_TRIGGER_BUY_PCT_LONG,
    flow_trigger_buy_pct_short: float = FLOW_TRIGGER_BUY_PCT_SHORT,
    flow_strong_delta_min: float = FLOW_STRONG_DELTA_MIN,
    flow_strong_buy_pct_long: float = FLOW_STRONG_BUY_PCT_LONG,
    flow_strong_buy_pct_short: float = FLOW_STRONG_BUY_PCT_SHORT,
    chase_flow_cfg: Optional[Dict[str, Any]] = None,
    trigger_flow_cfg: Optional[Dict[str, Any]] = None,
    rsi_gate_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Aggressive mode: enter on 30s trigger when trend aligns — no setup arming required."""
    if not aggressive_mode:
        return None
    pb_trend = _trend_pullback(row_5m, adx_min=adx_min_pullback, trend_mode=trend_mode)
    if pb_trend != direction:
        return None
    o = float(row_30s["open"])
    h = float(row_30s["high"])
    l = float(row_30s["low"])
    c = float(row_30s["close"])
    prev_h = float(prev_30s["high"])
    prev_l = float(prev_30s["low"])
    prev_c = float(prev_30s["close"])
    if not _trigger_fired(
        o, c, prev_h, prev_l, direction,
        prev_close=prev_c,
        h=h, l=l,
        aggressive=True,
        setup_mode="burst",
        flow_snap=flow_snap,
        micro_break_pts=micro_break_pts,
        flow_trigger_delta_min=flow_trigger_delta_min,
        flow_trigger_buy_pct_long=flow_trigger_buy_pct_long,
        flow_trigger_buy_pct_short=flow_trigger_buy_pct_short,
        flow_strong_delta_min=flow_strong_delta_min,
        flow_strong_buy_pct_long=flow_strong_buy_pct_long,
        flow_strong_buy_pct_short=flow_strong_buy_pct_short,
        trigger_flow_cfg=trigger_flow_cfg,
    ):
        return None
    chased, _ = _chase_check(
        row_30s, row_1m, direction, flow_snap,
        body_mult=chase_body_mult, ema_atr=chase_ema_atr,
        chase_flow_cfg=chase_flow_cfg,
    )
    if chased:
        return None
    if rsi_entry_block_reason(
        direction, row_1m, row_5m, row_30s, prev_30s, flow_snap,
        adx_min_pullback=adx_min_pullback,
        trend_mode=trend_mode,
        rsi_gate_cfg=rsi_gate_cfg,
    ):
        return None
    _set_side_state(state, direction, PHASE_WAIT, "", None)
    return "trigger"


def hybrid_trigger_no_signal_reason(
    row_1m: pd.Series,
    row_5m: pd.Series,
    row_30s: pd.Series,
    prev_30s: pd.Series,
    *,
    direction: int,
    aggressive_mode: bool = False,
    adx_min_pullback: int = ADX_MIN_PULLBACK,
    trend_mode: str = TREND_MODE_BOTH,
    chase_body_mult: float = CHASE_BODY_MULT,
    chase_ema_atr: float = CHASE_EMA_ATR,
    flow_snap: Optional[Dict] = None,
    micro_break_pts: float = TRIGGER_MICRO_BREAK_PTS,
    flow_trigger_delta_min: float = FLOW_TRIGGER_DELTA_MIN,
    flow_trigger_buy_pct_long: float = FLOW_TRIGGER_BUY_PCT_LONG,
    flow_trigger_buy_pct_short: float = FLOW_TRIGGER_BUY_PCT_SHORT,
    flow_strong_delta_min: float = FLOW_STRONG_DELTA_MIN,
    flow_strong_buy_pct_long: float = FLOW_STRONG_BUY_PCT_LONG,
    flow_strong_buy_pct_short: float = FLOW_STRONG_BUY_PCT_SHORT,
    chase_flow_cfg: Optional[Dict[str, Any]] = None,
    trigger_flow_cfg: Optional[Dict[str, Any]] = None,
    rsi_gate_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Explain why trigger=yes did not produce a hybrid entry (for live logs)."""
    pb_trend = _trend_pullback(row_5m, adx_min=adx_min_pullback, trend_mode=trend_mode)
    if pb_trend != direction:
        adx = float(row_5m.get("adx", 0))
        return f"trend dir={pb_trend} (need {direction}, ADX={adx:.0f})"
    o = float(row_30s["open"])
    h = float(row_30s["high"])
    l = float(row_30s["low"])
    c = float(row_30s["close"])
    prev_h = float(prev_30s["high"])
    prev_l = float(prev_30s["low"])
    prev_c = float(prev_30s["close"])
    fired, trigger_reason = _trigger_eval(
        o, c, prev_h, prev_l, prev_c, direction,
        h=h, l=l,
        aggressive=aggressive_mode,
        setup_mode="burst" if aggressive_mode else "",
        flow_snap=flow_snap,
        micro_break_pts=micro_break_pts,
        flow_trigger_delta_min=flow_trigger_delta_min,
        flow_trigger_buy_pct_long=flow_trigger_buy_pct_long,
        flow_trigger_buy_pct_short=flow_trigger_buy_pct_short,
        flow_strong_delta_min=flow_strong_delta_min,
        flow_strong_buy_pct_long=flow_strong_buy_pct_long,
        flow_strong_buy_pct_short=flow_strong_buy_pct_short,
        trigger_flow_cfg=trigger_flow_cfg,
    )
    if not fired:
        return f"30s trigger no ({trigger_reason})"
    chased, chase_reason = _chase_check(
        row_30s, row_1m, direction, flow_snap,
        body_mult=chase_body_mult, ema_atr=chase_ema_atr,
        chase_flow_cfg=chase_flow_cfg,
    )
    if chased:
        return f"chase block — {chase_reason}"
    rsi_block = rsi_entry_block_reason(
        direction, row_1m, row_5m, row_30s, prev_30s, flow_snap,
        adx_min_pullback=adx_min_pullback,
        trend_mode=trend_mode,
        rsi_gate_cfg=rsi_gate_cfg,
    )
    if rsi_block:
        return rsi_block
    if not aggressive_mode:
        return "setup not armed (pullback/continuation required)"
    return "unknown gate"


def _try_side_entry(
    direction: int,
    row_1m: pd.Series,
    row_5m: pd.Series,
    prev_5m: Optional[pd.Series],
    row_30s: pd.Series,
    prev_30s: pd.Series,
    state: ScalpHybridState,
    *,
    pullback_enabled: bool,
    continuation_enabled: bool,
    adx_min_pullback: int,
    adx_min_continuation: int,
    pullback_atr: float,
    setup_bars: int,
    setup_window_sec: int,
    trigger_bar_seconds: int,
    trend_mode: str,
    continuation_volume_strict: bool,
    cont_volume_min_ratio: float,
    chase_body_mult: float,
    chase_ema_atr: float,
    aggressive_mode: bool = False,
    flow_snap: Optional[Dict] = None,
    micro_break_pts: float = TRIGGER_MICRO_BREAK_PTS,
    flow_trigger_delta_min: float = FLOW_TRIGGER_DELTA_MIN,
    flow_trigger_buy_pct_long: float = FLOW_TRIGGER_BUY_PCT_LONG,
    flow_trigger_buy_pct_short: float = FLOW_TRIGGER_BUY_PCT_SHORT,
    flow_strong_delta_min: float = FLOW_STRONG_DELTA_MIN,
    flow_strong_buy_pct_long: float = FLOW_STRONG_BUY_PCT_LONG,
    flow_strong_buy_pct_short: float = FLOW_STRONG_BUY_PCT_SHORT,
    chase_flow_cfg: Optional[Dict[str, Any]] = None,
    trigger_flow_cfg: Optional[Dict[str, Any]] = None,
    rsi_gate_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Update state and return mode label if entry fires."""
    ts_30s = _bar_ts(row_30s)
    close_1m = float(row_1m["close"])
    ema20_1m = float(row_1m.get("ema_20", float("nan")))
    atr_1m = float(row_1m.get("atr", 0))
    window = _setup_window_sec(setup_bars, setup_window_sec, trigger_bar_seconds)

    def _confirm_entry(entry_mode: str) -> Optional[str]:
        if rsi_entry_block_reason(
            direction, row_1m, row_5m, row_30s, prev_30s, flow_snap,
            adx_min_pullback=adx_min_pullback,
            trend_mode=trend_mode,
            rsi_gate_cfg=rsi_gate_cfg,
        ):
            return None
        _set_side_state(state, direction, PHASE_WAIT, "", None)
        return entry_mode

    pb_trend = _trend_pullback(row_5m, adx_min=adx_min_pullback, trend_mode=trend_mode)
    cont_ok, _ = _continuation_ok(
        row_5m, prev_5m, direction, adx_min=adx_min_continuation, trend_mode=trend_mode,
        volume_strict=continuation_volume_strict, volume_min_ratio=cont_volume_min_ratio,
    )
    in_pb = _in_pullback_zone(close_1m, ema20_1m, atr_1m, direction, pb_limit=pullback_atr)

    phase, mode, setup_until = _get_side_state(state, direction)

    if phase == PHASE_WAIT:
        if aggressive_mode and pb_trend == direction:
            direct = _try_aggressive_direct_trigger(
                direction, row_1m, row_5m, row_30s, prev_30s, state,
                adx_min_pullback=adx_min_pullback,
                trend_mode=trend_mode,
                chase_body_mult=chase_body_mult,
                chase_ema_atr=chase_ema_atr,
                aggressive_mode=True,
                flow_snap=flow_snap,
                micro_break_pts=micro_break_pts,
                flow_trigger_delta_min=flow_trigger_delta_min,
                flow_trigger_buy_pct_long=flow_trigger_buy_pct_long,
                flow_trigger_buy_pct_short=flow_trigger_buy_pct_short,
                flow_strong_delta_min=flow_strong_delta_min,
                flow_strong_buy_pct_long=flow_strong_buy_pct_long,
                flow_strong_buy_pct_short=flow_strong_buy_pct_short,
                chase_flow_cfg=chase_flow_cfg,
                trigger_flow_cfg=trigger_flow_cfg,
                rsi_gate_cfg=rsi_gate_cfg,
            )
            if direct:
                return direct
            adx = float(row_5m.get("adx", 0))
            if (
                continuation_enabled
                and not pd.isna(adx)
                and adx >= adx_min_continuation
                and not in_pb
            ):
                o = float(row_30s["open"])
                h = float(row_30s["high"])
                l = float(row_30s["low"])
                c = float(row_30s["close"])
                prev_h = float(prev_30s["high"])
                prev_l = float(prev_30s["low"])
                prev_c = float(prev_30s["close"])
                if _trigger_fired(
                    o, c, prev_h, prev_l, direction,
                    prev_close=prev_c,
                    h=h, l=l,
                    aggressive=True,
                    setup_mode="continuation",
                    flow_snap=flow_snap,
                    micro_break_pts=micro_break_pts,
                    flow_trigger_delta_min=flow_trigger_delta_min,
                    flow_trigger_buy_pct_long=flow_trigger_buy_pct_long,
                    flow_trigger_buy_pct_short=flow_trigger_buy_pct_short,
                    flow_strong_delta_min=flow_strong_delta_min,
                    flow_strong_buy_pct_long=flow_strong_buy_pct_long,
                    flow_strong_buy_pct_short=flow_strong_buy_pct_short,
                    trigger_flow_cfg=trigger_flow_cfg,
                ) and ts_30s is not None:
                    _set_side_state(
                        state, direction, PHASE_SETUP, "continuation",
                        ts_30s + pd.Timedelta(seconds=window),
                    )
        if pullback_enabled and pb_trend == direction and in_pb:
            if ts_30s is not None:
                _set_side_state(
                    state, direction, PHASE_SETUP, "pullback",
                    ts_30s + pd.Timedelta(seconds=window),
                )
                if aggressive_mode:
                    phase, mode, setup_until = _get_side_state(state, direction)
            else:
                return None
            if aggressive_mode and phase == PHASE_SETUP:
                o = float(row_30s["open"])
                h = float(row_30s["high"])
                l = float(row_30s["low"])
                c = float(row_30s["close"])
                prev_h = float(prev_30s["high"])
                prev_l = float(prev_30s["low"])
                prev_c = float(prev_30s["close"])
                if _trigger_fired(
                    o, c, prev_h, prev_l, direction,
                    prev_close=prev_c,
                    h=h, l=l,
                    aggressive=True,
                    setup_mode=mode,
                    flow_snap=flow_snap,
                    micro_break_pts=micro_break_pts,
                    flow_trigger_delta_min=flow_trigger_delta_min,
                    flow_trigger_buy_pct_long=flow_trigger_buy_pct_long,
                    flow_trigger_buy_pct_short=flow_trigger_buy_pct_short,
                    flow_strong_delta_min=flow_strong_delta_min,
                    flow_strong_buy_pct_long=flow_strong_buy_pct_long,
                    flow_strong_buy_pct_short=flow_strong_buy_pct_short,
                    trigger_flow_cfg=trigger_flow_cfg,
                ):
                    chased, _ = _chase_check(
                        row_30s, row_1m, direction, flow_snap,
                        body_mult=chase_body_mult, ema_atr=chase_ema_atr,
                        chase_flow_cfg=chase_flow_cfg,
                    )
                    if not chased:
                        return _confirm_entry(mode)
            return None
        if continuation_enabled and cont_ok and not in_pb:
            if ts_30s is not None:
                _set_side_state(
                    state, direction, PHASE_SETUP, "continuation",
                    ts_30s + pd.Timedelta(seconds=window),
                )
                if aggressive_mode:
                    phase, mode, setup_until = _get_side_state(state, direction)
                    o = float(row_30s["open"])
                    c = float(row_30s["close"])
                    prev_h = float(prev_30s["high"])
                    prev_l = float(prev_30s["low"])
                    prev_c = float(prev_30s["close"])
                    if _trigger_fired(
                        o, c, prev_h, prev_l, direction,
                        prev_close=prev_c,
                        h=h, l=l,
                        aggressive=True,
                        setup_mode=mode,
                        flow_snap=flow_snap,
                        micro_break_pts=micro_break_pts,
                        flow_trigger_delta_min=flow_trigger_delta_min,
                        flow_trigger_buy_pct_long=flow_trigger_buy_pct_long,
                        flow_trigger_buy_pct_short=flow_trigger_buy_pct_short,
                        flow_strong_delta_min=flow_strong_delta_min,
                        flow_strong_buy_pct_long=flow_strong_buy_pct_long,
                        flow_strong_buy_pct_short=flow_strong_buy_pct_short,
                        trigger_flow_cfg=trigger_flow_cfg,
                    ):
                        chased, _ = _chase_check(
                            row_30s, row_1m, direction, flow_snap,
                            body_mult=chase_body_mult, ema_atr=chase_ema_atr,
                            chase_flow_cfg=chase_flow_cfg,
                        )
                        if not chased:
                            return _confirm_entry(mode)
            return None
        return None

    phase, mode, setup_until = _get_side_state(state, direction)
    if ts_30s is not None and setup_until is not None and ts_30s > setup_until:
        _set_side_state(state, direction, PHASE_WAIT, "", None)
        return None

    if mode == "pullback":
        if pb_trend != direction or not in_pb:
            _set_side_state(state, direction, PHASE_WAIT, "", None)
            return None
    elif mode == "continuation":
        if not cont_ok:
            _set_side_state(state, direction, PHASE_WAIT, "", None)
            return None

    o = float(row_30s["open"])
    h = float(row_30s["high"])
    l = float(row_30s["low"])
    c = float(row_30s["close"])
    prev_h = float(prev_30s["high"])
    prev_l = float(prev_30s["low"])
    prev_c = float(prev_30s["close"])
    if not _trigger_fired(
        o, c, prev_h, prev_l, direction,
        prev_close=prev_c,
        h=h, l=l,
        aggressive=aggressive_mode,
        setup_mode=mode,
        flow_snap=flow_snap,
        micro_break_pts=micro_break_pts,
        flow_trigger_delta_min=flow_trigger_delta_min,
        flow_trigger_buy_pct_long=flow_trigger_buy_pct_long,
        flow_trigger_buy_pct_short=flow_trigger_buy_pct_short,
        flow_strong_delta_min=flow_strong_delta_min,
        flow_strong_buy_pct_long=flow_strong_buy_pct_long,
        flow_strong_buy_pct_short=flow_strong_buy_pct_short,
        trigger_flow_cfg=trigger_flow_cfg,
    ):
        return None

    chased, _ = _chase_check(
        row_30s, row_1m, direction, flow_snap,
        body_mult=chase_body_mult, ema_atr=chase_ema_atr,
        chase_flow_cfg=chase_flow_cfg,
    )
    if chased:
        return None

    return _confirm_entry(mode)


def _build_signal(
    symbol: str,
    direction: int,
    row_30s: pd.Series,
    row_1m: pd.Series,
    mode: str,
    sl_pts: float,
    tp_pts: float,
) -> Dict[str, Any]:
    entry_price = float(row_30s["close"])
    atr = float(row_1m.get("atr", 0))
    if direction == 1:
        sl = entry_price - sl_pts
        tp = entry_price + tp_pts
        dir_str = "long"
    else:
        sl = entry_price + sl_pts
        tp = entry_price - tp_pts
        dir_str = "short"
    return {
        "symbol": symbol,
        "direction": dir_str,
        "entry": entry_price,
        "sl": sl,
        "tp": tp,
        "atr": atr if not pd.isna(atr) else sl_pts,
        "scalp_mode": mode,
        "scalp_hybrid": True,
        "structure_capped": False,
    }


def check_hybrid_entry(
    symbol: str,
    row_1m: pd.Series,
    row_5m: pd.Series,
    prev_5m: Optional[pd.Series],
    row_30s: pd.Series,
    prev_30s: pd.Series,
    state: ScalpHybridState,
    *,
    pullback_enabled: bool = True,
    continuation_enabled: bool = True,
    adx_min_pullback: int = ADX_MIN_PULLBACK,
    adx_min_continuation: int = ADX_MIN_CONTINUATION,
    pullback_atr: float = PULLBACK_ATR,
    setup_bars: int = MAX_SETUP_BARS,
    setup_window_sec: int = 0,
    trigger_bar_seconds: int = 30,
    sl_pts: float = 8.0,
    tp_pts: float = 15.0,
    trend_mode: str = TREND_MODE_BOTH,
    continuation_volume_strict: bool = True,
    cont_volume_min_ratio: float = 0.85,
    chase_body_mult: float = CHASE_BODY_MULT,
    chase_ema_atr: float = CHASE_EMA_ATR,
    momentum_burst_enabled: bool = False,
    momentum_burst_adx: int = 25,
    aggressive_mode: bool = False,
    flow_snap: Optional[Dict] = None,
    flow_burst_delta_min: float = 50,
    micro_break_pts: float = TRIGGER_MICRO_BREAK_PTS,
    flow_trigger_delta_min: float = FLOW_TRIGGER_DELTA_MIN,
    flow_trigger_buy_pct_long: float = FLOW_TRIGGER_BUY_PCT_LONG,
    flow_trigger_buy_pct_short: float = FLOW_TRIGGER_BUY_PCT_SHORT,
    flow_strong_delta_min: float = FLOW_STRONG_DELTA_MIN,
    flow_strong_buy_pct_long: float = FLOW_STRONG_BUY_PCT_LONG,
    flow_strong_buy_pct_short: float = FLOW_STRONG_BUY_PCT_SHORT,
    chase_flow_cfg: Optional[Dict[str, Any]] = None,
    trigger_flow_cfg: Optional[Dict[str, Any]] = None,
    rsi_gate_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], ScalpHybridState]:
    """Evaluate hybrid entry; returns signal dict or None."""
    st = _copy_state(state)
    hybrid_kwargs = dict(
        pullback_enabled=pullback_enabled,
        continuation_enabled=continuation_enabled,
        adx_min_pullback=adx_min_pullback,
        adx_min_continuation=adx_min_continuation,
        pullback_atr=pullback_atr,
        setup_bars=setup_bars,
        setup_window_sec=setup_window_sec,
        trigger_bar_seconds=trigger_bar_seconds,
        trend_mode=trend_mode,
        continuation_volume_strict=continuation_volume_strict,
        cont_volume_min_ratio=cont_volume_min_ratio,
        chase_body_mult=chase_body_mult,
        chase_ema_atr=chase_ema_atr,
        aggressive_mode=aggressive_mode,
        flow_snap=flow_snap,
        micro_break_pts=micro_break_pts,
        flow_trigger_delta_min=flow_trigger_delta_min,
        flow_trigger_buy_pct_long=flow_trigger_buy_pct_long,
        flow_trigger_buy_pct_short=flow_trigger_buy_pct_short,
        flow_strong_delta_min=flow_strong_delta_min,
        flow_strong_buy_pct_long=flow_strong_buy_pct_long,
        flow_strong_buy_pct_short=flow_strong_buy_pct_short,
        chase_flow_cfg=chase_flow_cfg,
        trigger_flow_cfg=trigger_flow_cfg,
        rsi_gate_cfg=rsi_gate_cfg,
    )
    for direction in (1, -1):
        mode = _try_side_entry(
            direction, row_1m, row_5m, prev_5m, row_30s, prev_30s, st,
            **hybrid_kwargs,
        )
        if mode:
            return _build_signal(symbol, direction, row_30s, row_1m, mode, sl_pts, tp_pts), st

    if momentum_burst_enabled:
        burst_adx = momentum_burst_adx
        if aggressive_mode:
            burst_adx = min(burst_adx, adx_min_pullback)
        for direction in (1, -1):
            phase, _, _ = _get_side_state(st, direction)
            if phase != PHASE_WAIT:
                continue
            if _try_momentum_burst(
                direction, row_5m, row_30s, prev_30s, row_1m,
                adx_min=burst_adx,
                trend_mode=trend_mode,
                chase_body_mult=chase_body_mult,
                chase_ema_atr=chase_ema_atr,
                adx_min_pullback=adx_min_pullback,
                flow_snap=flow_snap,
                aggressive=aggressive_mode,
                flow_burst_delta_min=flow_burst_delta_min,
                micro_break_pts=micro_break_pts,
                flow_trigger_delta_min=flow_trigger_delta_min,
                flow_trigger_buy_pct_long=flow_trigger_buy_pct_long,
                flow_trigger_buy_pct_short=flow_trigger_buy_pct_short,
                flow_strong_delta_min=flow_strong_delta_min,
                flow_strong_buy_pct_long=flow_strong_buy_pct_long,
                flow_strong_buy_pct_short=flow_strong_buy_pct_short,
                chase_flow_cfg=chase_flow_cfg,
                trigger_flow_cfg=trigger_flow_cfg,
            ):
                return _build_signal(
                    symbol, direction, row_30s, row_1m, "burst", sl_pts, tp_pts,
                ), st

    return None, st


def hybrid_block_summary(
    row_1m: pd.Series,
    row_5m: pd.Series,
    prev_5m: Optional[pd.Series],
    row_30s: Optional[pd.Series],
    prev_30s: Optional[pd.Series],
    state: ScalpHybridState,
    *,
    pullback_enabled: bool = True,
    continuation_enabled: bool = True,
    adx_min_pullback: int = ADX_MIN_PULLBACK,
    adx_min_continuation: int = ADX_MIN_CONTINUATION,
    pullback_atr: float = PULLBACK_ATR,
    setup_bars: int = MAX_SETUP_BARS,
    setup_window_sec: int = 0,
    trigger_bar_seconds: int = 30,
    sl_pts: float = 8.0,
    tp_pts: float = 15.0,
    trend_mode: str = TREND_MODE_BOTH,
    continuation_volume_strict: bool = True,
    cont_volume_min_ratio: float = 0.85,
    chase_body_mult: float = CHASE_BODY_MULT,
    chase_ema_atr: float = CHASE_EMA_ATR,
    momentum_burst_enabled: bool = False,
    momentum_burst_adx: int = 25,
    aggressive_mode: bool = False,
    flow_snap: Optional[Dict] = None,
    flow_burst_delta_min: float = 50,
    micro_break_pts: float = TRIGGER_MICRO_BREAK_PTS,
    flow_trigger_delta_min: float = FLOW_TRIGGER_DELTA_MIN,
    flow_trigger_buy_pct_long: float = FLOW_TRIGGER_BUY_PCT_LONG,
    flow_trigger_buy_pct_short: float = FLOW_TRIGGER_BUY_PCT_SHORT,
    flow_strong_delta_min: float = FLOW_STRONG_DELTA_MIN,
    flow_strong_buy_pct_long: float = FLOW_STRONG_BUY_PCT_LONG,
    flow_strong_buy_pct_short: float = FLOW_STRONG_BUY_PCT_SHORT,
    chase_flow_cfg: Optional[Dict[str, Any]] = None,
    trigger_flow_cfg: Optional[Dict[str, Any]] = None,
    rsi_gate_cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """One-line reason why hybrid is not entering (for fast scan log)."""
    if row_30s is None or prev_30s is None:
        return "block: no 30s trigger bars"
    st = _copy_state(state)
    signal, _ = check_hybrid_entry(
        "X", row_1m, row_5m, prev_5m, row_30s, prev_30s, st,
        pullback_enabled=pullback_enabled,
        continuation_enabled=continuation_enabled,
        adx_min_pullback=adx_min_pullback,
        adx_min_continuation=adx_min_continuation,
        pullback_atr=pullback_atr,
        setup_bars=setup_bars,
        setup_window_sec=setup_window_sec,
        trigger_bar_seconds=trigger_bar_seconds,
        sl_pts=sl_pts,
        tp_pts=tp_pts,
        trend_mode=trend_mode,
        continuation_volume_strict=continuation_volume_strict,
        cont_volume_min_ratio=cont_volume_min_ratio,
        chase_body_mult=chase_body_mult,
        chase_ema_atr=chase_ema_atr,
        momentum_burst_enabled=momentum_burst_enabled,
        momentum_burst_adx=momentum_burst_adx,
        aggressive_mode=aggressive_mode,
        flow_snap=flow_snap,
        flow_burst_delta_min=flow_burst_delta_min,
        micro_break_pts=micro_break_pts,
        flow_trigger_delta_min=flow_trigger_delta_min,
        flow_trigger_buy_pct_long=flow_trigger_buy_pct_long,
        flow_trigger_buy_pct_short=flow_trigger_buy_pct_short,
        flow_strong_delta_min=flow_strong_delta_min,
        flow_strong_buy_pct_long=flow_strong_buy_pct_long,
        flow_strong_buy_pct_short=flow_strong_buy_pct_short,
        chase_flow_cfg=chase_flow_cfg,
        trigger_flow_cfg=trigger_flow_cfg,
        rsi_gate_cfg=rsi_gate_cfg,
    )
    if signal:
        return f"READY {signal['direction'].upper()} ({signal.get('scalp_mode', 'hybrid')})"
    pb_trend = _trend_pullback(row_5m, adx_min=adx_min_pullback, trend_mode=trend_mode)
    adx = float(row_5m.get("adx", 0))
    trend_label = "VWAP" if trend_mode == TREND_MODE_VWAP else "VWAP+EMA20"
    if pb_trend == 0:
        if aggressive_mode and momentum_burst_enabled and flow_snap:
            for d in (1, -1):
                if _flow_confirms_burst(flow_snap, d, delta_min=flow_burst_delta_min):
                    side = "LONG" if d == 1 else "SHORT"
                    o = float(row_30s["open"])
                    h = float(row_30s["high"])
                    l = float(row_30s["low"])
                    c = float(row_30s["close"])
                    prev_h = float(prev_30s["high"])
                    prev_l = float(prev_30s["low"])
                    prev_c = float(prev_30s["close"])
                    if _trigger_fired(
                        o, c, prev_h, prev_l, d,
                        prev_close=prev_c,
                        h=h, l=l,
                        aggressive=aggressive_mode,
                        setup_mode="burst",
                        flow_snap=flow_snap,
                        micro_break_pts=micro_break_pts,
                        flow_trigger_delta_min=flow_trigger_delta_min,
                        flow_trigger_buy_pct_long=flow_trigger_buy_pct_long,
                        flow_trigger_buy_pct_short=flow_trigger_buy_pct_short,
                        flow_strong_delta_min=flow_strong_delta_min,
                        flow_strong_buy_pct_long=flow_strong_buy_pct_long,
                        flow_strong_buy_pct_short=flow_strong_buy_pct_short,
                        trigger_flow_cfg=trigger_flow_cfg,
                    ):
                        chased, chase_reason = _chase_check(
                            row_30s, row_1m, d, flow_snap,
                            body_mult=chase_body_mult, ema_atr=chase_ema_atr,
                            chase_flow_cfg=chase_flow_cfg,
                        )
                        if chased:
                            return f"block: flow {side} burst chase — {chase_reason}"
                        rsi_block = rsi_entry_block_reason(
                            d, row_1m, row_5m, row_30s, prev_30s, flow_snap,
                            adx_min_pullback=adx_min_pullback,
                            trend_mode=trend_mode,
                            rsi_gate_cfg=rsi_gate_cfg,
                        )
                        if rsi_block:
                            return rsi_block
                        return f"block: flow {side} burst armed — waiting re-scan"
                    return f"block: flow confirms {side}, waiting 30s break"
            return f"block: trend dir=0 ({trend_label}, ADX={adx:.0f}), weak flow"
        return f"block: trend dir=0 ({trend_label}, ADX={adx:.0f})"
    side = "LONG" if pb_trend == 1 else "SHORT"
    direction = pb_trend
    phase, mode, _ = _get_side_state(st, direction)
    trigger_kw = dict(
        aggressive=aggressive_mode,
        flow_snap=flow_snap,
        micro_break_pts=micro_break_pts,
        flow_trigger_delta_min=flow_trigger_delta_min,
        flow_trigger_buy_pct_long=flow_trigger_buy_pct_long,
        flow_trigger_buy_pct_short=flow_trigger_buy_pct_short,
        flow_strong_delta_min=flow_strong_delta_min,
        flow_strong_buy_pct_long=flow_strong_buy_pct_long,
        flow_strong_buy_pct_short=flow_strong_buy_pct_short,
        trigger_flow_cfg=trigger_flow_cfg,
    )
    o = float(row_30s["open"])
    h = float(row_30s["high"])
    l = float(row_30s["low"])
    c = float(row_30s["close"])
    prev_h = float(prev_30s["high"])
    prev_l = float(prev_30s["low"])
    prev_c = float(prev_30s["close"])
    eval_mode = mode if phase == PHASE_SETUP and mode else (
        "burst" if aggressive_mode else ""
    )
    fired, trigger_reason = _trigger_eval(
        o, c, prev_h, prev_l, prev_c, direction,
        h=h, l=l,
        setup_mode=eval_mode,
        **trigger_kw,
    )
    if fired:
        chased, chase_reason = _chase_check(
            row_30s, row_1m, direction, flow_snap,
            body_mult=chase_body_mult, ema_atr=chase_ema_atr,
            chase_flow_cfg=chase_flow_cfg,
        )
        if chased:
            return f"block: {side} trigger yes — chase: {chase_reason}"
        rsi_block = rsi_entry_block_reason(
            direction, row_1m, row_5m, row_30s, prev_30s, flow_snap,
            adx_min_pullback=adx_min_pullback,
            trend_mode=trend_mode,
            rsi_gate_cfg=rsi_gate_cfg,
        )
        if rsi_block:
            return rsi_block
        if phase == PHASE_SETUP and mode:
            return f"block: {side} {mode} setup armed — trigger yes ({trigger_reason})"
        return f"block: {side} trigger yes ({trigger_reason}) — entry pending"
    if phase == PHASE_SETUP and mode:
        return f"block: {side} {mode} setup — waiting 30s trigger ({trigger_reason})"
    if aggressive_mode and adx >= adx_min_pullback:
        return f"block: {side} trend OK (ADX={adx:.0f}) — waiting 30s trigger ({trigger_reason})"
    if momentum_burst_enabled and adx >= min(momentum_burst_adx, adx_min_pullback):
        return f"block: {side} trend OK (ADX={adx:.0f}) — waiting 30s break or pullback"
    return f"block: {side} trend (ADX={adx:.0f}) — no setup armed"


def add_30s_body_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add avg_body column for chase protection."""
    out = df.copy()
    out["body"] = (out["close"] - out["open"]).abs()
    out["avg_body"] = out["body"].rolling(CANDLE_AVG_PERIOD).mean()
    return out
