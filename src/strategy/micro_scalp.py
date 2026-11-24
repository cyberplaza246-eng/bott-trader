"""
MNQ micro-scalp: enter with trend, hold at least 10 seconds, take $20+.

Lucid flags microscalping when >50% of profits come from holds <=5s.
This bot never arms the $20 take-profit until MIN_HOLD_SEC (default 10).

1 MNQ: $20 = 10 points. Stop is ATR-sized (clip 6-12 pts). Time stop after MAX_HOLD_SEC.
Paper / Lucid sim only — not wired to funded live.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz

from src.strategy.day_hybrid import (
    entry_pullback,
    prepare_arrays,
    stack_side_15_5_1,
)
from src.strategy.mnq_15m_ema_eod import (
    ENTRY_MINUTE,
    FLAT_MINUTE,
    ONE_MIN_MAX_BODY,
    sl_pts_from_atr,
)

ET = pytz.timezone("US/Eastern")
SESSION_NAME = "micro_scalp"

POINT_VALUE = 2.0  # MNQ $/point
TP_DOLLARS = float(os.getenv("MICRO_SCALP_TP_USD", "20"))
MIN_HOLD_SEC = float(os.getenv("MICRO_SCALP_MIN_HOLD_SEC", "10"))
MAX_HOLD_SEC = float(os.getenv("MICRO_SCALP_MAX_HOLD_SEC", "180"))
SL_ATR_MULT = float(os.getenv("MICRO_SCALP_SL_ATR_MULT", "1.0"))
SL_MIN = float(os.getenv("MICRO_SCALP_SL_MIN", "6"))
SL_MAX = float(os.getenv("MICRO_SCALP_SL_MAX", "12"))
MAX_TRADES_DAY = int(os.getenv("MICRO_SCALP_MAX_DAY", "30"))
COOLDOWN_SEC = float(os.getenv("MICRO_SCALP_COOLDOWN_SEC", "30"))
BODY_MAX = float(os.getenv("MICRO_SCALP_BODY_MAX", str(ONE_MIN_MAX_BODY)))


def micro_scalp_from_env() -> bool:
    return str(os.getenv("PAPER_MICRO_SCALP", "")).strip().lower() in (
        "1", "true", "yes", "on",
    ) or str(os.getenv("MICRO_SCALP", "")).strip().lower() in (
        "1", "true", "yes", "on",
    )


def tp_pts_from_dollars(dollars: float = TP_DOLLARS, point_value: float = POINT_VALUE) -> float:
    return float(dollars) / float(point_value)


def is_rth_scan_et(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(pytz.UTC)
    et = now.astimezone(ET)
    if et.weekday() >= 5:
        return False
    m = et.hour * 60 + et.minute
    return ENTRY_MINUTE <= m < FLAT_MINUTE


def should_flatten_rth(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(pytz.UTC)
    et = now.astimezone(ET)
    if et.weekday() >= 5:
        return True
    return (et.hour * 60 + et.minute) >= FLAT_MINUTE


@dataclass
class MicroScalpState:
    day: Optional[date] = None
    trades_today: int = 0
    last_exit_ts: float = 0.0
    last_signal_bar: Optional[pd.Timestamp] = None
    open_side: int = 0  # 1 long, -1 short, 0 flat


def reset_day_if_needed(state: MicroScalpState, now: datetime) -> MicroScalpState:
    d = now.astimezone(ET).date()
    if state.day != d:
        state.day = d
        state.trades_today = 0
        state.last_signal_bar = None
    return state


def recipe_banner_lines() -> List[str]:
    tp_pts = tp_pts_from_dollars()
    return [
        "=== MNQ MICRO SCALP (paper / Lucid sim) ===",
        "Stack: completed 15m+5m+1m EMA8/21 same side",
        f"Trigger: quiet 1m pullback (body <= {BODY_MAX:g})",
        f"Stop: 1m ATR x{SL_ATR_MULT:g} clip {SL_MIN:g}-{SL_MAX:g} pts (broker SL)",
        f"TP: ${TP_DOLLARS:g} ({tp_pts:g} pts) — armed AFTER {MIN_HOLD_SEC:g}s hold",
        f"Time stop: {MAX_HOLD_SEC:g}s | max {MAX_TRADES_DAY}/day | cooldown {COOLDOWN_SEC:g}s",
        "1 MNQ | flatten leftover 15:50 ET | not funded live",
    ]


def _stop_pts(a: Dict[str, Any], i: int) -> float:
    atr1 = float(a["atr1"][i])
    if atr1 != atr1 or atr1 <= 0:
        return float("nan")
    return sl_pts_from_atr(atr1, mult=SL_ATR_MULT, sl_min=SL_MIN, sl_max=SL_MAX)


def check_entry(
    df_1m: pd.DataFrame,
    state: MicroScalpState,
    *,
    now: Optional[datetime] = None,
    open_count: int = 0,
) -> Tuple[Optional[Dict[str, Any]], MicroScalpState]:
    """Return entry signal dict or None. Uses completed bars only."""
    now = now or datetime.now(pytz.UTC)
    state = reset_day_if_needed(state, now)
    if open_count > 0 or state.open_side != 0:
        return None, state
    if not is_rth_scan_et(now):
        return None, state
    if state.trades_today >= MAX_TRADES_DAY:
        return None, state
    if state.last_exit_ts and (time.time() - state.last_exit_ts) < COOLDOWN_SEC:
        return None, state
    if df_1m is None or len(df_1m) < 80:
        return None, state

    a = prepare_arrays(df_1m)
    i = len(a["cl"]) - 1
    if i < 2:
        return None, state
    if int(a["dow"][i]) >= 5:
        return None, state
    m = int(a["mins"][i])
    if m < ENTRY_MINUTE or m >= FLAT_MINUTE:
        return None, state

    side = stack_side_15_5_1(i, a)
    if side == 0:
        return None, state
    if not entry_pullback(side, i, a):
        return None, state

    bar_ts = pd.Timestamp(a["dt"][i])
    if state.last_signal_bar is not None and bar_ts == state.last_signal_bar:
        return None, state

    sl_pts = _stop_pts(a, i)
    if sl_pts != sl_pts or sl_pts <= 0:
        return None, state

    tp_pts = tp_pts_from_dollars()
    entry = float(a["cl"][i])
    direction = "long" if side > 0 else "short"
    if side > 0:
        sl = entry - sl_pts
        tp = entry + tp_pts
    else:
        sl = entry + sl_pts
        tp = entry - tp_pts

    state.last_signal_bar = bar_ts
    return {
        "direction": direction,
        "side": side,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "sl_pts": float(sl_pts),
        "tp_pts": float(tp_pts),
        "tp_dollars": float(TP_DOLLARS),
        "min_hold_sec": float(MIN_HOLD_SEC),
        "max_hold_sec": float(MAX_HOLD_SEC),
        "session": SESSION_NAME,
        "reason": "15_5_1_quiet_pullback",
    }, state


def unrealized_pnl_usd(
    direction: str,
    entry: float,
    last: float,
    *,
    lots: int = 1,
    point_value: float = POINT_VALUE,
) -> float:
    pts = (last - entry) if direction == "long" else (entry - last)
    return float(pts) * point_value * lots


def check_quote_exit(
    pos: Dict[str, Any],
    last: float,
    *,
    held_sec: float,
    flatten_eod: bool = False,
) -> Optional[Tuple[str, float]]:
    """
    Manage open scalp on live quotes.
    Stop always live. $TP armed only after min_hold_sec.
    """
    direction = str(pos["direction"])
    entry = float(pos["entry"])
    sl = float(pos["sl"])
    tp_dollars = float(pos.get("tp_dollars", TP_DOLLARS))
    min_hold = float(pos.get("min_hold_sec", MIN_HOLD_SEC))
    max_hold = float(pos.get("max_hold_sec", MAX_HOLD_SEC))

    if flatten_eod:
        return "flat_eod", float(last)

    if direction == "long":
        if last <= sl:
            return "stop", float(sl)
    else:
        if last >= sl:
            return "stop", float(sl)

    if held_sec >= max_hold:
        return "time", float(last)

    if held_sec >= min_hold:
        pnl = unrealized_pnl_usd(direction, entry, last)
        if pnl >= tp_dollars:
            return "target", float(last)

    return None


def on_closed(state: MicroScalpState) -> MicroScalpState:
    state.trades_today += 1
    state.last_exit_ts = time.time()
    state.open_side = 0
    return state
