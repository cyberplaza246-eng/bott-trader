"""MNQ micro-scalp: min hold >10s, take profit from $20 up, hard stop."""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

# MNQ = $2 per point
MNQ_POINT_VALUE = float(os.getenv("MICRO_SCALP20_POINT_VALUE", "2.0"))
TP_USD = float(os.getenv("MICRO_SCALP20_TP_USD", "20.0"))
TP_PTS = TP_USD / MNQ_POINT_VALUE  # 10 pts at $2
SL_PTS = float(os.getenv("MICRO_SCALP20_SL_PTS", "20.0"))
MIN_HOLD_SEC = float(os.getenv("MICRO_SCALP20_MIN_HOLD_SEC", "10.0"))
MAX_HOLD_SEC = float(os.getenv("MICRO_SCALP20_MAX_HOLD_SEC", "180.0"))
COOLDOWN_SEC = float(os.getenv("MICRO_SCALP20_COOLDOWN_SEC", "15.0"))
LOOKBACK_SEC = float(os.getenv("MICRO_SCALP20_LOOKBACK_SEC", "25.0"))
IMPULSE_PTS = float(os.getenv("MICRO_SCALP20_IMPULSE_PTS", "2.5"))
EMA_FAST = int(os.getenv("MICRO_SCALP20_EMA_FAST", "5"))
EMA_SLOW = int(os.getenv("MICRO_SCALP20_EMA_SLOW", "12"))
# Legacy far placeholder (compat only). Real broker TP uses BROKER_TP_PTS.
FAR_TP_PTS = float(os.getenv("MICRO_SCALP20_FAR_TP_PTS", "500.0"))
# Real Lucid take-profit distance — must match $20 target so a failed software
# flatten cannot leave you long into a loser.
BROKER_TP_PTS = float(os.getenv("MICRO_SCALP20_BROKER_TP_PTS", str(TP_PTS)))


def unrealized_usd(direction: str, entry: float, last: float) -> float:
    pts = (last - entry) if direction == "long" else (entry - last)
    return pts * MNQ_POINT_VALUE


def real_tp_price(direction: str, fill: float, tp_pts: float = BROKER_TP_PTS) -> float:
    return fill + tp_pts if direction == "long" else fill - tp_pts


def brackets_from_fill(direction: str, fill: float, sl_pts: float = SL_PTS) -> Tuple[float, float]:
    """Hard stop + real $20-class TP on the broker (visible LIMIT)."""
    tp = real_tp_price(direction, fill)
    if direction == "long":
        return fill - sl_pts, tp
    return fill + sl_pts, tp


@dataclass
class TickBuffer:
    max_age_sec: float = 120.0
    ticks: Deque[Tuple[float, float]] = field(default_factory=deque)

    def push(self, last: float, ts: Optional[float] = None) -> None:
        if last <= 0:
            return
        now = time.time() if ts is None else float(ts)
        self.ticks.append((now, float(last)))
        cutoff = now - self.max_age_sec
        while self.ticks and self.ticks[0][0] < cutoff:
            self.ticks.popleft()

    def last(self) -> Optional[float]:
        return self.ticks[-1][1] if self.ticks else None

    def impulse_pts(self, lookback_sec: float = LOOKBACK_SEC) -> float:
        if len(self.ticks) < 2:
            return 0.0
        now = self.ticks[-1][0]
        px_now = self.ticks[-1][1]
        px_then = None
        for ts, px in reversed(self.ticks):
            if now - ts >= lookback_sec:
                px_then = px
                break
        if px_then is None:
            px_then = self.ticks[0][1]
        return px_now - px_then

    def ema_pair(self, fast: int = EMA_FAST, slow: int = EMA_SLOW) -> Tuple[Optional[float], Optional[float]]:
        if len(self.ticks) < slow:
            return None, None
        prices = [px for _, px in self.ticks]

        def ema(n: int) -> float:
            k = 2.0 / (n + 1.0)
            v = prices[0]
            for p in prices[1:]:
                v = p * k + v * (1.0 - k)
            return v

        return ema(fast), ema(slow)


@dataclass
class OpenPos:
    direction: str
    entry: float
    sl: float
    tp_far: float  # broker take-profit price (real $20 target)
    entry_ts: float
    ticket: str = ""
    exit_pending: bool = False  # True until Lucid confirms flat
    exit_reason: str = ""


@dataclass
class ScalpState:
    buf: TickBuffer = field(default_factory=TickBuffer)
    open: Optional[OpenPos] = None
    cooldown_until: float = 0.0
    last_exit_reason: str = ""


def recipe_banner_lines() -> List[str]:
    return [
        "=" * 60,
        "  MNQ MICRO SCALP $20  (paper / Lucid TEST)",
        f"  Min hold {MIN_HOLD_SEC:.0f}s then TP ${TP_USD:.0f}+ ({TP_PTS:.0f} pts)",
        f"  Hard stop {SL_PTS:.0f} pts (~${SL_PTS * MNQ_POINT_VALUE:.0f}) from fill",
        f"  Entry: {LOOKBACK_SEC:.0f}s impulse >= {IMPULSE_PTS:.1f} pts + EMA{EMA_FAST}/{EMA_SLOW} align",
        "  Lucid TEST: SL=STOP / TP=LIMIT working orders (sim has no SL/TP labels)",
        "  One position · Lucid sim · not funded live",
        "=" * 60,
    ]


def rth_only_enabled() -> bool:
    return str(os.getenv("MICRO_SCALP20_RTH_ONLY", "false")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def in_rth_et(now_et_hour_min: Tuple[int, int]) -> bool:
    h, m = now_et_hour_min
    mins = h * 60 + m
    return (9 * 60 + 35) <= mins < (15 * 60 + 50)


def check_entry(state: ScalpState, now: Optional[float] = None) -> Optional[Dict]:
    """Return signal dict or None."""
    if state.open is not None:
        return None
    ts = time.time() if now is None else float(now)
    if ts < state.cooldown_until:
        return None
    last = state.buf.last()
    if last is None:
        return None
    impulse = state.buf.impulse_pts()
    fast, slow = state.buf.ema_pair()
    if fast is None or slow is None:
        return None
    direction = None
    if impulse >= IMPULSE_PTS and fast > slow:
        direction = "long"
    elif impulse <= -IMPULSE_PTS and fast < slow:
        direction = "short"
    if not direction:
        return None
    sl, tp_far = brackets_from_fill(direction, last)
    return {
        "direction": direction,
        "entry": last,
        "sl": sl,
        "tp_far": tp_far,
        "sl_pts": SL_PTS,
        "tp_usd": TP_USD,
        "impulse": impulse,
        "ema_fast": fast,
        "ema_slow": slow,
    }


def check_exit(pos: OpenPos, last: float, now: Optional[float] = None) -> Optional[Tuple[str, float]]:
    """
    Returns (reason, exit_px) or None.
    TP only after MIN_HOLD_SEC and unrealized >= TP_USD.
    SL always live. MAX_HOLD forces flat.
    """
    ts = time.time() if now is None else float(now)
    hold = ts - pos.entry_ts
    # Stop first (always)
    if pos.direction == "long" and last <= pos.sl:
        return "SL", pos.sl
    if pos.direction == "short" and last >= pos.sl:
        return "SL", pos.sl
    # Max hold
    if hold >= MAX_HOLD_SEC:
        return "MAX_HOLD", last
    # Take profit only after min hold
    if hold >= MIN_HOLD_SEC:
        u = unrealized_usd(pos.direction, pos.entry, last)
        if u >= TP_USD:
            return "TP", last
    return None


def on_closed(state: ScalpState, reason: str, now: Optional[float] = None) -> None:
    ts = time.time() if now is None else float(now)
    state.open = None
    state.last_exit_reason = reason
    state.cooldown_until = ts + COOLDOWN_SEC
